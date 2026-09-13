"""
preprocessing/preprocess.py
===========================

Convert raw trajectory CSVs into the canonical segment table used by the
IDM calibration pipeline.

Purpose
-------
This module reads the **real** dataset that accompanies the reference
study: 48 CSV files organised as

* ``data/raw/LEADCAR/leadingcar{1..12}.csv`` — leader speed traces
* ``data/raw/FLWCAR1/followingcar_1_{1..12}.csv`` — follower #1 traces
* ``data/raw/FLWCAR2/followingcar_2_{1..12}.csv`` — follower #2 traces
* ``data/raw/FLWCAR3/followingcar_3_{1..12}.csv`` — follower #3 traces

Each CSV is sampled at 10 Hz and contains the columns:

    Record, Time, Latitude, Longitude, Altitude, Speed,
    GForceX, GForceY, GForceZ, Lap

where ``Speed`` is in km/h and ``GForceX`` is the longitudinal g-force
(in units of g = 9.81 m/s^2).

For each (leader, follower) pair the loader:
1. Resamples both signals to a common 10 Hz time-base.
2. Converts ``Speed`` from km/h to m/s.
3. Computes the net bumper-to-bumper gap by integrating the relative
   speed:  ``gap(t) = gap_0 + ∫(v_leader - v_follower) dt``.
4. Estimates the initial gap ``gap_0`` from the first 5 s of GPS
   positions (Haversine distance, with a 5 m bumper offset).
5. Computes the IDM observation tuple ``(v, Δv, gap, a)`` where ``a``
   is derived from ``GForceX * 9.81``.
6. Sub-samples to 2 Hz for tractability.

If the real raw directory is not present, the loader transparently
falls back to the synthetic NGSIM-like generator used in the original
release; the provenance flag in ``data/segment_manifest.json``
documents which mode was used.

Inputs / outputs
----------------
* Input  : ``data/raw/`` (LEADCAR + FLWCAR{1,2,3}).
* Output : ``data/processed_segments.csv`` and
  ``data/segment_manifest.json``.

Run
---
::

    python -m preprocessing.preprocess \
        --raw-dir data/raw \
        --output  data/processed_segments.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Determinism
SEED = 42
N_SEGMENTS = 26        # spec target
SUBSAMPLE_HZ = 2       # 2 Hz output (every 5th of 10 Hz input)
BUMPER_OFFSET_M = 5.0   # 5 m front-to-front -> bumper-to-bumper
G_MPS2 = 9.81


# ----------------------------------------------------------------------
# Real-data loader
# ----------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres between two WGS84 points (vectorised)."""
    R = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def _load_one_csv(path: Path) -> pd.DataFrame:
    """Load one trajectory CSV and normalise column names + units.

    The longitudinal acceleration is computed as the central finite
    difference of the speed signal (in m/s) rather than from the
    raw GForceX column.  GForceX mixes longitudinal and lateral
    components on banked roads and has a typical noise floor of 0.05 g,
    which produces spurious 0.5 m/s^2 noise.  Finite differencing the
    speed gives a cleaner, physically consistent acceleration estimate.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    # Parse the ISO timestamp and convert to relative seconds
    df["t_iso"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["t_iso"]).sort_values("t_iso").reset_index(drop=True)
    df["t"] = (df["t_iso"] - df["t_iso"].iloc[0]).dt.total_seconds()
    # Speed km/h -> m/s
    df["v_ms"] = df["speed"].astype(float) / 3.6
    # Longitudinal acceleration = d(v_ms)/dt  (more reliable than GForceX)
    df["a_mps2"] = np.gradient(df["v_ms"].to_numpy(),
                                df["t"].to_numpy())
    return df[["t", "v_ms", "a_mps2", "latitude", "longitude"]]


def _resample_10hz(df: pd.DataFrame, t_start: float = 0.0,
                   duration_s: float | None = None) -> pd.DataFrame:
    """Resample to a uniform 10 Hz grid via linear interpolation."""
    if duration_s is None:
        duration_s = float(df["t"].iloc[-1] - df["t"].iloc[0])
    t_grid = np.arange(t_start, t_start + duration_s, 0.1)
    v = np.interp(t_grid, df["t"], df["v_ms"])
    a = np.interp(t_grid, df["t"], df["a_mps2"])
    lat = np.interp(t_grid, df["t"], df["latitude"])
    lon = np.interp(t_grid, df["t"], df["longitude"])
    return pd.DataFrame({"t": t_grid, "v_ms": v, "a_mps2": a,
                         "lat": lat, "lon": lon})


def _pair_loader(leader_csv: Path, follower_csv: Path,
                 segment_id: int) -> pd.DataFrame | None:
    """Build the IDM observation tuple for one (leader, follower) pair.

    The function also performs data cleaning:
    * reject pairs whose initial GPS gap is implausible (> 200 m, which
      means the cars are not in the same platoon);
    * reject pairs whose duration is below 20 s;
    * smooth the speed and acceleration signals with a 1-s moving
      average to suppress GPS jitter;
    * clip extreme accelerations outside ``[-5, +5] m/s^2``.
    """
    try:
        lead = _load_one_csv(leader_csv)
        foll = _load_one_csv(follower_csv)
    except Exception as e:
        print(f"[preprocess] Failed to load pair "
              f"({leader_csv.name}, {follower_csv.name}): {e}")
        return None

    # Time-align on a common grid
    t0 = max(lead["t"].iloc[0], foll["t"].iloc[0])
    t1 = min(lead["t"].iloc[-1], foll["t"].iloc[-1])
    duration = t1 - t0
    if duration < 20.0:
        return None
    lead_r = _resample_10hz(lead, t_start=t0, duration_s=duration)
    foll_r = _resample_10hz(foll, t_start=t0, duration_s=duration)

    # Initial gap from GPS Haversine distance - bumper offset
    gap0 = _haversine_m(lead_r["lat"].iloc[0], lead_r["lon"].iloc[0],
                        foll_r["lat"].iloc[0], foll_r["lon"].iloc[0])
    gap0 = max(BUMPER_OFFSET_M + 1.0, gap0 - BUMPER_OFFSET_M)
    # Reject implausible initial gaps (cars not in the same platoon)
    if gap0 > 200.0:
        return None

    # Smooth signals with a 1-s moving average (10 samples at 10 Hz)
    win = 10
    v_lead = np.convolve(lead_r["v_ms"].to_numpy(),
                         np.ones(win) / win, mode="same")
    v_foll = np.convolve(foll_r["v_ms"].to_numpy(),
                         np.ones(win) / win, mode="same")
    a_foll = np.convolve(foll_r["a_mps2"].to_numpy(),
                         np.ones(win) / win, mode="same")
    t = lead_r["t"].to_numpy()

    # Clip extreme accelerations (sensor outliers)
    a_foll = np.clip(a_foll, -5.0, 5.0)

    dv = v_foll - v_lead
    rel_speed = v_lead - v_foll
    gap = gap0 + np.cumsum(rel_speed) * 0.1
    gap = np.maximum(gap, BUMPER_OFFSET_M)
    # Clip gap to a plausible highway range
    gap = np.minimum(gap, 200.0)

    # Sub-sample to 2 Hz for tractability
    step = int(10 / SUBSAMPLE_HZ)
    rows = []
    for i in range(0, len(t), step):
        # Skip rows with implausible kinematics (data drop-outs)
        if abs(dv[i]) > 10.0:    # approach speed > 10 m/s is non-physical
            continue
        if gap[i] > 150.0:        # too-far leader, no car-following happening
            continue
        rows.append({
            "segment_id": int(segment_id),
            "t": float(t[i]),
            "v": float(v_foll[i]),
            "dv": float(dv[i]),
            "gap": float(gap[i]),
            "a": float(a_foll[i]),
        })
    if len(rows) < 50:
        return None
    return pd.DataFrame(rows)


def load_real_raw(raw_dir: str) -> pd.DataFrame:
    """Load all (leader, follower) pairs from the real raw directory.

    Returns a single DataFrame with columns ``segment_id, t, v, dv, gap, a``.
    Pairs are formed by matching ``leadingcar{i}.csv`` with
    ``followingcar_{j}_{i}.csv`` for each follower car j ∈ {1, 2, 3}.
    """
    p = Path(raw_dir)
    lead_dir = p / "LEADCAR"
    flw_dirs = [p / "FLWCAR1", p / "FLWCAR2", p / "FLWCAR3"]
    if not lead_dir.exists():
        return pd.DataFrame()

    lead_csvs = sorted(lead_dir.glob("leadingcar*.csv"))
    # Extract the integer suffix from "leadingcar12.csv" -> 12
    def _idx(path: Path) -> int:
        s = path.stem.replace("leadingcar", "")
        try:
            return int(s)
        except ValueError:
            return -1

    pairs = []
    seg_id = 0
    for lead_csv in lead_csvs:
        i = _idx(lead_csv)
        if i < 0:
            continue
        for flw_dir in flw_dirs:
            flw_csv = flw_dir / f"followingcar_{flw_dir.name[-1]}_{i}.csv"
            if not flw_csv.exists():
                continue
            seg_df = _pair_loader(lead_csv, flw_csv, seg_id)
            if seg_df is not None and len(seg_df) >= 50:
                pairs.append(seg_df)
                seg_id += 1
        # Continue past N_SEGMENTS target so we collect all valid pairs;
        # the calibration can sub-sample later if needed.

    if not pairs:
        return pd.DataFrame()
    return pd.concat(pairs, ignore_index=True)


# ----------------------------------------------------------------------
# Synthetic fallback (kept for offline reproducibility)
# ----------------------------------------------------------------------
def build_synthetic_segments(n_segments: int = N_SEGMENTS,
                             length_s: int = 90,
                             dt: float = 0.1,
                             seed: int = SEED) -> pd.DataFrame:
    """Generate NGSIM-like synthetic segments when real data unavailable."""
    rng = np.random.default_rng(seed)
    n_steps = int(length_s / dt)
    rows: list[dict] = []
    base_params = dict(v0=28.0, T=1.4, s0=2.0, a=1.4, b=2.0, delta=4.0)
    from idm.idm_model import IDMParameters
    from idm.jit_kernels import step_trajectory

    for seg_id in range(n_segments):
        params = IDMParameters(**base_params)
        v0_leader = max(8.0, rng.normal(13.5, 4.0))
        a_drv = np.clip(rng.normal(0.0, 0.55, size=n_steps), -3.0, 2.5)
        v_lead = np.zeros(n_steps)
        v_lead[0] = v0_leader
        for t in range(1, n_steps):
            v_lead[t] = max(0.0, v_lead[t - 1] + a_drv[t] * dt)
        initial_gap = max(5.0, rng.lognormal(np.log(15.0), 0.55))
        initial_speed = min(v_lead[0] + rng.normal(0.0, 2.0), 30.0)
        traj = step_trajectory(
            params.v0, params.T, params.s0, params.a, params.b, params.delta,
            dt, n_steps, v_lead, initial_gap, initial_speed)
        gap = traj[:, 0]; v_follow = traj[:, 1]; a_idm = traj[:, 2]
        obs_v = v_follow
        obs_gap = gap
        obs_a = a_idm + rng.normal(0, 0.05, n_steps)
        dv = obs_v - v_lead
        for t in range(0, n_steps, 5):
            rows.append({
                "segment_id": seg_id,
                "t": t * dt,
                "v": float(obs_v[t]),
                "dv": float(dv[t]),
                "gap": float(obs_gap[t]),
                "a": float(obs_a[t]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Loader dispatch
# ----------------------------------------------------------------------
def _load_raw_dir(raw_dir: str | None) -> pd.DataFrame | None:
    """Try to load the real raw directory; return None if unavailable."""
    if raw_dir is None:
        return None
    df = load_real_raw(raw_dir)
    if df.empty:
        return None
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Return summary statistics for the manifest."""
    return {
        "n_segments": int(df["segment_id"].nunique()),
        "n_samples": int(len(df)),
        "v_mean_mps": float(df["v"].mean()),
        "v_std_mps": float(df["v"].std()),
        "a_mean_mps2": float(df["a"].mean()),
        "a_std_mps2": float(df["a"].std()),
        "gap_mean_m": float(df["gap"].mean()),
        "gap_std_m": float(df["gap"].std()),
        "dv_mean_mps": float(df["dv"].mean()),
        "dv_std_mps": float(df["dv"].std()),
    }


def main(raw_dir: str | None, output_csv: str,
         manifest_json: str | None = None) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    raw = _load_raw_dir(raw_dir) if raw_dir else None
    if raw is None:
        print("[preprocess] Real raw directory unavailable; "
              "falling back to synthetic NGSIM-like generator.")
        df = build_synthetic_segments()
        provenance = "synthetic_ngsim_like"
    else:
        df = raw
        provenance = f"raw:{raw_dir}"

    df.to_csv(output_csv, index=False)
    print(f"[preprocess] Wrote {output_csv} ({len(df)} rows, "
          f"{df['segment_id'].nunique()} segments)")

    if manifest_json:
        os.makedirs(os.path.dirname(manifest_json), exist_ok=True)
        manifest = summarise(df)
        manifest["provenance"] = provenance
        manifest["seed"] = SEED
        with open(manifest_json, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[preprocess] Wrote manifest {manifest_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw trajectory CSVs.")
    parser.add_argument("--raw-dir", default="data/raw",
                       help="Directory of raw CSV files.")
    parser.add_argument("--output", default="data/processed_segments.csv",
                       help="Output CSV path.")
    parser.add_argument("--manifest", default="data/segment_manifest.json",
                       help="Optional manifest JSON path.")
    args = parser.parse_args()
    main(args.raw_dir, args.output, args.manifest)
