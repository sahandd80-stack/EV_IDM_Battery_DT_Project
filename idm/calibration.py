"""
idm/calibration.py
==================

Maximum-likelihood calibration of the IDM parameters against real
vehicle trajectory data.

Purpose
-------
Take a pre-processed trajectory table (follower speed ``v``,
approach speed ``dv``, gap ``s``, observed acceleration ``a``) and
search for the IDM parameter vector that minimises the RMSE between
the modelled and observed accelerations.  The optimiser is a
bounded Nelder-Mead search seeded from a highway-style default,
followed by a fine polish with L-BFGS-B.

Inputs / outputs
----------------
The default entry point reads a CSV file produced by
``preprocessing/preprocess.py`` and writes ``results/calibration_results.json``.

Run
---
::

    python -m idm.calibration \
        --input data/processed_segments.csv \
        --output results/calibration_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Deterministic seed for calibration, as required by the spec.
SEED = 42

from .idm_model import IDMModel, IDMParameters


# Bounds used by L-BFGS-B (highway-driving style, Kia Soul EV)
# Tighter bounds to prevent degenerate solutions on noisy real data.
DEFAULT_BOUNDS = [
    (15.0, 45.0),  # v0 (m/s)   -- 54-162 km/h (real data has v_mean ~ 18)
    (0.8, 3.0),    # T  (s)    -- safe time gap, must be positive
    (0.5, 5.0),    # s0 (m)
    (0.5, 3.0),    # a  (m/s^2)
    (1.0, 5.0),    # b  (m/s^2)
    (3.0, 5.0),    # delta
]


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def pearson_corr(pred: np.ndarray, obs: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if pred.std() < 1e-9 or obs.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(pred, obs)[0, 1])


def objective(theta: np.ndarray,
              v_obs: np.ndarray,
              dv_obs: np.ndarray,
              gap_obs: np.ndarray,
              a_obs: np.ndarray,
              bounds=DEFAULT_BOUNDS) -> float:
    """Mean-squared-error objective with a quadratic penalty on bound
    violations.  The penalty is essential because Nelder-Mead is an
    unbounded optimiser; without it the algorithm explores physically
    implausible regions (e.g. negative time gap).
    """
    # Quadratic penalty for out-of-bounds parameters
    penalty = 0.0
    for i, (lo, hi) in enumerate(bounds):
        if theta[i] < lo:
            penalty += (lo - theta[i]) ** 2 * 1e3
        elif theta[i] > hi:
            penalty += (theta[i] - hi) ** 2 * 1e3
    params = IDMParameters.from_array(theta)
    model = IDMModel(params)
    pred = model.acceleration_batch(v_obs, dv_obs, gap_obs)
    return float(np.mean((pred - a_obs) ** 2)) + penalty


def calibrate(df: pd.DataFrame,
              bounds=DEFAULT_BOUNDS,
              seed: int = SEED) -> dict:
    """Calibrate IDM parameters against the observation table.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``v``, ``dv``, ``gap``, ``a``.
    bounds : list of tuple
        Parameter bounds for L-BFGS-B.
    seed : int
        Random seed used to perturb the initial guess.

    Returns
    -------
    dict
        Dictionary with keys ``params``, ``rmse``, ``corr``,
        ``n_samples``, ``seed``.
    """
    rng = np.random.default_rng(seed)
    v_obs = df["v"].to_numpy(dtype=np.float64)
    dv_obs = df["dv"].to_numpy(dtype=np.float64)
    gap_obs = df["gap"].to_numpy(dtype=np.float64)
    a_obs = df["a"].to_numpy(dtype=np.float64)

    # Initial guess -- use the empirical mean speed as v0 (better starting
    # point than the highway default when the data has been collected on
    # urban / arterial roads with v_mean < 30 m/s).
    v0_init = float(np.clip(np.percentile(v_obs, 95), 15.0, 45.0))
    x0 = np.array([v0_init, 1.5, 2.0, 1.4, 2.0, 4.0])
    # Small perturbation seeded by the calibration seed
    x0 = x0 + rng.normal(0.0, 0.05, size=x0.shape[0])
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Stage 1: differential evolution for a robust global search
    from scipy.optimize import differential_evolution
    res_de = differential_evolution(
        objective, bounds=bounds,
        args=(v_obs, dv_obs, gap_obs, a_obs),
        seed=seed, maxiter=80, popsize=15, tol=1e-8,
        mutation=(0.5, 1.0), recombination=0.7,
        polish=False, workers=1)

    # Stage 2: Nelder-Mead polish starting from DE solution
    res_nm = minimize(objective, res_de.x,
                     args=(v_obs, dv_obs, gap_obs, a_obs),
                     method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-9,
                              "maxiter": 4000, "maxfev": 4000})
    # Stage 3: L-BFGS-B bounded polish
    res_lb = minimize(objective, res_nm.x,
                     args=(v_obs, dv_obs, gap_obs, a_obs),
                     method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 1000, "ftol": 1e-12})

    candidates = [res_de, res_nm, res_lb]
    theta_best = min(candidates, key=lambda r: r.fun).x

    params = IDMParameters.from_array(theta_best)
    model = IDMModel(params)
    pred = model.acceleration_batch(v_obs, dv_obs, gap_obs)
    calib_rmse = rmse(pred, a_obs)
    calib_corr = pearson_corr(pred, a_obs)

    # Trajectory-based speed RMSE: simulate the IDM forward for each
    # segment and compare the simulated follower speed to the observed
    # follower speed.  This is the "training speed RMSE" reported in the
    # reference paper.
    speed_rmse_all = []
    speed_pred_all = []
    speed_obs_all = []
    for seg_id, seg_df in df.groupby("segment_id"):
        if len(seg_df) < 10:
            continue
        v0_seg = seg_df["v"].iloc[0]
        gap0 = seg_df["gap"].iloc[0]
        # Reconstruct leader speed from follower + dv
        v_lead_seg = seg_df["v"].to_numpy() - seg_df["dv"].to_numpy()
        # Use the calibrated model to simulate forward
        traj = model.simulate(v_lead_seg,
                              initial_gap=gap0,
                              initial_speed=v0_seg,
                              dt=0.5)  # data is at 2 Hz (sub-sampled at every 5 steps of 0.1)
        speed_pred_all.extend(traj[:, 1].tolist())
        speed_obs_all.extend(seg_df["v"].to_numpy().tolist())
        speed_rmse_all.append(rmse(traj[:, 1], seg_df["v"].to_numpy()))
    speed_rmse = float(np.mean(speed_rmse_all)) if speed_rmse_all else calib_rmse
    speed_corr = pearson_corr(np.array(speed_pred_all), np.array(speed_obs_all))

    return {
        "params": params.to_dict(),
        "rmse_mps": calib_rmse,
        "pearson_r": calib_corr,
        "speed_rmse_mps": speed_rmse,
        "speed_pearson_r": speed_corr,
        "n_samples": int(v_obs.shape[0]),
        "seed": int(seed),
        "de_fun": float(res_de.fun),
        "nelder_mead_fun": float(res_nm.fun),
        "lbfgsb_fun": float(res_lb.fun),
        "bounds": bounds,
    }


def main(input_csv: str, output_json: str) -> None:
    df = pd.read_csv(input_csv)
    print(f"[calibrate] Loaded {len(df)} observations from {input_csv}")
    results = calibrate(df)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[calibrate] Wrote {output_json}")
    print(f"[calibrate] RMSE = {results['rmse_mps']:.4f} m/s  "
          f"r = {results['pearson_r']:.4f}")
    print(f"[calibrate] params = {results['params']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate IDM parameters.")
    parser.add_argument("--input", required=True,
                       help="Path to processed segments CSV.")
    parser.add_argument("--output", default="results/calibration_results.json",
                       help="Output JSON file.")
    args = parser.parse_args()
    main(args.input, args.output)
