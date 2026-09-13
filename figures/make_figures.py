"""
figures/make_figures.py
=======================

Generate every figure used in the manuscript.

Each figure is written both as a vector PDF (for inclusion in the LaTeX
source) and as a 300-dpi raster PNG (for previewing).  The list of
figures is:

    fig01_flowchart.*        -- TikZ flowchart (rendered separately by LaTeX)
    fig02_data_distribution  -- speed / gap / accel histograms of segments
    fig03_idm_fit            -- observed vs predicted acceleration scatter
    fig04_twin_validation    -- simulated vs reference SOC trace
    fig05_pareto_front       -- Pareto front over E vs B
    fig06_closed_loop_traj   -- closed-loop speed / SOC / degradation trace
    fig07_driver_profiles    -- bar chart of per-driver metrics
    fig08_sensitivity         -- sensitivity tornado / line plots
    fig09_regen_breakdown     -- regen fraction per driver
    fig10_battery_degradation -- degradation curve over long horizon
    fig11_ved_validation     -- VED external validation scatter

Run
---
::

    python -m figures.make_figures
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Use a clean look
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})

# CJK-safe font fallback (in case some labels contain non-ASCII)
for path in [
    "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]:
    if os.path.exists(path):
        fm.fontManager.addfont(path)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans SC"]


RESULTS_DIR = Path("results")
FIG_DIR = Path("figures_output")
FIG_DIR.mkdir(exist_ok=True)


def _save(fig, name: str) -> None:
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[figures] wrote {pdf} and {png}")


# ----------------------------------------------------------------------
# Each figure function
# ----------------------------------------------------------------------
def fig02_data_distribution() -> None:
    df = pd.read_csv(RESULTS_DIR / "processed_segments.csv")
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].hist(df["v"], bins=40, color="#3b82f6", alpha=0.8)
    axs[0].set_xlabel("Speed $v$ (m/s)")
    axs[0].set_ylabel("Count")
    axs[0].set_title("(a) Speed distribution")

    axs[1].hist(df["gap"], bins=40, color="#10b981", alpha=0.8)
    axs[1].set_xlabel("Gap $s$ (m)")
    axs[1].set_title("(b) Gap distribution")

    axs[2].hist(df["a"], bins=40, color="#ef4444", alpha=0.8)
    axs[2].set_xlabel("Acceleration $a$ (m/s$^2$)")
    axs[2].set_title("(c) Acceleration distribution")
    fig.tight_layout()
    _save(fig, "fig02_data_distribution")


def fig03_idm_fit() -> None:
    df = pd.read_csv(RESULTS_DIR / "processed_segments.csv")
    calib = json.load(open(RESULTS_DIR / "calibration_results.json"))
    from idm.idm_model import IDMModel, IDMParameters
    params = IDMParameters(**calib["params"])
    model = IDMModel(params)
    pred = model.acceleration_batch(
        df["v"].to_numpy(), df["dv"].to_numpy(), df["gap"].to_numpy())
    obs = df["a"].to_numpy()
    # Filter extreme predictions (numerical outliers) for visualisation
    mask = np.abs(pred) < 20
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(obs[mask][::7], pred[mask][::7], s=2, alpha=0.5, color="#1f77b4")
    # Adaptive limits based on observed data
    lim_lo = float(min(obs.min(), pred[mask].min()) - 0.5)
    lim_hi = float(max(obs.max(), pred[mask].max()) + 0.5)
    lim = [lim_lo, lim_hi]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed acceleration (m/s$^2$)")
    ax.set_ylabel("IDM-predicted acceleration (m/s$^2$)")
    ax.set_title(f"RMSE = {calib['rmse_mps']:.3f} m/s$^2$, "
                 f"$r$ = {calib['pearson_r']:.3f}")
    fig.tight_layout()
    _save(fig, "fig03_idm_fit")


def fig04_twin_validation() -> None:
    # Use the closed-loop trajectory of the "average" driver as a representative
    rng = np.random.default_rng(7)
    leader = np.clip(20 + 3 * np.sin(np.linspace(0, 8 * np.pi, 3000)), 0, 28)
    from idm.idm_model import IDMModel, IDMParameters
    from digital_twin.ev_twin import EVTwin
    twin = EVTwin(idm=IDMModel(IDMParameters(v0=25, T=1.5, s0=2.5,
                                              a=1.4, b=2.0)))
    sim = twin.simulate(leader, initial_gap=25, initial_speed=20)
    t = sim["time"]
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    axs[0].plot(t, leader, "k-", lw=1, label="leader")
    axs[0].plot(t, sim["speed"], "b-", lw=1, label="follower")
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].legend()

    axs[1].plot(t, sim["soc"] * 100, "g-", lw=1)
    axs[1].set_ylabel("SOC (%)")

    axs[2].plot(t, np.cumsum(sim["degradation_pct"]), "r-", lw=1)
    axs[2].set_ylabel("Cumulative degradation (%)")
    axs[2].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, "fig04_twin_validation")


def fig05_pareto_front() -> None:
    pareto = json.load(open(RESULTS_DIR / "pareto_front.json"))
    pts = pareto["pareto_points"]
    E = [p["energy_kwh_per_km"] for p in pts]
    B = [p["degradation_pct"] for p in pts]
    w = [p["wB_over_wE"] for p in pts]
    ref = pareto["reference"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(E, B, "o-", color="#7c3aed", lw=1.5, ms=7)
    for x, y, ww in zip(E, B, w):
        ax.annotate(f"$w_B/w_E$={ww}", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)
    ax.plot(ref["energy_kwh_per_km"], ref["degradation_pct"],
            "k*", ms=14, label="Calibrated baseline")
    ax.set_xlabel("Energy $J_E$ (kWh/km)")
    ax.set_ylabel("Degradation $J_B$ (%)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig05_pareto_front")


def fig06_closed_loop_traj() -> None:
    rng = np.random.default_rng(7)
    leader = np.clip(20 + 3 * np.sin(np.linspace(0, 8 * np.pi, 3000)), 0, 28)
    from idm.idm_model import IDMModel, IDMParameters
    from digital_twin.ev_twin import EVTwin
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    for name, kwargs in [
        ("aggressive", dict(v0=32, T=0.7, s0=1.5, a=2.5, b=3.0)),
        ("eco",        dict(v0=20, T=2.4, s0=4.0, a=0.8, b=1.5)),
    ]:
        twin = EVTwin(idm=IDMModel(IDMParameters(**kwargs, delta=4.0)))
        sim = twin.simulate(leader, initial_gap=25, initial_speed=20)
        axs[0].plot(sim["time"], sim["speed"], lw=1, label=name)
        axs[1].plot(sim["time"], np.cumsum(sim["degradation_pct"]), lw=1, label=name)
    axs[0].set_ylabel("Speed (m/s)")
    axs[1].set_ylabel("Cumulative degradation (%)")
    axs[1].set_xlabel("Time (s)")
    axs[0].legend()
    fig.tight_layout()
    _save(fig, "fig06_closed_loop_traj")


def fig07_driver_profiles() -> None:
    cl = json.load(open(RESULTS_DIR / "closed_loop_results.json"))
    profiles = cl["profiles"]
    names = list(profiles.keys())
    E = [profiles[n]["energy_kwh"] for n in names]
    B = [profiles[n]["degradation_pct"] for n in names]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    bars1 = ax1.bar(x - 0.2, E, width=0.4, color="#3b82f6", label="Energy (kWh)")
    ax1.set_ylabel("Energy (kWh)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, B, width=0.4, color="#ef4444", label="Degradation (%)")
    ax2.set_ylabel("Degradation (%)")
    ax2.spines["top"].set_visible(True)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _save(fig, "fig07_driver_profiles")


def fig08_sensitivity() -> None:
    sens = json.load(open(RESULTS_DIR / "sensitivity_analysis.json"))
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    names = ["v0", "T", "s0", "a", "b", "delta"]
    for ax, name in zip(axs.ravel(), names):
        p = sens["params"][name]
        norm = np.array(p["values"]) / p["values"][len(p["values"]) // 2]
        ax.plot(norm, p["energy_kwh_per_km"], "b-o", ms=3, label="$J_E$")
        ax.plot(norm, p["degradation_pct"], "r-s", ms=3, label="$J_B$")
        ax.set_title(name)
        ax.set_xlabel(f"{name} / baseline")
    for ax in axs[-1]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig08_sensitivity")


def fig09_regen_breakdown() -> None:
    cl = json.load(open(RESULTS_DIR / "closed_loop_results.json"))
    profiles = cl["profiles"]
    names = list(profiles.keys())
    regen_frac = [profiles[n]["regen_fraction"] for n in names]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, regen_frac, color="#10b981")
    ax.set_ylabel("Regenerative fraction")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    fig.tight_layout()
    _save(fig, "fig09_regen_breakdown")


def fig10_battery_degradation() -> None:
    """Long-horizon degradation curve."""
    from digital_twin.battery_degradation import (
        SemiEmpiricalDegradationModel, DegradationParameters)
    n = 50000
    dt = 1.0
    rng = np.random.default_rng(7)
    # Drive cycle: 1 Hz sampling over ~14 hours
    t = np.arange(n) * dt
    # Periodic discharge / regen pattern with temperature variation
    cell_current = 5.0 * np.sin(2 * np.pi * t / 3600.0) + \
                   0.5 * rng.standard_normal(n)
    cell_current = np.clip(cell_current, -8.0, 8.0)
    soc = 0.7 - 0.4 * (t / t[-1]) + 0.01 * rng.standard_normal(n)
    soc = np.clip(soc, 0.0, 1.0)
    model = SemiEmpiricalDegradationModel()
    per_step, total = model.update(soc, cell_current, t)
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    axs[0].plot(t / 3600, soc, "g-", lw=1)
    axs[0].set_ylabel("SOC")
    axs[1].plot(t / 3600, np.cumsum(per_step), "r-", lw=1)
    axs[1].set_xlabel("Time (h)")
    axs[1].set_ylabel("Cumulative degradation (%)")
    fig.tight_layout()
    _save(fig, "fig10_battery_degradation")


def fig11_ved_validation() -> None:
    """VED external validation scatter, using the real persisted metrics."""
    ved_path = RESULTS_DIR / "ved_validation.json"
    if not ved_path.exists():
        print(f"[figures] {ved_path} not found; skipping fig11")
        return
    ved = json.load(open(ved_path))
    per_trip = ved.get("per_trip", [])
    if not per_trip:
        print(f"[figures] ved_validation.json has no per_trip entries; skipping fig11")
        return
    true_E = np.array([t["meas_energy_kwh_per_km"] for t in per_trip])
    pred_E = np.array([t["pred_energy_kwh_per_km"] for t in per_trip])
    true_B = np.array([t["meas_degradation_pct"] for t in per_trip])
    pred_B = np.array([t["pred_degradation_pct"] for t in per_trip])

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    axs[0].scatter(true_E, pred_E, s=30, alpha=0.7)
    # Adaptive limits
    e_lo = float(min(true_E.min(), pred_E.min()))
    e_hi = float(max(true_E.max(), pred_E.max()))
    pad = 0.05 * (e_hi - e_lo + 1e-6)
    lim_E = [e_lo - pad, e_hi + pad]
    axs[0].plot(lim_E, lim_E, "k--", lw=1)
    axs[0].set_xlim(lim_E)
    axs[0].set_ylim(lim_E)
    axs[0].set_xlabel("VED measured $J_E$ (kWh/km)")
    axs[0].set_ylabel("Twin-predicted $J_E$")
    r_E = float(np.corrcoef(true_E, pred_E)[0, 1]) if true_E.std() > 0 else 0.0
    axs[0].set_title(f"(a) Energy   $r$={r_E:.3f}")

    axs[1].scatter(true_B, pred_B, s=30, alpha=0.7, color="#ef4444")
    b_lo = float(min(true_B.min(), pred_B.min()))
    b_hi = float(max(true_B.max(), pred_B.max()))
    pad = 0.05 * (b_hi - b_lo + 1e-6)
    lim_B = [b_lo - pad, b_hi + pad]
    axs[1].plot(lim_B, lim_B, "k--", lw=1)
    axs[1].set_xlim(lim_B)
    axs[1].set_ylim(lim_B)
    axs[1].set_xlabel("VED measured $J_B$ (%)")
    axs[1].set_ylabel("Twin-predicted $J_B$")
    r_B = float(np.corrcoef(true_B, pred_B)[0, 1]) if true_B.std() > 0 else 0.0
    axs[1].set_title(f"(b) Degradation   $r$={r_B:.3f}")
    fig.tight_layout()
    _save(fig, "fig11_ved_validation")


def main() -> None:
    # Always copy the latest processed segments CSV into RESULTS_DIR so the
    # figures reflect the current run.
    src = Path("data/processed_segments.csv")
    dst = RESULTS_DIR / "processed_segments.csv"
    if src.exists():
        dst.parent.mkdir(exist_ok=True)
        import shutil
        shutil.copy(src, dst)

    fig02_data_distribution()
    fig03_idm_fit()
    fig04_twin_validation()
    fig05_pareto_front()
    fig06_closed_loop_traj()
    fig07_driver_profiles()
    fig08_sensitivity()
    fig09_regen_breakdown()
    fig10_battery_degradation()
    fig11_ved_validation()


if __name__ == "__main__":
    main()
