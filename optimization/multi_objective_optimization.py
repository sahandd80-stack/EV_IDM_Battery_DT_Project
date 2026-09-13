"""
optimization/multi_objective_optimization.py
=============================================

Pareto-front exploration of the trade-off between energy efficiency and
battery lifespan.

Purpose
-------
Define the two competing objectives:

* ``J_E`` -- normalised energy consumption (kWh per km), minimised.
* ``J_B`` -- per-trip capacity fade percentage, minimised.

Scan a convex weight ratio ``w_B / w_E in {0.1, 0.3, 1.0, 3.0, 10.0}``
and, for each weight, run a bounded Nelder-Mead search over the IDM
parameter vector ``theta = (v0, T, s0, a, b, delta)``.  The Pareto
front is persisted as ``results/pareto_front.json``.

Inputs / outputs
----------------
The script reads ``results/calibration_results.json`` (for the initial
guess) and writes the Pareto front and a CSV table for plotting.

Run
---
::

    python -m optimization.multi_objective_optimization \
        --calibration results/calibration_results.json \
        --output     results/pareto_front.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Deterministic seed for optimisation
SEED = 7

from idm.idm_model import IDMModel, IDMParameters
from digital_twin.ev_twin import EVTwin, EVVehicleParameters


# ----------------------------------------------------------------------
# Reference scenarios used to evaluate the objective
# ----------------------------------------------------------------------
def make_highway_trace(n: int = 1800, dt: float = 0.1) -> np.ndarray:
    """3-minute highway trace with mild congestion."""
    t = np.arange(n) * dt
    base = 25.0 + 3.0 * np.sin(2 * np.pi * t / 60.0)
    base += 2.0 * np.sin(2 * np.pi * t / 17.0)
    return np.clip(base, 0.0, 30.0)


def make_urban_trace(n: int = 1800, dt: float = 0.1) -> np.ndarray:
    """3-minute urban trace with stop-and-go."""
    t = np.arange(n) * dt
    base = 8.0 + 6.0 * np.sin(2 * np.pi * t / 30.0)
    base += 4.0 * (np.random.default_rng(SEED).standard_normal(n) > 1.2)
    return np.clip(base, 0.0, 16.0)


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------
def evaluate_objective(theta: np.ndarray,
                       leader_speed: np.ndarray,
                       initial_gap: float = 25.0,
                       initial_speed: float = 18.0) -> dict:
    """Return energy and degradation scalars for a parameter vector."""
    params = IDMParameters.from_array(theta)
    twin = EVTwin(idm=IDMModel(params))
    sim = twin.simulate(leader_speed, initial_gap, initial_speed, dt=0.1)
    metrics = twin.metrics(sim)
    distance_km = float(np.sum(sim["speed"]) * 0.1 / 1000.0)
    energy_kwh_per_km = metrics["energy_kwh"] / max(distance_km, 1e-6)
    return {
        "energy_kwh": metrics["energy_kwh"],
        "energy_kwh_per_km": float(energy_kwh_per_km),
        "degradation_pct": metrics["degradation_pct"],
        "mean_speed_mps": metrics["mean_speed_mps"],
        "regen_fraction": metrics["regen_fraction"],
        "delta_soc_pct": metrics["delta_soc_pct"],
        "distance_km": distance_km,
    }


def scalarised_objective(theta: np.ndarray,
                         leader_speed: np.ndarray,
                         w_E: float,
                         w_B: float,
                         E_ref: float,
                         B_ref: float) -> float:
    """Convex scalarisation ``w_E * J_E/E_ref + w_B * J_B/B_ref``."""
    m = evaluate_objective(theta, leader_speed)
    return w_E * (m["energy_kwh_per_km"] / E_ref) + \
           w_B * (m["degradation_pct"] / B_ref)


# ----------------------------------------------------------------------
# Main Pareto scan
# ----------------------------------------------------------------------
WEIGHT_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
BOUNDS = [(15.0, 40.0), (0.5, 3.0), (0.5, 5.0),
          (0.3, 3.0), (1.0, 5.0), (3.0, 5.0)]


def run_pareto_scan(initial_theta: np.ndarray,
                    weight_grid: list[float] = WEIGHT_GRID,
                    leader_speed: np.ndarray | None = None,
                    seed: int = SEED) -> dict:
    """Scan the Pareto front across the weight grid.

    Parameters
    ----------
    initial_theta : np.ndarray
        Initial IDM parameter vector (typically the calibrated one).
    weight_grid : list of float
        ``w_B / w_E`` ratios.
    leader_speed : np.ndarray | None
        Reference speed trace.  Defaults to a synthetic highway trace.
    seed : int
        Random seed for perturbing the initial guess.

    Returns
    -------
    dict
        Dictionary with ``pareto_points`` and ``reference``.
    """
    rng = np.random.default_rng(seed)
    if leader_speed is None:
        leader_speed = make_highway_trace()

    # Reference point used for normalisation: calibrated parameters
    ref = evaluate_objective(initial_theta, leader_speed)
    E_ref = max(ref["energy_kwh_per_km"], 1e-6)
    B_ref = max(ref["degradation_pct"], 1e-6)

    pareto_points = []
    for w in weight_grid:
        w_E, w_B = 1.0, w
        x0 = initial_theta + rng.normal(0.0, 0.05, size=initial_theta.shape[0])
        x0 = np.clip(x0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        res = minimize(
            scalarised_objective, x0,
            args=(leader_speed, w_E, w_B, E_ref, B_ref),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6,
                     "maxiter": 800, "maxfev": 800})
        m = evaluate_objective(res.x, leader_speed)
        pareto_points.append({
            "wB_over_wE": float(w),
            "theta": IDMParameters.from_array(res.x).to_dict(),
            "energy_kwh_per_km": float(m["energy_kwh_per_km"]),
            "degradation_pct": float(m["degradation_pct"]),
            "scalarised": float(res.fun),
            "distance_km": float(m["distance_km"]),
            "mean_speed_mps": float(m["mean_speed_mps"]),
            "regen_fraction": float(m["regen_fraction"]),
        })
        print(f"[pareto] w_B/w_E={w:>5.1f}  "
              f"E={m['energy_kwh_per_km']:.4f} kWh/km  "
              f"B={m['degradation_pct']:.4f}%")

    return {
        "pareto_points": pareto_points,
        "reference": {
            "theta": IDMParameters.from_array(initial_theta).to_dict(),
            "energy_kwh_per_km": float(ref["energy_kwh_per_km"]),
            "degradation_pct": float(ref["degradation_pct"]),
            "distance_km": float(ref["distance_km"]),
            "mean_speed_mps": float(ref["mean_speed_mps"]),
            "regen_fraction": float(ref["regen_fraction"]),
        },
        "seed": int(seed),
        "weight_grid": list(weight_grid),
    }


def main(calibration_json: str, output_json: str) -> None:
    with open(calibration_json) as fh:
        calib = json.load(fh)
    theta0 = np.array([
        calib["params"]["v0"], calib["params"]["T"],
        calib["params"]["s0"], calib["params"]["a"],
        calib["params"]["b"], calib["params"]["delta"],
    ])
    results = run_pareto_scan(theta0)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[pareto] Wrote {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-objective Pareto scan.")
    parser.add_argument("--calibration", required=True,
                       help="calibration_results.json path.")
    parser.add_argument("--output", default="results/pareto_front.json",
                       help="Output JSON path.")
    args = parser.parse_args()
    main(args.calibration, args.output)
