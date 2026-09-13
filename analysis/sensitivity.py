"""
analysis/sensitivity.py
=======================

One-at-a-time (OAT) sensitivity analysis around the joint optimised
IDM parameter vector.

Purpose
-------
Vary each of the six IDM parameters in turn over a +/-15 percent band
(11 sample points each) while holding the other five at their baseline
values, and report the induced change in the two objectives (energy
kWh/km and degradation percent).  The resulting partial-dependency
curves are saved to ``results/sensitivity_analysis.json``.

Run
---
::

    python -m analysis.sensitivity \
        --baseline results/pareto_front.json \
        --output    results/sensitivity_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np

from idm.idm_model import IDMModel, IDMParameters
from digital_twin.ev_twin import EVTwin
from optimization.multi_objective_optimization import make_highway_trace


PARAM_NAMES = ["v0", "T", "s0", "a", "b", "delta"]
PERTURBATION = 0.15
N_POINTS = 11


def oat_sensitivity(baseline_theta: np.ndarray,
                    leader_speed: np.ndarray | None = None,
                    perturbation: float = PERTURBATION,
                    n_points: int = N_POINTS) -> dict:
    """Run a one-at-a-time sensitivity sweep.

    Parameters
    ----------
    baseline_theta : np.ndarray
        Baseline IDM parameter vector.
    leader_speed : np.ndarray | None
        Leader trace; defaults to a synthetic highway trace.
    perturbation : float
        Fractional perturbation band (e.g. 0.15 for +/-15 percent).
    n_points : int
        Number of sample points per parameter.

    Returns
    -------
    dict
        Per-parameter sensitivity curves.
    """
    if leader_speed is None:
        leader_speed = make_highway_trace()

    def evaluate(theta: np.ndarray) -> dict:
        params = IDMParameters.from_array(theta)
        twin = EVTwin(idm=IDMModel(params))
        sim = twin.simulate(leader_speed, initial_gap=25.0,
                            initial_speed=18.0)
        m = twin.metrics(sim)
        distance_km = float(np.sum(sim["speed"]) * 0.1 / 1000.0)
        return {
            "energy_kwh_per_km": float(m["energy_kwh"] / max(distance_km, 1e-6)),
            "degradation_pct": float(m["degradation_pct"]),
            "regen_fraction": float(m["regen_fraction"]),
            "mean_speed_mps": float(m["mean_speed_mps"]),
        }

    baseline_metrics = evaluate(baseline_theta)
    out = {
        "baseline_theta": IDMParameters.from_array(baseline_theta).to_dict(),
        "baseline_metrics": baseline_metrics,
        "perturbation": float(perturbation),
        "n_points": int(n_points),
        "params": {},
    }

    for i, name in enumerate(PARAM_NAMES):
        lo = baseline_theta[i] * (1.0 - perturbation)
        hi = baseline_theta[i] * (1.0 + perturbation)
        grid = np.linspace(lo, hi, n_points)
        per_param = {
            "values": grid.tolist(),
            "energy_kwh_per_km": [],
            "degradation_pct": [],
            "regen_fraction": [],
            "mean_speed_mps": [],
        }
        for val in grid:
            theta = baseline_theta.copy()
            theta[i] = val
            m = evaluate(theta)
            per_param["energy_kwh_per_km"].append(m["energy_kwh_per_km"])
            per_param["degradation_pct"].append(m["degradation_pct"])
            per_param["regen_fraction"].append(m["regen_fraction"])
            per_param["mean_speed_mps"].append(m["mean_speed_mps"])
        # Elasticity at baseline (midpoint)
        e_idx = n_points // 2
        d_in = (per_param["values"][e_idx] - per_param["values"][0]) / \
               max(per_param["values"][e_idx], 1e-6)
        d_E = (per_param["energy_kwh_per_km"][e_idx] -
               per_param["energy_kwh_per_km"][0]) / \
              max(abs(per_param["energy_kwh_per_km"][e_idx]), 1e-6)
        d_B = (per_param["degradation_pct"][e_idx] -
               per_param["degradation_pct"][0]) / \
              max(abs(per_param["degradation_pct"][e_idx]), 1e-6)
        per_param["elasticity_energy"] = float(d_E / d_in) if abs(d_in) > 1e-9 else 0.0
        per_param["elasticity_degradation"] = float(d_B / d_in) if abs(d_in) > 1e-9 else 0.0
        out["params"][name] = per_param
        print(f"[sensitivity] {name:>6s}: "
              f"elasticity_E={per_param['elasticity_energy']:+.3f} "
              f"elasticity_B={per_param['elasticity_degradation']:+.3f}")
    return out


def main(baseline_json: str, output_json: str,
         weight_index: int = 2) -> None:
    """Use the third Pareto point (w_B/w_E=1.0) as the joint baseline."""
    with open(baseline_json) as fh:
        pareto = json.load(fh)
    point = pareto["pareto_points"][weight_index]
    theta = np.array([
        point["theta"]["v0"], point["theta"]["T"], point["theta"]["s0"],
        point["theta"]["a"], point["theta"]["b"], point["theta"]["delta"],
    ])
    results = oat_sensitivity(theta)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[sensitivity] Wrote {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OAT sensitivity analysis.")
    parser.add_argument("--baseline", required=True,
                       help="Path to pareto_front.json.")
    parser.add_argument("--output", default="results/sensitivity_analysis.json")
    parser.add_argument("--weight-index", type=int, default=2,
                       help="Index into the Pareto point list (default 2 = w_B/w_E=1).")
    args = parser.parse_args()
    main(args.baseline, args.output, args.weight_index)
