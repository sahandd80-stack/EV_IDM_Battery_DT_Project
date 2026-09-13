"""
experiments/closed_loop.py
==========================

Closed-loop evaluation of five driver parameterisations.

Purpose
-------
For each of the five driver archetypes (Aggressive, Sporty, Average,
Cautious, Eco) defined in the reference paper (Table IV), run the
calibrated EV digital twin over a 20-minute mixed highway/urban trace
and persist the trajectory, battery, and degradation metrics.

Run
---
::

    python -m experiments.closed_loop \
        --calibration results/calibration_results.json \
        --output     results/closed_loop_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

from idm.idm_model import IDMModel, IDMParameters
from digital_twin.ev_twin import EVTwin, EVVehicleParameters


# Five driver archetypes (Table IV of the reference paper)
DRIVER_PROFILES = {
    "aggressive": dict(v0=32.0, T=0.7, s0=1.5, a=2.5, b=3.0, delta=4.0),
    "sporty":     dict(v0=29.0, T=1.0, s0=2.0, a=2.0, b=2.5, delta=4.0),
    "average":    dict(v0=25.0, T=1.5, s0=2.5, a=1.4, b=2.0, delta=4.0),
    "cautious":   dict(v0=22.0, T=2.0, s0=3.5, a=1.0, b=1.7, delta=4.0),
    "eco":        dict(v0=20.0, T=2.4, s0=4.0, a=0.8, b=1.5, delta=4.0),
}


def make_mixed_trace(seed: int = 7, n: int = 12000, dt: float = 0.1) -> np.ndarray:
    """20-minute mixed highway/urban trace."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    base = np.zeros(n)
    # Highway phase: minutes 0-7, 13-20
    highway = (t < 7 * 60) | (t >= 13 * 60)
    base[highway] = 25.0 + 3.0 * np.sin(2 * np.pi * t[highway] / 90.0)
    # Urban phase: minutes 7-13
    urban = ~highway
    base[urban] = 8.0 + 5.0 * np.sin(2 * np.pi * t[urban] / 25.0)
    # Random micro-perturbations (traffic-light-like)
    base += 1.5 * rng.standard_normal(n)
    return np.clip(base, 0.0, 32.0)


def run_closed_loop(calibration: dict,
                    leader_speed: np.ndarray,
                    dt: float = 0.1) -> dict:
    """Run all five driver profiles over the same leader trace."""
    calib_params = IDMParameters(**calibration["params"])
    out = {"profiles": {}, "trace": {
        "duration_s": float(leader_speed.shape[0] * dt),
        "n_steps": int(leader_speed.shape[0]),
        "dt": float(dt),
    }}

    for name, kwargs in DRIVER_PROFILES.items():
        # Merge calibrated delta with the archetype override
        params = IDMParameters(**kwargs)
        twin = EVTwin(idm=IDMModel(params))
        sim = twin.simulate(leader_speed,
                           initial_gap=25.0,
                           initial_speed=20.0,
                           dt=dt)
        metrics = twin.metrics(sim)
        out["profiles"][name] = {
            "params": params.to_dict(),
            **metrics,
            "min_speed_mps": float(np.min(sim["speed"])),
            "max_speed_mps": float(np.max(sim["speed"])),
            "max_accel_mps2": float(np.max(sim["accel"])),
            "min_accel_mps2": float(np.min(sim["accel"])),
            "max_cell_current_a": float(np.max(np.abs(sim["cell_current"]))),
        }
        print(f"[closed-loop] {name:>10s}: "
              f"E={metrics['energy_kwh']:.3f} kWh  "
              f"B={metrics['degradation_pct']:.4f}%  "
              f"v_mean={metrics['mean_speed_mps']:.2f} m/s")
    return out


def main(calibration_json: str, output_json: str) -> None:
    with open(calibration_json) as fh:
        calib = json.load(fh)
    leader = make_mixed_trace()
    results = run_closed_loop(calib, leader)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[closed-loop] Wrote {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Closed-loop driver-profile evaluation.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", default="results/closed_loop_results.json")
    args = parser.parse_args()
    main(args.calibration, args.output)
