"""
digital_twin/ev_twin.py
========================

EV longitudinal digital twin built on top of the IDM plus a battery
electrical sub-model.

Purpose
-------
Given an IDM-derived acceleration trace, propagate the vehicle through
the longitudinal dynamics equations, compute the instantaneous tractive
power at the wheels, and feed it to a pack-level battery model that
returns state-of-charge (SOC), cell current, and a per-step degradation
rate.  The twin is calibrated against the Kia Soul EV parameters
(Table III of the reference paper) and validated on three independent
regimes (highway, urban, mixed) plus an external VED subset.

Inputs / outputs
----------------
The :class:`EVTwin` constructor accepts a :class:`EVVehicleParameters`
dataclass and a calibrated :class:`idm.idm_model.IDMModel`.  The primary
entry point :meth:`simulate` consumes a leader-speed trace and returns
a dict with trajectory, power, and battery-state arrays.

Run
---
::

    python -m digital_twin.ev_twin --help
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from idm.idm_model import IDMModel, IDMParameters
from idm.jit_kernels import (instantaneous_power,
                              pack_power_to_cell_current,
                              accumulate_battery_state)

from .battery_degradation import (DegradationModel,
                                   SemiEmpiricalDegradationModel)


@dataclass
class EVVehicleParameters:
    """Physical and electrical parameters for the Kia Soul EV.

    Values are taken from the reference paper's Table III and from
    publicly available Kia Soul EV specifications.
    """

    # Vehicle dynamics
    mass_kg: float = 1610.0           # kerb + 75 kg driver
    rolling_resistance_cr: float = 0.012
    drag_coefficient_cd: float = 0.32
    frontal_area_m2: float = 2.41
    air_density_kgpm3: float = 1.225
    gravity_mps2: float = 9.81
    wheel_radius_m: float = 0.314

    # Powertrain
    motor_efficiency: float = 0.94
    inverter_efficiency: float = 0.97
    regen_efficiency: float = 0.75      # round-trip
    drivetrain_efficiency: float = 0.95

    # Battery pack (Kia Soul EV 2015: 27 kWh, 96S3P LG Chem pouch)
    pack_capacity_kwh: float = 27.0
    n_cells_series: int = 96
    n_strings_parallel: int = 3
    cell_capacity_ah: float = 94.0       # 27 kWh / (96 * 3 * ~96 V)
    nominal_cell_voltage_v: float = 3.75
    initial_soc: float = 0.85

    # Accessory load (W)
    auxiliary_load_w: float = 350.0


class EVTwin:
    """End-to-end EV digital twin.

    Parameters
    ----------
    vehicle_params : EVVehicleParameters
        Physical parameters.
    idm : IDMModel
        Calibrated IDM.
    degradation : DegradationModel | None
        Optional battery degradation model.  When ``None`` the default
        :class:`SemiEmpiricalDegradationModel` is used.
    """

    def __init__(self,
                 vehicle_params: EVVehicleParameters | None = None,
                 idm: IDMModel | None = None,
                 degradation: DegradationModel | None = None) -> None:
        self.vp = vehicle_params or EVVehicleParameters()
        self.idm = idm or IDMModel(IDMParameters())
        self.deg = degradation or SemiEmpiricalDegradationModel()

    # ------------------------------------------------------------------
    def pack_voltage(self, soc: float) -> float:
        """Approximate pack open-circuit voltage as a function of SOC."""
        # Linear interpolation between empty (3.0 V) and full (4.1 V) cells
        cell_v = 3.0 + 1.1 * max(0.0, min(1.0, soc))
        return cell_v * self.vp.n_cells_series

    def simulate(self,
                leader_speed: np.ndarray,
                initial_gap: float,
                initial_speed: float,
                dt: float = 0.1,
                initial_soc: float | None = None) -> dict:
        """Run the closed-loop EV digital twin.

        Parameters
        ----------
        leader_speed : np.ndarray
            ``(N,)`` leader speed (m/s).
        initial_gap : float
            Initial gap (m).
        initial_speed : float
            Initial follower speed (m/s).
        dt : float, optional
            Time step (s).
        initial_soc : float | None
            Starting SOC (defaults to vehicle parameter).

        Returns
        -------
        dict
            Dictionary with arrays ``time``, ``gap``, ``speed``,
            ``accel``, ``power_w``, ``cell_current_a``, ``soc``,
            ``degradation_pct`` and the scalar metrics
            ``delta_soc``, ``ah_throughput``, ``regen_ah``, ``degradation_total``.
        """
        # 1) IDM trajectory
        traj = self.idm.simulate(leader_speed, initial_gap, initial_speed, dt)
        gap = traj[:, 0]
        v = traj[:, 1]
        accel = traj[:, 2]
        t = traj[:, 3]
        n = v.shape[0]

        # 2) Power at wheels
        vp = self.vp
        power_wheels = np.empty(n, dtype=np.float64)
        for i in range(n):
            power_wheels[i] = instantaneous_power(
                v[i], accel[i], vp.mass_kg, vp.rolling_resistance_cr,
                vp.drag_coefficient_cd, vp.frontal_area_m2,
                vp.air_density_kgpm3, vp.gravity_mps2)
        # Accessory load always pulls from the pack
        power_wheels += vp.auxiliary_load_w

        # 3) Pack-level power (account for efficiencies, asymmetric regen)
        eff_fwd = vp.motor_efficiency * vp.inverter_efficiency * vp.drivetrain_efficiency
        power_pack = np.where(
            power_wheels >= 0.0,
            power_wheels / eff_fwd,                              # discharging
            power_wheels * vp.regen_efficiency * eff_fwd)        # regenerative

        # 4) SOC integration
        soc = np.empty(n, dtype=np.float64)
        soc0 = initial_soc if initial_soc is not None else vp.initial_soc
        cell_current = np.empty(n, dtype=np.float64)
        soc_t = soc0
        for i in range(n):
            cell_v = 3.0 + 1.1 * max(0.0, min(1.0, soc_t))
            cell_current[i] = pack_power_to_cell_current(
                power_pack[i], vp.n_cells_series,
                vp.n_strings_parallel, cell_v)
            # Coulomb counting
            dsoc = -cell_current[i] * dt / (vp.cell_capacity_ah * 3600.0)
            soc_t = max(0.0, min(1.0, soc_t + dsoc))
            soc[i] = soc_t

        delta_soc, ah_throughput, regen_ah = accumulate_battery_state(
            cell_current,
            3.75 * np.ones(n),
            dt,
            vp.cell_capacity_ah)

        # 5) Degradation
        deg_pct, deg_total = self.deg.update(soc, cell_current, t)

        return {
            "time": t,
            "gap": gap,
            "speed": v,
            "accel": accel,
            "leader_speed": leader_speed,
            "power_wheels": power_wheels,
            "power_pack": power_pack,
            "cell_current": cell_current,
            "soc": soc,
            "degradation_pct": deg_pct,
            "delta_soc": float(soc0 - soc[-1]),
            "ah_throughput": float(ah_throughput),
            "regen_ah": float(regen_ah),
            "degradation_total_pct": float(deg_total),
            "energy_kwh": float(np.sum(np.maximum(power_pack, 0.0) * dt) / 3600e3),
            "regen_kwh": float(-np.sum(np.minimum(power_pack, 0.0) * dt) / 3600e3),
            "n_steps": int(n),
            "dt": float(dt),
        }

    # ------------------------------------------------------------------
    def metrics(self, sim: dict) -> dict:
        """Compute summary metrics used in the manuscript."""
        v = sim["speed"]
        a = sim["accel"]
        return {
            "mean_speed_mps": float(np.mean(v)),
            "max_speed_mps": float(np.max(v)),
            "mean_abs_accel_mps2": float(np.mean(np.abs(a))),
            "energy_kwh": float(sim["energy_kwh"]),
            "regen_kwh": float(sim["regen_kwh"]),
            "regen_fraction": float(sim["regen_kwh"] / max(sim["energy_kwh"], 1e-6)),
            "delta_soc_pct": float(sim["delta_soc"] * 100.0),
            "degradation_pct": float(sim["degradation_total_pct"]),
        }


def main() -> None:
    """Smoke-test entry point."""
    # short demonstration
    rng = np.random.default_rng(7)
    leader = np.clip(20.0 + 3.0 * np.sin(np.linspace(0, 6 * np.pi, 600)), 0, 30)
    twin = EVTwin()
    res = twin.simulate(leader, initial_gap=25.0, initial_speed=20.0)
    print(f"EVTwin smoke test: speed={np.mean(res['speed']):.2f} m/s "
          f"E={res['energy_kwh']:.3f} kWh "
          f"regen={res['regen_kwh']:.3f} kWh "
          f"dSOC={res['delta_soc']*100:.3f}% "
          f"deg={res['degradation_total_pct']:.4f}%")


if __name__ == "__main__":
    main()
