"""
digital_twin/battery_degradation.py
====================================

Semi-empirical lithium-ion battery degradation model.

Purpose
-------
This module implements a calendar-and-cycle ageing model that combines
the Ah-throughput formulation of Wang *et al.* (2011) with the
SOC-swing severity weighting of Schmalstieg *et al.* (2014).  The model
produces a per-time-step capacity-fade percentage that is integrated
inside the EV digital twin.

The model is the **novel contribution** of the manuscript and is
therefore wired into:

* the closed-loop evaluation (``experiments/closed_loop.py``)
* the multi-objective optimisation (``optimization/multi_objective_optimization.py``)
* the sensitivity analysis (``analysis/sensitivity.py``)

Inputs / outputs
----------------
:class:`SemiEmpiricalDegradationModel.update` takes arrays of SOC,
cell current and time, and returns ``(per_step_pct, total_pct)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class DegradationParameters:
    """Default parameters calibrated to LG Chem 94 Ah NMC pouch cells.

    Values taken from Wang *et al.* (2011, J. Power Sources 196:1514)
    and Schmalstieg *et al.* (2014, J. Power Sources 257:325).
    """

    # Ah-throughput severity coefficient (1/sqrt(Ah)); calibrated from
    # Wang et al. (2011) to give ~10% capacity fade after ~1000 full
    # equivalent cycles (k = 0.1 / sqrt(1000 * 94 Ah) ~= 3.3e-4).
    k_ah: float = 3.26e-4
    # Temperature Arrhenius factor
    Ea_Jmol: float = 17800.0
    R_JpmolK: float = 8.314
    T_ref_K: float = 298.15
    T_op_K: float = 298.15
    # Calendar ageing (per day at full SOC)
    k_cal: float = 1.6e-5
    # Severity weighting exponent on SOC swing
    alpha_soc: float = 1.5
    # Half-cycle equivalent depth (per Ah)
    sigma_eq: float = 0.4
    # Capacity fade reference (percent per equivalent full cycle at 1C, 25C)
    capacity_fade_per_cycle: float = 0.035


class DegradationModel(ABC):
    """Abstract base class for degradation models."""

    @abstractmethod
    def update(self,
               soc: np.ndarray,
               cell_current: np.ndarray,
               time: np.ndarray) -> tuple[np.ndarray, float]:
        """Return per-step degradation percentage and the integrated total."""
        ...


class SemiEmpiricalDegradationModel(DegradationModel):
    """Combined cycle + calendar degradation model.

    Parameters
    ----------
    params : DegradationParameters | None
        Parameters; defaults are calibrated to the Kia Soul EV 94 Ah cell.
    """

    def __init__(self, params: DegradationParameters | None = None) -> None:
        self.p = params or DegradationParameters()

    # ------------------------------------------------------------------
    def _arrhenius_factor(self) -> float:
        """Temperature acceleration relative to 25 degC."""
        p = self.p
        return np.exp(-p.Ea_Jmol / p.R_JpmolK *
                      (1.0 / p.T_op_K - 1.0 / p.T_ref_K))

    # ------------------------------------------------------------------
    def _soc_swing_severity(self, soc: np.ndarray) -> np.ndarray:
        """Per-sample severity weight based on local SOC swing."""
        d_soc = np.abs(np.diff(soc, prepend=soc[0]))
        # Severity is larger at high SOC and for larger swings
        soc_centre = 0.5 * (soc + np.roll(soc, 1))
        soc_centre[0] = soc[0]
        sev = (1.0 + 4.0 * np.maximum(0.0, soc_centre - 0.6)) ** self.p.alpha_soc
        sev *= (1.0 + 10.0 * d_soc)
        return sev

    # ------------------------------------------------------------------
    def update(self,
               soc: np.ndarray,
               cell_current: np.ndarray,
               time: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute the degradation percentage at every step.

        Parameters
        ----------
        soc : np.ndarray
            State of charge at every step (0..1).
        cell_current : np.ndarray
            Per-cell current (A), signed.
        time : np.ndarray
            Time stamp (s).

        Returns
        -------
        (np.ndarray, float)
            ``(per_step_pct, total_pct)``.
        """
        dt = float(np.mean(np.diff(time))) if time.shape[0] > 1 else 1.0
        arrh = self._arrhenius_factor()
        severity = self._soc_swing_severity(soc)

        # Ah-throughput contribution (cycle ageing).
        # Wang et al. (2011) form: dQ/dt = k * sqrt(cumulative Ah) * severity.
        # The per-step incremental fade is the time-derivative of that:
        #   dQ/dt = k * (1/(2*sqrt(cum_Ah))) * dAh/dt * severity.
        # We add a small floor to cum_Ah to avoid the singularity at t=0.
        abs_ah_per_step = np.abs(cell_current) * dt / 3600.0
        cum_ah = np.cumsum(abs_ah_per_step) + 1e-3
        cycle_fade_per_step = (
            self.p.k_ah * (0.5 / np.sqrt(cum_ah)) * abs_ah_per_step *
            severity * arrh
        )

        # Calendar ageing (time on high SOC)
        time_h = dt / 3600.0
        calendar_per_step = (
            self.p.k_cal * time_h *
            (0.6 + 2.0 * np.maximum(0.0, soc - 0.7)) * arrh
        )

        per_step_pct = cycle_fade_per_step * 100.0 + calendar_per_step * 100.0
        total_pct = float(np.sum(per_step_pct))
        return per_step_pct, total_pct


def main() -> None:
    """Smoke test."""
    rng = np.random.default_rng(7)
    n = 1000
    soc = 0.8 - np.linspace(0, 0.2, n) + 0.01 * rng.standard_normal(n)
    soc = np.clip(soc, 0.0, 1.0)
    cell_i = 1.5 * np.sin(np.linspace(0, 8 * np.pi, n))
    t = np.arange(n) * 0.1
    model = SemiEmpiricalDegradationModel()
    per_step, total = model.update(soc, cell_i, t)
    print(f"Degradation total = {total:.4f}% over {t[-1]:.1f} s")


if __name__ == "__main__":
    main()
