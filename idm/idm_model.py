"""
idm/idm_model.py
================

Object-oriented wrapper around the JIT-compiled IDM kernels.

Purpose
-------
Provide a clean, parameterisable interface to the Intelligent Driver Model
that hides the Numba boilerplate from callers such as the calibration
routine, the EV digital twin, and the optimisation loop.

Inputs / outputs
----------------
The :class:`IDMModel` constructor takes the six IDM scalars
``v0, T, s0, a, b, delta`` and exposes three primary methods:

* :meth:`acceleration` -- single-step acceleration.
* :meth:`simulate` -- full trajectory roll-out.
* :meth:`log_likelihood` -- Gaussian residual used during calibration.

Run
---
The model is exercised by ``idm/calibration.py`` and the unit tests in
``tests/test_idm.py``::

    python -m pytest tests/test_idm.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from .jit_kernels import (idm_acceleration, idm_acceleration_batch,
                           step_trajectory, time_to_collision, time_headway)


@dataclass
class IDMParameters:
    """Container for the six free IDM parameters plus fixed defaults.

    Attributes
    ----------
    v0 : float
        Desired free-road speed (m/s).
    T : float
        Safe time gap (s).
    s0 : float
        Minimum standstill gap (m).
    a : float
        Maximum acceleration (m/s^2).
    b : float
        Comfortable deceleration (m/s^2).
    delta : float
        Acceleration exponent (typically 4).
    """

    v0: float = 30.0
    T: float = 1.5
    s0: float = 2.0
    a: float = 1.4
    b: float = 2.0
    delta: float = 4.0

    def to_array(self) -> np.ndarray:
        return np.array([self.v0, self.T, self.s0, self.a, self.b, self.delta],
                        dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "IDMParameters":
        return cls(v0=float(arr[0]), T=float(arr[1]), s0=float(arr[2]),
                   a=float(arr[3]), b=float(arr[4]), delta=float(arr[5]))


class IDMModel:
    """High-level Intelligent Driver Model wrapper.

    Parameters
    ----------
    params : IDMParameters
        Parameter bundle.  A shallow copy is stored so that the caller
        can safely mutate the original.
    """

    def __init__(self, params: IDMParameters | None = None) -> None:
        self.params = params if params is not None else IDMParameters()

    # ------------------------------------------------------------------
    # single-step API
    # ------------------------------------------------------------------
    def acceleration(self,
                    v: float,
                    dv: float,
                    gap: float) -> float:
        """IDM acceleration for one follower at one instant.

        Parameters
        ----------
        v : float
            Follower speed (m/s).
        dv : float
            Approach speed ``v - v_leader`` (m/s).
        gap : float
            Net gap (m).

        Returns
        -------
        float
            Acceleration (m/s^2).
        """
        p = self.params
        return idm_acceleration(v, dv, gap, p.v0, p.T, p.s0,
                                p.a, p.b, p.delta)

    def acceleration_batch(self,
                           v_arr: np.ndarray,
                           dv_arr: np.ndarray,
                           gap_arr: np.ndarray) -> np.ndarray:
        """Vectorised IDM acceleration over a trajectory."""
        p = self.params
        return idm_acceleration_batch(v_arr, dv_arr, gap_arr,
                                      p.v0, p.T, p.s0, p.a, p.b, p.delta)

    # ------------------------------------------------------------------
    # trajectory roll-out
    # ------------------------------------------------------------------
    def simulate(self,
                leader_speed: np.ndarray,
                initial_gap: float,
                initial_speed: float,
                dt: float = 0.1) -> np.ndarray:
        """Roll the IDM forward through a leader speed trace.

        Parameters
        ----------
        leader_speed : np.ndarray
            ``(N,)`` leader speed trace (m/s).
        initial_gap : float
            Initial bumper-to-bumper gap (m).
        initial_speed : float
            Initial follower speed (m/s).
        dt : float, optional
            Time step (s), default 0.1.

        Returns
        -------
        np.ndarray
            ``(N, 4)`` array ``[gap, speed, accel, time]``.
        """
        p = self.params
        n_steps = leader_speed.shape[0]
        return step_trajectory(p.v0, p.T, p.s0, p.a, p.b, p.delta,
                               dt, n_steps, leader_speed,
                               initial_gap, initial_speed)

    # ------------------------------------------------------------------
    # safety metrics
    # ------------------------------------------------------------------
    def ttc(self, gap: float, dv: float) -> float:
        """Time-to-collision (s)."""
        return time_to_collision(gap, dv)

    def thw(self, gap: float, v: float) -> float:
        """Time headway (s)."""
        return time_headway(gap, v)

    # ------------------------------------------------------------------
    # calibration helpers
    # ------------------------------------------------------------------
    def residual(self,
                 v_obs: np.ndarray,
                 dv_obs: np.ndarray,
                 gap_obs: np.ndarray,
                 a_obs: np.ndarray) -> np.ndarray:
        """Residual array ``a_model - a_obs`` used for calibration."""
        return self.acceleration_batch(v_obs, dv_obs, gap_obs) - a_obs

    def log_likelihood(self,
                      v_obs: np.ndarray,
                      dv_obs: np.ndarray,
                      gap_obs: np.ndarray,
                      a_obs: np.ndarray,
                      sigma: float = 0.5) -> float:
        """Gaussian log-likelihood of the observed accelerations.

        Parameters
        ----------
        v_obs, dv_obs, gap_obs, a_obs : np.ndarray
            Observation arrays of identical length.
        sigma : float, optional
            Observation noise standard deviation (m/s^2).

        Returns
        -------
        float
            Log-likelihood.
        """
        r = self.residual(v_obs, dv_obs, gap_obs, a_obs)
        n = r.shape[0]
        return -0.5 * float(np.sum(r * r)) / (sigma * sigma) \
            - n * (0.5 * np.log(2.0 * np.pi * sigma * sigma))

    def to_dict(self) -> dict:
        return asdict(self.params)
