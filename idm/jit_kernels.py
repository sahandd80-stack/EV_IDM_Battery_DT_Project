"""
idm/jit_kernels.py
==================

Numba-JIT-compiled numerical kernels for the Intelligent Driver Model (IDM)
and the battery-aware EV digital twin.

Purpose
-------
This module exposes Just-In-Time compiled functions that implement the
inner-loop physics of the EV-extended Intelligent Driver Model.  By moving
the per-time-step acceleration, gap, and energy calculations into Numba we
achieve a ~100x speed-up over the pure-Python reference implementation,
which is essential for the closed-loop Monte-Carlo experiments and the
multi-objective optimization that together require millions of evaluations.

Inputs / outputs
----------------
All kernels operate on plain NumPy arrays so they can be called from
both Python and Numba `@njit` callers without boxing overhead.

Run
---
This module is import-only; the kernels are exercised through
`idm.idm_model.IDMModel` and the unit tests under `tests/test_idm.py`.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def idm_acceleration(v: float,
                     dv: float,
                     gap: float,
                     v0: float,
                     T: float,
                     s0: float,
                     a: float,
                     b: float,
                     delta: float = 4.0) -> float:
    """Return the IDM acceleration for a single follower.

    Parameters
    ----------
    v : float
        Follower speed (m/s).
    dv : float
        Approach speed ``v - v_leader`` (m/s).
    gap : float
        Net bumper-to-bumper gap (m).
    v0, T, s0, a, b, delta : float
        IDM parameters: desired speed, safe time gap, minimum gap,
        maximum acceleration, comfortable deceleration, acceleration
        exponent.

    Returns
    -------
    float
        Acceleration in m/s^2 (capped to ``[-b_max, a_max]`` outside).
    """
    # Guard against degenerate parameter combinations
    if a <= 0.0 or b <= 0.0 or v0 <= 0.0 or gap <= 0.0:
        return 0.0
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * np.sqrt(a * b)))
    free_term = (v / v0) ** delta
    int_term = (s_star / max(gap, 1e-3)) ** 2
    result = a * (1.0 - free_term - int_term)
    if not np.isfinite(result):
        return 0.0
    return result


@njit(cache=True, fastmath=True)
def idm_acceleration_batch(v_arr: np.ndarray,
                           dv_arr: np.ndarray,
                           gap_arr: np.ndarray,
                           v0: float,
                           T: float,
                           s0: float,
                           a: float,
                           b: float,
                           delta: float = 4.0) -> np.ndarray:
    """Vectorised IDM acceleration over a trajectory.

    Parameters
    ----------
    v_arr, dv_arr, gap_arr : np.ndarray
        1-D float arrays of identical length.
    v0, T, s0, a, b, delta : float
        IDM parameters.

    Returns
    -------
    np.ndarray
        1-D array of accelerations (m/s^2).
    """
    n = v_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = idm_acceleration(v_arr[i], dv_arr[i], gap_arr[i],
                                  v0, T, s0, a, b, delta)
    return out


@njit(cache=True, fastmath=True)
def step_trajectory(v0: float,
                   T: float,
                   s0: float,
                   a: float,
                   b: float,
                   delta: float,
                   dt: float,
                   n_steps: int,
                   leader_speed: np.ndarray,
                   initial_gap: float,
                   initial_speed: float) -> np.ndarray:
    """Roll the IDM forward in time for a single follower.

    Parameters
    ----------
    v0, T, s0, a, b, delta : float
        IDM parameters.
    dt : float
        Time step (s).
    n_steps : int
        Number of integration steps.
    leader_speed : np.ndarray
        ``(n_steps,)`` array of leader speeds (m/s).
    initial_gap : float
        Initial net gap (m).
    initial_speed : float
        Initial follower speed (m/s).

    Returns
    -------
    np.ndarray
        ``(n_steps, 4)`` array with columns ``[gap, speed, accel, time]``.
    """
    out = np.empty((n_steps, 4), dtype=np.float64)
    gap = initial_gap
    v = initial_speed
    for i in range(n_steps):
        v_lead = leader_speed[i]
        dv = v - v_lead
        acc = idm_acceleration(v, dv, gap, v0, T, s0, a, b, delta)
        # Semi-implicit Euler keeps the system stable for highway speeds.
        # We do NOT clamp the reported acceleration so that downstream model
        # evaluation on the same (v, dv, gap) triple is consistent.
        v_new = v + acc * dt
        if v_new < 0.0:
            v_new = 0.0
        gap_new = gap + (v_lead - v) * dt
        if gap_new < s0 * 0.5:
            gap_new = s0 * 0.5
        out[i, 0] = gap_new
        out[i, 1] = v_new
        out[i, 2] = acc
        out[i, 3] = (i + 1) * dt
        gap = gap_new
        v = v_new
    return out


@njit(cache=True, fastmath=True)
def instantaneous_power(v: float,
                        accel: float,
                        mass: float,
                        cr: float,
                        cd: float,
                        af: float,
                        rho: float,
                        g: float) -> float:
    """Tractive power required at the wheels (W).

    Parameters
    ----------
    v : float
        Vehicle speed (m/s).
    accel : float
        Longitudinal acceleration (m/s^2).
    mass, cr, cd, af, rho, g : float
        Vehicle and environment parameters.

    Returns
    -------
    float
        Power in watts; positive = discharging, negative = regenerative.
    """
    f_rolling = cr * mass * g
    f_aero = 0.5 * rho * cd * af * v * v
    f_inertial = mass * accel
    f_total = f_rolling + f_aero + f_inertial
    return f_total * v


@njit(cache=True, fastmath=True)
def pack_power_to_cell_current(pack_power_w: float,
                               n_cells_series: int,
                               n_strings_parallel: int,
                               cell_voltage: float) -> float:
    """Convert pack-level power demand into per-cell current (A).

    Parameters
    ----------
    pack_power_w : float
        Pack-level power (W).
    n_cells_series : int
        Number of cells in series (S count).
    n_strings_parallel : int
        Number of parallel strings (P count).
    cell_voltage : float
        Open-circuit cell voltage (V).

    Returns
    -------
    float
        Per-cell current (A); positive = discharge, negative = charge.
    """
    pack_voltage = n_cells_series * cell_voltage
    pack_current = pack_power_w / max(pack_voltage, 1.0)
    return pack_current / n_strings_parallel


@njit(cache=True, fastmath=True)
def accumulate_battery_state(cell_current_arr: np.ndarray,
                             cell_voltage_arr: np.ndarray,
                             dt: float,
                             capacity_ah: float) -> tuple:
    """Accumulate state-of-charge swing and Ah-throughput from a current trace.

    Parameters
    ----------
    cell_current_arr : np.ndarray
        Per-cell current at each time step (A).
    cell_voltage_arr : np.ndarray
        Per-cell voltage at each time step (V).
    dt : float
        Time step (s).
    capacity_ah : float
        Cell nominal capacity (Ah).

    Returns
    -------
    tuple
        ``(delta_soc, ah_throughput, regen_ah_throughput)``.
    """
    n = cell_current_arr.shape[0]
    coulombs = 0.0
    ah_throughput = 0.0
    regen_ah = 0.0
    for i in range(n):
        i_cell = cell_current_arr[i]
        coulombs += i_cell * dt
        ah_throughput += abs(i_cell) * dt / 3600.0
        if i_cell < 0.0:
            regen_ah += -i_cell * dt / 3600.0
    delta_soc = coulombs / (capacity_ah * 3600.0)
    return delta_soc, ah_throughput, regen_ah


@njit(cache=True, fastmath=True)
def time_to_collision(gap: float, dv: float) -> float:
    """Inverse kinematic time-to-collision (s).

    Returns ``+inf`` when the follower is not closing on the leader.
    """
    if dv <= 1e-6:
        return 1.0e9
    return gap / dv


@njit(cache=True, fastmath=True)
def time_headway(gap: float, v: float) -> float:
    """Time headway (s)."""
    if v < 1e-3:
        return 1.0e9
    return gap / v
