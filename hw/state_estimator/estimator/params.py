"""The filter's parameters and its output.  Dataclasses, no logic.

The six noise defaults are MIT's Mini Cheetah numbers under shorter names.
`config/lkf.yaml` carries the same ten keys.

    lam_p          imu_process_noise_position      Q, trunk position
    lam_v          imu_process_noise_velocity      Q, trunk velocity
    lam_f          foot_process_noise_position     Q, each foot
    rho_p          foot_sensor_noise_position      R, p - foot_i
    rho_v          foot_sensor_noise_velocity      R, v from a leg
    rho_h          foot_height_sensor_noise        R, foot_i,z = 0

    trust_window   phase width of the ramp at each end of a stance, (0, 0.5]
    suspect_gain   an untrusted leg's noise is scaled by
                   s = 1 + suspect_gain * (1 - trust)
    P0             the covariance `reset` starts from, P = P0 * I
    g              m/s^2, what a level accelerometer at rest reads

All six noise terms are variance scales; `matrices` is where each one lands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ("LKFParams", "LKFOutput")


@dataclass(frozen=True)
class LKFParams:
    """Every number the filter has.  Frozen: tune with `dataclasses.replace`."""

    lam_p: float = 0.02
    lam_v: float = 0.02
    lam_f: float = 0.002
    rho_p: float = 0.001
    rho_v: float = 0.1
    rho_h: float = 0.001
    trust_window: float = 0.2
    suspect_gain: float = 100.0
    P0: float = 100.0
    g: float = 9.81


@dataclass(frozen=True)
class LKFOutput:
    """One update's result.  Every array is a fresh copy, safe to keep or edit.

    WORLD is the frame `reset` fixed: axes are R_wb's world axes, the origin
    is under the trunk origin at the moment of reset, and z = 0 is the mean
    height of the planted feet's `r` points.  So p_w[2] is the height of the
    trunk ORIGIN above THOSE POINTS, not above the floor -- if `r` is a foot
    ball's centre, the floor is one ball radius lower.
    """

    p_w: np.ndarray      # (3,)   trunk origin, WORLD, m.  x/y drift: odometry
    v_w: np.ndarray      # (3,)   trunk origin velocity, WORLD, m/s
    v_b: np.ndarray      # (3,)   the same velocity in the TRUNK frame, R_wb^T v_w
    foot_w: np.ndarray   # (4, 3) the four foot points, WORLD, m, FL FR RL RR
    trust: np.ndarray    # (4,)   tau_i in [0, 1]: 1 = planted, 0 = not believed
    innov: np.ndarray    # (28,)  y - C x_pre, rows as `matrices.build_C`
