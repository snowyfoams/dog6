"""The filter's four matrices.  Pure functions: arguments in, a new array out.

THE LAYOUT EVERY SLICE BELOW REFERS TO, all WORLD frame, legs FL FR RL RR
    x  (18,)   [ p (3) | v (3) | foot_0 .. foot_3 (4 x 3) ]
    y  (28,)   [ p - foot_i (4 x 3) | v (4 x 3) | foot_i,z (4) ]

    Every position row is a DIFFERENCE, so absolute x/y is unobservable and
    drifts like any leg odometry.  z is observable because the last four rows
    put each planted foot on z = 0.

A, B and C are the same for every leg and every contact state.  Contact
enters only Q and R, through the per-leg suspicion s_i >= 1:

    s_i = 1 + suspect_gain * (1 - tau_i)      1 planted, 101 not believed

with tau_i from `measurement.trapezoid_trust`.
"""
from __future__ import annotations

import numpy as np

from .params import LKFParams

__all__ = ("build_C", "build_A_B", "build_Q", "build_R")


def build_C() -> np.ndarray:
    """Measurement matrix, (28, 18).  A constant: no leg, time or contact enters.

    For leg i = 0..3:

        rows 3i : 3i+3         p - foot_i      C[., 0:3] = I3,  C[., 6+3i:9+3i] = -I3
        rows 12+3i : 15+3i     v               C[., 3:6] = I3
        row  24+i              foot_i,z        C[., 8+3i] = 1
    """
    C = np.zeros((28, 18))
    for i in range(4):
        C[3 * i:3 * i + 3, 0:3] = np.eye(3)
        C[3 * i:3 * i + 3, 6 + 3 * i:9 + 3 * i] = -np.eye(3)
        C[12 + 3 * i:15 + 3 * i, 3:6] = np.eye(3)
        C[24 + i, 8 + 3 * i] = 1.0
    return C


def build_A_B(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Transition A (18, 18) and input B (18, 3) for one step of `dt` seconds.

    The trunk integrates the WORLD acceleration a_w; the feet stand still:

        p'      = p + dt v            A[0:3, 3:6] = dt I3
        v'      = v + dt a_w          B[3:6, :]   = dt I3
        foot_i' = foot_i              the rest of A is I

    There is no dt^2/2 a_w term in p.  MIT's A leaves it out; so does this.
    """
    assert 0.0 < dt < np.inf, f"dt must be a positive finite period in s, got {dt}"
    A = np.eye(18)
    A[0:3, 3:6] = dt * np.eye(3)
    B = np.zeros((18, 3))
    B[3:6, :] = dt * np.eye(3)
    return A, B


def build_Q(dt: float, s: np.ndarray, p: LKFParams) -> np.ndarray:
    """Process noise, (18, 18), diagonal.  `s` (4,) is each leg's suspicion.

        Q[0:3, 0:3]                 dt / 20 * lam_p         I3    trunk position
        Q[3:6, 3:6]                 9.8 * dt / 20 * lam_v   I3    trunk velocity
        Q[6+3i:9+3i, 6+3i:9+3i]     dt * lam_f * s_i        I3    foot i

    A foot that is not believed (s_i = 101) gets 101 times the process noise
    of a planted one, which is what lets its state follow the leg through a
    swing.

    The 9.8 is MIT's literal, not `p.g`.  It is a noise scale, not gravity,
    and the two differ by 0.1 %.
    """
    assert 0.0 < dt < np.inf, f"dt must be a positive finite period in s, got {dt}"
    s = np.asarray(s, dtype=float)
    assert s.shape == (4,), f"s must be (4,), got {s.shape}"
    q = np.concatenate((np.full(3, dt / 20.0 * p.lam_p),
                        np.full(3, 9.8 * dt / 20.0 * p.lam_v),
                        np.repeat(dt * p.lam_f * s, 3)))
    return np.diag(q)


def build_R(s: np.ndarray, p: LKFParams) -> np.ndarray:
    """Measurement noise, (28, 28), diagonal, rows in `build_C`'s order.

        R[0:12, 0:12]                    rho_p         I12   p - foot_i, every leg
        R[12+3i:15+3i, 12+3i:15+3i]      rho_v * s_i   I3    v, from leg i
        R[24+i, 24+i]                    rho_h * s_i         foot_i,z = 0

    THE RELATIVE-POSITION ROWS ARE NOT SCALED BY s_i.  Kinematics say where a
    foot is whether or not it carries load.  What contact changes is whether
    the foot is still -- so its velocity is the trunk's -- and whether it is on
    the floor, and those are the rows `s` inflates.
    """
    s = np.asarray(s, dtype=float)
    assert s.shape == (4,), f"s must be (4,), got {s.shape}"
    r = np.concatenate((np.full(12, p.rho_p),
                        np.repeat(p.rho_v * s, 3),
                        p.rho_h * s))
    return np.diag(r)
