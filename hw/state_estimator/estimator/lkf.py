"""LinearKFPosVelEstimator: one predict and one correct per control sweep.

ONE STEP, IN ORDER
    1  trust     tau_i from contact phase, s_i from tau_i.  Q, R and y all
                 depend on it, so it comes first.
    2  measure   y from this sweep's sensors AND x as it stood before step 3
    3  predict   x_pre = A x + B a_w           P_pre = A P A^T + Q
    4  correct   K = P_pre C^T S^-1, by a solve against S, never an inverse
    5  bleed     MIT's cap on the x/y covariance

    STEP 2 READS x BEFORE STEP 3 MOVES IT.  That is MIT's order, and swapping
    the two is not a refactor: an untrusted leg's rows fall back on v_prev and
    p_prev, so the order changes what those rows say.  `tests/test_lkf.py`
    pins it against a line-by-line transcription of MIT's `run()`.

    a_w = R_wb acc_b + (0, 0, -g): the accelerometer reads SPECIFIC force,
    which is +g straight up for a trunk at rest.  Subtracting g is the
    filter's job; getting the sign of acc_b right is the adapter's.
"""
from __future__ import annotations

import numpy as np

from .matrices import build_A_B, build_C, build_Q, build_R
from .measurement import build_measurement, trapezoid_trust
from .params import LKFOutput, LKFParams

__all__ = ("LinearKFPosVelEstimator",)


class LinearKFPosVelEstimator:
    """MIT Cheetah's linear KF for trunk position and velocity, WORLD frame.

    Holds four things and nothing else -- `__slots__` makes that structural:

        x       (18,)      [ p_w | v_w | foot_w FL FR RL RR ], WORLD
        P       (18, 18)   its covariance
        C       (28, 18)   `build_C()`, the one matrix that never changes
        params  LKFParams

    Everything that varies per sweep -- dt, contact, the sensors -- arrives as
    an argument, and every other matrix is rebuilt from those arguments by a
    pure function in `matrices`.

    Before `reset` the state is MIT's own `setup()`: x = 0, P = P0 * I.  The
    filter converges from there, but `reset` starts it where the legs are.
    """

    __slots__ = ("x", "P", "C", "params")

    def __init__(self, params: LKFParams | None = None) -> None:
        self.params = LKFParams() if params is None else params
        self.C = build_C()
        self.x = np.zeros(18)
        self.P = self.params.P0 * np.eye(18)

    def reset(self, R_wb: np.ndarray, r: np.ndarray) -> None:
        """Start at rest, over the feet the legs report.  All four assumed planted.

        Args:
            R_wb  (3, 3)  world <- trunk
            r     (4, 3)  foot points about the trunk origin, TRUNK frame, m --
                          the same `r` that `update` takes

            pf      = R_wb r_i                      (4, 3)
            p       = (0, 0, -mean_i pf_i,z)        world origin under the trunk,
                                                    z = 0 at the feet's mean height
            v       = 0
            foot_i  = p + pf_i
            P       = P0 * I
        """
        R_wb = np.asarray(R_wb, dtype=float)
        r = np.asarray(r, dtype=float)
        assert R_wb.shape == (3, 3), f"R_wb must be (3, 3), got {R_wb.shape}"
        assert r.shape == (4, 3), f"r must be (4, 3), got {r.shape}"

        pf = r @ R_wb.T                          # (4, 3): row i is R_wb @ r[i]
        x = np.zeros(18)
        x[2] = -pf[:, 2].mean()
        x[6:18] = (x[0:3] + pf).ravel()
        self.x = x
        self.P = self.params.P0 * np.eye(18)

    def update(self, R_wb: np.ndarray, omega_b: np.ndarray, acc_b: np.ndarray,
               r: np.ndarray, rd: np.ndarray,
               contact_phase: np.ndarray, dt: float) -> LKFOutput:
        """One sweep: trust, measure, predict, correct.  Returns the new estimate.

        Args -- SI, TRUNK frame unless marked, legs FL FR RL RR:
            R_wb           (3, 3)  world <- trunk.  The SAME matrix the
                                   controller uses this sweep.
            omega_b        (3,)    angular velocity, rad/s
            acc_b          (3,)    specific force, m/s^2: (0, 0, +g) level at rest
            r              (4, 3)  foot points about the trunk ORIGIN, m, hip
                                   offset already included
            rd             (4, 3)  J_i qd_i, m/s -- NOT including omega x r
            contact_phase  (4,)    stance progress in [0, 1]; 0 in swing; 0.5 on
                                   all four for a balance stand
            dt             float   the MEASURED period, s, clamped by the caller

        Nothing is converted here.  A wrong unit or axis in any argument is
        not detectable from inside the filter; it becomes a bias.
        """
        R_wb = np.asarray(R_wb, dtype=float)
        omega_b = np.asarray(omega_b, dtype=float)
        acc_b = np.asarray(acc_b, dtype=float)
        r = np.asarray(r, dtype=float)
        rd = np.asarray(rd, dtype=float)
        contact_phase = np.asarray(contact_phase, dtype=float)
        assert R_wb.shape == (3, 3), f"R_wb must be (3, 3), got {R_wb.shape}"
        assert omega_b.shape == (3,), f"omega_b must be (3,), got {omega_b.shape}"
        assert acc_b.shape == (3,), f"acc_b must be (3,), got {acc_b.shape}"
        assert r.shape == (4, 3), f"r must be (4, 3), got {r.shape}"
        assert rd.shape == (4, 3), f"rd must be (4, 3), got {rd.shape}"
        assert contact_phase.shape == (4,), \
            f"contact_phase must be (4,), got {contact_phase.shape}"
        assert 0.0 < dt < np.inf, f"dt must be a positive finite period in s, got {dt}"
        p = self.params
        C = self.C

        # 1  trust.  Q, R and y all depend on it.
        tau = trapezoid_trust(contact_phase, p.trust_window)
        s = 1.0 + p.suspect_gain * (1.0 - tau)

        # 2  measure, against x as it stood BEFORE this step's predict.
        y, _ = build_measurement(R_wb, omega_b, r, rd, tau,
                                 self.x[0:3], self.x[3:6])
        a_w = R_wb @ acc_b + np.array([0.0, 0.0, -p.g])

        # 3  predict.
        A, B = build_A_B(dt)
        x_pre = A @ self.x + B @ a_w
        P_pre = A @ self.P @ A.T + build_Q(dt, s, p)

        # 4  correct.  S and P_pre are symmetric, so K^T = S^-1 C P_pre.
        e = y - C @ x_pre
        S = C @ P_pre @ C.T + build_R(s, p)
        K = np.linalg.solve(S, C @ P_pre).T
        x = x_pre + K @ e
        P = (np.eye(18) - K @ C) @ P_pre
        P = 0.5 * (P + P.T)

        # 5  bleed.  Every position row is a difference, so the x/y variance
        # random-walks without bound.  Once the block's determinant passes
        # 1e-6 -- about 3 cm standard deviation per axis -- MIT drops its
        # correlations and shrinks it tenfold.  A heuristic, not a Kalman
        # step, kept exactly as MIT has it.
        if np.linalg.det(P[0:2, 0:2]) > 1e-6:
            P[0:2, 2:] = 0.0
            P[2:, 0:2] = 0.0
            P[0:2, 0:2] /= 10.0

        self.x = x
        self.P = P
        v_w = x[3:6].copy()
        return LKFOutput(p_w=x[0:3].copy(), v_w=v_w, v_b=R_wb.T @ v_w,
                         foot_w=x[6:18].reshape(4, 3).copy(),
                         trust=tau, innov=e)
