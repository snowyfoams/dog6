"""How far each leg is believed, and what it says.  Pure functions.

ONE LEG OVER ONE STANCE
    phase   0 ........ w ...................... 1-w ........ 1
    tau     0  ramps   1   planted, believed     1   ramps   0

    At tau = 1 a leg says two things: the trunk moves at minus its foot's
    velocity relative to the trunk, and its foot is on the floor.  At tau = 0
    it says neither.  Its velocity row falls back to the filter's own previous
    v, its height row to where the kinematics already put the foot, and `s`
    down-weights both 101-fold.  In between it blends, so a touchdown fades in
    over `w` of the stance instead of arriving as a step.
"""
from __future__ import annotations

import numpy as np

__all__ = ("trapezoid_trust", "build_measurement")


def trapezoid_trust(phase: np.ndarray, window: float) -> np.ndarray:
    """Per-leg trust tau in [0, 1] from contact phase, (4,) -> (4,).

        phase < window          tau = phase / window           touching down
        phase > 1 - window      tau = (1 - phase) / window     lifting off
        otherwise               tau = 1                        mid-stance

    `phase` is each leg's progress through its stance: 0 at touchdown, 1 at
    liftoff, 0 for a leg in swing.  A four-foot stand passes 0.5 on every leg,
    which is tau = 1.

    CLIPPED TO [0, 1] FIRST.  MIT clamps the top with fmin(phase, 1): a gait
    clock a hair past 1 has to read as "lifting off", not as negative trust.
    The bottom is clamped for the same reason.

    `window` must lie in (0, 0.5]; past 0.5 the ramps overlap and the
    trapezoid never reaches 1.
    """
    phase = np.asarray(phase, dtype=float)
    assert phase.shape == (4,), f"phase must be (4,), got {phase.shape}"
    assert 0.0 < window <= 0.5, f"window must be in (0, 0.5], got {window}"
    phi = np.clip(phase, 0.0, 1.0)
    return np.where(phi < window, phi / window,
                    np.where(phi > 1.0 - window, (1.0 - phi) / window, 1.0))


def build_measurement(R_wb: np.ndarray, omega_b: np.ndarray, r: np.ndarray,
                      rd: np.ndarray, tau: np.ndarray, p_prev: np.ndarray,
                      v_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The measurement y (28,), in `matrices.build_C`'s row order, and pf (4, 3).

    Args -- TRUNK frame unless marked WORLD, SI, legs FL FR RL RR:
        R_wb     (3, 3)  world <- trunk
        omega_b  (3,)    trunk angular velocity, rad/s
        r        (4, 3)  each foot point about the trunk ORIGIN, m
        rd       (4, 3)  J_i qd_i: the foot's velocity with the trunk held
                         still, m/s.  Trunk rotation is added here, not there.
        tau      (4,)    trust, from `trapezoid_trust`
        p_prev   (3,)    WORLD trunk position BEFORE this step's predict
        v_prev   (3,)    WORLD trunk velocity BEFORE this step's predict

    Returns:
        y   (28,)    [ y_p (4 x 3) | y_v (4 x 3) | y_h (4) ]
        pf  (4, 3)   R_wb r_i, each foot about the trunk origin, WORLD axes

    Per leg:
        pf_i  = R_wb r_i
        vf_i  = R_wb (omega_b x r_i + rd_i)          foot velocity relative to
                                                     the trunk, WORLD axes
        y_p_i = -pf_i                                reads p - foot_i
        y_v_i = (1 - tau_i) v_prev + tau_i (-vf_i)   reads v: a planted foot is
                                                     still, so v = -vf_i
        y_h_i = (1 - tau_i) (p_prev,z + pf_i,z)      reads foot_i,z: 0 planted

    THE omega x r TERM IS NOT OPTIONAL.  rd is the leg folding under a still
    trunk; omega x r is the trunk turning over a still foot.  Its z component
    cancels in the mean over a level, square stance, so leaving it out passes
    a level bench test of the height rate.  Its x and y components do not --
    a trunk pitching over its feet moves its origin -- and per leg nothing
    cancels.
    """
    R_wb = np.asarray(R_wb, dtype=float)
    omega_b = np.asarray(omega_b, dtype=float)
    r = np.asarray(r, dtype=float)
    rd = np.asarray(rd, dtype=float)
    tau = np.asarray(tau, dtype=float)
    p_prev = np.asarray(p_prev, dtype=float)
    v_prev = np.asarray(v_prev, dtype=float)
    assert R_wb.shape == (3, 3), f"R_wb must be (3, 3), got {R_wb.shape}"
    assert omega_b.shape == (3,), f"omega_b must be (3,), got {omega_b.shape}"
    assert r.shape == (4, 3), f"r must be (4, 3), got {r.shape}"
    assert rd.shape == (4, 3), f"rd must be (4, 3), got {rd.shape}"
    assert tau.shape == (4,), f"tau must be (4,), got {tau.shape}"
    assert p_prev.shape == (3,), f"p_prev must be (3,), got {p_prev.shape}"
    assert v_prev.shape == (3,), f"v_prev must be (3,), got {v_prev.shape}"

    pf = r @ R_wb.T                                  # (4, 3): row i is R_wb @ r[i]
    vf = (np.cross(omega_b, r) + rd) @ R_wb.T        # (4, 3)
    t = tau[:, None]                                 # (4, 1), broadcasts per leg

    y_p = -pf
    y_v = (1.0 - t) * v_prev + t * (-vf)
    y_h = (1.0 - tau) * (p_prev[2] + pf[:, 2])
    return np.concatenate((y_p.ravel(), y_v.ravel(), y_h)), pf
