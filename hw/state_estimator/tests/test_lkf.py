"""The estimator end to end: the static stand, the measurement's row order, and
agreement with MIT's own `run()`.

    python -m pytest hw/state_estimator/tests

The spec's two checks (5, 6) are at R = I, where a transposed rotation and a
dropped omega x r are both exactly zero error.  So each has a twin at a tilt,
with the trunk turning and moving -- the case a level bench cannot show.
"""
from __future__ import annotations

import numpy as np
import pytest

from hw.state_estimator.adapters import rpy_zyx_to_R
from hw.state_estimator.estimator import (LinearKFPosVelEstimator, LKFParams,
                                          build_C, build_measurement)

DT = 0.002

#: DOG6's four foot-ball centres at `coordinates.Q_STAND`, TRUNK frame, m --
#: `hw.kinematics.all_foot_positions`, printed.  z is FOOT_RADIUS - STAND_HEIGHT.
STAND_FEET_B = np.array([[+0.182, +0.065, -0.177535],
                         [+0.182, -0.065, -0.177535],
                         [-0.182, +0.065, -0.177535],
                         [-0.182, -0.065, -0.177535]])


def _kinematic_sweep(rng: np.random.Generator):
    """A trunk that is tilted, turning and moving over four still feet on z = 0.

    Returns (x (18,), R_wb, omega_b, r, rd): x is the true state, and r and rd
    are exactly what ideal leg kinematics would report for it.

        r_i  = R^T (foot_i - p)
        rd_i = d/dt r_i = -omega_b x r_i - R^T v        (feet still)
    """
    p = np.array([0.3, -0.2, 0.19]) + rng.normal(scale=0.02, size=3)
    v = rng.normal(scale=0.3, size=3)
    feet = np.column_stack((STAND_FEET_B[:, 0:2] + p[0:2]
                            + rng.normal(scale=0.02, size=(4, 2)), np.zeros(4)))
    R_wb = rpy_zyx_to_R(rng.uniform(-1.0, 1.0, size=3) * [0.3, 0.3, np.pi])
    omega_b = rng.normal(scale=0.8, size=3)
    r = (feet - p) @ R_wb                     # row i is R_wb^T (foot_i - p)
    rd = -np.cross(omega_b, r) - v @ R_wb
    return np.concatenate((p, v, feet.ravel())), R_wb, omega_b, r, rd


# -- 5  the static stand -----------------------------------------------------
def test_static_level_stand_holds_height_and_zero_velocity():
    """Spec 5: R = I, omega = 0, rd = 0, acc = (0, 0, g), phase 0.5, 500 steps."""
    p = LKFParams()
    est = LinearKFPosVelEstimator(p)
    R_wb, omega_b, rd = np.eye(3), np.zeros(3), np.zeros((4, 3))
    acc_b = np.array([0.0, 0.0, p.g])
    phase = np.full(4, 0.5)
    est.reset(R_wb, STAND_FEET_B)
    z_expected = -(STAND_FEET_B @ R_wb.T)[:, 2].mean()

    for _ in range(500):
        out = est.update(R_wb, omega_b, acc_b, STAND_FEET_B, rd, phase, DT)
        np.testing.assert_array_equal(est.P, est.P.T)

    assert out.p_w[2] == pytest.approx(z_expected, abs=1e-12)
    np.testing.assert_allclose(out.p_w[0:2], 0.0, atol=1e-12)
    np.testing.assert_allclose(out.v_w, 0.0, atol=1e-12)
    np.testing.assert_array_equal(out.trust, 1.0)
    assert np.linalg.eigvalsh(est.P).min() > 0.0


def test_tilted_stand_recovers_from_a_kicked_state():
    """Spec 5 at a tilt and from a wrong start.

    Level, R and R^T are the same matrix; here they are not.  The trunk sits
    at a fixed tilt over feet on a flat floor, the accelerometer reads gravity
    in the tilted frame, and the state is knocked 30 mm high and 0.25 m/s off
    after it has settled.  Height has to come back to the floor-derived value
    and velocity to zero.
    """
    p = LKFParams()
    R_wb = rpy_zyx_to_R(np.array([0.12, -0.18, 0.7]))
    p_true = np.array([0.0, 0.0, -STAND_FEET_B[0, 2]])
    feet_w = np.column_stack((STAND_FEET_B[:, 0:2], np.zeros(4)))
    r = (feet_w - p_true) @ R_wb                    # what the legs report
    acc_b = R_wb.T @ np.array([0.0, 0.0, p.g])      # at rest, tilted
    omega_b, rd, phase = np.zeros(3), np.zeros((4, 3)), np.full(4, 0.5)

    est = LinearKFPosVelEstimator(p)
    est.reset(R_wb, r)
    np.testing.assert_allclose(est.x[0:3], p_true, atol=1e-12)
    np.testing.assert_allclose(est.x[6:18].reshape(4, 3), feet_w, atol=1e-12)

    for _ in range(200):
        est.update(R_wb, omega_b, acc_b, r, rd, phase, DT)
    est.x = est.x + np.concatenate(([0.0, 0.0, 0.03], [0.2, -0.1, 0.12], np.zeros(12)))
    for _ in range(500):
        out = est.update(R_wb, omega_b, acc_b, r, rd, phase, DT)

    # measured: 2e-9 m and 1.3e-7 m/s after the 500 steps
    assert out.p_w[2] == pytest.approx(p_true[2], abs=1e-6)
    np.testing.assert_allclose(out.v_w, 0.0, atol=1e-5)
    np.testing.assert_allclose(out.v_b, R_wb.T @ out.v_w, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(est.P, est.P.T)
    assert np.linalg.eigvalsh(est.P).min() > 0.0


# -- 6  the measurement's row order ------------------------------------------
def test_measurement_rows_line_up_with_C_in_the_static_stand():
    """Spec 6: in spec 5's stand, y and C x agree row for row -- innovation ~ 0."""
    p = LKFParams()
    est = LinearKFPosVelEstimator(p)
    R_wb, omega_b, rd = np.eye(3), np.zeros(3), np.zeros((4, 3))
    acc_b = np.array([0.0, 0.0, p.g])
    est.reset(R_wb, STAND_FEET_B)

    y, pf = build_measurement(R_wb, omega_b, STAND_FEET_B, rd, np.ones(4),
                              est.x[0:3], est.x[3:6])
    np.testing.assert_allclose(y, est.C @ est.x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pf, STAND_FEET_B, rtol=0, atol=1e-15)
    for _ in range(500):
        out = est.update(R_wb, omega_b, acc_b, STAND_FEET_B, rd,
                         np.full(4, 0.5), DT)
        assert np.abs(out.innov).max() < 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_measurement_is_C_x_for_a_tilted_turning_moving_trunk(seed):
    """Spec 6 where it can fail: four different legs, a tilt, and omega != 0.

    With ideal kinematics every row of y is exactly the matching row of C x,
    at ANY trust -- so a swapped leg, a transposed R or a dropped omega x r
    shows up as a row that disagrees.
    """
    rng = np.random.default_rng(seed)
    x, R_wb, omega_b, r, rd = _kinematic_sweep(rng)
    tau = rng.uniform(0.0, 1.0, size=4)
    tau[rng.integers(4)] = 0.0
    tau[rng.integers(4)] = 1.0

    y, pf = build_measurement(R_wb, omega_b, r, rd, tau, x[0:3], x[3:6])
    np.testing.assert_allclose(y, build_C() @ x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pf, x[6:18].reshape(4, 3) - x[0:3], rtol=0, atol=1e-12)

    # and the check has teeth: the same sweep without the omega x r term
    y_flat, _ = build_measurement(R_wb, np.zeros(3), r, rd, np.ones(4),
                                  x[0:3], x[3:6])
    assert np.abs(y_flat[12:24] - y[12:24]).max() > 1e-3


# -- against MIT -------------------------------------------------------------
def _mit_run(xhat, P, prm, R_wb, omega_b, acc_b, r, rd, phase, dt):
    """One call of MIT's `LinearKFPositionVelocityEstimator::run()`, transcribed.

    Blocks, a per-leg loop and an explicit inverse -- the C++'s shape, not this
    package's -- so agreeing with it is evidence rather than a tautology.
    `setup()`'s members are rebuilt inline.
    """
    # setup()
    A = np.zeros((18, 18))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 3:6] = dt * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    A[6:18, 6:18] = np.eye(12)
    B = np.zeros((18, 3))
    B[3:6, 0:3] = dt * np.eye(3)
    C1 = np.hstack((np.eye(3), np.zeros((3, 3))))
    C2 = np.hstack((np.zeros((3, 3)), np.eye(3)))
    C = np.zeros((28, 18))
    for row in (0, 3, 6, 9):
        C[row:row + 3, 0:6] = C1
    C[0:12, 6:18] = -np.eye(12)
    for row in (12, 15, 18, 21):
        C[row:row + 3, 0:6] = C2
    C[27, 17] = C[26, 14] = C[25, 11] = C[24, 8] = 1.0
    Q0 = np.eye(18)
    Q0[0:3, 0:3] = (dt / 20.0) * np.eye(3)
    Q0[3:6, 3:6] = (dt * 9.8 / 20.0) * np.eye(3)
    Q0[6:18, 6:18] = dt * np.eye(12)
    R0 = np.eye(28)

    # run()
    Q = np.eye(18)
    Q[0:3, 0:3] = Q0[0:3, 0:3] * prm.lam_p
    Q[3:6, 3:6] = Q0[3:6, 3:6] * prm.lam_v
    Q[6:18, 6:18] = Q0[6:18, 6:18] * prm.lam_f
    R = np.eye(28)
    R[0:12, 0:12] = R0[0:12, 0:12] * prm.rho_p
    R[12:24, 12:24] = R0[12:24, 12:24] * prm.rho_v
    R[24:28, 24:28] = R0[24:28, 24:28] * prm.rho_h

    a = R_wb @ acc_b + np.array([0.0, 0.0, -prm.g])
    ps, vs, pzs = np.zeros(12), np.zeros(12), np.zeros(4)
    p0, v0 = xhat[0:3].copy(), xhat[3:6].copy()
    for i in range(4):
        i1 = 3 * i
        p_rel, dp_rel = r[i], rd[i]
        p_f = R_wb @ p_rel
        dp_f = R_wb @ (np.cross(omega_b, p_rel) + dp_rel)
        qindex, rindex2, rindex3 = 6 + i1, 12 + i1, 24 + i
        trust = 1.0
        ph = min(phase[i], 1.0)
        if ph < prm.trust_window:
            trust = ph / prm.trust_window
        elif ph > 1.0 - prm.trust_window:
            trust = (1.0 - ph) / prm.trust_window
        k = 1.0 + (1.0 - trust) * prm.suspect_gain
        Q[qindex:qindex + 3, qindex:qindex + 3] *= k
        R[rindex2:rindex2 + 3, rindex2:rindex2 + 3] *= k
        R[rindex3, rindex3] *= k
        ps[i1:i1 + 3] = -p_f
        vs[i1:i1 + 3] = (1.0 - trust) * v0 + trust * (-dp_f)
        pzs[i] = (1.0 - trust) * (p0[2] + p_f[2])

    y = np.concatenate((ps, vs, pzs))
    xhat = A @ xhat + B @ a
    Pm = A @ P @ A.T + Q
    ey = y - C @ xhat
    S = C @ Pm @ C.T + R
    S_inv = np.linalg.inv(S)
    xhat = xhat + Pm @ C.T @ S_inv @ ey
    P = (np.eye(18) - Pm @ C.T @ S_inv @ C) @ Pm
    P = (P + P.T) / 2.0
    if np.linalg.det(P[0:2, 0:2]) > 0.000001:
        P[0:2, 2:18] = 0.0
        P[2:18, 0:2] = 0.0
        P[0:2, 0:2] /= 10.0
    return xhat, P, ey


def _corr_gap(P_a: np.ndarray, P_b: np.ndarray) -> float:
    """max_ij |P_a - P_b| / sqrt(P_b,ii P_b,jj): a covariance gap in correlation
    units, so a small entry is not held to a tolerance sized for a large one."""
    scale = np.sqrt(np.outer(np.diag(P_b), np.diag(P_b)))
    return float((np.abs(P_a - P_b) / scale).max())


def test_update_is_MITs_run_step_for_step():
    """400 steps of tilts, turns, noisy legs and every trust regime.

    Both sides start every step from the estimator's own (x, P), so what is
    compared is the one-step map.  It pins the order (y from x BEFORE the
    predict), the trust ramps, K by a solve, and the x/y bleed -- which fires
    on the first steps after reset and not after, so both branches are met.

    THE TOLERANCES ARE ~100x THE ROUNDING MEASURED OVER 40 SEEDS x 400 STEPS:
    x 5e-13 m, innovation 2e-16, P 4e-11 of a correlation.  A logic
    difference -- x_pre where x_prev belongs, a block on the wrong leg -- is
    1e-4 or more, on every step.

    EXCEPT P ON THE FIRST STEP AFTER reset, which is held at 1e-4.  One
    (I - KC) P_pre takes P from P0 = 100 to ~1e-3, cancelling five digits,
    and MIT's inverse and this solve then part at up to 1.3e-6 of a
    correlation.  That is the update form the spec and MIT share, not a
    difference between them; one step later the gap is back to rounding.
    """
    prm = LKFParams()
    rng = np.random.default_rng(11)
    est = LinearKFPosVelEstimator(prm)
    est.reset(rpy_zyx_to_R(np.array([0.05, -0.1, 0.4])), STAND_FEET_B)
    phases = np.array([0.0, 0.03, 0.1, 0.2, 0.5, 0.8, 0.9, 0.97, 1.0])

    bled = 0
    for step in range(400):
        R_wb = rpy_zyx_to_R(rng.normal(scale=[0.1, 0.1, 0.5]))
        omega_b = rng.normal(scale=0.5, size=3)
        acc_b = R_wb.T @ np.array([0.0, 0.0, prm.g]) + rng.normal(scale=0.5, size=3)
        r = STAND_FEET_B + rng.normal(scale=0.01, size=(4, 3))
        rd = rng.normal(scale=0.2, size=(4, 3))
        phase = rng.choice(phases, size=4)
        dt = rng.uniform(0.0015, 0.0035)

        x, P = est.x.copy(), est.P.copy()
        out = est.update(R_wb, omega_b, acc_b, r, rd, phase, dt)
        x_mit, P_mit, e_mit = _mit_run(x, P, prm, R_wb, omega_b, acc_b, r, rd,
                                       phase, dt)
        np.testing.assert_allclose(est.x, x_mit, rtol=0, atol=1e-10)
        np.testing.assert_allclose(out.innov, e_mit, rtol=0, atol=1e-12)
        assert _corr_gap(est.P, P_mit) < (1e-4 if step == 0 else 1e-8)
        bled += not est.P[0:2, 2:].any()
    assert 0 < bled < 400


# -- the contract around the numbers -----------------------------------------
def test_output_is_a_copy_and_the_class_holds_four_members():
    p = LKFParams()
    est = LinearKFPosVelEstimator(p)
    est.reset(np.eye(3), STAND_FEET_B)
    out = est.update(np.eye(3), np.zeros(3), np.array([0.0, 0.0, p.g]),
                     STAND_FEET_B, np.zeros((4, 3)), np.full(4, 0.5), DT)
    x_before = est.x.copy()
    for field in (out.p_w, out.v_w, out.v_b, out.foot_w, out.trust, out.innov):
        field[...] = 99.0
    np.testing.assert_array_equal(est.x, x_before)

    with pytest.raises(AttributeError):
        est.last_output = out


@pytest.mark.parametrize("name, bad", [
    ("R_wb", np.eye(4)),
    ("omega_b", np.zeros(4)),
    ("acc_b", np.zeros((3, 1))),
    ("r", np.zeros(12)),                 # flattened: the classic one
    ("rd", np.zeros((3, 4))),            # transposed
    ("contact_phase", np.zeros(3)),
])
def test_update_refuses_a_wrong_shape(name, bad):
    args = dict(R_wb=np.eye(3), omega_b=np.zeros(3),
                acc_b=np.array([0.0, 0.0, 9.81]), r=STAND_FEET_B,
                rd=np.zeros((4, 3)), contact_phase=np.full(4, 0.5), dt=DT)
    args[name] = bad
    with pytest.raises(AssertionError):
        LinearKFPosVelEstimator().update(**args)
