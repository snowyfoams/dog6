"""The pure functions: build_C, build_A_B, build_Q, build_R, trapezoid_trust.

    python -m pytest hw/state_estimator/tests
"""
from __future__ import annotations

import numpy as np
import pytest

from hw.state_estimator.estimator import (LKFParams, build_A_B, build_C,
                                          build_Q, build_R, trapezoid_trust)

DT = 0.002


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x (18,) -> p (3,), v (3,), feet (4, 3)."""
    return x[0:3], x[3:6], x[6:18].reshape(4, 3)


# -- 1  C --------------------------------------------------------------------
def test_C_reads_relative_position_then_velocity_then_foot_height():
    C = build_C()
    assert C.shape == (28, 18)
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.normal(size=18)
        p, v, feet = _split(x)
        expected = np.concatenate(((p - feet).ravel(), np.tile(v, 4), feet[:, 2]))
        np.testing.assert_allclose(C @ x, expected, rtol=0, atol=1e-15)


def test_C_is_rebuilt_not_shared():
    a, b = build_C(), build_C()
    a[0, 0] = 99.0
    np.testing.assert_array_equal(build_C(), b)


# -- 2  A, B -----------------------------------------------------------------
@pytest.mark.parametrize("dt", [0.001, 0.002, 0.0045])
def test_A_B_integrate_the_trunk_and_hold_the_feet(dt):
    A, B = build_A_B(dt)
    assert A.shape == (18, 18)
    assert B.shape == (18, 3)
    rng = np.random.default_rng(2)
    for _ in range(50):
        x, a = rng.normal(size=18), rng.normal(size=3)
        p, v, feet = _split(x)
        p1, v1, feet1 = _split(A @ x + B @ a)
        np.testing.assert_allclose(p1, p + dt * v, rtol=0, atol=1e-15)
        np.testing.assert_allclose(v1, v + dt * a, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(feet1, feet)


@pytest.mark.parametrize("dt", [0.0, -0.002, np.nan, np.inf])
def test_A_B_and_Q_refuse_a_period_that_is_not_positive_and_finite(dt):
    with pytest.raises(AssertionError):
        build_A_B(dt)
    with pytest.raises(AssertionError):
        build_Q(dt, np.ones(4), LKFParams())


# -- 3  trust ----------------------------------------------------------------
def test_trust_at_the_spec_points():
    tau = trapezoid_trust(np.array([0.0, 0.5, 0.1, 0.9]), 0.2)
    np.testing.assert_allclose(tau, [0.0, 1.0, 0.5, 0.5], rtol=0, atol=1e-12)


def test_trust_is_a_continuous_trapezoid():
    w = 0.2
    ramps = trapezoid_trust(np.array([0.05, 0.15, 0.85, 0.95]), w)
    np.testing.assert_allclose(ramps, [0.25, 0.75, 0.75, 0.25], rtol=0, atol=1e-12)
    corners = trapezoid_trust(np.array([w, 1.0 - w, w - 1e-9, 1.0 - w + 1e-9]), w)
    np.testing.assert_allclose(corners, 1.0, rtol=0, atol=1e-8)
    ends = trapezoid_trust(np.array([0.0, 1.0, 0.35, 0.65]), w)
    np.testing.assert_allclose(ends, [0.0, 0.0, 1.0, 1.0], rtol=0, atol=1e-12)


def test_trust_clips_phase_to_the_unit_interval():
    """MIT's fmin(phase, 1), and the same at the bottom: never negative trust."""
    tau = trapezoid_trust(np.array([-0.3, 1.2, 1.0 + 1e-12, -1e-12]), 0.2)
    np.testing.assert_array_equal(tau, 0.0)


@pytest.mark.parametrize("window", [0.0, -0.1, 0.51])
def test_trust_refuses_a_window_with_no_plateau(window):
    with pytest.raises(AssertionError):
        trapezoid_trust(np.full(4, 0.5), window)


# -- 4  Q, R -----------------------------------------------------------------
def test_Q_and_R_diagonals_match_the_spec_entry_by_entry():
    p = LKFParams()
    s = np.array([1.0, 7.0, 42.0, 101.0])
    Q, R = build_Q(DT, s, p), build_R(s, p)
    assert Q.shape == (18, 18)
    assert R.shape == (28, 28)
    assert np.count_nonzero(Q - np.diag(np.diag(Q))) == 0
    assert np.count_nonzero(R - np.diag(np.diag(R))) == 0

    q = np.diag(Q)
    np.testing.assert_allclose(q[0:3], DT / 20 * p.lam_p, rtol=1e-14)
    np.testing.assert_allclose(q[3:6], 9.8 * DT / 20 * p.lam_v, rtol=1e-14)
    for i in range(4):
        np.testing.assert_allclose(q[6 + 3 * i:9 + 3 * i], DT * p.lam_f * s[i],
                                   rtol=1e-14)

    r = np.diag(R)
    np.testing.assert_allclose(r[0:12], p.rho_p, rtol=1e-14)
    for i in range(4):
        np.testing.assert_allclose(r[12 + 3 * i:15 + 3 * i], p.rho_v * s[i],
                                   rtol=1e-14)
        np.testing.assert_allclose(r[24 + i], p.rho_h * s[i], rtol=1e-14)


@pytest.mark.parametrize("swing", [(0,), (1,), (2,), (3,), (0, 1, 2, 3)])
def test_a_leg_in_swing_is_101x_on_foot_velocity_and_height_not_position(swing):
    """s = 101 scales exactly that leg's foot, velocity and height blocks.

    One leg at a time, so a block landing on the wrong leg's rows fails too.
    """
    p = LKFParams()
    phase = np.full(4, 0.5)
    phase[list(swing)] = 0.0
    s = 1.0 + p.suspect_gain * (1.0 - trapezoid_trust(phase, p.trust_window))
    np.testing.assert_array_equal(s, np.where(phase == 0.0, 101.0, 1.0))

    ones = np.ones(4)
    q_ratio = np.diag(build_Q(DT, s, p)) / np.diag(build_Q(DT, ones, p))
    r_ratio = np.diag(build_R(s, p)) / np.diag(build_R(ones, p))

    q_expected = np.ones(18)
    r_expected = np.ones(28)            # r_expected[0:12] stays 1: position
    for i in swing:
        q_expected[6 + 3 * i:9 + 3 * i] = 101.0
        r_expected[12 + 3 * i:15 + 3 * i] = 101.0
        r_expected[24 + i] = 101.0
    np.testing.assert_allclose(q_ratio, q_expected, rtol=1e-12)
    np.testing.assert_allclose(r_ratio, r_expected, rtol=1e-12)
