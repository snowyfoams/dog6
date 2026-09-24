"""The world z datum: ball centres inside the filter, the FLOOR on the way out.

    python -m pytest hw/state_estimator/tests/test_height_datum.py

`hw.kinematics` reports the foot ball's CENTRE and `build_measurement` tells
the filter a planted foot sits at z = 0, so the filter's world z is measured
from the plane of the centres -- one ball radius above the floor.
`adapters.to_floor` is the one place that radius is spent, and these tests
pin it against `hw.balance.state`'s `z_origin`, which is the height every
other part of `hw` already means.

No pytest helpers and no robot: plain asserts, so this file also runs under
`python -m hw.state_estimator.tests.test_height_datum` on a host without
pytest installed.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from hw.state_estimator.adapters import rpy_zyx_to_R, run_once, to_floor
from hw.state_estimator.estimator import LinearKFPosVelEstimator, LKFParams

from sim import params as P

#: The four ball CENTRES on flat ground, WORLD axes, about the mean centre.
#: z = 0 on every row: four balls of one radius on one floor sit at one
#: height, whatever the trunk above them is doing.
CENTRES_W = np.array([[+0.182, +0.065, 0.0], [+0.182, -0.065, 0.0],
                      [-0.182, +0.065, 0.0], [-0.182, -0.065, 0.0]])

#: m, the trunk ORIGIN above the FLOOR -- what `z_origin` reports.
TRUNK_ABOVE_FLOOR = 0.25

#: m/s^2.  AT REST MEANS THE FILTER'S OWN g, not 9.81: `lkf` forms
#: a_w = R acc_b + (0, 0, -g), so a trunk that is really standing still reads
#: exactly `LKFParams.g` and any other number is an acceleration.  Written as
#: 9.81 here, these tests fail by 23 um of climb over 0.1 s -- which is the
#: bias `params.g` carries DOG6's measured reading to avoid.
G_AT_REST = LKFParams().g

#: The tilts every datum claim is made at.  A level trunk is the case that
#: cannot tell a world-frame shift from a trunk-frame one, so it is here to
#: be the control, not the evidence.
TILTS = [np.array([0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 1.9]),
         np.array([0.17, 0.0, 0.0]),
         np.array([0.0, -0.17, 0.4]),
         np.array([0.12, 0.21, -2.7])]


def _scene(rpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R_wb and r for a trunk at `TRUNK_ABOVE_FLOOR` on flat ground, tilted.

    The feet are placed in the WORLD and the legs are solved backwards, which
    is the only way to get a scene a real robot could stand in: the trunk
    tilts, the floor does not, and all four centres stay at one height.
    """
    R = rpy_zyx_to_R(rpy)
    d = CENTRES_W - np.array([0.0, 0.0, TRUNK_ABOVE_FLOOR - P.FOOT_RADIUS])
    return R, d @ R                              # row i is R.T @ d_i


def _at_rest(est, R, r, n=50):
    """`n` sweeps of a still trunk, four feet down.  Returns the last output."""
    acc_b = R.T @ np.array([0.0, 0.0, G_AT_REST])
    out = None
    for _ in range(n):
        out = est.update(R, np.zeros(3), acc_b, r, np.zeros((4, 3)),
                         np.full(4, 0.5), 0.002)
    return out


def test_to_floor_moves_world_z_and_leaves_every_other_field_alone():
    R, r = _scene(TILTS[3])
    est = LinearKFPosVelEstimator()
    est.reset(R, r)
    raw = _at_rest(est, R, r, n=5)
    moved = to_floor(raw, P.FOOT_RADIUS)

    assert moved.p_w[2] == raw.p_w[2] + P.FOOT_RADIUS
    np.testing.assert_array_equal(moved.p_w[0:2], raw.p_w[0:2])
    np.testing.assert_array_equal(moved.foot_w[:, 2], raw.foot_w[:, 2] + P.FOOT_RADIUS)
    np.testing.assert_array_equal(moved.foot_w[:, 0:2], raw.foot_w[:, 0:2])
    for field in ("v_w", "v_b", "trust", "innov"):
        np.testing.assert_array_equal(getattr(moved, field), getattr(raw, field))

    # A frame shift, not a filter step: x and P are untouched, so the next
    # sweep starts where MIT's filter would have.
    assert est.x[2] == raw.p_w[2]


def test_to_floor_does_not_alias_the_estimator_state():
    R, r = _scene(TILTS[0])
    est = LinearKFPosVelEstimator()
    est.reset(R, r)
    raw = _at_rest(est, R, r, n=1)
    moved = to_floor(raw, P.FOOT_RADIUS)
    moved.p_w[2] += 1.0
    moved.foot_w[:, 2] += 1.0
    assert est.x[2] == raw.p_w[2]
    np.testing.assert_array_equal(est.x[6:18].reshape(4, 3), raw.foot_w)


def test_the_floor_datum_is_balance_states_z_origin_at_every_tilt():
    """The claim, checked against the code that already owns the number.

    `hw.balance.state._height` is private, and it is called here on purpose:
    a copy of `z_origin = FOOT_RADIUS - mean (R x^b)_z` written out in this
    file would agree by inspection, which is exactly what this project does
    not trust.  Only the height rows are exercised, so the jacobians and the
    joint rates are zero and no robot is needed.
    """
    from hw.balance import config as cfg
    from hw.balance import state as BS

    for rpy in TILTS:
        R, r = _scene(rpy)
        est = LinearKFPosVelEstimator()
        est.reset(R, r)
        out = to_floor(_at_rest(est, R, r), P.FOOT_RADIUS)

        x_w = r @ R.T                            # the feet as `state` sees them
        z_origin = BS._height(x_w, np.zeros((4, 3, 3)), np.zeros((4, 3)), R,
                              np.zeros(3), cfg.SRB, None)["z_origin"]
        assert abs(z_origin - TRUNK_ABOVE_FLOOR) < 1e-12, rpy
        assert abs(out.p_w[2] - z_origin) < 1e-9, (rpy, out.p_w[2], z_origin)

        # And the feet come back where they are: one radius above the floor.
        np.testing.assert_allclose(out.foot_w[:, 2], P.FOOT_RADIUS,
                                   rtol=0, atol=1e-9)


def test_taking_the_radius_off_r_in_the_trunk_frame_is_a_different_number():
    """The shortcut `robot_io` warns about, with its error measured.

    Subtracting the radius from `r[:, 2]` moves the point along the TRUNK's z,
    so what reaches the floor is `foot_radius * cos(roll) cos(pitch)` of it.
    Level, the two are the same and no bench test can tell them apart; at a
    tilt they are not.
    """
    level, tilted = np.array([0.0, 0.0, 0.3]), np.array([0.175, 0.0, 0.3])
    for rpy, floor in ((level, True), (tilted, False)):
        R, r = _scene(rpy)
        shortcut = r - np.array([0.0, 0.0, P.FOOT_RADIUS])

        est_a, est_b = LinearKFPosVelEstimator(), LinearKFPosVelEstimator()
        est_a.reset(R, r)
        est_b.reset(R, shortcut)
        right = to_floor(_at_rest(est_a, R, r), P.FOOT_RADIUS).p_w[2]
        wrong = _at_rest(est_b, R, shortcut).p_w[2]

        gap = right - wrong
        predicted = P.FOOT_RADIUS * (1.0 - R[2, 2])
        assert abs(gap - predicted) < 1e-9, (rpy, gap, predicted)
        if floor:
            assert abs(gap) < 1e-12                      # 10 deg is 0.23 mm
        else:
            assert 2.2e-4 < gap < 2.4e-4


class _StillImu:
    def __init__(self, sample):
        self.sample = sample

    def read(self):
        return self.sample


class _StillLegs:
    def __init__(self, sample):
        self.sample = sample

    def read(self):
        return self.sample


def test_run_once_hands_back_the_floor_datum():
    """`run_once` is `update` then `to_floor`, field for field."""
    rpy = TILTS[4]
    R, r = _scene(rpy)
    acc_b = R.T @ np.array([0.0, 0.0, G_AT_REST])
    rd = np.zeros((4, 3))
    imu = _StillImu((rpy, np.zeros(3), acc_b))
    legs = _StillLegs((r, rd))

    via_adapter, direct = LinearKFPosVelEstimator(), LinearKFPosVelEstimator()
    via_adapter.reset(R, r)
    direct.reset(R, r)
    for _ in range(20):
        a = run_once(imu, legs, via_adapter, 0.002, P.FOOT_RADIUS)
        b = to_floor(direct.update(R, np.zeros(3), acc_b, r, rd,
                                   np.full(4, 0.5), 0.002), P.FOOT_RADIUS)
        for field in dataclasses.fields(type(a)):
            np.testing.assert_array_equal(getattr(a, field.name),
                                          getattr(b, field.name), field.name)


def test_run_once_will_not_silently_default_the_radius():
    """The argument is required: a forgotten datum reads 15 mm low forever."""
    rpy = TILTS[0]
    R, r = _scene(rpy)
    imu = _StillImu((rpy, np.zeros(3), R.T @ np.array([0.0, 0.0, G_AT_REST])))
    legs = _StillLegs((r, np.zeros((4, 3))))
    est = LinearKFPosVelEstimator()
    est.reset(R, r)
    try:
        run_once(imu, legs, est, 0.002)
    except TypeError:
        pass
    else:
        raise AssertionError("run_once took no foot_radius and did not complain")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
