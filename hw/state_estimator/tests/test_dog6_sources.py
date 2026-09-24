"""The two concrete sources, the gait's phases, and the whole tap end to end.

    python -m pytest hw/state_estimator/tests/test_dog6_sources.py
    python -m hw.state_estimator.tests.test_dog6_sources     no pytest needed

`adapters.OrientationImu` and `adapters.BodyLegs` are where `hw`'s own objects
become the filter's arrays, and `hw.trot_esti.EstimatorTap` is the only caller.
A wrong axis, a wrong row order or a wrong Jacobian is absorbed by the filter
as a bias and the innovation stays plausible, so each one is checked against
something that is NOT a second copy of the same expression:

    rpy      against the R the control law itself multiplies by
    rd       against a FINITE DIFFERENCE of the forward kinematics
    phase    against the gait's own `contact`, and against the trust it makes
    the tap  against `hw.balance.state`'s height, over a whole sequence

No robot and no IMU: plain asserts, hand-built states, `TrunkOrientation`
stands in for the sensor.
"""
from __future__ import annotations

import math

import numpy as np

from hw import imu as IMU
from hw import kinematics as HK
from hw import trot_esti as TE
from hw.balance import gait as GAIT
from hw.balance import posture as POSE
from hw.balance import state as BSTATE
from hw.state_estimator.adapters import (BodyLegs, OrientationImu, run_once,
                                         stance_phase)
from hw.state_estimator.estimator import (LinearKFPosVelEstimator,
                                          trapezoid_trust)
from sim import coordinates as C
from sim import params as P

#: The attitudes every claim is made at.  A level trunk cannot tell a right
#: frame from several wrong ones, so it is the control and not the evidence.
RPYS = [(0.0, 0.0, 0.0), (0.09, -0.05, 0.0), (0.0, 0.0, 0.7),
        (-0.12, 0.17, -2.1)]


def _body(q=None, qd=None, rpy=(0.0, 0.0, 0.0)):
    """One sweep's `BodyState` from joint angles and an attitude.  No hardware."""
    q = POSE.NOMINAL.q if q is None else q
    qd = np.zeros(C.N_JOINTS) if qd is None else qd
    roll, pitch, yaw = rpy
    orientation = IMU.TrunkOrientation(
        R=IMU.trunk_rotation(roll, pitch, yaw), omega_b=np.zeros(3),
        roll=roll, pitch=pitch, yaw=yaw, age_s=0.0,
        acc_b=IMU.trunk_rotation(roll, pitch, yaw).T
        @ np.array([0.0, 0.0, IMU.G_AT_REST]), acc_age_s=0.0)
    return BSTATE.read(C.flat(q), qd, orientation), orientation


# -- OrientationImu -----------------------------------------------------------
def test_orientation_imu_hands_back_the_law_s_own_rotation():
    """rpy is read out of R, so `rpy_zyx_to_R` must rebuild R exactly.

    `run_once` turns the triple back into a matrix, and that matrix is the
    world the filter's every foot and velocity is expressed in.  If the round
    trip is not the law's R, the filter is standing in a different world from
    the controller and nothing in either says so.
    """
    for rpy in RPYS:
        _, orientation = _body(rpy=rpy)
        out_rpy, omega_b, acc_b = OrientationImu(orientation).read()
        assert out_rpy.shape == (3,)
        np.testing.assert_allclose(out_rpy, rpy, rtol=0, atol=1e-12)
        from hw.state_estimator.adapters import rpy_zyx_to_R
        np.testing.assert_allclose(rpy_zyx_to_R(out_rpy), orientation.R,
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(omega_b, orientation.omega_b)
        np.testing.assert_array_equal(acc_b, orientation.acc_b)


def test_orientation_imu_refuses_a_non_finite_accelerometer():
    """NaN until the first 0x40 packet -- and one NaN poisons x and P."""
    _, orientation = _body()
    blind = IMU.TrunkOrientation(
        R=orientation.R, omega_b=orientation.omega_b, roll=0.0, pitch=0.0,
        yaw=0.0, age_s=0.0)                  # acc_b defaults to NaN
    assert not np.all(np.isfinite(blind.acc_b))
    try:
        OrientationImu(blind).read()
    except ValueError as refusal:
        assert "0x40" in str(refusal), refusal
    else:
        raise AssertionError("a NaN accelerometer was handed to the filter")


def test_a_level_trunk_at_rest_is_the_filter_s_own_g():
    """`TrunkOrientation.level()` is the `--no-imu` sweep: (0, 0, +g), not 0."""
    rpy, _, acc_b = OrientationImu(IMU.TrunkOrientation.level()).read()
    np.testing.assert_array_equal(rpy, np.zeros(3))
    np.testing.assert_allclose(acc_b, [0.0, 0.0, TE.LKFParams().g],
                               rtol=0, atol=1e-12)


# -- BodyLegs -----------------------------------------------------------------
def test_body_legs_r_is_the_foot_about_the_trunk_origin():
    body, _ = _body()
    r, _ = BodyLegs(body).read()
    assert r.shape == (C.N_LEGS, 3)
    for i in range(C.N_LEGS):
        x_b, _ = HK.leg_state(i, C.unflat(POSE.NOMINAL.q)[i])
        np.testing.assert_allclose(r[i], x_b, rtol=0, atol=1e-15)


def test_body_legs_rd_is_the_foot_velocity_the_legs_alone_make():
    """rd against a finite difference of the FK -- not against J qd again.

    `build_measurement` adds omega x r itself, so rd must be the foot's motion
    with the TRUNK HELD STILL: exactly d/dt fk(q + t qd) at t = 0.
    """
    rng = np.random.default_rng(7)
    q = POSE.NOMINAL.q
    for _ in range(5):
        qd4 = rng.uniform(-1.5, 1.5, size=(C.N_LEGS, 3))
        body, _ = _body(q=q, qd=C.flat(qd4))
        _, rd = BodyLegs(body).read()
        h = 1e-7
        for i in range(C.N_LEGS):
            ahead, _ = HK.leg_state(i, C.unflat(q)[i] + h * qd4[i])
            behind, _ = HK.leg_state(i, C.unflat(q)[i] - h * qd4[i])
            np.testing.assert_allclose(rd[i], (ahead - behind) / (2.0 * h),
                                       rtol=1e-6, atol=1e-9)


# -- stance_phase -------------------------------------------------------------
def test_stance_phase_is_zero_exactly_where_the_gait_lifts_the_foot():
    gait = GAIT.TrotGait(period=0.8)
    gait.reset(0.0)
    for t in np.arange(0.0, 4.0, 0.004):
        ph = gait.phase(t)
        phase = stance_phase(ph, gait.duty)
        contact = gait.contact(t)
        assert phase.shape == (C.N_LEGS,)
        assert np.all(phase[~contact] == 0.0), (t, phase, contact)
        assert np.all(phase[contact] >= 0.0) and np.all(phase[contact] <= 1.0)
        # Stance progress, not cycle progress: the same ordering as the gait's
        # own phase within the stance, reaching 1 as the foot is about to lift.
        np.testing.assert_allclose(phase[contact], ph[contact] / gait.duty)


def test_the_filter_s_trust_ramp_is_wider_than_the_gait_s_load_ramp():
    """MEASURED, not assumed: at the four-foot window trust is 0.9375, not 1.

    The gait's own load ramp is `config.CONTACT_RAMP` = 0.15 of stance and the
    filter's trust ramp is MIT's `trust_window` = 0.20 of it, so at the freeze
    point -- where the gait has every foot at FULL weight, and where the settle
    re-levels -- the filter is still fading two of them in.  0.9375 of trust is
    s = 1 + 100 (1 - tau) = 7.25, which down-weights those legs' rows sevenfold
    at exactly the instant they are carrying everything.

    Nothing is tuned here to fix that: `trust_window` is MIT's number and
    `CONTACT_RAMP` is DOG5's flown one, and the day the filter enters the loop
    is the day one of them has to give.  This test is where it will be noticed.
    """
    gait = GAIT.TrotGait(period=0.8)
    gait.reset(10.0)
    np.testing.assert_allclose(gait.contact_weight(10.0), np.ones(C.N_LEGS))
    tau = trapezoid_trust(stance_phase(gait.phase(10.0), gait.duty),
                          TE.LKFParams().trust_window)
    np.testing.assert_allclose(tau, np.full(C.N_LEGS, 0.9375), rtol=0,
                               atol=1e-12)

    # Past the ramps the trapezoid is flat: a leg between `window` and
    # 1 - window of its stance is believed completely.  There is no instant
    # where BOTH diagonals are there -- they are half a cycle apart -- so the
    # claim is made per leg, over three cycles.
    window = TE.LKFParams().trust_window
    flat = 0
    for t in np.arange(10.0, 12.4, 0.002):
        phase = stance_phase(gait.phase(t), gait.duty)
        tau = trapezoid_trust(phase, window)
        inner = (phase >= window) & (phase <= 1.0 - window)
        np.testing.assert_allclose(tau[inner], np.ones(inner.sum()), rtol=0,
                                   atol=1e-12)
        flat += int(inner.sum())
    assert flat > 0, "no leg was ever past the trust ramp"


def test_a_swinging_diagonal_is_not_believed_at_all():
    gait = GAIT.TrotGait(period=0.8)
    gait.reset(0.0)
    window = TE.LKFParams().trust_window
    swung = np.zeros(C.N_LEGS, dtype=bool)
    for t in np.arange(0.0, 1.6, 0.002):
        tau = trapezoid_trust(stance_phase(gait.phase(t), gait.duty), window)
        air = ~gait.contact(t)
        assert np.all(tau[air] == 0.0), (t, tau, air)
        swung |= air
    assert np.all(swung), "no leg ever left the floor in 1.6 s of trot"


def test_run_once_takes_the_gait_s_phases_without_reordering_them():
    body, orientation = _body(rpy=RPYS[1])
    r, rd = BodyLegs(body).read()
    phase = np.array([0.0, 0.5, 0.5, 0.0])       # FR/RL down, FL/RR in the air
    a, b = LinearKFPosVelEstimator(), LinearKFPosVelEstimator()
    a.reset(body.R, r)
    b.reset(body.R, r)
    for _ in range(10):
        one = run_once(OrientationImu(orientation), BodyLegs(body), a, 0.004,
                       P.FOOT_RADIUS, phase)
        rpy, omega_b, acc_b = OrientationImu(orientation).read()
        two = b.update(body.R, omega_b, acc_b, r, rd, phase, 0.004)
        np.testing.assert_allclose(one.trust, [0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(one.v_w, two.v_w, rtol=0, atol=1e-12)
        assert abs(one.p_w[2] - (two.p_w[2] + P.FOOT_RADIUS)) < 1e-12


# -- the tap ------------------------------------------------------------------
class _FakeStand:
    """The three attributes `EstimatorTap` reads off `StandSequence`."""

    def __init__(self, phase_name="hold", yaw_offset=0.0, gait=None):
        self.phase_name = phase_name
        self.yaw_offset = yaw_offset
        self.gait = gait
        self.step_gait = gait


def _run_tap(tap, stand, body, orientation, sweeps, t0=0.0, dt=0.004):
    lines = []
    for k in range(sweeps):
        said = tap.update(t0 + k * dt, stand, body, orientation)
        if said:
            lines.append(said)
    return lines


def test_the_tap_converges_on_balance_state_s_own_height():
    """The claim the whole read-out is for, at four tilts.

    A still robot on four feet: the filter's trunk height must come back on
    `BodyState.h` -- the number `hw.stand` prints and a ruler reads -- and its
    velocity must stay at zero.  The accelerometer here is exactly gravity, so
    any height that walks is a datum or a frame error, not noise.
    """
    for rpy in RPYS:
        body, orientation = _body(rpy=rpy)
        tap = TE.EstimatorTap(None)
        _run_tap(tap, _FakeStand(), body, orientation, 400)
        h = BSTATE.origin_to_height(tap.out.p_w[2])
        assert abs(h - body.h) < 1e-3, (rpy, 1e3 * h, 1e3 * body.h)
        assert np.all(np.abs(tap.out.v_w) < 1e-3), (rpy, tap.out.v_w)
        np.testing.assert_allclose(tap.out.trust, np.ones(C.N_LEGS))


def test_the_tap_resets_when_the_heading_is_zeroed_at_the_handover():
    """The world frame turns once; the filter's stored feet must not be left
    behind in the old one."""
    body, orientation = _body(rpy=(0.03, -0.02, 0.9))
    tap = TE.EstimatorTap(None)
    stand = _FakeStand(phase_name="crouch")
    _run_tap(tap, stand, body, orientation, 200)
    before = tap.out.p_w[2]
    assert tap.world == 0.0

    stand.phase_name, stand.yaw_offset = "rise", 0.9      # the handover
    lines = _run_tap(tap, stand, body, orientation, 200)
    assert any("reset" in line for line in lines), lines
    assert tap.world == 0.9
    # The height is yaw-free, so the turn must not have moved it at all.
    assert abs(tap.out.p_w[2] - before) < 1e-6, (before, tap.out.p_w[2])
    # And the world the filter reports in is the run's: the trunk's heading
    # reads zero there, so a foot ahead of the trunk is at +x.
    assert tap.out.foot_w[0, 0] > tap.out.foot_w[2, 0]    # FL ahead of RL


def test_the_tap_waits_for_the_accelerometer_and_then_starts():
    body, _ = _body()
    blind = IMU.TrunkOrientation(R=body.R, omega_b=np.zeros(3), roll=0.0,
                                pitch=0.0, yaw=0.0, age_s=0.0)
    tap = TE.EstimatorTap(None)
    lines = _run_tap(tap, _FakeStand(), body, blind, 50)
    assert tap.out is None and "0x40" in tap.refusal
    assert len(lines) == 1 and lines[0].startswith("estimator waiting"), lines
    assert "waiting" in tap.status() or "0x40" in tap.status()

    good = IMU.TrunkOrientation.level()
    _run_tap(tap, _FakeStand(), body, good, 200, t0=0.2)
    assert tap.out is not None and tap.refusal is None


def test_the_tap_prints_one_reading_per_phase():
    body, orientation = _body()
    tap = TE.EstimatorTap(None)
    stand = _FakeStand(phase_name="limp")
    said = []
    for phase in ("limp", "settle", "crouch"):
        stand.phase_name = phase
        said += [line for line in _run_tap(tap, stand, body, orientation, 100)
                 if line.startswith("estimator, ")]
    assert len(said) == 3, said
    for phase, line in zip(("limp", "settle", "crouch"), said):
        assert line.startswith("estimator, " + phase), line
    assert "\n" not in "".join(said), "a per-phase reading is one line"
    assert len(tap.status().splitlines()) == 2, tap.status()


def test_a_broken_tap_disables_itself_instead_of_ending_the_run():
    """A read-out is not allowed to drop the robot."""
    body, orientation = _body()
    tap = TE.EstimatorTap(None)
    _run_tap(tap, _FakeStand(), body, orientation, 10)

    class _Exploding(_FakeStand):
        @property
        def phase_name(self):
            raise RuntimeError("boom")

        @phase_name.setter
        def phase_name(self, value):
            pass

    said = _run_tap(tap, _Exploding(), body, orientation, 5)
    assert len(said) == 1 and "estimator OFF" in said[0], said
    assert "boom" in tap.broken and "OFF" in tap.status()
    # And it stays off, silently, for the rest of the run.
    assert _run_tap(tap, _FakeStand(), body, orientation, 5) == []


def test_the_tap_clamps_a_stalled_sweep():
    """dt is the MEASURED period, and one stalled sweep must not integrate its
    whole gap: past the clamp the gap is integrated as the clamp."""
    body, orientation = _body()
    tap = TE.EstimatorTap(None)
    stand = _FakeStand()
    tap.update(0.0, stand, body, orientation)
    tap.update(0.004, stand, body, orientation)
    tap.update(4.004, stand, body, orientation)           # a 4 s stall
    assert tap.dt == TE.DT_CLAMP[1], tap.dt


def test_the_tap_holds_no_reference_to_the_sweep_s_arrays():
    """The filter's output is a copy: editing it cannot reach the estimator."""
    body, orientation = _body()
    tap = TE.EstimatorTap(None)
    _run_tap(tap, _FakeStand(), body, orientation, 20)
    z = tap.out.p_w[2]
    tap.out.p_w[2] += 1.0
    assert tap.est.x[2] == z - P.FOOT_RADIUS


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)


def test_the_tap_reads_a_forward_walk_as_plus_x_at_any_heading():
    """THE 2026-09-24 SIGN FLIP.  The legs came through `rezero_yaw` and the
    IMU did not, so the filter integrated in the magnetometer's world while
    the run's had x forward.  At that day's 170 deg heading, forward printed
    as -x and left as -y.  A trunk walking forward over planted feet must read
    +x whatever the magnetometer calls north.
    """
    from dataclasses import replace
    v, dt, sweeps = 0.05, 0.004, 500
    for heading in (0.0, np.radians(170.0), np.radians(-95.0)):
        body0, orientation = _body(rpy=(0.02, -0.01, heading))
        stand = _FakeStand(phase_name="hold", yaw_offset=heading)
        tap = TE.EstimatorTap(None)
        tap.update(0.0, stand, body0, orientation)             # the reset
        assert tap.world == heading
        for k in range(1, sweeps + 1):
            slide = np.array([-v * k * dt, 0.0, 0.0])          # feet go -x ...
            qd4 = np.array([np.linalg.solve(body0.jac[i], [-v, 0.0, 0.0])
                            for i in range(C.N_LEGS)])          # ... at -v
            body = replace(body0, x_b=body0.x_b + slide, qd=C.flat(qd4))
            tap.update(k * dt, stand, body, orientation)
        walked = v * sweeps * dt
        x, y = tap.out.p_w[0], tap.out.p_w[1]
        assert abs(x - walked) < 0.01, (np.degrees(heading), x, walked)
        assert abs(y) < 0.01, (np.degrees(heading), y)
        assert abs(tap.out.v_w[0] - v) < 0.01, (np.degrees(heading), tap.out.v_w)
