"""Where hardware becomes the estimator's arrays.  No bus, no thread, no clock.

    ImuSource.read()                     ->  rpy (3,), omega_b (3,), acc_b (3,)
    LegSource.read()                     ->  r (4, 3), rd (4, 3)
    OrientationImu(orientation)          ->  an ImuSource over ONE sweep's IMU
    BodyLegs(body)                       ->  a LegSource over ONE sweep's legs
    stance_phase(gait_phase, duty)       ->  contact phase (4,), the filter's
    to_floor(out, foot_radius)           ->  LKFOutput, world z from the FLOOR
    run_once(imu, leg, est, dt, r_f)     ->  LKFOutput, world z from the FLOOR

    The two concrete sources hold a PACKET AND A POSE, never a device: they
    are built fresh each sweep from the objects the control law is given that
    sweep, and read once.  Nothing in this file can block, and nothing in it
    can go stale between two calls, which is what keeps a 333 us slot a 333 us
    slot.  `hw.trot_esti` is the caller.

EVERY CONVERSION HAPPENS IN A read(), NEVER IN THE FILTER
    `estimator` takes SI, the TRUNK frame (x forward, y left, z up, origin at
    the centre of the abduction-axis plane) and FL FR RL RR row order.  Unit
    conversion, the sensor-to-trunk rotation and the hip offsets all happen
    before `run_once` sees an array, because inside the filter a wrong axis is
    indistinguishable from a biased sensor.

WHAT IN `hw` PRODUCES THESE -- AND `OrientationImu` / `BodyLegs` ARE IT, TYPED
    r, rd     `hw.kinematics.leg_state(i, q_i)` -> (x_b[i], J_i), hip included.
              r[i] = x_b[i] and rd[i] = J_i @ qd_i.  `hw.balance.state`'s
              `BodyState` already carries x_b, jac and qd, so a stand running
              the balance law has both for free.
    omega_b   `hw.imu.ImuDog.orientation().omega_b`.
    rpy       `sim.coordinates.zyx_from_rot(orientation.R)`, the TRUNK's
              triple.  `orientation.roll/.pitch/.yaw` are trunk-frame too --
              `hw.imu.sensor_to_trunk` has already turned NED into FLU and
              taken the trim off -- but they are not read back out of R, so
              the two part by R_BODY_IMU once the board is mounted at an
              angle.  R_BODY_IMU is the identity BY MEASUREMENT, so either
              route works today; taking it from R is the one that stays right.
    acc_b     `hw.imu.ImuDog.orientation().acc_b`, with `acc_age_s` beside it.
              The DETA10's 0x40 packet carries the specific force and
              `SENSOR_TO_TRUNK` puts it in the trunk, the same chain as the
              gyro.  MEASURED 2026-09-24, level and at rest: (0.020, 0.043,
              +9.764) -- specific force, +z up, SO NOTHING IS NEGATED.  Its
              magnitude is `hw.imu.G_AT_REST` and `LKFParams.g` carries the
              same number; see the note there for why the pair has to agree.
              The 0x40 stream is ~100 Hz on its own clock, independent of the
              0x41 attitude stream, which is why it is aged separately: check
              `acc_age_s`, not `age_s`, before believing it.  NaN means no
              packet has arrived -- reject it in `read()`, per the contract
              below.  The lever arm from the board to the trunk origin is
              ignored, as MIT ignores it.

THE HEIGHT DATUM, AND WHO SPENDS IT
    With `hw.kinematics`, r is the foot ball's CENTRE, so the filter's z = 0
    is the plane of the ball centres -- `build_measurement` pins it there, by
    telling the filter a planted foot sits at z = 0 -- and every world z it
    reports is one ball radius low.  `to_floor` moves the datum and `run_once`
    has already applied it, so what an adapter hands back is on the same
    number as `hw.balance.state`'s `z_origin`:

        z_origin = FOOT_RADIUS  - mean_i (R x_i^b)_z        state._height
        p_w[2]   = foot_radius  - mean_i (R r_i)_z          reset + to_floor

    IT IS A WORLD-FRAME SHIFT, AND ONLY THERE IS IT A CONSTANT.  A ball on
    flat ground touches exactly `foot_radius` below its centre along WORLD z
    whatever the trunk is doing.  Taking the radius off `r[:, 2]` inside a
    `LegSource` instead -- the obvious shortcut -- moves the point in the
    TRUNK frame, and two things go wrong at a tilt: the datum itself is off by
    foot_radius * (1 - cos roll cos pitch), which is only 0.23 mm at 10 deg,
    but r also stops being the point rd was differentiated from, so the
    omega x r term and the foot-height rows carry a tilt-dependent bias
    (15 mm/s per rad/s of turn, at a 15 mm ball).  The filter does not know it
    has feet with a radius.  Keep it that way and pay for it here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from sim import coordinates as C

from ..estimator import LinearKFPosVelEstimator, LKFOutput
from .rotation import rpy_zyx_to_R

__all__ = ("ImuSource", "LegSource", "OrientationImu", "BodyLegs",
           "stance_phase", "to_floor", "run_once")


class ImuSource(ABC):
    """The IMU as the filter needs it.  Subclass and implement `read`."""

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The latest sample, SI, already in the TRUNK frame.

        Returns:
            rpy      (3,)  rad    the TRUNK's ZYX (roll, pitch, yaw), the
                                  convention of `rotation.rpy_zyx_to_R`
            omega_b  (3,)  rad/s  angular velocity, TRUNK axes
            acc_b    (3,)  m/s^2  specific force, TRUNK axes

        Must not block.  The IMU streams on its own clock, so return the most
        recent packet whatever its age -- a sweep that waits for the IMU
        misses its CAN deadline.  Reject non-finite values here: one NaN
        reaching `update` poisons x and P until the next `reset`.
        """
        raise NotImplementedError


class LegSource(ABC):
    """The legs as the filter needs them.  Subclass and implement `read`."""

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """This sweep's feet, SI, TRUNK frame, rows FL FR RL RR.

        Returns:
            r   (4, 3)  m    each foot point about the trunk ORIGIN, hip
                             offset included -- forward kinematics, not a
                             hip-frame vector
            rd  (4, 3)  m/s  J_i @ qd_i: the foot's velocity with the trunk
                             held still.  Leave omega x r out; the filter
                             adds it.
        """
        raise NotImplementedError


class OrientationImu(ImuSource):
    """ONE sweep's `hw.imu.TrunkOrientation` as an `ImuSource`.

    Built per sweep from the same object the control law is handed that sweep,
    so the filter and the law cannot be looking at two different packets.  It
    holds no device: `read` does arithmetic, and the only way it fails is the
    one below.

    rpy IS READ BACK OUT OF R, not taken from `.roll/.pitch/.yaw`.  Those three
    are trunk-frame as well and R is built from them today, so the two routes
    agree exactly -- but they part company by `R_BODY_IMU` the day the board is
    mounted at an angle, and R is the matrix the law multiplies by, so R is what
    the filter's world has to be defined against.  `rpy_zyx_to_R` rebuilds it to
    1e-12 (`tests/test_layers.py` gates the round trip, `test_dog6_sources.py`
    gates this class against it).

    THE TRIM IS ON R AND NOT ON acc_b.  `hw.imu` subtracts the mount/floor trim
    from roll and pitch before building R; the 0x40 accelerometer comes through
    untrimmed.  With no `hw/imu_calib.json` the trim is 0.0 and the two agree
    exactly, which is DOG6 today.  Capture one and this pair is off by it:
    a_w = R acc_b + (0, 0, -g) then carries g*sin(trim) of horizontal
    acceleration, 0.17 m/s^2 per degree.  `hw.velocity_estimator` takes the
    UNTRIMMED attitude for exactly this reason -- gravity does not care about
    the floor -- and this class is the place to do the same if a trim is ever
    captured.
    """

    def __init__(self, orientation) -> None:
        self.orientation = orientation

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        o = self.orientation
        acc_b = np.asarray(o.acc_b, dtype=float).reshape(3)
        if not bool(np.all(np.isfinite(acc_b))):
            raise ValueError(
                "the accelerometer reads %s: no 0x40 packet has arrived on "
                "this DETA10, or the stream stopped.  ONE NaN REACHING THE "
                "FILTER POISONS x AND P until the next reset(), so the sweep "
                "is refused here instead." % np.array2string(acc_b))
        R = np.asarray(o.R, dtype=float).reshape(3, 3)
        return (np.array(C.zyx_from_rot(R)),
                np.asarray(o.omega_b, dtype=float).reshape(3),
                acc_b)


class BodyLegs(LegSource):
    """ONE sweep's `hw.balance.state.BodyState` as a `LegSource`.

    `x_b` is already the foot point about the trunk origin with the hip offset
    in it, and `jac` is already the trunk-frame Jacobian, so the whole of this
    class is one einsum: the balance law pays for the kinematics and the filter
    reads them for free, off the same sweep's measurement.

    `qd` is the FILTERED encoder difference the law damps with, not the
    driver's speed field -- `hw.calibration.joint_state` says why that field is
    not trusted alone.  Whatever noise is on it lands on rd, and rd's rows are
    the ones the velocity estimate is made of.
    """

    def __init__(self, body) -> None:
        self.body = body

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        body = self.body
        r = np.asarray(body.x_b, dtype=float).reshape(C.N_LEGS, 3)
        jac = np.asarray(body.jac, dtype=float).reshape(C.N_LEGS, 3, 3)
        qd4 = C.unflat(np.asarray(body.qd, dtype=float))
        return r, np.einsum("kij,kj->ki", jac, qd4)


def stance_phase(gait_phase: np.ndarray, duty: float) -> np.ndarray:
    """`balance.gait.TrotGait.phase` -> the filter's contact phase, (4,) -> (4,).

        gait phase   0 touchdown ....... duty liftoff ....... 1 touchdown
        contact      0 ramps up ....... 1    0 all the way through swing

    The gait counts a whole CYCLE from 0 to 1 with liftoff at `duty`.  The
    filter counts one STANCE from 0 to 1 and wants exactly 0 for a leg in the
    air (`estimator.trapezoid_trust`, and MIT's own convention).  So stance
    progress is phase/duty and the swing window collapses onto 0 -- which is
    the only trust a swinging leg can be given: its foot is not on the floor,
    so neither its velocity row nor its height row means anything.

    THE SCHEDULE IS NOT A CONTACT SENSOR.  This says what the clock INTENDED,
    which is the same thing `hw.balance.state.on_stance` averages the height
    over.  A foot that misses the floor is trusted here anyway -- and the trust
    ramp at each end of the stance is what keeps that from arriving as a step.
    """
    phase = np.asarray(gait_phase, dtype=float).reshape(C.N_LEGS)
    assert 0.0 < duty < 1.0, f"duty must lie in (0, 1), got {duty}"
    return np.where(phase < duty, phase / duty, 0.0)


def to_floor(out: LKFOutput, foot_radius: float) -> LKFOutput:
    """`out` with world z measured from the FLOOR instead of the ball centres.

    The whole world z datum moves up by one ball radius, which is a shift of
    the frame and not a correction of the estimate: `p_w[2]` and every
    `foot_w[:, 2]` gain `foot_radius`, and `v_w`, `v_b`, `trust` and `innov`
    come back untouched, because a constant datum has no rate and the filter
    was never told about it.  The estimator's own x and P are not touched
    either -- it goes on working in the frame its measurement rows define, and
    every sweep is shifted the same way on the way out.

    `foot_radius` is `sim.params.FOOT_RADIUS` (15 mm) for DOG6's ball feet.
    Pass 0.0 only for a `LegSource` whose r is already the contact point --
    and see the module docstring before building one that way.
    """
    assert np.isfinite(foot_radius) and foot_radius >= 0.0, \
        f"foot_radius must be a finite radius in m, got {foot_radius}"
    p_w = out.p_w.copy()
    p_w[2] += foot_radius
    foot_w = out.foot_w.copy()
    foot_w[:, 2] += foot_radius
    return replace(out, p_w=p_w, foot_w=foot_w)


def run_once(imu: ImuSource, leg: LegSource, est: LinearKFPosVelEstimator,
             dt: float, foot_radius: float,
             contact_phase: np.ndarray = None) -> LKFOutput:
    """This sweep's estimate, made BEFORE anything controls on it.

        out = run_once(imu, leg, est, dt, R_F)  1  read the sensors, estimate
        tau = controller(out, ...)              2  control on THIS sweep's state
        bus.send(tau)                           3  then actuate

    Estimating after control would hand the controller the previous sweep's
    state -- one period of delay on every feedback path, with nothing in the
    output to say so.

    FOUR FEET DOWN IS THE DEFAULT.  `contact_phase` None is 0.5 on every leg,
    which is trust 1 on every leg: that is the balance stand.  A gait passes
    its own schedule instead -- `stance_phase(gait.phase(now), gait.duty)` --
    because a swinging leg that is handed 0.5 tells the filter its foot is
    planted on the floor it is 40 mm above.

    THIS IS THE ONE PLACE THE CALL ORDER IS WRITTEN DOWN, which is why the
    gait goes through here rather than reaching for `est.update` itself:
    `tests/test_layers.py` pins the order against MIT's `run()`, and a second
    copy of it in a caller is a copy that can drift from the pinned one.

    `dt` is the measured sweep period, already clamped by the caller: one
    stalled sweep taken at face value integrates its whole gap in one step.
    Nothing here reads a clock.

    `foot_radius` is required rather than defaulted because a silent 0.0 is
    the bug this argument exists to prevent: the estimate stays plausible,
    stays smooth, and reads 15 mm low against every other height in `hw`.
    """
    rpy, omega_b, acc_b = imu.read()
    r, rd = leg.read()
    R_wb = rpy_zyx_to_R(rpy)
    if contact_phase is None:
        contact_phase = np.full(C.N_LEGS, 0.5)
    out = est.update(R_wb, omega_b, acc_b, r, rd, contact_phase, dt)
    return to_floor(out, foot_radius)
