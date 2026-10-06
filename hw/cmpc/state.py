"""What the MPC is handed on the robot, and where each number comes from.

    hw.balance.state.BodyState (sensors)  ->  sim.cmpc.controller.BodyState

THE SIMULATOR HANDS THE CONTROLLER GROUND TRUTH.  THE ROBOT HAS NONE.
    `sim.cmpc` reads the twelve-state body pose straight out of MuJoCo and
    says so in its first line.  On DOG6 the same twelve numbers have to be
    built from an IMU that reports attitude and rates, and encoders that
    report joint angles -- there is no position sensor and there is no
    velocity sensor.  This file is the smallest construction that closes the
    loop, and it is honest about which of its outputs are measured, which are
    derived under an assumption, and which are integrated:

        rotation, roll, pitch   MEASURED.  The DETA10's, through `hw.imu`,
                                with the mounting rotation applied.
        yaw                     INTEGRATED from the gyro.  The magnetometer
                                heading is logged beside it and never used:
                                it sits beside twelve motors and a steel
                                frame, `hw.imu` labels it untrusted, and the
                                MPC's yaw row is weighted 10 -- a heading
                                that jumps is a yaw moment the robot will
                                try to produce.  The integral drifts slowly
                                with gyro bias, which the reference does not
                                see because it is anchored on the same yaw at
                                the handover.  The world x axis is the trunk's
                                heading at that instant.
        omega (world)           MEASURED, R omega^b.
        z                       DERIVED from the stance feet under the
                                assumption that they are on the floor:
                                z = FOOT_RADIUS - mean_stance (R x_i^b)_z,
                                `hw.balance.state`'s expression, rotated,
                                over the feet the schedule says are down.
        velocity                DERIVED from the stance feet under the same
                                assumption: a planted foot is still in the
                                world, so the trunk moves at minus the foot's
                                velocity relative to it,
                                    v = -mean_stance (R J_i qdot_i + omega x R x_i^b),
                                low-passed at `config.VELOCITY_FILTER_HZ`.
        x, y                    INTEGRATED from that velocity.  Leg odometry,
                                nothing else; it drifts with every slip, and
                                the MPC's position weights (2.0 against z's
                                50) are the paper's own acknowledgement that
                                the horizontal reference is bookkeeping.

    Every one of the DERIVED rows is wrong the moment a foot the schedule
    calls planted is not.  That is the same assumption `hw.balance.state`
    makes, the one the 2026-09-15 runs showed the robot can violate, and
    nothing here can detect it -- there is no contact sensor.  The IMU
    accelerometer is not fused; a filter that uses it belongs in its own
    module when the kinematic estimate has been seen to be the limit.

WHAT GOES WITH THE STATE
    `feet_body` and `jacobians_body` -- the closed-form kinematics already
    computed for `hw.balance.state.read` -- ride along, so the controller's
    250 Hz half does no chain walk.  See `sim.cmpc.controller.BodyState`.

    `imu_fresh` is derived from the packet age: an age that did not grow
    since the last sweep is a new packet.  The no-IMU `TrunkOrientation.level`
    has age zero always and so reads fresh every sweep, which is right -- its
    R is constant and holding it costs nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__ in (None, ""):        # allow `python hw/cmpc/state.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.cmpc"

from sim import coordinates as C             # noqa: E402
from sim import params as P                  # noqa: E402
from sim.cmpc import controller as CTL       # noqa: E402

from . import config as cfg                  # noqa: E402

__all__ = ["Estimate", "KinematicOdometry", "euler_yaw_rate"]


def euler_yaw_rate(roll: float, pitch: float, omega_b) -> float:
    """psi_dot for ZYX Euler angles from the body rates.  Exact, not the
    small-angle Rz(psi)^T omega the MPC's own model uses -- the integral has
    to be right over minutes, the model only over a 0.25 s horizon."""
    wx, wy, wz = (float(v) for v in np.asarray(omega_b, dtype=float).reshape(3))
    return (wy * np.sin(roll) + wz * np.cos(roll)) / np.cos(pitch)


@dataclass
class Estimate:
    """One sweep's estimate, with the provenance in the field names."""

    state: CTL.BodyState         # what the controller is handed
    imu_fresh: bool              # a new IMU packet arrived this sweep
    imu_held: bool               # the IMU is stale; attitude is the last one
    yaw_mag: float               # rad, the magnetometer's -- LOGGED, not used
    n_stance: int                # feet the z and velocity were taken over


class KinematicOdometry:
    """The estimator.  `reset` at the handover, `update` once per sweep."""

    def __init__(self, velocity_filter_hz: float = cfg.VELOCITY_FILTER_HZ):
        self.velocity_filter_hz = float(velocity_filter_hz)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.R = np.eye(3)
        self.roll = 0.0
        self.pitch = 0.0
        self.t_prev: float | None = None
        self.imu_age_prev = float("inf")
        self.started = False

    def reset(self, body, now: float) -> None:
        """Latch the start: xy and yaw at zero, z from the feet, no velocity."""
        self.__init__(self.velocity_filter_hz)
        self.roll, self.pitch = float(body.roll), float(body.pitch)
        self.R = C.rot_zyx(self.roll, self.pitch, 0.0)
        x_w = body.x_b @ self.R.T
        self.position = np.array([0.0, 0.0, P.FOOT_RADIUS - x_w[:, 2].mean()])
        self.t_prev = float(now)
        self.imu_age_prev = float(body.imu_age_s)
        self.started = True

    def update(self, body, now: float, contacts) -> Estimate:
        """`body` is `hw.balance.state.BodyState`; `contacts` is (4,) bool."""
        if not self.started:
            self.reset(body, now)
        dt = max(0.0, float(now) - self.t_prev)
        self.t_prev = float(now)

        # -- attitude: measured roll and pitch, integrated yaw ---------------
        fresh = float(body.imu_age_s) <= self.imu_age_prev
        self.imu_age_prev = float(body.imu_age_s)
        held = bool(body.imu_stale)
        if not held:
            self.roll, self.pitch = float(body.roll), float(body.pitch)
            if abs(self.pitch) < cfg.YAW_RATE_PITCH_LIMIT_RAD:
                self.yaw_rate = euler_yaw_rate(self.roll, self.pitch,
                                               body.omega_b)
        self.yaw += self.yaw_rate * dt
        R = C.rot_zyx(self.roll, self.pitch, self.yaw)
        self.R = R
        omega_w = R @ np.asarray(body.omega_b, dtype=float)

        # -- the stance feet, in the world the estimate defines ---------------
        contacts = np.asarray(contacts, dtype=bool).reshape(C.N_LEGS)
        stance = np.flatnonzero(contacts)
        x_w = body.x_b @ R.T                                 # (4, 3)
        if stance.size:
            qd4 = C.unflat(body.qd)
            xdot_b = np.einsum("kij,kj->ki", body.jac, qd4)  # J_i qdot_i
            xdot_w = xdot_b @ R.T + np.cross(omega_w, x_w)
            v_raw = -xdot_w[stance].mean(axis=0)
            # z is read, not integrated: a planted foot is a height sensor.
            z = float(P.FOOT_RADIUS - x_w[stance, 2].mean())
        else:                                 # no foot down: coast
            v_raw = self.velocity
            z = self.position[2] + self.velocity[2] * dt

        alpha = 1.0 - np.exp(-2.0 * np.pi * self.velocity_filter_hz * dt)
        self.velocity = self.velocity + alpha * (v_raw - self.velocity)
        self.position = np.array([
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
            z])

        state = CTL.BodyState(
            position=self.position.copy(), rotation=R,
            rpy=np.array([self.roll, self.pitch, self.yaw]),
            velocity=self.velocity.copy(), omega=omega_w,
            q=C.unflat(body.q).copy(), qd=C.unflat(body.qd).copy(),
            feet_body=np.asarray(body.x_b, dtype=float),
            jacobians_body=np.asarray(body.jac, dtype=float))
        return Estimate(state=state, imu_fresh=bool(fresh), imu_held=held,
                        yaw_mag=float(body.yaw), n_stance=int(stance.size))

    def status(self) -> str:
        return ("est z %6.1f mm  v %+.3f/%+.3f  yaw %+6.1f deg"
                % (1e3 * self.position[2], self.velocity[0], self.velocity[1],
                   np.degrees(self.yaw)))
