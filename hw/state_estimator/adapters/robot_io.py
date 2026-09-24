"""Where hardware becomes the estimator's arrays.  Interfaces only -- no driver.

    ImuSource.read()              ->  rpy (3,), omega_b (3,), acc_b (3,)
    LegSource.read()              ->  r (4, 3), rd (4, 3)
    run_once(imu, leg, est, dt)   ->  LKFOutput

EVERY CONVERSION HAPPENS IN A read(), NEVER IN THE FILTER
    `estimator` takes SI, the TRUNK frame (x forward, y left, z up, origin at
    the centre of the abduction-axis plane) and FL FR RL RR row order.  Unit
    conversion, the sensor-to-trunk rotation and the hip offsets all happen
    before `run_once` sees an array, because inside the filter a wrong axis is
    indistinguishable from a biased sensor.

WHAT IN `hw` ALREADY PRODUCES THESE, AND WHAT DOES NOT
    r, rd     `hw.kinematics.leg_state(i, q_i)` -> (x_b[i], J_i), hip included.
              r[i] = x_b[i] and rd[i] = J_i @ qd_i.  `hw.balance.state`'s
              `BodyState` already carries x_b, jac and qd, so a stand running
              the balance law has both for free.
    omega_b   `hw.imu.ImuDog.orientation().omega_b`.
    rpy       `sim.coordinates.zyx_from_rot(orientation.R)`, the TRUNK's
              triple.  NOT (orientation.roll, .pitch, .yaw): that is the
              BOARD's triple, and it parts from the trunk's by R_BODY_IMU the
              moment the board is mounted at an angle.  At today's identity
              placeholder the two agree, so no bench test can tell them apart.
    acc_b     NOTHING YET.  `TrunkOrientation` has no accelerometer field and
              `ImuDog` subscribes to the AHRS stream only.  The DETA10's
              specific force has to be read and carried into the trunk by the
              same SENSOR_TO_TRUNK as the gyro.  Check it level and at rest:
              (0, 0, +9.81).  -9.81 means the sensor reports gravity, not
              specific force -- negate it in read(), never in the filter.
              The lever arm from the board to the trunk origin is ignored,
              as MIT ignores it.

THE HEIGHT DATUM
    With `hw.kinematics`, r is the foot ball's CENTRE, so the filter's z = 0
    is the plane of the ball centres and p_w[2] is `params.FOOT_RADIUS`
    (15 mm) below `hw.balance.state`'s z_origin.  Add it where a floor height
    is wanted; the filter does not know it has feet with a radius.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..estimator import LinearKFPosVelEstimator, LKFOutput
from .rotation import rpy_zyx_to_R

__all__ = ("ImuSource", "LegSource", "run_once")


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


def run_once(imu: ImuSource, leg: LegSource, est: LinearKFPosVelEstimator,
             dt: float) -> LKFOutput:
    """This sweep's estimate, made BEFORE anything controls on it.

        out = run_once(imu, leg, est, dt)      1  read the sensors, estimate
        tau = controller(out, ...)             2  control on THIS sweep's state
        bus.send(tau)                          3  then actuate

    Estimating after control would hand the controller the previous sweep's
    state -- one period of delay on every feedback path, with nothing in the
    output to say so.

    FOUR FEET DOWN.  Every leg's contact phase is 0.5, which is trust 1: this
    is the balance stand.  A gait has its own phases and calls `est.update`
    with them directly.

    `dt` is the measured sweep period, already clamped by the caller: one
    stalled sweep taken at face value integrates its whole gap in one step.
    Nothing here reads a clock.
    """
    rpy, omega_b, acc_b = imu.read()
    r, rd = leg.read()
    R_wb = rpy_zyx_to_R(rpy)
    contact_phase = np.full(4, 0.5)
    return est.update(R_wb, omega_b, acc_b, r, rd, contact_phase, dt)
