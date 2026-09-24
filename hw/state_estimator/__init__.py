"""Trunk position and velocity from the IMU and the legs: MIT's linear KF.

    python -m pytest hw/state_estimator/tests        gate it, no robot needed

A port of `LinearKFPositionVelocityEstimator` from MIT's Cheetah-Software.
The accelerometer drives the prediction; every leg then reports where the
trunk is relative to its foot and, if the foot is planted, how fast the trunk
is moving over it.  Each leg's contact phase decides how far it is believed.

TWO LAYERS, AND THE BOUNDARY IS THE POINT
    estimator/   the filter.  numpy and the standard library, nothing else --
                 no IMU, no CAN, no FK, no clock, no log, no file.  Arrays and
                 floats in, arrays and a dataclass out.  `tests/test_layers.py`
                 reads the source and fails the day that stops being true.
    adapters/    where hardware becomes those arrays.  Interfaces only:
                 `ImuSource`, `LegSource`, and `run_once` for the call order.
    config/      lkf.yaml -- the same ten keys as `estimator.LKFParams`.

    A Kalman filter cannot tell a wrong frame from a noisy sensor: a flipped
    axis is absorbed as a bias and the innovation stays plausible.  So every
    unit, axis and offset is settled in an adapter, before the filter sees
    it, and each `estimator` docstring says exactly what it assumes it was
    given.

THE HEIGHT DATUM
    `r` is the foot ball's CENTRE, so the filter's z = 0 is the plane of the
    centres and the floor is one ball radius lower.  `adapters.to_floor`
    shifts the world z datum by that radius and `adapters.run_once` applies
    it, which puts p_w[2] on the same number as `hw.balance.state`'s
    `z_origin`.  The shift belongs in the WORLD frame; `adapters.robot_io`
    says what taking it off `r[:, 2]` costs instead.

WHO CALLS IT, AND WHAT IS STILL OPEN
    `hw.trot_esti` does, and ONLY to print: `EstimatorTap` steps the filter
    once per sweep off the same `BodyState` and `TrunkOrientation` the balance
    law is given, and reads it out in every phase.  Nothing in the control path
    reads a number that comes out of here.

    WHAT STANDS BETWEEN THAT AND THE LOOP IS 290 us.  One `update` is 290 us on
    the Pi (2026-09-24) and the balance law already fills the 333 us CAN slot it
    would have to share; `hw.stand.ESTIMATOR_SLOT` buys the read-out a slot of
    its own, which is a luxury a controller does not have -- the law needs the
    estimate before it acts.  The 28x28 solve in `lkf.update` is most of that
    cost and is where the work is.

    Yaw is magnetometer-based and `hw.imu` calls it untrusted.  It enters
    through R_wb, so the x/y estimates are only as good as it is.  z and v_z
    do not depend on yaw at all -- a turn about world z leaves every z
    component alone -- so the stand's height channel is yaw-free.  `trot_esti`
    hands it the world frame the run pins at the handover rather than the
    magnetometer's own, and resets the filter on the sweep that frame turns.
"""
from __future__ import annotations

__all__ = ("estimator", "adapters")
