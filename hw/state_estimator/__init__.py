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

WHAT IS STILL OPEN
    acc_b has no source yet.  `hw.imu.TrunkOrientation` carries R and
    omega^b but no accelerometer, and `ImuDog` subscribes to the AHRS stream
    only.  See `adapters.robot_io`.

    Yaw is magnetometer-based and `hw.imu` calls it untrusted.  It enters
    through R_wb, so the x/y estimates are only as good as it is.  z and v_z
    do not depend on yaw at all -- a turn about world z leaves every z
    component alone -- so the stand's height channel is yaw-free.
"""
from __future__ import annotations

__all__ = ("estimator", "adapters")
