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
    `hw.trot_esti` and `hw.fold_trot` do, ONLY to print: `EstimatorTap` steps
    the filter once per sweep off the same `BodyState` and `TrunkOrientation`
    the balance law is given, and reads it out in every phase.

    `hw.fully_trot` WAS THE FIRST CONTROL PATH THAT READS IT (2026-10-01): its
    trot closes the wrench's x/y rows on the filter's x/y and rate
    (`hw.balance.law.BalanceLaw.est_xy`).  SINCE 2026-10-02 EVERY TROT DOES,
    on request: `trot.trot_options` brings `hw.trot_esti.EstimatorFeed` with
    `--est-xy` on by default, `--no-est-xy` for print only -- and their HOLD
    closes x/y on it too (the same day, on request).  STILL 290 us,
    STILL ITS OWN SLOT:
    one `update` is 290 us on the Pi (2026-09-24) and the balance law already
    fills the 333 us CAN slot it would have to share, so the filter runs in
    `hw.stand.ESTIMATOR_SLOT` and the law reads its answer one sweep late --
    which a 0.5 Hz x/y loop can take and the attitude loop could not.  The
    28x28 solve in `lkf.update` is most of that cost and is where the work is
    if anything faster is ever to feed on it.

    AND ITS x/y ARE LEG ODOMETRY.  In MuJoCo the fold trot's drift came
    through planted feet sliding, which no row of this filter can see, so a
    loop closed on its x/y holds the robot where the legs say it is
    (`hw.fully_trot` has the numbers).

    Yaw is magnetometer-based and `hw.imu` calls it untrusted.  It enters
    through R_wb, so the x/y estimates are only as good as it is.  z and v_z
    do not depend on yaw at all -- a turn about world z leaves every z
    component alone -- so the stand's height channel is yaw-free.  `trot_esti`
    hands it the world frame the run pins at the handover rather than the
    magnetometer's own, and resets the filter on the sweep that frame turns.
"""
from __future__ import annotations

__all__ = ("estimator", "adapters")
