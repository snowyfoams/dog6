"""Hardware -> the estimator's arrays.  The conversions and the call order.

    rotation    rpy_zyx_to_R -- numpy only, gated against `sim.coordinates`
    robot_io    ImuSource, LegSource (abstract), OrientationImu, BodyLegs,
                stance_phase, to_floor, run_once

No bus, no thread and no clock: `OrientationImu` and `BodyLegs` are built per
sweep from one `hw.imu.TrunkOrientation` and one `hw.balance.state.BodyState`,
and the caller owns both.
"""
from __future__ import annotations

from .robot_io import (BodyLegs, ImuSource, LegSource, OrientationImu,
                       run_once, stance_phase, to_floor)
from .rotation import rpy_zyx_to_R

__all__ = ("ImuSource", "LegSource", "OrientationImu", "BodyLegs",
           "stance_phase", "run_once", "to_floor", "rpy_zyx_to_R")
