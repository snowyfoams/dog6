"""The filter itself: numpy arrays in, numpy arrays and a dataclass out.

    params        LKFParams, LKFOutput.  Dataclasses, no logic.
    matrices      build_C, build_A_B, build_Q, build_R.  Pure functions.
    measurement   trapezoid_trust, build_measurement.  Pure functions.
    lkf           LinearKFPosVelEstimator.  Holds x, P, C, params -- nothing else.

THIS PACKAGE IMPORTS numpy AND THE STANDARD LIBRARY AND NOTHING ELSE.  No
sensor, bus, kinematics, clock, logger or file appears in it, in a signature
or in a body, and there is no module-level state.  `tests/test_layers.py`
enforces all of that by reading the source.  Frames, units and offsets are
settled before anything arrives here; that is `adapters/`, one level up.
"""
from __future__ import annotations

from .lkf import LinearKFPosVelEstimator
from .matrices import build_A_B, build_C, build_Q, build_R
from .measurement import build_measurement, trapezoid_trust
from .params import LKFOutput, LKFParams

__all__ = ("LKFParams", "LKFOutput", "build_C", "build_A_B", "build_Q",
           "build_R", "trapezoid_trust", "build_measurement",
           "LinearKFPosVelEstimator")
