"""DOG6 simulation module -- the robot as a model, independent of the robot.

This module is what is true of the DESIGN: the CAD's geometry, its mass
properties, the frames everything is expressed in, and the maps between joint
angles and foot positions.  It was written before a single motor was powered.
Nothing here talks to hardware; that is `hw`, whose wiring was measured on the
assembled DOG6 on 2026-09-15.

    What that measurement reached back into this module is `coordinates`:
    the flat zero's leg fold is fixed by the BUILT robot, and where it and the
    CAD pipeline disagree, the robot wins.  `coordinates.R_BODY_IMU` is a
    separate matter and is still an identity PLACEHOLDER -- the stand sequence
    used no IMU, so nothing has checked it.

    coordinates   frames, signs, naming, pose conventions.  No numbers.
    params        masses, inertias, lengths, limits, actuator spec.  No logic.
    kinematics    FK, Jacobians, IK, leg statics, the composite inertia.
    selftest      all three, gated against model/dog6.xml.

    model/        dog6.xml, 20 meshes, robot_export.json.  GENERATED -- see
                  the note below.

THE SPLIT IS THE POINT
    These could be one file; DOG5's equivalent nearly was.  They are three
    because the three kinds of mistake are different.  A frame error is a sign
    that survives every scalar check.  A parameter error is a plausible number
    in the right place.  A kinematics error is neither -- it is caught by
    comparing against an integrator.  Keeping them apart means `selftest` can
    check each in the way that actually catches it, and means a controller can
    import a convention without importing a table of inertias.

    The dependency runs one way and never back:

        coordinates  <-  params  <-  kinematics  <-  selftest

MODEL/ IS A GENERATED ARTIFACT AND DOES NOT BELONG TO THIS PROJECT
    `model/dog6.xml` comes out of the Fusion 360 design `quadruped_robot` in
    two stages, and both stages live in D:\\mujoco\\dog6_description:

        Fusion 360  --(MCP read)-->  dog6_raw.json + meshes/
                    --(build_dog6_mjcf.py)-->  dog6.xml, robot_export.json

    When the CAD moves: rebuild there, copy dog6.xml, meshes/ and
    robot_export.json into model/, then run

        python -m sim.selftest

    which is what will tell you which numbers in `params` went stale.
"""
from __future__ import annotations

import importlib

__all__ = ["coordinates", "params", "kinematics"]


def __getattr__(name: str):
    """Resolve `sim.params` and friends on first use.

    LAZY, NOT EAGER, AND THE REASON IS `python -m sim.params`.  Importing the
    submodules here would make runpy import them once as `sim.params` and then
    execute them again as `__main__`, which it warns about ("found in
    sys.modules ... prior to execution") and which leaves two copies of every
    constant alive.  `from sim import params` resolves the submodule anyway;
    this only restores plain `import sim` followed by `sim.params`.
    """
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
