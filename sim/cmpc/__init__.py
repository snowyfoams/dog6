"""Convex model-predictive control for DOG6.

A reproduction of Di Carlo, Wensing, Katz, Bledt and Kim, "Dynamic Locomotion
in the MIT Cheetah 3 Through Convex Model-Predictive Control" (IROS 2018),
written against `sim`'s description of DOG6.

    config       every constant, and where each one came from
    gait         the contact schedule -- a pure function of the clock
    trajectory   operator command -> reference states
    gamepad      an Xbox pad as the operator, through XInput
    dynamics     the linearised body model, and its ZOH discretisation
    qp           condensation, constraint assembly, the OSQP solve
    swing        foot placement (33) and the swing control law (1)-(3)
    controller   the two loops: 20 Hz MPC over 250 Hz torques
    run          drive it in MuJoCo, from the pad, the keys or a script
    teleop       the pad driving the reference alone, animated
    selftest     all of the above, gated

TWO ASSUMPTIONS MAKE THIS SIMPLER THAN THE PAPER, ON PURPOSE
    The body state is READ, not estimated -- no IMU, no leg odometry, no
    filter -- and the gait is a pure TIMETABLE with no contact sensing.  Both
    are stated again at the top of `config`.  They are what makes a failure
    here attributable: anything that goes wrong is the MPC formulation or the
    swing law, not an estimator.

THE PIECE THAT IS NOT CONVEX, AND WHERE IT HIDES
    The QP is convex given the contact schedule and the foot positions over
    the horizon.  Both are supplied from outside it: the schedule by `gait`,
    the positions by predicting where the feet will be.  Choosing WHERE to put
    a foot -- `swing.foot_placement`, eq (33) -- is not part of the
    optimisation at all.  So the convexity is real but it is convexity of a
    subproblem; the placement heuristic feeding it is the part of this method
    that is still a heuristic.
"""
from __future__ import annotations

import importlib

__all__ = ["config", "gait", "trajectory", "gamepad", "dynamics", "qp",
           "swing", "controller", "run", "teleop", "selftest"]


def __getattr__(name: str):
    """Resolve submodules on first use -- see `sim.__init__` for why lazily."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
