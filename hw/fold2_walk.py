"""Walk from the KNEES-OUT fold stand: `hw.fold2_trot`, steered from the keyboard.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold2_walk --fake --auto 1 --no-imu     the whole path, no robot
    $V -m hw.fold2_walk --log fold2_walk.npz         on the robot
    $V -m hw.fold2_walk --ff-armature 0.005          M0 with a fitted rotor

    limp -> settle -> crouch -> rise -> hold --T--> trot (walk) --T--> hold -> park

`hw.fold2_trot` -- `posture.FOLD2`, every knee outboard of its hip, the
straight rise, the 160 mm hold, the 20 mm apex, the estimator closing x/y,
the joint layer -- walked exactly as `hw.fold_walk` walks `hw.fold_trot`:
the same hook, keys and box (`fold_walk.walking`), the QP, one reference for
the x/y rows, the heading, the joint layer's world-anchored stance targets
and the footholds, the task-space swing at `config.WN_SWING_OSC`, and the
walk's clock -- 0.6 s at duty 0.70, no settle.  `hw.fold_walk`'s docstring
says why each of those, and doc/walk/README.md has the runs.

WHAT IS FOLD2'S OWN, AND WHAT IT DOES TO THE WALK
    The footholds are placed about the stance's OWN sites, latched from
    q_hold at the first trot sweep (walk.py) -- FOLD2's feet at trunk x
    +-219.5 mm, y +-65, where the fold's stand has the front pair at +149.5.
    The SRB model is pinned at FOLD2's hold (`hw.stand` builds it from the
    crouch), and FOLD2's CoM is ON both trot diagonals (0.0 mm off each,
    the fold's 5.7): the moment a diagonal pair cannot make about its own
    line is zero here.  The swing leg is the same leg mirrored -- the foot
    is the same 0.31 / 0.36 / 5.8 kg in x / y / z (Lambda at the hold), so
    the torque-rate budget of doc/walk section 4 is the fold's to the digit.

    WHAT MUJOCO SAYS (doc/walk/README.md section 7): the better walker.  Six
    gait phases a case: in place, 0.05-0.15 m/s forward, 0.10 back, 0.05 /
    0.08 / 0.10 sideways, 20 and 40 deg/s turns and 0.10 m/s with a 20 deg/s
    turn, none tipped, all under 8 deg (the fold tipped 3 of 6 at 0.15 m/s
    forward); 0.20 m/s forward is the edge, 4 of 6 to the tilt stop.  Joint
    friction with a late IMU, a mu 0.5 floor, DOG6's lighter rotor with
    DOG5's M0 or a fitted one: none tipped.  The keys' box is the fold's,
    0.10 / 0.05 m/s and 20 deg/s, for the first flights; `--v-max 0.15 0.10
    --yaw-rate-max 40` is inside what MuJoCo walked.

THE ARMATURE IS THE SWING'S GAIN, here as in `hw.fold_walk`: fit it with
`hw.swing_bench --analyse` and pass `--ff-armature` before walking.

[SIM-TUNED, NOT FLOWN].

KEYS
    W / S  forward / back      A / D  left / right      Q / E  turn left / right
    SPACE  stop                T      trot / leave the trot
    ENTER  phases, REFUSED while trotting
    X      E-STOP.  From rise, hold or trot it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/fold2_walk.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from . import fold2_trot as F2       # noqa: E402
from . import fold_walk as FW        # noqa: E402
from . import stand as STAND         # noqa: E402

__all__ = ["main", "walk_options"]


def walk_options() -> dict:
    """`hw.fold2_trot`'s options, walking (`fold_walk.walking`)."""
    return FW.walking(F2.stand_options())


def main(argv=None) -> int:
    """`hw.fold2_trot`'s stand and trot, steered by WASD/QE."""
    return STAND.main(argv, **walk_options())


if __name__ == "__main__":
    raise SystemExit(main())
