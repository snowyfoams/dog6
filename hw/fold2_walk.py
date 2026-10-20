"""Walk from the KNEES-OUT fold stand: `hw.fold2_trot`, steered from the keyboard.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold2_walk --fake --auto 1 --no-imu     the whole path, no robot
    $V -m hw.fold2_walk --log fold2_walk.npz         on the robot
    $V -m hw.fold2_walk --ff-armature 0.005          M0 with a fitted rotor
    $V -m hw.fold2_walk --kp-swing-walk-rl 150 150 600
                                                     ONE LEG'S impedance, as in
                                                     `hw.fold_walk`
    $V -m hw.fold2_walk --swing-law osc --settle 0.2 the task-space swing and
                                                     DOG5's re-level, the
                                                     defaults before 2026-10-20
    $V -m hw.fold2_walk --no-touchdown               the clock's word for which
                                                     foot is down: the walk as
                                                     flown on 2026-10-07

    limp -> settle -> crouch -> rise -> hold --T--> trot (walk) --T--> hold -> park

`hw.fold2_trot` -- `posture.FOLD2`, every knee outboard of its hip, the
straight rise, the 160 mm hold, the 20 mm apex, the estimator closing x/y,
the joint layer -- walked exactly as `hw.fold_walk` walks `hw.fold_trot`:
the same hook, keys and box (`fold_walk.walking`), the QP, one reference for
the x/y rows, the heading, the joint layer's world-anchored stance targets
and the footholds, the Cartesian impedance swing (the default since
2026-10-20, on the robot; `--swing-law osc` is the task-space swing at
`config.WN_SWING_OSC`), and the walk's clock -- 0.6 s at duty 0.70, no
settle (DOG6 needs none, the operator's test of 2026-10-20).  `hw.fold_walk`'s docstring
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
    0.12 / 0.06 m/s and 12 deg/s, 0.06 m/s or 6 deg/s a press (config.WALK_*;
    x/y doubled 2026-10-20 on request from 0.06 / 0.03 and 0.03 a press);
    `--v-max 0.15 0.10 --yaw-rate-max 40` is still inside what MuJoCo walked.

TOUCHDOWN DETECTION ON, 2026-10-20 -- inherited from `hw.fold2_trot`
    (`TOUCHDOWN`; scheme A, hw/balance/contact.py; `--no-touchdown` is the
    walk as flown).  Walking is where it earns its keep: the walk's swing
    lands 15 mm (median) / 27 mm (p90) off its arc on this robot, and the
    2026-10-06 fall at 629 s was a LATE landing -- the FR/RL swing lagging
    its arc, the clock handing the allocator a foot still in the air, the
    trunk rolling +0.9 -> +7.6 deg in 80 ms on the force that foot could not
    give.  With the detector that foot stays out of the allocator and probes
    down to the floor; the diagonal that is down carries the trunk the
    while.  And a foot the rolled trunk lands EARLY is handed over the sweep
    it is felt rather than pushed on into the floor by the arc.  The
    world-fixed footholds are kept through the probe: the arc's end is
    carried straight down in the trunk's z, its x/y where the foothold was
    placed.  `hw.fold2_trot`'s docstring has the mechanism; the exit report
    and the log's `td_*` columns say what it did.  [SIM-TUNED, NOT FLOWN]

THE ARMATURE IS THE osc SWING'S GAIN, here as in `hw.fold_walk`: under
`--swing-law osc`, fit it with `hw.swing_bench --analyse` and pass
`--ff-armature` before walking.

THE BEST FLIGHT, 2026-10-07 (the operator's report: it ran a very long time
with no fall -- the end state of DOG6):

    $V -m hw.fold2_walk --kp-joint 4 --kp-joint-abd 5 --swing-law impedance \\
        --duty 0.8 --contact-ramp 0.2 --settle 0.2 --swing-roll --tau-slew 300

    The defaults are NOT this command: the walk's stance Kp is 0
    (config.WALK_KP_JOINT), duty 0.70, slew 120.  Its swing law and its
    settle are, since 2026-10-20: the impedance is the default, and the
    operator's test that day dropped the settle (`--settle 0.2` is it).  `--swing-roll` (default on
    since this day) does nothing in a walk -- the walk's swing replaces the
    arc it levels.  Still open: walking with no joint hold, the kf problem,
    the swing trajectory and its Kp/Kd.

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
