"""Trot in place from the WIDE crouch at the CAD stand height: the posture the
2026-10-01 study (doc/trot_posture) found best for DOG6's trot.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.wide_trot --fake --auto 1 --no-imu      the whole path, no robot
    $V -m hw.wide_trot --log wide.npz                on the robot: 160 mm,
                                                     0.5 s / 20 mm, spring on
    $V -m hw.wide_trot --period 0.8 --swing-height 40    hw.trot's clock
    $V -m hw.wide_trot --track 95                    the feet 10 mm further out
    $V -m hw.wide_trot --kp-stance-xy 0 --kd-stance-xy 0   the spring off

    limp -> settle -> crouch -> rise -> hold --T--> trot --T--> hold -> park
                                               (exit at the next four-foot window)

WHAT IT FLIES, AND WHY EACH PIECE
    THE POSTURE   `posture.WIDE`: NOMINAL's X configuration (front knees
                  forward, rear knees back, feet 81 mm ahead of the front
                  hips and 81 mm behind the rear hips, shins vertical at the
                  crouch), the feet 20 mm further out (|y| 85 mm), crouch
                  40 mm.  Front and rear legs MIRROR, so both diagonal
                  support lines pass through the CoM (the fold's are 20 mm
                  off), the rise has no forward glide (a parallel fold
                  glides ~100 mm, doc/trot_posture/README.md section 1), the
                  hold torque is a quarter of the fold's, and the crouch
                  rests the knee motors nowhere near the floor.
    THE HEIGHT    160 mm floor to trunk bottom (`HEIGHT_MM`, `--height`),
                  the CAD stand's 157.5 rounded, against hw.trot's 145.  In
                  simulation the worst-case tilt under a 10 mm lateral CoM
                  error falls ~0.1 deg per mm of height (7.96 -> 6.28 deg at
                  0.8 s / 40 mm) with the hold torque unchanged (0.38 N m);
                  above ~165 mm the knee comes within 25 deg of straight and
                  the vertical Jacobian goes singular, so no higher.  The
                  SRB model is re-pinned at the flown height (`pinned_at`);
                  `posture.WIDE.srb` is pinned at config.H_LIFT.
    THE TRACK     `--track 85` (WIDE) by default; 95 and 105 are the same
                  posture with the feet 10 / 20 mm further out, built here
                  the way `posture.WIDE` is built.  Wider is better in the
                  model (~0.08 deg per mm of half-track) at the price of
                  abduction torque in HOLD (0.38 / 0.48 / 0.62 N m) and a
                  slow lateral walk under a CoM error.  All three use the
                  40 mm crouch: 8 mm lower the left and right knee motors
                  touch at the handover sag.
    THE SPRING    `--kp-stance-xy 400 --kd-stance-xy 25` ON by default
                  (`STANCE_XY`): every planted foot held at its site in the
                  trunk frame, x and y, through the rise, the hold and the
                  trot.  Not needed against the glide here (a mirrored
                  stance has none) but it is what makes the stand repeatable
                  to the millimetre, and in the trot it trades a slow walk
                  toward a CoM offset for ~0.5 deg less roll.  Pass 0 0 for
                  hw.trot's behaviour.
    THE REST      `hw.fold_trot`'s: `hw.trot.trot_options` (cap 9 N m, slew
                  60, no overspeed trip), roll 290 / 23
                  (`fold_stand.ROLL_GAINS`), limits OFF (`fold_stand.LIMITS`,
                  `--limits` puts them back), the Cartesian swing, the joint
                  layer, `hw.trot_esti`'s Kalman filter printed under the
                  status line.  The attitude setpoint is LATCHED in limp as
                  hw.trot's (the robot starts flat on its belly).
    THE CLOCK     0.5 s / 20 mm (`PERIOD_S`, `SWING_HEIGHT`), the operator's
                  best clock on the robot; in simulation this posture also
                  survives every perturbation at 0.8 s / 40 mm.

WHAT THE STUDY COULD NOT SETTLE (doc/trot_posture/README.md section 6)
    Everything above is simulation of the CAD model.  The real lateral CoM
    offset, the handover sag, and whether the abduction motors mind the
    hold torque are hardware questions; the README lists the tests.

KEYS
    ENTER  the stand's phases, as ever.  REFUSED while trotting.
    T      from HOLD, start trotting.  While trotting, latch the exit.
    X      E-STOP.  From rise, hold or trot it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/wide_trot.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

import argparse                      # noqa: E402
import dataclasses                   # noqa: E402
import sys                           # noqa: E402

import numpy as np                   # noqa: E402

from sim import kinematics as SK     # noqa: E402
from sim import params as P          # noqa: E402

from . import fold_stand as FS       # noqa: E402
from . import kinematics as HK       # noqa: E402
from . import stand as STAND         # noqa: E402
from . import trot as TROT           # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import posture as POSE  # noqa: E402
from .trot_esti import EstimatorTap  # noqa: E402

__all__ = ["main", "wide_posture", "pinned_at", "HEIGHT_MM", "PERIOD_S",
           "SWING_HEIGHT", "STANCE_XY", "TRACKS", "CROUCH_H"]

#: mm, floor to trunk bottom, the stand -- `--height`'s default here.
HEIGHT_MM = 160.0
#: The clock and apex, the operator's best on the robot (hw.fold_trot's).
PERIOD_S = 0.5
SWING_HEIGHT = 0.020                 # m
#: The stance xy spring, per leg: N/m, N s/m.  `--kp-stance-xy 0
#: --kd-stance-xy 0` switches it off.
STANCE_XY = (400.0, 25.0)
#: m, the crouch for every track: the WIDE crouch.
CROUCH_H = POSE.WIDE_H
#: |y| of the feet, mm, for `--track`: 85 is `posture.WIDE` itself.
TRACKS = (85, 95, 105)


def wide_posture(track_mm: float = 85.0) -> POSE.CrouchPose:
    """`posture.WIDE` at |y| = `track_mm` mm: NOMINAL's feet splayed out to
    that track, trunk `CROUCH_H` off the floor.  85 IS `posture.WIDE`."""
    if abs(track_mm - 85.0) < 1e-9:
        return POSE.WIDE
    hip = SK.hip_to_foot_stance(POSE.NOMINAL.q)
    splay = 1e-3 * track_mm - np.abs(hip[0, 1]) - np.abs(P.HIP_OFFSET[0, 1])
    hip[:, 1] += np.sign(hip[:, 1]) * splay
    hip[:, 2] -= CROUCH_H - POSE.NOMINAL.h
    return POSE.CrouchPose.from_hip_sites(
        "wide%d" % round(track_mm), hip, q_seed=POSE.NOMINAL.q,
        note="NOMINAL's feet at |y| %.0f mm, trunk %.0f mm off the floor"
             % (track_mm, 1e3 * CROUCH_H))


def pinned_at(pose: POSE.CrouchPose, height_mm: float) -> POSE.CrouchPose:
    """`pose` with its SRB model (c^b, I^b) derived at the stand height it
    will actually be flown at, instead of config.H_LIFT."""
    p = np.zeros((4, 3))
    p[:, :2] = pose.foot_xy
    p[:, 2] = -(1e-3 * height_mm + BCFG.TRUNK_BOTTOM_OFFSET - P.FOOT_RADIUS)
    srb = BCFG.SrbModel.from_pose("%s@%.0f" % (pose.name, height_mm),
                                  HK.all_leg_ik(p, q_seed=pose.q))
    return dataclasses.replace(pose, srb_pinned=srb)


def _defaults(argv: list, pairs) -> list:
    """`argv` with `--flag value` appended for every flag not already there."""
    out = list(argv)
    for flag, value in pairs:
        if not any(a == flag or a.startswith(flag + "=") for a in out):
            out += [flag, str(value)]
    return out


def main(argv=None) -> int:
    """`hw.fold_trot`'s run from `wide_posture(--track)`, at `--height`
    (160 mm), the stance xy spring on, 0.5 s / 20 mm."""
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--track", type=float, default=TRACKS[0], choices=TRACKS,
                     help="|y| of the feet, mm: 85 is posture.WIDE")
    pre.add_argument("--height", type=float, default=HEIGHT_MM)
    own, rest = pre.parse_known_args(argv)
    rest = _defaults(rest, (("--height", own.height),
                            ("--kp-stance-xy", STANCE_XY[0]),
                            ("--kd-stance-xy", STANCE_XY[1])))
    crouch = pinned_at(wide_posture(own.track), own.height)
    return STAND.main(rest, crouch=crouch, only_law="srb",
                      tilt_stop=FS.TILT_STOP_DEG, roll_gains=FS.ROLL_GAINS,
                      track_stop=FS.TRACK_STOP_DEG, velocity=False,
                      estimator=EstimatorTap, limits=FS.LIMITS,
                      swing_height=SWING_HEIGHT,
                      **TROT.trot_options(PERIOD_S))


if __name__ == "__main__":
    raise SystemExit(main())
