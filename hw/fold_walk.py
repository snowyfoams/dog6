"""Walk from the fold stand: `hw.fold_trot`, steered from the keyboard.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold_walk --fake --auto 1 --no-imu      the whole path, no robot
    $V -m hw.fold_walk --log fold_walk.npz           on the robot
    $V -m hw.fold_walk --alloc wls                   the least squares instead
    $V -m hw.fold_walk --swing-law impedance --kd-swing-walk-rl 6 6 60
                                                     ONE LEG'S impedance (RL's
                                                     Kd), the other three on
                                                     --kd-swing-walk

    limp -> settle -> crouch -> rise -> hold --T--> trot (walk) --T--> hold -> park

`hw.fold_trot` -- the stand on the slanted rise, the 20 mm apex, the
estimator closing x/y, the joint layer -- with five things on top
(2026-10-04, on request), each a file in `hw.balance`:

    keys.py            W/S x, A/D y, Q/E yaw, SPACE stop -- `sim.cmpc.run`'s
    trajectory.py      `sim.cmpc.trajectory`'s generator, slewed and leashed
    footstep.py        eq (33) + feedback, the arc in the world
    swing_control.py   the swing leg: task-space computed torque (`--swing-law
                       osc`, the default) or the Cartesian impedance
    qp.py              the QP allocator -- every entry point's default since
                       2026-10-04 (--alloc wls is the least squares)

and walk.py tying them to ONE reference: the x/y rows, the heading, the joint
layer's targets and the footholds are all computed from the same sample.

THE CLOCK IS THE WALK'S, NOT THE TROT'S: 0.6 s at duty 0.70, so 180 ms of
swing where `hw.fold_trot` has 120.  At 120 ms no swing law walked in MuJoCo:
the foot is 5.8 kg in z through the reflected rotor and the 20 mm arc then
asks the knee for more torque rate than the gate's 120 N*m/s (config.py,
WALK_DUTY; doc/walk/README.md).  `--duty 0.8` puts the trot's clock back.

THE ARMATURE IS THE SWING'S GAIN.  The task-space law multiplies by M0, and
M0's rotor is DOG5's -- the bench read DOG6's as ~1/1.7 of it.  Fit it with
`hw.swing_bench --analyse` and pass `--ff-armature`; or fly `--swing-law
impedance`, which does not read M0 unless `--swing-ff`.

ONE LEG'S OWN IMPEDANCE, 2026-10-05, on request: `--kp-swing-walk-fl` ..
`--kd-swing-walk-rr` replace `--kp-swing-walk` / `--kd-swing-walk` on that
leg alone -- `hw.trot`'s per-leg swing gains ("ONE LEG'S SWING GAINS"), on
the walk's impedance.  `--swing-law osc` refuses them: its knobs are
`--wn-swing` / `--zeta-swing`.  `hw.fold2_walk` has them through this hook.
MuJoCo, on the next flight's flags (`--swing-law impedance --duty 0.8
--contact-ramp 0.15 --settle 0.2`), RL's motors making 60 % of the torque:
in place RL's trunk-frame apex 7.1 -> 3.1 mm, and RL's own Kp_z 700 / Kd_z
70 (or Kd_z 80 alone) gave back 6.0; walking 0.05 m/s 4.7 -> 1.8 -> 5.0;
FOLD2 in place 3.2 -> 6.1.  The other three legs did not move.  NOT FLOWN.

THE KEYS ACT WHILE TROTTING.  In HOLD they accumulate a command and the
status line shows it, but the reference only moves while the robot trots,
and T stops feeding it the moment the exit is latched: the reference slews
to rest over the swing or two before the four-foot window and the HOLD that
follows (the walk stays engaged there: world-anchored targets, the x/y rows
on the reference).  Stop walking with SPACE, let it trot in place a cycle or
two -- the footholds come back to the stance's own sites -- then T, then
ENTER: the park is a joint ramp to the crouch and drags any foot that is off
its site.

THE KEYS' BOX IS 60 % OF WHAT MUJOCO WALKED, since 2026-10-05 on request:
0.06 m/s in x, 0.03 in y, 12 deg/s, and a press is 0.03 m/s or 6 deg/s
(config.WALK_*; --v-max, --yaw-rate-max, --v-step, --yaw-step).  MuJoCo
walked the 0.10 / 0.05 m/s and 20 deg/s box at 0.05 m/s and 10 deg/s a
press.

AT ZERO COMMAND THE SWING IS hw.fold_trot's, 2026-10-05 (on request).  On the
robot, standing, the filter's v_hat is rms 0.05-0.08 m/s with spikes to 0.4,
and the foothold carried 0.27 s of it, re-aimed every sweep: the stiff swing
chased a target that wandered 10-60 mm inside one swing and threw the feet
(fold_walk_imp.npz).  Now, with the reference at rest, each swing flies the
walk's arc in the TRUNK frame from the foot to its own site, no velocity
term, the filter nowhere in it (footstep.py).  Any key places the NEXT swing
again; after SPACE the swings already in the air land where they were
placed.  The clock, the swing law and the world-anchored stance targets are
still the walk's.  `--place-at-rest` is the walk as flown before.

THE PLACEMENT'S VELOCITY IS LOW-PASSED, 2026-10-05 (on request): walking,
the foothold reads the filter's x/y velocity through a first-order low-pass,
`--v-filter-hz` (config.WALK_V_FILTER_HZ; 0 reads it raw).  The x/y rows
still read it raw.  The status line shows both, `est v` and `lp`.

[SIM-TUNED, NOT FLOWN]: doc/walk/README.md has what MuJoCo says it does.

KEYS
    W / S  forward / back      A / D  left / right      Q / E  turn left / right
    SPACE  stop                T      trot / leave the trot
    ENTER  phases, REFUSED while trotting
    X      E-STOP.  From rise, hold or trot it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/fold_walk.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

import numpy as np                   # noqa: E402

from . import fold_trot as FT        # noqa: E402
from . import stand as STAND         # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import gait as GAIT    # noqa: E402
from .balance import swing as BSWING  # noqa: E402
from .balance import swing_control as SC   # noqa: E402
from .balance.keys import WalkKeys   # noqa: E402
from .balance.trajectory import WalkReference   # noqa: E402
from .balance.walk import WalkPlan   # noqa: E402

__all__ = ["main", "WalkHook", "walk_options", "walking"]


class WalkHook:
    """`hw.stand.main`'s hook: the walking keys, and the law's walk plan."""

    #: The law's walking branch flies every swinging leg from the first trot
    #: sweep (walk.py, swing_control.py); `hw.stand` says so in its banner.
    replaces_swing = True

    def __init__(self):
        self.keys = WalkKeys()
        self.plan: WalkPlan | None = None
        #: the impedance's (4, 3) gains and the legs with their own, from
        #: `configure`
        self.kp_rows = self.kd_rows = None
        self.own_swing: list[str] = []

    # -- hw.stand.main's hook -----------------------------------------------
    def add_arguments(self, ap) -> None:
        g = ap.add_argument_group(
            "the walk", "W/S x, A/D y, Q/E yaw, SPACE stop -- while trotting")
        g.add_argument("--v-max", type=float, nargs=2,
                       default=[BCFG.WALK_VX_MAX, BCFG.WALK_VY_MAX],
                       metavar=("VX", "VY"), help="m/s the keys may reach")
        g.add_argument("--yaw-rate-max", type=float,
                       default=np.degrees(BCFG.WALK_YAW_RATE_MAX),
                       metavar="DEG_S")
        g.add_argument("--v-step", type=float, default=BCFG.WALK_V_STEP,
                       metavar="M_S", help="one W/S/A/D press")
        g.add_argument("--yaw-step", type=float,
                       default=np.degrees(BCFG.WALK_YAW_STEP), metavar="DEG_S",
                       help="one Q/E press")
        g.add_argument("--walk-acc", type=float, default=BCFG.WALK_ACC_MAX,
                       metavar="M_S2", help="the command's slew")
        g.add_argument("--leash", type=float, default=1e3 * BCFG.WALK_LEASH,
                       metavar="MM", help="the reference's distance from the "
                                          "filter's CoM, at most")
        g.add_argument("--step-kv", type=float, default=BCFG.STEP_KV,
                       metavar="SECONDS", help="Raibert's velocity feedback "
                                               "in the foothold")
        g.add_argument("--kp-swing-walk", type=float, nargs=3,
                       default=list(BCFG.KP_SWING_WALK), metavar="N_PER_M")
        g.add_argument("--kd-swing-walk", type=float, nargs=3,
                       default=list(BCFG.KD_SWING_WALK), metavar="NS_PER_M")
        # --kp-swing-walk-fl .. --kd-swing-walk-rr: one leg's own impedance.
        STAND.add_swing_leg_flags(g, "swing-walk")
        g.add_argument("--hold-xy-max", type=float,
                       default=1e3 * BCFG.HOLD_XY_ERR_MAX, metavar="MM",
                       help="per-leg clamp on the walking joint target's "
                            "x/y error")
        g.add_argument("--swing-law", choices=SC.SWING_LAWS,
                       default=BCFG.WALK_SWING_LAW,
                       help="the swing leg while walking: 'osc' the task-space "
                            "computed torque (acceleration gains --wn-swing; "
                            "M0 is its gain, so --ff-armature matters), "
                            "'impedance' the Cartesian PD (--kp/--kd-swing-walk, "
                            "+ --swing-ff)")
        g.add_argument("--wn-swing", type=float, nargs=3,
                       default=list(BCFG.WN_SWING_OSC),
                       metavar=("X", "Y", "Z"),
                       help="'osc' bandwidth per trunk axis, rad/s")
        g.add_argument("--zeta-swing", type=float,
                       default=BCFG.ZETA_SWING_OSC, help="'osc' damping ratio")
        g.add_argument("--place-at-rest", action="store_true",
                       default=BCFG.WALK_PLACE_AT_REST,
                       help="keep eq (33)'s and Raibert's velocity terms in "
                            "the foothold at zero command too -- the walk as "
                            "flown before 2026-10-05, whose swing chases the "
                            "filter's v_hat standing still.  Default: at rest "
                            "each swing lands on its own site, trunk frame, "
                            "hw.fold_trot's in-place target")
        g.add_argument("--v-filter-hz", type=float,
                       default=BCFG.WALK_V_FILTER_HZ, metavar="HZ",
                       help="corner of the first-order low-pass on the "
                            "filter's x/y velocity as the foot placement "
                            "reads it; 0 reads it raw.  The x/y rows read it "
                            "raw either way")

    def configure(self, args) -> None:
        self.keys = WalkKeys(v_step=args.v_step,
                             yaw_step=np.radians(args.yaw_step),
                             vx_max=args.v_max[0], vy_max=args.v_max[1],
                             yaw_rate_max=np.radians(args.yaw_rate_max))
        reference = WalkReference(vx_max=args.v_max[0], vy_max=args.v_max[1],
                                  yaw_rate_max=np.radians(args.yaw_rate_max),
                                  acc_max=args.walk_acc,
                                  leash=1e-3 * args.leash)
        ff = getattr(args, "swing_ff", False)
        armature = getattr(args, "ff_armature", None)
        # ONE LEG'S OWN impedance, 2026-10-05, on request: a (4, 3) table,
        # `--kp-swing-walk` in every row.  The osc law has no Kp/Kd to take.
        self.kp_rows, self.kd_rows, self.own_swing = STAND.swing_leg_gains(
            args, "swing-walk")
        if self.own_swing and args.swing_law != "impedance":
            raise ValueError(
                "--kp/--kd-swing-walk-%s are the impedance's gains (--swing-law "
                "impedance); the '%s' swing is tuned by --wn-swing / "
                "--zeta-swing" % (self.own_swing[0].lower(), args.swing_law))
        self.plan = WalkPlan(reference=reference, kv=args.step_kv,
                             kp_swing=self.kp_rows,
                             kd_swing=self.kd_rows, swing_ff=ff,
                             ff_inertia=(None if armature is None else
                                         BSWING.feedforward_inertia(armature)),
                             hold_xy_max=1e-3 * args.hold_xy_max,
                             swing_law=args.swing_law,
                             wn_swing=args.wn_swing,
                             zeta_swing=args.zeta_swing,
                             place_at_rest=args.place_at_rest,
                             v_filter_hz=args.v_filter_hz)

    def law_kwargs(self, args) -> dict:
        return dict(walk=self.plan)

    def banner(self, args) -> list:
        swing = ("osc, wn %s rad/s, zeta %.2f, M0 armature %s"
                 % (np.array2string(np.asarray(args.wn_swing), precision=0),
                    args.zeta_swing,
                    "inherited (DOG5's)" if getattr(args, "ff_armature", None)
                    is None else "%.5f" % args.ff_armature)
                 if args.swing_law == "osc" else
                 "impedance, Kp %s Kd %s%s"
                 % (np.array2string(np.asarray(args.kp_swing_walk),
                                    precision=0),
                    np.array2string(np.asarray(args.kd_swing_walk),
                                    precision=0),
                    " + feedforward" if getattr(args, "swing_ff", False)
                    else ""))
        return ["WALK while trotting: %s" % WalkKeys.help_line(),
                "     limits vx %.2f vy %.2f m/s, yaw %.0f deg/s; slew %.2f "
                "m/s^2; leash %.0f mm; foothold eq (33) + %.2f s feedback; "
                "allocator %s" % (args.v_max[0], args.v_max[1],
                                  args.yaw_rate_max, args.walk_acc, args.leash,
                                  args.step_kv, args.alloc),
                "     swing %s; gait %.2f s at duty %.2f (%.0f ms of swing)"
                % (swing, args.period, args.duty,
                   1e3 * args.period * (1.0 - args.duty)),
                ] + ([STAND.swing_leg_banner(self.kp_rows, self.kd_rows,
                                             self.own_swing, indent="     ")]
                     if self.own_swing else []) + [
                "     at zero command: %s"
                % ("placed as when walking, the filter's v_hat in the "
                   "foothold (--place-at-rest)" if args.place_at_rest else
                   "each swing lands on its own site, trunk frame, no "
                   "velocity term -- hw.fold_trot's in-place target"),
                "     placement velocity: %s"
                % ("the filter's x/y low-passed at %.1f Hz (tau %.0f ms); "
                   "the x/y rows read it raw"
                   % (args.v_filter_hz, 1e3 / (2.0 * np.pi * args.v_filter_hz))
                   if args.v_filter_hz > 0.0 else
                   "the filter's x/y, raw (--v-filter-hz 0)")]

    def owns(self, ch) -> bool:
        return self.keys.owns(ch)

    def key(self, ch: str, now: float, stand) -> str | None:
        said = self.keys.press(ch)
        if said is not None and not stand.trotting:
            said += "  (applies while trotting -- T)"
        return said

    def blocks_advance(self, stand) -> str | None:
        return None

    def sweep(self, now: float, stand) -> None:
        # The keys reach the reference while trotting -- and stop reaching it
        # the moment T latches the exit, so the reference is already slewing
        # to rest across the swing or two before the four-foot window.
        walking = stand.trotting and not getattr(stand, "trot_exit", False)
        if self.plan is not None:
            self.plan.set_command(self.keys.command if walking else None)

    def status(self, now: float, stand) -> str:
        out = stand.out
        ref = None if out is None else out.ref
        if ref is None:
            return "walk: %s  (engages at T)" % self.keys.describe_command()
        line = ("walk: keys %s | ref v %+.3f %+.3f m/s  yaw %+6.1f deg  r %+5.1f "
                "deg/s" % (self.keys.describe_command(), ref.v[0], ref.v[1],
                           np.degrees(ref.yaw), np.degrees(ref.yaw_rate)))
        if out.estimate is not None:
            e = np.asarray(out.estimate.v_w, dtype=float)
            line += " | est v %+.3f %+.3f" % (e[0], e[1])
            if self.plan is not None and self.plan.v_filter_hz > 0.0:
                line += " lp %+.3f %+.3f" % (self.plan.v_lp[0],
                                             self.plan.v_lp[1])
        if ref.leashed:
            line += "  LEASHED"
        return line


def walking(trot_options: dict) -> dict:
    """A trot entry point's `hw.stand.main` options, made a WALK: the
    walking hook, the QP, and the walk's own clock -- the trot's period at
    `config.WALK_DUTY`, the contact ramp fitted to its four-foot window, no
    settle (`--period`, `--duty`, `--contact-ramp`, `--settle` still
    override).  `hw.fold_walk` is this over `hw.fold_trot`, `hw.fold2_walk`
    over `hw.fold2_trot`."""
    opts = dict(trot_options)
    clock = opts["gait"]
    opts["gait"] = GAIT.TrotGait(period=clock.period, duty=BCFG.WALK_DUTY,
                                 offsets=clock.offsets,
                                 ramp=BCFG.WALK_CONTACT_RAMP,
                                 settle_s=BCFG.WALK_SETTLE_S,
                                 settle_every=clock.settle_every)
    return dict(opts, hook=WalkHook(), alloc="qp")


def walk_options() -> dict:
    """`hw.fold_trot`'s options, walking (`walking`)."""
    return walking(FT.stand_options())


def main(argv=None) -> int:
    """`hw.fold_trot`'s stand and trot, steered by WASD/QE."""
    return STAND.main(argv, **walk_options())


if __name__ == "__main__":
    raise SystemExit(main())
