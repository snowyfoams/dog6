"""`hw.fold2_walk` with the convex MPC as the trot's force law, on cmpc's clock.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold2_cmpc --fake --auto 1 --no-imu     the whole path, no robot
    $V -m hw.fold2_cmpc --log fold2_cmpc.npz         on the robot
    $V -m hw.fold2_cmpc --no-mpc                     the A/B: fold2_walk's SRB
                                                     law on this same clock
    $V -m hw.fold2_cmpc --mpc-hz 50                  the solve rate

    limp -> settle -> crouch -> rise -> hold --T--> trot (walk) --T--> hold -> park

2026-11-10, on request: "cmpc only for the trot and the walk; the duty and
the ramp are cmpc's own; what is aligned with fold2_walk is the crouch, the
stand, the swing control, the trajectory and the sweep handling".

WHAT IS cmpc's -- on the sweeps the gait clock runs, and only those
    the foot forces   `sim.cmpc`'s QP (its model, Q, R, horizon 10, 40 Hz),
                      solved inside the law's slot and held between solves:
                      `hw.balance.mpc.MpcForces`, in place of the SRB wrench
                      and its allocator
    the clock         0.5 s at duty 0.5, NO contact ramp, no settle
                      (`sim.cmpc.config`): the diagonals hand over on one
                      sweep, there is no four-foot window, the FR/RL pair
                      lifts on T's first sweep and T leaves the trot at a
                      handover.  250 ms of swing.  `--period` / `--duty` /
                      `--contact-ramp` still override

WHAT IS fold2_walk's, THE SAME OBJECTS, NOT A COPY
    the crouch and the stand   `posture.FOLD2`, settle, crouch, the tracked
                               straight rise to 160 mm, HOLD and its joint
                               layer, the CoM re-centring at T, the park --
                               the SRB law and the QP allocator, as flown
    the swing control          `swing_control.py`'s impedance, Kp 150/150/400,
                               Kd 6/6/40, `--kp/--kd-swing-walk[-<leg>]`, no
                               feedforward unless `--swing-ff`, apex 20 mm,
                               no joint damper on the swinging leg
    the trajectory             one walking reference (`walk.WalkPlan`:
                               slewed, clipped, leashed to the filter), the
                               footholds about the stance's own sites and the
                               arc (footstep.py: trunk-frame at rest, eq 33 +
                               Raibert while walking), the planted legs'
                               world-anchored joint targets, damped not sprung
    the sweep                  `hw.stand.run`: the law at slot 0, the gate
                               (cap, slew 120, limit block), the estimator's
                               feed, `hw.realtime`, `hw.can_link`, the fall
                               hold, the governor, the keys, the log

    The MPC reads what that law already has -- the IMU's attitude, the
    filter's x/y, the stance feet's height -- and tracks the same reference
    the swing and the joint layer take their targets from (mpc.py).

THE LOG is fold2_walk's, one script reads both: on a trot sweep `f_w` / `fz`
    are the MPC's forces, `b_d` what the SRB law would have asked of the
    same state, and `mpc_ms` the QP's cost on the sweeps one ran.

NOT FLOWN.  The solve runs inside the sweep and delays every motor behind it
    by its whole duration once per MPC period (~5 ms on the Pi); the exit
    report prints what it cost.

KEYS
    W / S  forward / back      A / D  left / right      Q / E  turn left / right
    SPACE  stop                T      trot / leave the trot
    ENTER  phases, REFUSED while trotting
    X      E-STOP.  From rise, hold or trot it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/fold2_cmpc.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim.cmpc import config as SIMCFG    # noqa: E402

from . import fold2_walk as F2W      # noqa: E402
from . import fold_walk as FW        # noqa: E402
from . import stand as STAND         # noqa: E402
from .balance import gait as GAIT    # noqa: E402
from .balance import mpc as MPC      # noqa: E402

__all__ = ["main", "CmpcHook", "cmpc_options", "MPC_HZ_MIN"]

#: Hz.  A trot's MPC must not go below this: `sim.cmpc.config.MPC_HZ` has the
#: measured subharmonic of the diagonal pendulum mode.
MPC_HZ_MIN = 40.0


class CmpcHook(FW.WalkHook):
    """`fold_walk.WalkHook` -- the keys, the walk plan -- plus the MPC the
    law flies the trot's forces on."""

    def __init__(self):
        super().__init__()
        self.mpc: MPC.MpcForces | None = None

    def add_arguments(self, ap) -> None:
        super().add_arguments(ap)
        g = ap.add_argument_group(
            "the MPC", "sim.cmpc's QP as the trot's foot forces (hw.balance.mpc)")
        g.add_argument("--mpc-hz", type=float, default=SIMCFG.MPC_HZ,
                       metavar="HZ", help="the solve rate; not below %.0f"
                                          % MPC_HZ_MIN)
        g.add_argument("--no-mpc", action="store_true",
                       help="the SRB wrench and the QP allocator in the trot "
                            "too -- hw.fold2_walk's law, on this entry "
                            "point's clock: the A/B")
        g.add_argument("--mpc-polish", action="store_true",
                       help="OSQP polishing on, as in sim.  Off by default: "
                            "it prints a line per solve where it has nothing "
                            "to do")

    def configure(self, args) -> None:
        super().configure(args)
        if args.no_mpc:
            self.mpc = None
            return
        if args.mpc_hz < MPC_HZ_MIN:
            raise ValueError("--mpc-hz %.0f: a trot below %.0f Hz excites the "
                             "diagonal pendulum mode (sim.cmpc.config.MPC_HZ)"
                             % (args.mpc_hz, MPC_HZ_MIN))
        if args.mpc_hz > 0.5 * args.rate:
            raise ValueError("--mpc-hz %.0f is faster than every other sweep "
                             "(--rate %.0f)" % (args.mpc_hz, args.rate))
        self.mpc = MPC.MpcForces(hz=args.mpc_hz, mu=args.mu,
                                 polish=args.mpc_polish)
        self.mpc.warm_up()               # OSQP's setup, before the bus opens

    def law_kwargs(self, args) -> dict:
        return dict(super().law_kwargs(args), mpc=self.mpc)

    def banner(self, args) -> list:
        return super().banner(args) + [
            "     TROT FORCES: %s"
            % ("the SRB wrench + allocator %s (--no-mpc): hw.fold2_walk's law"
               % args.alloc if self.mpc is None else
               "THE MPC -- " + self.mpc.describe()),
            "     (the rise, HOLD and the park are the SRB law's, allocator "
            "%s, as hw.fold2_walk)" % args.alloc]


def cmpc_options() -> dict:
    """`hw.fold2_walk`'s options with the MPC's hook and cmpc's own clock:
    `sim.cmpc.config`'s period and duty, no contact ramp, no settle."""
    opts = dict(F2W.walk_options())
    clock = opts["gait"]
    opts["gait"] = GAIT.TrotGait(period=MPC.GAIT_PERIOD, duty=MPC.GAIT_DUTY,
                                 offsets=clock.offsets, ramp=0.0, settle_s=0.0,
                                 settle_every=clock.settle_every)
    return dict(opts, hook=CmpcHook())


def one_blas_thread() -> None:
    """Re-run this very command with OPENBLAS_NUM_THREADS=1, unless it is set.

    MEASURED ON THE PI, 2026-11-10 (the law offline, FOLD2, three cycles):
    with numpy's default four BLAS threads the MPC's solve is 5-10 ms and
    then, in runs of several, 20-40 ms -- the thread pool waking up for a
    120 x 130 product -- which is past the 25 ms CAN gap line; with one
    thread it is 5.7-6.9 ms every time.  The variable is read when numpy
    loads, and `hw` has loaded it before this module runs, so the process
    replaces itself once."""
    import os
    import sys
    if os.environ.get("OPENBLAS_NUM_THREADS") == "1":
        return
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.orig_argv[1:])


def main(argv=None) -> int:
    """`hw.fold2_walk`'s stand; the trot's forces from the MPC."""
    if argv is None:                     # the command line, not a caller
        one_blas_thread()
    return STAND.main(argv, **cmpc_options())


if __name__ == "__main__":
    raise SystemExit(main())
