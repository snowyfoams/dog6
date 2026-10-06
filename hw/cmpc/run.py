"""The convex MPC on the robot: `sim.cmpc.controller` as one phase of the stand.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.cmpc.run --fake --auto 1 --no-imu        the whole path, no robot
    $V -m hw.cmpc.run --tau-cap 2.5 --log run.npz     stand gait, the first run
    $V -m hw.cmpc.run --tau-cap 2.5 --mpc-height 140  ...and rise to 140 mm

Seven phases.  ENTER steps them; X is an E-STOP at any point:

    limp     0xA1 iq=0 keep-alives.  NO TORQUE
    settle   0xA4, holding the pose latched at ENTER          } the stand's
    crouch   0xA4, smoothstep to `sim.stand.Q_CROUCH`         } JOINT HOLD,
    lift     0xA1 TORQUE, `hw.balance`'s SRB law, to H_LIFT   } inherited
    mpc      0xA1 TORQUE, `sim.cmpc.controller.Controller`      <- added here
    park     0xA4, back to Q_CROUCH from wherever the MPC left it
    done     0xA4, holding Q_CROUCH.  ENTER exits and stops the motors

`hw.stand` explains the sequence and `hw.cmpc` (the package docstring) what
is different about the MPC on the robot.  This file is the phase machine's
one new phase and the command line that reaches it.


THE HANDOVER, IN BOTH DIRECTIONS
    lift -> mpc   The MPC's height command is LATCHED at the trunk-origin
                  height the estimator measures on entry -- the same latch the
                  SRB law makes at its own arming, for the same reason.  The
                  reference's xy and yaw anchor on the estimate's (0, 0, 0).
                  The `SafetyGate` is the same object the lift used, not
                  restarted: its cap is already at `--tau-cap` and its slew
                  limiter (5 N*m/s by default) is what blends the SRB law's
                  last torque into the MPC's first.  At the handover both
                  laws are holding the same robot at the same height, so the
                  difference it blends is their disagreement about load
                  sharing, not the weight.
    mpc -> park   `HardwareStand.advance` latches the measured pose and
                  ramps to Q_CROUCH in driver position mode -- exactly what
                  it does after the lift.  A trip in the MPC phase ends in
                  `MotorBus.close()` like every other trip: the robot goes
                  limp from the lift height.  Run the first MPC phases with
                  the robot supported, as the first lifts were.

THE GAIT IS A STAND, AND THAT IS THE LADDER
    `--gait stand` runs the MPC with four feet down at every step of the
    horizon: the same QP, a constant constraint structure, no swing leg.  It
    is the controller `sim.cmpc` was gated with in its "stand" row (2.3 mm,
    0.18 deg) with the trot taken out, and it is the A/B against the SRB law
    that just lifted the robot -- same robot, same height, same gate, same
    log columns, a horizon of planned wrenches against a closed-form PD.

    `--gait trot` is the timetable.  It is REFUSED without `--trot-anyway
    "why"`, because a diagonal needs 4.46 N*m and `SafetyGate` stops at 3.0:
    the robot would plan a trot it cannot deliver and the first swing would
    drop the trunk onto one diagonal at the cap.  Raising TAU_STAGED_MAX is a
    decision taken from a log, not from this flag.  With the trot allowed,
    W/S/A/D/Q/E command velocity and SPACE zeroes it, exactly as in
    `sim.cmpc.run`, under `config.V_MAX_HW`.

THE TIMING, AND WHY THE SOLVE IS A TRIP
    The whole controller runs at slot 0 of the 4 ms sweep, like the SRB law.
    Its 250 Hz half is cheap on this path -- the kinematics arrive
    precomputed -- but every `mpc_dt` the QP runs inside the same slot and
    delays every motor behind it by its whole duration.  `hw.stand.run`'s
    gap stop line is 25 ms; a solve past `config.SOLVE_BUDGET_S` (12 ms) for
    `SOLVE_OVER_STREAK` solves in a row is a trip here, before a motor's
    input-lost latch is the thing that reports it.  `--mpc-hz 20` halves the
    load and the stand gait allows it; the exit report prints what the solve
    actually cost.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/cmpc/run.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.cmpc"

from sim import coordinates as C             # noqa: E402
from sim.cmpc import config as SIMCFG        # noqa: E402
from sim.cmpc import controller as CTL       # noqa: E402
from sim.cmpc import gait as GAIT            # noqa: E402
from sim.cmpc import trajectory as TRAJ      # noqa: E402

from .. import CONFIRMED_ON_DOG6             # noqa: E402
from .. import hardware_map as HM            # noqa: E402
from .. import safety as SAFE                # noqa: E402
from .. import stand as HS                   # noqa: E402
from ..balance import config as BCFG         # noqa: E402
from ..balance import controller as BCTRL    # noqa: E402
from ..balance import law as BLAW            # noqa: E402
from ..balance import reference as REF       # noqa: E402
from ..balance import state as BSTATE        # noqa: E402
from ..balance import torque as TRQ          # noqa: E402
from . import config as cfg                  # noqa: E402
from . import state as EST                   # noqa: E402

__all__ = ["HardwareCmpc", "CmpcLog", "MpcTiming", "main"]

class MpcTiming:
    """Solve and sweep cost, kept as samples so p95 is a real number."""

    LIMIT = 20000

    def __init__(self) -> None:
        self.solve: list[float] = []
        self.sweep: list[float] = []
        self.over_budget = 0

    @staticmethod
    def _add(store: list, seconds: float) -> None:
        if len(store) < MpcTiming.LIMIT:
            store.append(float(seconds))

    @staticmethod
    def _summary(store: list) -> str:
        if not store:
            return "no samples"
        a = np.asarray(store)
        return ("n %d  p50 %.2f ms  p95 %.2f ms  max %.2f ms"
                % (a.size, 1e3 * np.percentile(a, 50),
                   1e3 * np.percentile(a, 95), 1e3 * a.max()))

    def report(self) -> str:
        return "\n".join([
            "  solve           %s  (%d over the %.0f ms budget)"
            % (self._summary(self.solve), self.over_budget,
               1e3 * cfg.SOLVE_BUDGET_S),
            "  sweep, no solve %s" % self._summary(self.sweep),
        ])


class HardwareCmpc(HS.HardwareStand):
    """`hw.stand.HardwareStand` with one more phase.

    Everything about limp, settle, crouch, lift, park and done -- the 0xA4
    joint hold, the ramps, the speed caps, the tracking trips -- is the
    parent's.  This class adds `mpc` after `lift`, and overrides exactly the
    three methods that have to know about it.
    """

    PHASES = ("limp", "settle", "crouch", "lift", "mpc", "park", "done")
    BLURB = {**HS.HardwareStand.BLURB,
             "mpc": "TORQUE mode -- convex MPC; ENTER parks"}

    def __init__(self, gate: SAFE.SafetyGate, *, law: str = "srb",
                 balance: BLAW.BalanceLaw | None = None,
                 gait: str = cfg.DEFAULT_GAIT, mpc_hz: float = cfg.MPC_HZ,
                 leg_gravity: bool = True, z_target: float | None = None,
                 polish: bool = False,
                 estimator: EST.KinematicOdometry | None = None):
        super().__init__(gate, law=law, balance=balance)
        if gait not in cfg.GAITS:
            raise ValueError("gait must be one of %s, got %r" % (cfg.GAITS, gait))
        self.gait = gait
        self.mpc_hz = float(mpc_hz)
        self.leg_gravity = bool(leg_gravity)
        #: Trunk-ORIGIN height to ramp to after the handover, or None to hold
        #: the latched one.  Bounded by config.Z_MIN / Z_MAX in `_mpc_arm`.
        self.z_target = z_target
        #: OSQP polishing.  OFF on the robot: see `Controller.__init__`.
        self.polish = bool(polish)
        self.estimator = estimator if estimator is not None else EST.KinematicOdometry()
        self.controller: CTL.Controller | None = None
        self.estimate: EST.Estimate | None = None
        self.operator = TRAJ.Command()
        self.timing = MpcTiming()
        self.t_mpc = 0.0
        self.z_ramp: REF.Quintic | None = None
        self.t_ramp = 0.0
        self._last_now = 0.0            # the clock of the last `update`
        self._solve_over_streak = 0
        self._qp_fail_streak = 0
        self._height_streak = 0
        self._imu_held_sweeps = 0
        self._imu_warned = False
        self.tau_grav = np.zeros(C.N_JOINTS)
        self.tau_mpc = np.zeros(C.N_JOINTS)

    # -- the hooks -----------------------------------------------------------
    def blurb(self) -> str:
        if self.phase_name == "mpc":
            return ("TORQUE, convex MPC -- %s gait, %.0f Hz solve, %s.  "
                    "ENTER parks" % (self.gait, self.mpc_hz,
                                     "+/- nudge height" if self.gait == "stand"
                                     else "WASD/QE drive, SPACE stops"))
        return super().blurb()

    def on_key(self, key: str) -> None:
        if self.phase_name != "mpc" or self.controller is None:
            return
        if key in ("+", "=", "-"):
            step = cfg.Z_NUDGE_M if key != "-" else -cfg.Z_NUDGE_M
            self.replan_height(self.controller.command.z + step)
            return
        if self.gait != "trot":
            return                       # a standing robot takes no velocity
        command = self.operator
        if key in ("w", "W"):
            command.vx += SIMCFG.V_STEP
        elif key in ("s", "S"):
            command.vx -= SIMCFG.V_STEP
        elif key in ("a", "A"):
            command.vy += SIMCFG.V_STEP
        elif key in ("d", "D"):
            command.vy -= SIMCFG.V_STEP
        elif key in ("q", "Q"):
            command.yaw_rate += SIMCFG.YAW_RATE_STEP
        elif key in ("e", "E"):
            command.yaw_rate -= SIMCFG.YAW_RATE_STEP
        elif key == " ":
            command.vx = command.vy = command.yaw_rate = 0.0
        else:
            return
        command.vx = float(np.clip(command.vx, -cfg.V_MAX_HW, cfg.V_MAX_HW))
        command.vy = float(np.clip(command.vy, -cfg.V_MAX_HW, cfg.V_MAX_HW))
        command.yaw_rate = float(np.clip(command.yaw_rate, -cfg.YAW_RATE_MAX_HW,
                                         cfg.YAW_RATE_MAX_HW))
        print("\n   command  %s" % command, flush=True)

    def log_extra(self) -> dict:
        if self.phase_name != "mpc" or self.controller is None:
            return {}
        tele = self.controller.telemetry
        est = self.estimate
        return dict(
            f_mpc=tele.forces, fz_mpc=tele.forces[:, 2],
            contacts=tele.contacts.astype(float),
            solve_ms=tele.solve_ms if tele.solved else np.nan,
            qp_ok=float(tele.status in cfg.QP_OK_STATUSES),
            clipped=float(tele.clipped),
            est_p=None if est is None else est.state.position,
            est_v=None if est is None else est.state.velocity,
            est_yaw=None if est is None else est.state.rpy[2],
            yaw_mag=None if est is None else est.yaw_mag,
            z_cmd=self.controller.command.z,
            tau_grav=self.tau_grav, tau_mpc=self.tau_mpc)

    def status_extra(self) -> str:
        if self.phase_name != "mpc" or self.controller is None:
            return super().status_extra()
        tele = self.controller.telemetry
        fz = tele.forces[:, 2]
        return ("  fz %.0f/%.0f/%.0f/%.0f N  %s  solve %.1f ms%s"
                % (*fz, self.estimator.status(), tele.solve_ms,
                   "" if tele.status in cfg.QP_OK_STATUSES
                   else "  QP:" + tele.status))

    # -- the phase machine ------------------------------------------------------
    def ramp_remaining(self, now: float) -> float:
        if self.phase_name == "mpc":
            if self.z_ramp is None:
                return 0.0
            return max(0.0, self.z_ramp.T - (now - self.t_ramp))
        return super().ramp_remaining(now)

    def advance(self, now: float, q) -> str | None:
        refused = super().advance(now, q)
        if refused is None and self.phase_name == "mpc":
            self._mpc_arm(now)
        return refused

    def _mpc_arm(self, now: float) -> None:
        """The lift -> mpc handover.  Latch everything from what is measured."""
        schedule = GAIT.SCHEDULES[self.gait]
        self.controller = CTL.Controller(schedule=schedule,
                                         mpc_dt=1.0 / self.mpc_hz,
                                         sensor_split=False,
                                         solver_settings={"polishing": self.polish})
        self.estimator.reset(self.body, now)
        self.t_mpc = now
        self.operator = TRAJ.Command()
        # The height command starts WHERE THE ROBOT IS.  `Command.z` is a
        # trunk-origin height, as the estimator's z is; the controller moves
        # it to the CoM itself.
        z0 = float(self.estimator.position[2])
        self.controller.command = TRAJ.Command(z=z0)
        self.z_ramp = None
        if self.z_target is not None:
            # No faster than 20 mm/s: the lift's own ramp is 115 mm in 3 s.
            self.replan_height(self.z_target, now, seconds=max(
                cfg.Z_RAMP_S, abs(self.z_target - z0) / 0.02))
        self.out = None                  # the balance columns go NaN
        self._solve_over_streak = self._qp_fail_streak = 0
        self._height_streak = 0

    def replan_height(self, z_target: float, now: float | None = None,
                      seconds: float = cfg.Z_RAMP_S) -> None:
        """Ramp the height command to `z_target` (trunk origin) with the
        balance package's C2 quintic, from wherever the command is NOW --
        including mid-ramp, where it continues from the current (h, hdot,
        hddot) rather than restarting.  `now` defaults to the last sweep's."""
        if self.controller is None:
            return
        now = self._last_now if now is None else float(now)
        z_target = float(np.clip(z_target, cfg.Z_MIN, cfg.Z_MAX))
        if self.z_ramp is not None:
            current = self.z_ramp.at(now - self.t_ramp)
            start = (current.h, current.hdot, current.hddot)
        else:
            start = (self.controller.command.z, 0.0, 0.0)
        self.z_ramp = REF.Quintic.between(*start, z_target, 0.0, 0.0, seconds)
        self.t_ramp = now
        print("\n   height -> %.1f mm (floor to trunk bottom %.1f) over %.1f s"
              % (1e3 * z_target, 1e3 * BSTATE.origin_to_height(z_target),
                 seconds), flush=True)

    def update(self, now: float, body):
        self._last_now = now
        if self.phase_name == "mpc":
            self.body = body
            self._sweep += 1
            return self._mpc(now, body)
        return super().update(now, body)

    # -- the phase ------------------------------------------------------------------
    def _mpc(self, now: float, body):
        """One sweep of the MPC.  Returns ``("torque", tau, trip)``."""
        started = time.perf_counter()
        controller = self.controller
        t = now - self.t_mpc

        # -- the command ---------------------------------------------------
        if self.z_ramp is not None:
            controller.command.z = self.z_ramp.at(now - self.t_ramp).h
        controller.command.vx = self.operator.vx
        controller.command.vy = self.operator.vy
        controller.command.yaw_rate = self.operator.yaw_rate
        self.h_cmd = controller.command.z

        # -- the state -----------------------------------------------------
        contacts = controller.schedule.contact(t)
        est = self.estimator.update(body, now, contacts)
        self.estimate = est
        if est.imu_held:
            self._imu_held_sweeps += 1

        # -- the controller, then the robot's own terms --------------------
        tau_ctl = controller.update(est.state, t, imu_fresh=est.imu_fresh)
        self.tau_mpc = C.flat(tau_ctl)
        if self.leg_gravity:
            self.tau_grav = C.flat(TRQ.all_leg_gravity_torque(body.q, est.state.rotation))
        else:
            self.tau_grav = np.zeros(C.N_JOINTS)
        self.tau_request = self.tau_mpc + self.tau_grav
        self.q_des = np.full(C.N_JOINTS, np.nan)     # no joint target here
        self.tau = self.gate.apply(self.tau_request, body.q, now)
        self.tau_peak = max(self.tau_peak, float(np.abs(self.tau).max()))

        # -- the books -----------------------------------------------------
        tele = controller.telemetry
        if tele.solved:
            self.timing._add(self.timing.solve, 1e-3 * tele.solve_ms)
            if 1e-3 * tele.solve_ms > cfg.SOLVE_BUDGET_S:
                self.timing.over_budget += 1
                self._solve_over_streak += 1
            else:
                self._solve_over_streak = 0
            if tele.status in cfg.QP_OK_STATUSES:
                self._qp_fail_streak = 0
            else:
                self._qp_fail_streak += 1
        else:
            self.timing._add(self.timing.sweep, time.perf_counter() - started)

        trip = self._trip(est, body, tele)
        if tele.solved:
            tele.solved = False         # consumed; the next solve sets it
        return "torque", self.tau, trip

    def _trip(self, est, body, tele) -> str | None:
        if tele.non_finite:
            return ("%d non-finite torque(s) out of the controller -- the "
                    "estimate or the QP produced NaN" % tele.non_finite)
        if body.tilt_deg > cfg.TILT_STOP_DEG:
            return ("tilt %.1f deg past the %.0f deg stop (roll %+.1f, "
                    "pitch %+.1f)" % (body.tilt_deg, cfg.TILT_STOP_DEG,
                                      np.degrees(body.roll),
                                      np.degrees(body.pitch)))
        if self._solve_over_streak >= cfg.SOLVE_OVER_STREAK:
            return ("the QP took %.1f ms, over the %.0f ms budget %d solves "
                    "in a row -- the CAN stop line is %.0f ms.  Lower --mpc-hz"
                    % (tele.solve_ms, 1e3 * cfg.SOLVE_BUDGET_S,
                       self._solve_over_streak, 1e3 * HS.GAP_ESTOP_S))
        if self._qp_fail_streak >= cfg.QP_FAIL_STREAK:
            return ("the QP returned '%s' %d solves in a row"
                    % (tele.status, self._qp_fail_streak))
        error = abs(est.state.position[2] - self.controller.command.z)
        self._height_streak = (self._height_streak + 1
                               if error > cfg.HEIGHT_STOP_M else 0)
        if self._height_streak >= cfg.HEIGHT_STOP_STREAK:
            return ("height %.1f mm from the %.1f mm commanded for %d sweeps"
                    % (1e3 * est.state.position[2],
                       1e3 * self.controller.command.z, self._height_streak))
        return None

    def report(self) -> str:
        lines = ["  gait            %s, %.0f Hz MPC, leg gravity %s"
                 % (self.gait, self.mpc_hz, "on" if self.leg_gravity else "OFF"),
                 self.timing.report(),
                 "  imu held        %d sweeps" % self._imu_held_sweeps]
        if self.controller is not None:
            lines.append("  reference ended at x %+.3f  y %+.3f  yaw %+.1f deg"
                         % (self.controller.reference.x,
                            self.controller.reference.y,
                            np.degrees(self.controller.reference.yaw)))
            lines.append("  " + self.estimator.status())
        return "\n".join(lines)


class CmpcLog(HS.StandLog):
    """`hw.stand.StandLog`'s columns -- so an MPC log and an SRB log read with
    one script -- plus the MPC's own, NaN in every non-MPC sweep."""

    EXTRA_FIELDS = {
        "f_mpc": (C.N_LEGS, 3), "fz_mpc": (C.N_LEGS,), "contacts": (C.N_LEGS,),
        "solve_ms": (), "qp_ok": (), "clipped": (),
        "est_p": (3,), "est_v": (3,), "est_yaw": (), "yaw_mag": (),
        "z_cmd": (), "tau_grav": (C.N_JOINTS,), "tau_mpc": (C.N_JOINTS,),
    }


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tau-cap", type=float, default=SAFE.TAU_START_MAX,
                    help="torque cap for the lift AND the MPC, N*m "
                         "(SafetyGate allows up to %.1f; standing needs ~2.2)"
                         % SAFE.TAU_STAGED_MAX)
    ap.add_argument("--tau-slew", type=float, default=SAFE.DEFAULT_TAU_SLEW_NM_S,
                    help="SafetyGate slew, N*m/s.  Also what blends the "
                         "handover")
    ap.add_argument("--rate", type=float, default=HS.RATE_HZ,
                    help="per-motor command rate, Hz")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--arm-timeout", type=float, default=30.0)
    ap.add_argument("--unconfirmed", default=None, metavar="WHY",
                    help="run although hw.CONFIRMED_ON_DOG6 is False")
    ap.add_argument("--fake", action="store_true",
                    help="hw.fake_bus instead of can0: protocol only, no "
                         "dynamics")
    ap.add_argument("--auto", type=float, default=None, metavar="SECONDS",
                    help="advance phases by themselves, SECONDS after each "
                         "ramp arrives (--fake only)")

    mpc = ap.add_argument_group("the MPC phase")
    mpc.add_argument("--gait", choices=cfg.GAITS, default=cfg.DEFAULT_GAIT,
                     help="stand: four feet down always.  trot: the timetable, "
                          "REFUSED without --trot-anyway")
    mpc.add_argument("--trot-anyway", default=None, metavar="WHY",
                     help="run the trot gait although a diagonal needs %.2f "
                          "N*m and the gate stops at %.1f"
                     % (cfg.TROT_DIAGONAL_NM, SAFE.TAU_STAGED_MAX))
    mpc.add_argument("--mpc-hz", type=float, default=cfg.MPC_HZ,
                     help="solve rate.  A trot may not go below %.0f"
                     % cfg.MPC_HZ_MIN_TROT)
    mpc.add_argument("--mpc-height", type=float, default=None, metavar="MM",
                     help="ramp to this height after the handover, mm FLOOR TO "
                          "TRUNK BOTTOM.  Default: hold the latched height")
    mpc.add_argument("--no-leg-gravity", action="store_true",
                     help="drop the closed-form leg-weight term: the "
                          "simulator's law exactly")
    mpc.add_argument("--polish", action="store_true",
                     help="OSQP polishing on, as in sim.  Off by default: it "
                          "prints a line per solve where it has nothing to do")

    lift = ap.add_argument_group("the lift (hw.stand's, unchanged)")
    lift.add_argument("--law", choices=HS.LAWS, default="srb")
    lift.add_argument("--rise", type=float, default=BCFG.T_RISE, metavar="SECONDS")
    lift.add_argument("--height", type=float, default=1e3 * BCFG.H_LIFT,
                      metavar="MM", help="lift target, mm floor to trunk bottom")

    sensing = ap.add_argument_group("sensing and recording")
    sensing.add_argument("--imu-port", default=None,
                         help="DETA10 serial port (default hw.imu.DEFAULT_PORT)")
    sensing.add_argument("--no-imu", action="store_true",
                         help="assume a perfectly level trunk.  AN ABLATION: "
                              "the MPC then regulates height and nothing else")
    sensing.add_argument("--log", default=None, metavar="FILE.npz",
                         help="record every torque-mode sweep, both laws")
    args = ap.parse_args(argv)

    from .. import imu as IMU
    if args.imu_port is None:
        args.imu_port = IMU.DEFAULT_PORT
    if args.auto is not None and not args.fake:
        ap.error("--auto is only allowed with --fake: on the robot a person "
                 "steps the phases")
    if args.law == "per-leg" and not args.no_imu:
        args.no_imu = True              # the baseline reads no IMU, as in hw.stand
    if args.gait == "trot":
        if not args.trot_anyway:
            ap.error("--gait trot: a trot diagonal needs %.2f N*m and "
                     "SafetyGate stops at %.1f.  The robot would plan a trot it "
                     "cannot deliver.  To run it anyway: --trot-anyway \"why\""
                     % (cfg.TROT_DIAGONAL_NM, SAFE.TAU_STAGED_MAX))
        if args.mpc_hz < cfg.MPC_HZ_MIN_TROT:
            ap.error("--mpc-hz %.0f: a trot below %.0f Hz excites the diagonal "
                     "pendulum mode (sim.cmpc.config.MPC_HZ)"
                     % (args.mpc_hz, cfg.MPC_HZ_MIN_TROT))
    if 1.0 / args.mpc_hz < 2.0 / args.rate:
        ap.error("--mpc-hz %.0f is faster than every other sweep" % args.mpc_hz)
    if args.rate * 2 * HS.STATUS_EVERY_SWEEPS < 1.0 / HS.GAP_ESTOP_S:
        ap.error("--rate %.0f Hz cannot keep every motor inside the %.0f ms "
                 "stop line" % (args.rate, 1e3 * HS.GAP_ESTOP_S))
    reason = args.unconfirmed or ("hw.fake_bus" if args.fake else None)

    # Everything that can refuse, refuses BEFORE the bus opens.
    ids = HM.motor_ids()
    try:
        gate = SAFE.SafetyGate(args.tau_cap, tau_slew=args.tau_slew,
                               unconfirmed_reason=reason)
    except (RuntimeError, ValueError) as refusal:
        print("[cmpc] REFUSED before opening the bus:\n  %s" % refusal,
              file=sys.stderr)
        return 2
    balance = BLAW.BalanceLaw(gains=BCTRL.BalanceGains(), rise_s=args.rise,
                              h_lift=1e-3 * args.height)
    z_target = (None if args.mpc_height is None
                else BSTATE.height_to_origin(1e-3 * args.mpc_height))
    stand = HardwareCmpc(gate, law=args.law, balance=balance, gait=args.gait,
                         mpc_hz=args.mpc_hz, leg_gravity=not args.no_leg_gravity,
                         z_target=z_target, polish=args.polish)
    log = CmpcLog() if args.log else None
    key = HS.KeyPoller()
    if not key.ok and not args.fake:
        print("[cmpc] stdin is not a terminal, so neither ENTER nor the X "
              "e-stop can reach this process.  Run it from a shell.",
              file=sys.stderr)
        return 2

    print("DOG6 convex MPC on %s" % ("hw.fake_bus" if args.fake else "can0"))
    print("  CAN ids %s  directions %s" % (ids, HM.directions()))
    print("  map confirmed: %s%s" % (CONFIRMED_ON_DOG6,
                                     "" if CONFIRMED_ON_DOG6
                                     else "  (running on: %r)" % reason))
    print("  tau cap %.2f N*m, slew %.1f N*m/s (standing needs ~2.2; a trot "
          "diagonal %.2f)" % (args.tau_cap, args.tau_slew, cfg.TROT_DIAGONAL_NM))
    print("  HEIGHTS BELOW ARE FLOOR TO TRUNK BOTTOM -- a ruler reaches them.")
    print("  lift: %s law to %.0f mm over %.1f s, then the MPC takes over "
          "at the height it measures" % (args.law, args.height, args.rise))
    print("  MPC: %s gait, %.0f Hz solve, horizon %d x %.3f s, %s"
          % (args.gait, args.mpc_hz, SIMCFG.HORIZON, 1.0 / args.mpc_hz,
             "hold the latched height" if args.mpc_height is None
             else "then rise to %.0f mm" % args.mpc_height))
    print("       Q diag %s" % np.array2string(SIMCFG.Q_DIAG[:12], precision=2))
    print("       leg gravity %s; kinematics closed-form; state: IMU roll/"
          "pitch, gyro yaw, feet for z and v"
          % ("ON (hw.balance.torque)" if not args.no_leg_gravity
             else "OFF -- the simulator's law exactly"))
    if args.gait == "trot":
        print("       TROT ALLOWED on: %r.  WASD/QE drive, SPACE stops, "
              "|v| <= %.2f m/s" % (args.trot_anyway, cfg.V_MAX_HW))
    if args.no_imu:
        print("       NO IMU: the trunk is ASSUMED level.  Roll, pitch and "
              "yaw are identically zero,")
        print("       so the MPC regulates height alone -- an ablation, not "
              "the controller this is.")
    print("  %.0f Hz per motor, %.1f ms sweep; stop line %.0f ms; solve "
          "budget %.0f ms" % (args.rate, 1e3 / args.rate, 1e3 * HS.GAP_ESTOP_S,
                              1e3 * cfg.SOLVE_BUDGET_S))
    print("  trips (mpc): tilt %.0f deg, height %.0f mm sustained, QP status, "
          "solve budget" % (cfg.TILT_STOP_DEG, 1e3 * cfg.HEIGHT_STOP_M))
    print("  ENTER steps the phase.  X is an E-STOP -- during lift or mpc it "
          "DROPS the robot.")
    if args.fake and args.law == "srb":
        print("  NOTE: hw.fake_bus has NO DYNAMICS, so the SRB lift's tracking "
              "trip fires partway up;")
        print("  run the whole fake path with --law per-leg, as hw.stand says.")

    mb = HS.open_bus(ids, fake=args.fake, bitrate=args.bitrate)
    imu = None
    if not args.no_imu:
        try:
            imu = HS.open_imu(args.imu_port)
        except Exception as failure:                     # noqa: BLE001
            print("[cmpc] no IMU: %s\n[cmpc] pass --no-imu to run the "
                  "level-trunk ablation deliberately." % failure,
                  file=sys.stderr)
            return 2

    stop = None
    try:
        with mb:
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[cmpc] not every motor armed", file=sys.stderr)
                return 1
            stop = HS.run(mb, stand, rate_hz=args.rate, key=key,
                          auto_s=args.auto, imu=imu, log=log)
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        key.restore()
        if imu is not None:
            imu.stop()

    print()
    if args.law == "srb" and stand.balance.armed:
        print("[cmpc] the lift, as the SRB law saw it:")
        print(stand.balance.report())
    if stand.controller is not None:
        print("[cmpc] the MPC phase:")
        print(stand.report())
    if log is not None:
        print("[cmpc] log: %s" % log.save(args.log))
    if stop is None:
        print("[cmpc] done; motors stopped in the crouch.  Peak torque "
              "%.2f N*m." % stand.tau_peak)
        return 0
    print("[cmpc] E-STOP in phase %s: %s" % (stand.phase_name, stop))
    print("[cmpc] motors stopped.")
    return 0 if stop == "operator X" else 1


if __name__ == "__main__":
    raise SystemExit(main())
