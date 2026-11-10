"""The convex MPC on the robot: `sim.cmpc.controller` as one phase of the stand.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.cmpc.run --fake --auto 1 --law per-leg   the whole path, no robot
    $V -m hw.cmpc.run --log run.npz                   stand gait
    $V -m hw.cmpc.run --mpc-height 140                ...and rise to 140 mm
    $V -m hw.cmpc.run --gait trot --log trot.npz      the trot, WASD/QE

NO SafetyGate (2026-11-10, on request): no torque cap, no slew, no ramp, no
joint-limit block, no overspeed or joint-limit stop, in the lift and the MPC
alike.  What the laws ask goes to the drivers, clipped only by the
controller's own 8 N*m (`sim.cmpc.config.TAU_MAX`) and the 0xA1 field's 9.94.
What still stops a run: X, NaN, over-temperature, CAN miss, a driver fault
code, the 0x80 input-lost latch, the CAN gap line, the torque read-back, and
the lift's own trips (hw.stand's tracking, the SRB law's 12 deg).

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
                  With no gate nothing blends the SRB law's last torque into
                  the MPC's first: the step is their disagreement about load
                  sharing, both holding the same robot at the same height.
    mpc -> park   `HardwareStand.advance` latches the measured pose and
                  ramps to Q_CROUCH in driver position mode -- exactly what
                  it does after the lift.  A stop in the MPC phase ends in
                  `MotorBus.close()` like every other: the robot goes limp
                  from the lift height.

THE FALL HOLD, IN PLACE OF A TILT STOP
    Past `--fall-hold` degrees of tilt (config.FALL_HOLD_DEG, 45) the MPC
    stops for the rest of the run and the phase reads "fall": the drivers
    hold every joint at the angle it had on that sweep, in their own 0xA4
    loop, and nothing stops.  ENTER parks from there; X drops.  This phase
    used to stop at the balance package's 12 deg, a drop from the lift
    height.  The lift before it is hw.stand's and still stops at 12.

THE GAITS
    `--gait stand` (the default) runs the MPC with four feet down at every
    step of the horizon: the same QP, a constant constraint structure, no
    swing leg -- `sim.cmpc`'s "stand" row (2.3 mm, 0.18 deg) with the trot
    taken out, and the A/B against the SRB law that just lifted the robot.

    `--gait trot` is the timetable.  THE SWING LEG IS THE BALANCE
    CONTROLLER'S (`hw.cmpc.swing`, 2026-11-10): the walk's Cartesian
    impedance as `hw.fold2_walk` flies it, Kp 150/150/400, Kd 6/6/40, no
    feedforward, apex 20 mm, its weight from the leg-gravity term every leg
    gets; sim.cmpc's foothold and arc feed it.  W/S/A/D/Q/E command velocity
    and SPACE zeroes it, exactly as in `sim.cmpc.run`, under
    `config.V_MAX_HW`.

THE TIMING
    The whole controller runs at slot 0 of the 4 ms sweep, like the SRB law.
    Every `mpc_dt` the QP runs inside the same slot and delays every motor
    behind it by its whole duration.  A solve past `config.SOLVE_BUDGET_S`
    (12 ms) is counted, not tripped; `hw.stand.run`'s 25 ms CAN gap line is
    what stops a loop that really stalls.  `--mpc-hz 20` halves the load and
    the stand gait allows it; the exit report prints what the solve cost.
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
from sim import kinematics as SK             # noqa: E402
from sim.cmpc import config as SIMCFG        # noqa: E402
from sim.cmpc import controller as CTL       # noqa: E402
from sim.cmpc import gait as GAIT            # noqa: E402
from sim.cmpc import trajectory as TRAJ      # noqa: E402

from .. import CONFIRMED_ON_DOG6             # noqa: E402
from .. import hardware_map as HM            # noqa: E402
from .. import stand as HS                   # noqa: E402
from ..balance import config as BCFG         # noqa: E402
from ..balance import controller as BCTRL    # noqa: E402
from ..balance import law as BLAW            # noqa: E402
from ..balance import reference as REF       # noqa: E402
from ..balance import state as BSTATE        # noqa: E402
from ..balance import torque as TRQ          # noqa: E402
from . import config as cfg                  # noqa: E402
from . import state as EST                   # noqa: E402
from . import swing as SWING                 # noqa: E402

__all__ = ["HardwareCmpc", "HardwareController", "CmpcLog", "MpcTiming", "main"]


class HardwareController(CTL.Controller):
    """`sim.cmpc.controller.Controller` with two things the robot's:

    THE SWING LEG IS THE BALANCE CONTROLLER'S (2026-11-10, on request).
        `hw.cmpc.swing`: the walk's Cartesian impedance as `hw.fold2_walk`
        flies it -- Kp 150/150/400, Kd 6/6/40 per leg, no feedforward, no leg
        weight inside (the run's leg gravity gives every leg its own once),
        apex 20 mm -- in place of sim.cmpc's eq (1)-(3).  The arc and the
        foothold are still sim.cmpc's.
    A FAILED QP KEEPS THE LAST GOOD FORCES.  The QP returns zero force on a
        failure; with the QP-failure trip gone (2026-11-10) that would drop
        the stance for a whole MPC period, where the previous plan is the
        zero-order hold the method assumes anyway.  `qp_failures` counts.

    `kp_swing` / `kd_swing` are (3,) for every leg or (4, 3) a row per leg."""

    def __init__(self, *args, kp_swing=None, kd_swing=None,
                 swing_height: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.qp_failures = 0
        self.kp_swing = np.asarray(SWING.KP_SWING_WALK if kp_swing is None
                                   else kp_swing, dtype=float)
        self.kd_swing = np.asarray(SWING.KD_SWING_WALK if kd_swing is None
                                   else kd_swing, dtype=float)
        self.swing_height = float(SWING.SWING_HEIGHT if swing_height is None
                                  else swing_height)

    def solve(self, state, t: float) -> None:
        previous = self._forces.copy()
        super().solve(state, t)
        if self.telemetry.status not in cfg.QP_OK_STATUSES:
            self.qp_failures += 1
            self._forces = previous
            self.telemetry.forces = previous.copy()

    def _swing_torque(self, state, leg: int, q, qd,
                      p_ref_body, v_ref_body, a_ref_body) -> np.ndarray:
        """The robot's swing law (`hw.cmpc.swing`).  `a_ref_body` is unused:
        no feedforward, as flown."""
        if state.feet_body is not None and state.jacobians_body is not None:
            foot_b, jac = state.feet_body[leg], state.jacobians_body[leg]
        else:                            # a state built without them
            foot_b, jac = SK.leg_state(leg, q)
        return SWING.swing_torque(foot_b, jac, qd, p_ref_body, v_ref_body,
                                  SWING.leg_gains(self.kp_swing, leg),
                                  SWING.leg_gains(self.kd_swing, leg))

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
    three methods that have to know about it -- and `phase_name`, which reads
    "fall" once the fall hold has taken the MPC phase over.
    """

    PHASES = ("limp", "settle", "crouch", "lift", "mpc", "park", "done")
    BLURB = {**HS.HardwareStand.BLURB,
             "mpc": "TORQUE mode -- convex MPC; ENTER parks",
             "fall": "driver position mode -- THE FALL HOLD: no MPC, every "
                     "joint held where it was.  ENTER parks"}

    def __init__(self, *, law: str = "srb",
                 balance: BLAW.BalanceLaw | None = None,
                 gait: str = cfg.DEFAULT_GAIT, mpc_hz: float = cfg.MPC_HZ,
                 leg_gravity: bool = True, z_target: float | None = None,
                 polish: bool = False,
                 fall_hold_deg: float | None = cfg.FALL_HOLD_DEG,
                 estimator: EST.KinematicOdometry | None = None,
                 kp_swing=None, kd_swing=None,
                 swing_height: float | None = None):
        # NO GATE: the lift and the MPC send what their laws ask.
        super().__init__(None, law=law, balance=balance)
        if gait not in cfg.GAITS:
            raise ValueError("gait must be one of %s, got %r" % (cfg.GAITS, gait))
        self.gait = gait
        self.mpc_hz = float(mpc_hz)
        #: The leg-weight term, on ALL FOUR legs: the stance legs' as the SRB
        #: law adds it, and the swinging legs' only weight term -- the robot's
        #: swing law (`hw.cmpc.swing`) has none of its own.
        self.leg_gravity = bool(leg_gravity)
        #: The swing law's gains and apex (`HardwareController`); None is
        #: `hw.cmpc.swing`'s, the robot's walk defaults.
        self.kp_swing = kp_swing
        self.kd_swing = kd_swing
        self.swing_height = swing_height
        #: Trunk-ORIGIN height to ramp to after the handover, or None to hold
        #: the latched one.  Bounded by config.Z_MIN / Z_MAX in `_mpc_arm`.
        self.z_target = z_target
        #: OSQP polishing.  OFF on the robot: see `Controller.__init__`.
        self.polish = bool(polish)
        self.estimator = estimator if estimator is not None else EST.KinematicOdometry()
        self.controller: HardwareController | None = None
        self.estimate: EST.Estimate | None = None
        self.operator = TRAJ.Command()
        self.timing = MpcTiming()
        self.t_mpc = 0.0
        self.z_ramp: REF.Quintic | None = None
        self.t_ramp = 0.0
        self._last_now = 0.0            # the clock of the last `update`
        self._imu_held_sweeps = 0
        self._imu_warned = False
        self.tau_grav = np.zeros(C.N_JOINTS)
        self.tau_mpc = np.zeros(C.N_JOINTS)
        #: Tilt, deg, past which the fall hold takes the MPC phase over
        #: (config.FALL_HOLD_DEG); None is off.  `q_fall` is its latch: the
        #: joint angles of the sweep it engaged on, None until then.
        self.fall_hold_deg = (None if fall_hold_deg is None or fall_hold_deg <= 0
                              else float(fall_hold_deg))
        self.q_fall: np.ndarray | None = None
        self.t_fall = 0.0
        self.fall_tilt_deg = 0.0
        self.fall_sweeps = 0
        self.fall_err_peak = 0.0

    @property
    def phase_name(self) -> str:
        """The parent's, except that the MPC phase reads "fall" once the fall
        hold has engaged -- so the status line and an X's report say so.
        ENTER from it is the parent's step to "park"."""
        name = self.PHASES[self.phase]
        return "fall" if name == "mpc" and self.q_fall is not None else name

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
        if self.phase_name == "fall":
            return ("  FALL HOLD, worst joint %.1f deg off its held angle"
                    % np.degrees(self.fall_err_peak))
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
        self.controller = HardwareController(schedule=schedule,
                                             mpc_dt=1.0 / self.mpc_hz,
                                             sensor_split=False,
                                             solver_settings={"polishing": self.polish},
                                             kp_swing=self.kp_swing,
                                             kd_swing=self.kd_swing,
                                             swing_height=self.swing_height)
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
        if self.phase_name == "fall":
            self.body = body
            self._sweep += 1
            return self._fall_hold(body)
        if self.phase_name == "mpc":
            self.body = body
            self._sweep += 1
            return self._mpc(now, body)
        return super().update(now, body)

    # -- the phase ------------------------------------------------------------------
    def _mpc(self, now: float, body):
        """One sweep of the MPC.  Returns ``("torque", tau, trip)`` -- or, on
        the sweep the tilt passes the fall hold, the hold's first command."""
        # -- the fall hold, BEFORE anything else reads the state ------------
        # On the sweep it engages as on every one after: not one more MPC
        # torque goes out once the trunk is past it.
        if self.fall_hold_deg is not None and body.tilt_deg > self.fall_hold_deg:
            self._fall(now, body)
            return self._fall_hold(body)
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
        self.tau = self.tau_request.copy()           # no gate: sent as asked
        self.tau_peak = max(self.tau_peak, float(np.abs(self.tau).max()))

        # -- the books -----------------------------------------------------
        tele = controller.telemetry
        if tele.solved:
            self.timing._add(self.timing.solve, 1e-3 * tele.solve_ms)
            if 1e-3 * tele.solve_ms > cfg.SOLVE_BUDGET_S:
                self.timing.over_budget += 1
        else:
            self.timing._add(self.timing.sweep, time.perf_counter() - started)

        trip = self._trip(tele)
        if tele.solved:
            tele.solved = False         # consumed; the next solve sets it
        return "torque", self.tau, trip

    # -- the fall hold --------------------------------------------------------------
    def _fall(self, now: float, body) -> None:
        """Engage the fall hold, ONCE: latch every joint's angle this sweep.
        From here `phase_name` reads "fall" until ENTER parks."""
        self.q_fall = np.array(body.q, dtype=float)
        self.t_fall = float(now)
        self.fall_tilt_deg = float(body.tilt_deg)
        self.max_dps = np.full(C.N_JOINTS, HS.SETTLE_MOTOR_DPS)
        self.tau = np.zeros(C.N_JOINTS)
        self.tau_request = np.zeros(C.N_JOINTS)
        self.h_cmd = float("nan")
        HS._print_phase(self)
        print("   tilt %.1f deg past %.0f (roll %+.1f, pitch %+.1f), %.1f s into "
              "the MPC: no stop, every joint held where it is"
              % (self.fall_tilt_deg, self.fall_hold_deg, np.degrees(body.roll),
                 np.degrees(body.pitch), now - self.t_mpc), flush=True)

    def _fall_hold(self, body):
        """One fall-hold sweep: every joint to `q_fall` in the drivers' 0xA4
        loop, at settle's speed cap.  No torque law runs and nothing here
        trips -- the stand's own trips in `hw.stand.run` still do."""
        self.fall_sweeps += 1
        self.fall_err_peak = max(self.fall_err_peak,
                                 float(np.abs(body.q - self.q_fall).max()))
        self.q_des = self.q_fall
        return "position", self.q_des, None

    def _trip(self, tele) -> str | None:
        """The MPC phase's one trip: a NaN out of the controller.  The height,
        solve-budget and QP-failure trips were removed on 2026-11-10."""
        if tele.non_finite:
            return ("%d non-finite torque(s) out of the controller -- the "
                    "estimate or the QP produced NaN" % tele.non_finite)
        return None

    def report(self) -> str:
        lines = ["  gait            %s, %.0f Hz MPC, leg gravity %s"
                 % (self.gait, self.mpc_hz, "on" if self.leg_gravity else "OFF"),
                 self.timing.report(),
                 "  QP failures     %d (each kept the last good forces)"
                 % (0 if self.controller is None else self.controller.qp_failures),
                 "  imu held        %d sweeps" % self._imu_held_sweeps]
        if self.fall_hold_deg is None:
            lines.append("  fall hold       OFF -- nothing acted on a tilt")
        elif self.q_fall is None:
            lines.append("  fall hold       at %.0f deg, never engaged"
                         % self.fall_hold_deg)
        else:
            lines.append("  fall hold       ENGAGED %.1f s into the MPC at "
                         "%.1f deg (past %.0f); held %d sweeps, worst joint "
                         "%.1f deg off its held angle"
                         % (self.t_fall - self.t_mpc, self.fall_tilt_deg,
                            self.fall_hold_deg, self.fall_sweeps,
                            np.degrees(self.fall_err_peak)))
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


def swing_leg_gains(args) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(4, 3) Kp, (4, 3) Kd -- a row per leg, FL FR RL RR: `--kp-swing-walk` /
    `--kd-swing-walk` in every row, a leg's own flags in its own -- and the
    legs that have their own.  hw.stand.swing_leg_gains on `cloud`."""
    kp = np.tile(np.asarray(args.kp_swing_walk, dtype=float), (C.N_LEGS, 1))
    kd = np.tile(np.asarray(args.kd_swing_walk, dtype=float), (C.N_LEGS, 1))
    own = []
    for i, leg in enumerate(C.LEGS):
        row_p = getattr(args, "kp_swing_walk_%s" % leg.lower(), None)
        row_d = getattr(args, "kd_swing_walk_%s" % leg.lower(), None)
        if row_p is not None:
            kp[i] = row_p
        if row_d is not None:
            kd[i] = row_d
        if row_p is not None or row_d is not None:
            own.append(leg)
    return kp, kd, own


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--rate", type=float, default=HS.RATE_HZ,
                    help="per-motor command rate, Hz")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--arm-timeout", type=float, default=30.0)
    ap.add_argument("--fake", action="store_true",
                    help="hw.fake_bus instead of can0: protocol only, no "
                         "dynamics")
    ap.add_argument("--auto", type=float, default=None, metavar="SECONDS",
                    help="advance phases by themselves, SECONDS after each "
                         "ramp arrives (--fake only)")

    mpc = ap.add_argument_group("the MPC phase")
    mpc.add_argument("--gait", choices=cfg.GAITS, default=cfg.DEFAULT_GAIT,
                     help="stand: four feet down always.  trot: the timetable, "
                          "WASD/QE drive")
    mpc.add_argument("--mpc-hz", type=float, default=cfg.MPC_HZ,
                     help="solve rate.  A trot may not go below %.0f"
                     % cfg.MPC_HZ_MIN_TROT)
    mpc.add_argument("--mpc-height", type=float, default=None, metavar="MM",
                     help="ramp to this height after the handover, mm FLOOR TO "
                          "TRUNK BOTTOM.  Default: hold the latched height")
    mpc.add_argument("--fall-hold", type=float, default=cfg.FALL_HOLD_DEG,
                     metavar="DEG",
                     help="past this tilt the MPC stops and the drivers hold "
                          "every joint where it is (0xA4): no stop, ENTER "
                          "parks.  0 = off, and nothing acts on a tilt")
    mpc.add_argument("--no-leg-gravity", action="store_true",
                     help="drop the closed-form leg-weight term from EVERY "
                          "leg -- stance and swing: the robot's swing law has "
                          "no weight term of its own")
    mpc.add_argument("--polish", action="store_true",
                     help="OSQP polishing on, as in sim.  Off by default: it "
                          "prints a line per solve where it has nothing to do")

    sw = ap.add_argument_group(
        "the swing leg (trot): the balance controller's, as hw.fold2_walk "
        "flies it -- hw.cmpc.swing")
    sw.add_argument("--kp-swing-walk", type=float, nargs=3,
                    default=SWING.KP_SWING_WALK.tolist(), metavar="N_PER_M",
                    help="swing impedance stiffness, trunk x y z")
    sw.add_argument("--kd-swing-walk", type=float, nargs=3,
                    default=SWING.KD_SWING_WALK.tolist(), metavar="NS_PER_M",
                    help="swing impedance damping, trunk x y z")
    for leg in C.LEGS:                   # hw.stand.add_swing_leg_flags on cloud
        sw.add_argument("--kp-swing-walk-%s" % leg.lower(), type=float, nargs=3,
                        default=None, metavar="N_PER_M",
                        help="%s's own stiffness, in place of --kp-swing-walk"
                             % leg)
        sw.add_argument("--kd-swing-walk-%s" % leg.lower(), type=float, nargs=3,
                        default=None, metavar="NS_PER_M",
                        help="%s's own damping, in place of --kd-swing-walk"
                             % leg)
    sw.add_argument("--swing-height", type=float,
                    default=1e3 * SWING.SWING_HEIGHT, metavar="MM",
                    help="the swing apex above the higher end of the arc")

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
    if args.gait == "trot" and args.mpc_hz < cfg.MPC_HZ_MIN_TROT:
        ap.error("--mpc-hz %.0f: a trot below %.0f Hz excites the diagonal "
                 "pendulum mode (sim.cmpc.config.MPC_HZ)"
                 % (args.mpc_hz, cfg.MPC_HZ_MIN_TROT))
    if 1.0 / args.mpc_hz < 2.0 / args.rate:
        ap.error("--mpc-hz %.0f is faster than every other sweep" % args.mpc_hz)
    if args.rate * 2 * HS.STATUS_EVERY_SWEEPS < 1.0 / HS.GAP_ESTOP_S:
        ap.error("--rate %.0f Hz cannot keep every motor inside the %.0f ms "
                 "stop line" % (args.rate, 1e3 * HS.GAP_ESTOP_S))

    ids = HM.motor_ids()
    balance = BLAW.BalanceLaw(gains=BCTRL.BalanceGains(), rise_s=args.rise,
                              h_lift=1e-3 * args.height)
    z_target = (None if args.mpc_height is None
                else BSTATE.height_to_origin(1e-3 * args.mpc_height))
    kp_swing, kd_swing, own_swing = swing_leg_gains(args)
    stand = HardwareCmpc(law=args.law, balance=balance, gait=args.gait,
                         mpc_hz=args.mpc_hz, leg_gravity=not args.no_leg_gravity,
                         z_target=z_target, polish=args.polish,
                         fall_hold_deg=args.fall_hold,
                         kp_swing=kp_swing, kd_swing=kd_swing,
                         swing_height=1e-3 * args.swing_height)
    log = CmpcLog() if args.log else None
    key = HS.KeyPoller()
    if not key.ok and not args.fake:
        print("[cmpc] stdin is not a terminal, so neither ENTER nor the X "
              "e-stop can reach this process.  Run it from a shell.",
              file=sys.stderr)
        return 2

    print("DOG6 convex MPC on %s" % ("hw.fake_bus" if args.fake else "can0"))
    print("  CAN ids %s  directions %s" % (ids, HM.directions()))
    print("  map confirmed: %s" % CONFIRMED_ON_DOG6)
    print("  NO SAFETY GATE: no cap, no slew, no ramp, no joint-limit block -- "
          "the laws' torque goes out as asked")
    print("  stops: X, NaN, overtemp, CAN miss, motor fault, input-lost, CAN "
          "gap, torque read-back, the lift's own trips")
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
          % ("ON (hw.balance.torque), every leg" if not args.no_leg_gravity
             else "OFF -- no leg weight on any leg, swing included"))
    if args.gait == "trot":
        print("       TROT: WASD/QE drive, SPACE stops, |v| <= %.2f m/s"
              % cfg.V_MAX_HW)
        print("       swing: the balance controller's impedance, Kp %s N/m, "
              "Kd %s N s/m, no feedforward, apex %.0f mm%s"
              % (args.kp_swing_walk, args.kd_swing_walk, args.swing_height,
                 "" if not own_swing else "; own gains on %s" % ", ".join(own_swing)))
    if args.no_imu:
        print("       NO IMU: the trunk is ASSUMED level.  Roll, pitch and "
              "yaw are identically zero,")
        print("       so the MPC regulates height alone -- an ablation, not "
              "the controller this is.")
    print("  %.0f Hz per motor, %.1f ms sweep; stop line %.0f ms; solve "
          "budget %.0f ms (counted, no trip)"
          % (args.rate, 1e3 / args.rate, 1e3 * HS.GAP_ESTOP_S,
             1e3 * cfg.SOLVE_BUDGET_S))
    print("  trips (mpc): NaN only; a failed QP keeps the last good forces")
    tilt_rule = ("no stop -- past %.0f deg the FALL HOLD: the MPC stops and "
                 "every joint is held where it is" % args.fall_hold
                 if args.fall_hold > 0 else "nothing acts on it (--fall-hold 0)")
    print("  tilt (mpc): %s" % tilt_rule)
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
