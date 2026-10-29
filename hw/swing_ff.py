"""The swing leg's inertial feedforward, A/B ON THE ROBOT: hung up, FOLD2's
hold, the legs swing on the clock -- ONE RUN WITH `--swing-ff`, ONE WITHOUT,
each to its own log.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.swing_ff --swings 10                   FF OFF: ENTER pose, ENTER swing, 10 per leg,
                                                    hold; X.  Log: swing_ff_off_<time>.npz
    $V -m hw.swing_ff --swings 10 --swing-ff        FF ON, the same.  Log: swing_ff_on_<time>.npz
    $V -m hw.swing_ff --legs RR --swings 10 --swing-ff --log rr_on.npz
                                                    one leg: doc/swing_height's case, on the robot
    $V -m hw.swing_ff --swings 10 --sched --sched-zeta 0.7
                                                    the swing PD's Kp, Kd scheduled on Lambda(q):
                                                    Kp = wn^2 Lambda, Kd = 2 zeta wn Lambda (below)

THE FEEDFORWARD IS A COMMAND-LINE DECISION, NOT A KEY: a run is FF off or FF
on from its first sweep to its last, its log says which (`ff_on`, and the
file name), and the A/B is two runs.  ENTER in hold swings again, the same
way.

HANG THE ROBOT UP FIRST.  No balance law, no height loop, no allocator and
NO IMU: the trunk is taken as level and nothing here carries it.  The legs
are the whole experiment.

THE QUESTION (doc/swing_height, branch swing-analysis, 2026-10-27).  The
one-leg study put the robot's swing code on MuJoCo's weighed RR leg
(`doc/swing_height/model/leg_RR.xml`) through the hardware chain, at the
walk's 20 mm / 0.8 s / duty 0.8 / ramp 0.15, and predicted:

    FF off   apex ~15 mm at s = 0.83, lag ~55 ms, h_td ~13 mm   (the robot flew
             19 / 50 / 13 in the gait study, FF off)
    FF on    apex ~23 mm at s = 0.52, lag ~33 ms, h_td ~ -2 mm, tau_ff peak 3.1 N*m

This script flies both on the robot, same hold, same clock, same gains, one
log, and prints the study's per-swing metrics for each so the two columns
can be put beside the study's table.

    limp    zero torque; the legs hang.                                 ENTER ->
    pose    POSITION mode, a smoothstep to FOLD2's hold over `--pose-s`:
            `posture.FOLD2`'s feet at their hip-frame sites, the trunk
            `--height` (160 mm, `hw.fold2_trot`'s stand) above a floor that
            is not there.
    hold    TORQUE, entered by itself when the ramp arrives: EVERY LEG ON
            THE JOINT LAYER the trots and walks fly, Kp (q_hold - q) - Kd qd
            (`config.KP_JOINT_HOLD` 5 / `KD_JOINT_HOLD` 0.2, `_ABD` for abd;
            `--kp-joint` and the rest are `hw.trot`'s flags), plus each
            leg's own gravity.  q_hold is the IK pose the ramp went to.
                                                                        ENTER ->
    swing   the gait clock runs -- TIME ALONE decides stance and swing, no
            touchdown detection.  A leg in `--legs` that the clock lifts
            leaves the joint layer entirely (the 2026-10-01 rule in
            `law.BalanceLaw`: a swinging leg gets no spring and no damper)
            and follows the arc on the Cartesian PD, `--swing-ff` adding
            M0 J^+ (a_ref - Jdot qd_ref).  The stance legs keep the joint
            layer.  ENTER latches a stop at the next four-foot window;
            `--swings N` does the same once every swinging leg has landed
            N times.  ENTER in hold swings again.
    X       e-stop, any time.  Motors stop; the legs drop to their hang.

THE SCHEDULED PD (`--sched`, 2026-10-28).  The swing's Cartesian PD has fixed
gains, 150 / 150 / 400 N/m and 6 / 6 / 40 N*s/m, on a foot whose apparent mass
Lambda = (J M0^-1 J^T)^-1 is 0.31 / 0.36 / 5.8 kg at the hold and moves with the
leg (z: 5.8 kg on the ground, 3.3 kg 20 mm up).  `--sched` re-reads Lambda from
every sweep's Jacobian and sends Kp = wn^2 Lambda, Kd = 2 zeta wn Lambda
(`swing.GainSchedule`): one natural frequency and one damping ratio per axis
whatever the configuration.  wn defaults to what makes Kp at the FOLD2 hold the
`--kp-swing` value, so at the hold nothing moves but Kd (zeta 0.7 asks z for
~67 N*s/m against 40, zeta 1.0 for ~95); `--sched-wn` sets it.  Lambda is
clamped to `--sched-lam-range` x its hold value (Kp floor and singularity cap).
`--kd-swing` is not used when scheduled.  The log holds kp_eff / kd_eff /
lam (4, 3, NaN off the swing) and the schedule's constants.

WHAT IT SHARES WITH THE WALK AND THE STUDY, EXACTLY
    `balance.swing.swing_reference_pva` (the arc), `swing_torque` (the PD, at
    the walk's `KP_SWING_WALK` / `KD_SWING_WALK`: for an in-place arc it is
    `swing_control.swing_control_torque`'s expression), `swing_feedforward`
    (`swing_control.inertial_feedforward` to 1e-13 N*m, study.py's check),
    `balance.gait.TrotGait` at the study's clock, `torque.leg_gravity_torque`,
    the joint layer's gains, and `safety.SafetyGate` at the study's cap and
    slew (9 N*m, 300 N*m/s: the 2026-10-21 line).  M0 is the law's own
    (`swing.JOINT_INERTIA_M0`: CAD links at the lift pose + `params.ARMATURE`);
    the study found M0 built from the weighed leg at this hold changes the
    apex by 0.2 mm, so it is not a knob here beyond `--ff-armature`.

WHAT IT PRINTS -- the study's metrics, per swing, heights above the leg's
RESTING SITE (the hold's target, the arc's zero; there is no floor):

    swing RR #3 FF on   apex 22.8 mm at s=0.51  lag 32 ms  rms 3.1 mm  h_td -1.5 mm
          vz_td -0.10 m/s  x 4.8 y 0.6 mm  tau_ff 3.0  pd 0.9  clip 0.2  peak 3.3 N*m

    apex    max height in the clock's window, and where on the arc
    lag     clock liftoff -> first sweep above 4 mm
    rms     |p_swing - x_b| over the window, 3-D
    h_td    height on the sweep the clock lands the foot
    vz_td   the foot's vertical speed on that sweep

At exit, one row per leg with the medians, beside the study's.

THE LOG -- ALWAYS WRITTEN, `--log FILE.npz` names it, the default is
swing_ff_<on|off>_<YYYYmmdd-HHMMSS>.npz in the working directory, so the two
runs never land in one file.  Every torque-mode sweep: t, phase, bout (the
ENTER count), q, qd, x_b, p_swing / v_swing / a_swing (NaN on a stance leg), swing_s,
contact, q_arc (the arc's own IK, the guard's), the torque split tau_g /
tau_hold / tau_pd / tau_ff, tau_req (before the gate), tau_cmd (sent),
tau_meas (driver iq), errors, temps, cap -- and the run's constants once:
ff_on, q_pose, rest, period, duty, ramp, swing_height, kp_swing, kd_swing,
kp_joint, kd_joint, m0, sched_on, sched_wn, sched_zeta, sched_lam_ref,
sched_lam_range, height, tau_cap, tau_slew, legs.  The column names
are `hw.swing_bench`'s where the two have the same thing.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/swing_ff.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402
from . import calibration as CAL     # noqa: E402
from . import can_link as CANL       # noqa: E402
from . import hardware_map as HM     # noqa: E402
from . import imu as IMU             # noqa: E402
from . import kinematics as HK       # noqa: E402
from . import realtime as RT         # noqa: E402
from . import safety as SAFE         # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import gait as GAIT    # noqa: E402
from .balance import sequence as SEQ  # noqa: E402
from .balance import state as BSTATE  # noqa: E402
from .balance import swing as SWING  # noqa: E402
from .balance import torque as TRQ   # noqa: E402
from .fold2_trot import HEIGHT as FOLD2_HEIGHT  # noqa: E402
from .gravity_bench import fold2_pose  # noqa: E402
from .stand import (GAP_ESTOP_S, QD_FILTER_HZ, RATE_HZ, STATUS_EVERY_SWEEPS,  # noqa: E402
                    STATUS_PERIOD_S, KeyPoller, _torque_lines)
from .swing_bench import POSE_S      # noqa: E402

__all__ = ["SwingFF", "SwingReport", "RunLog", "run", "main", "PERIOD_S", "DUTY",
           "CONTACT_RAMP", "SWING_HEIGHT", "TAU_CAP", "TAU_SLEW", "CLEAR_MM"]

#: The study's clock (doc/swing_height `hwsim_leg.Setup`; the gait study's
#: 2026-10-21 line, `hw.fold2_walk --duty 0.8 --contact-ramp 0.15`).
PERIOD_S = 0.80
DUTY = 0.80
CONTACT_RAMP = 0.15
SWING_HEIGHT = 0.020                 # m
#: The study's gate: the walk's cap and the flown slew.
TAU_CAP = SAFE.TAU_HARD_NM           # 9 N*m
TAU_SLEW = 300.0                     # N*m/s
#: mm above the resting site that counts as "the foot has left" (lag).
CLEAR_MM = 4.0

PHASES = ("limp", "pose", "hold", "swing")
PHASE_BLURB = {
    "limp": "NO TORQUE -- the legs hang.  ENTER ramps them to FOLD2's hold.",
    "pose": "POSITION mode, smoothstep to the hold.  Torque follows by itself.",
    "hold": "TORQUE -- every leg on the joint layer + gravity.  ENTER swings; X stops.",
    "swing": "the clock runs; ENTER latches a stop at the next four-foot window; "
             "X is an E-STOP.",
}


class SwingReport:
    """One swing, liftoff to the clock's touchdown: the study's metrics from the FK."""

    def __init__(self, leg: int, bout: int, ff: bool, t0: float, rest) -> None:
        self.leg, self.bout, self.ff, self.t0 = leg, int(bout), bool(ff), float(t0)
        self.rest = np.asarray(rest, dtype=float).copy()
        self.z0 = float("nan")            # height at liftoff (the hold's sag)
        self.apex = -np.inf
        self.s_apex = 0.0
        self.lag = float("nan")
        self.sq_err = 0.0
        self.x_dev = self.y_dev = 0.0
        self.ff_peak = self.pd_peak = self.clip_peak = self.tau_peak = 0.0
        self.h_td = self.vz_td = float("nan")
        self.last = None
        self.sweeps = 0

    def add(self, now, x_b, p_ref, s, ff, pd, clip, tau) -> None:
        d = np.asarray(x_b, dtype=float) - self.rest
        if self.sweeps == 0:
            self.z0 = d[2]
        if d[2] > self.apex:
            self.apex, self.s_apex = d[2], float(s)
        if np.isnan(self.lag) and d[2] > 1e-3 * CLEAR_MM:
            self.lag = now - self.t0
        self.sq_err += float(np.sum((np.asarray(p_ref, float) - x_b) ** 2))
        self.x_dev = max(self.x_dev, abs(d[0]))
        self.y_dev = max(self.y_dev, abs(d[1]))
        self.ff_peak = max(self.ff_peak, float(np.abs(ff).max()))
        self.pd_peak = max(self.pd_peak, float(np.abs(pd).max()))
        self.clip_peak = max(self.clip_peak, float(clip))
        self.tau_peak = max(self.tau_peak, float(np.abs(tau).max()))
        if self.last is not None and now > self.last[0]:
            self.vz_td = (d[2] - self.last[1]) / (now - self.last[0])
        self.last = (float(now), d[2])
        self.h_td = d[2]
        self.sweeps += 1

    @property
    def rms(self) -> float:
        return float(np.sqrt(self.sq_err / max(self.sweeps, 1)))

    def line(self, count: int) -> str:
        return ("swing %s #%d FF %-3s  apex %5.1f mm at s=%.2f  lag %3.0f ms  rms %4.1f mm  "
                "h_td %+5.1f mm  vz_td %+.2f m/s  x %.1f y %.1f mm  tau_ff %.1f  pd %.1f  "
                "clip %.1f  peak %.1f N*m  (%d sweeps, z0 %+.1f)"
                % (C.LEGS[self.leg], count, "on" if self.ff else "off", 1e3 * self.apex,
                   self.s_apex, 1e3 * self.lag, 1e3 * self.rms, 1e3 * self.h_td,
                   self.vz_td, 1e3 * self.x_dev, 1e3 * self.y_dev, self.ff_peak,
                   self.pd_peak, self.clip_peak, self.tau_peak, self.sweeps, 1e3 * self.z0))


class SwingFF:
    """The four phases and the torque they ask for.  No I/O in here."""

    def __init__(self, gate: SAFE.SafetyGate, gait: GAIT.TrotGait, *, legs,
                 swing_height: float = SWING_HEIGHT, swing_ff: bool = False,
                 kp_swing=None, kd_swing=None, kp_joint=None, kd_joint=None,
                 ff_inertia=None,
                 schedule: "SWING.GainSchedule | None" = None,
                 height: float = FOLD2_HEIGHT,
                 pose_s: float = POSE_S, swings: int = 0,
                 swing_stop_deg: float = 30.0) -> None:
        self.gate = gate
        self.gait = gait
        self.legs = np.zeros(C.N_LEGS, dtype=bool)
        self.legs[list(legs)] = True
        if not self.legs.any():
            raise ValueError("no leg to swing")
        self.swing_height = float(swing_height)
        self.swing_ff = bool(swing_ff)
        self.kp_swing = np.asarray(BCFG.KP_SWING_WALK if kp_swing is None else kp_swing,
                                   dtype=float)
        self.kd_swing = np.asarray(BCFG.KD_SWING_WALK if kd_swing is None else kd_swing,
                                   dtype=float)
        #: (3,) abd / pitch / knee -- the joint layer's, `hw.gravity_bench.KP`.
        self.kp_joint = np.asarray(
            [BCFG.KP_JOINT_HOLD_ABD, BCFG.KP_JOINT_HOLD, BCFG.KP_JOINT_HOLD]
            if kp_joint is None else kp_joint, dtype=float)
        self.kd_joint = np.asarray(
            [BCFG.KD_JOINT_HOLD_ABD, BCFG.KD_JOINT_HOLD, BCFG.KD_JOINT_HOLD]
            if kd_joint is None else kd_joint, dtype=float)
        self.m0 = (SWING.JOINT_INERTIA_M0 if ff_inertia is None
                   else np.asarray(ff_inertia, dtype=float))
        #: None: the fixed `kp_swing` / `kd_swing`; else Kp, Kd from Lambda(q) every sweep.
        self.schedule = schedule
        self.height = float(height)
        self.pose_s = float(pose_s)
        self.swings_wanted = int(swings)
        self.swing_stop_deg = float(swing_stop_deg)
        #: The hold: FOLD2's feet, the trunk `height` up -- what the ramp
        #: goes to, what the joint layer holds, where the arc leaves from.
        self.q_pose = fold2_pose(self.height)
        self.rest = np.array([HK.foot_position(i, self.q_pose[i]) for i in range(C.N_LEGS)])
        self.phase = "limp"
        self.t_phase = 0.0
        self.q_from = None
        self.max_dps = None
        self.trip = None
        self.notice = None
        self.bout = 0                           # ENTERs into swing
        self.stop_latched = False
        self.tau = np.zeros(C.N_JOINTS)         # requested, before the gate
        self.tau_g = np.zeros(C.N_JOINTS)
        self.tau_hold = np.zeros(C.N_JOINTS)
        self.tau_pd = np.zeros(C.N_JOINTS)
        self.tau_ff = np.zeros(C.N_JOINTS)
        self.p_swing = np.full((C.N_LEGS, 3), np.nan)
        self.v_swing = np.full((C.N_LEGS, 3), np.nan)
        self.a_swing = np.full((C.N_LEGS, 3), np.nan)
        self.q_arc = np.full((C.N_LEGS, 3), np.nan)
        #: the swing PD's gains and the foot's apparent mass this sweep (NaN off the swing)
        self.kp_eff = np.full((C.N_LEGS, 3), np.nan)
        self.kd_eff = np.full((C.N_LEGS, 3), np.nan)
        self.lam_eff = np.full((C.N_LEGS, 3), np.nan)
        self.swing_s = np.zeros(C.N_LEGS)
        self.contact = np.ones(C.N_LEGS, dtype=bool)
        self.reports: dict[int, SwingReport] = {}
        self.done: list[SwingReport] = []
        self.counts = np.zeros(C.N_LEGS, dtype=int)      # this bout, per leg

    # -- keys ---------------------------------------------------------------
    def enter(self, now: float, q) -> str:
        if self.phase == "limp":
            self.q_from = np.asarray(q, dtype=float).reshape(C.N_LEGS, 3).copy()
            self.max_dps = SEQ.ramp_motor_dps(self.q_from, self.q_pose, self.pose_s)
            self._go("pose", now)
            return ("POSE: position ramp to FOLD2's hold at %.0f mm over %.1f s; torque "
                    "takes over by itself when it arrives." % (1e3 * self.height, self.pose_s))
        if self.phase == "pose":
            return "ENTER ignored: the pose ramp is still running"
        if self.phase == "hold":
            self.gait.reset(now)
            self.stop_latched = False
            self.counts[:] = 0
            self.bout += 1
            self._go("swing", now)
            return ("SWING bout %d, FF %s: %s on %s; the first foot lifts in %.2f s.  ENTER "
                    "stops at a four-foot window, X is an E-STOP."
                    % (self.bout, "ON" if self.swing_ff else "OFF", self.gait,
                       "/".join(C.LEGS[i] for i in np.flatnonzero(self.legs)),
                       self.gait.entry_s))
        if self.stop_latched:
            return "ENTER ignored: stop already latched -- HOLD at the next four-foot window"
        self.stop_latched = True
        return "stop latched: HOLD at the next four-foot window"

    def _go(self, phase: str, now: float) -> None:
        self.phase = phase
        self.t_phase = float(now)

    def take_notice(self):
        notice, self.notice = self.notice, None
        return notice

    # -- the sweep ------------------------------------------------------------
    def update(self, now: float, body):
        """One sweep.  Returns ``(mode, values)``: keepalive / position / torque."""
        elapsed = now - self.t_phase
        if self.phase == "limp":
            return "keepalive", None
        if self.phase == "pose":
            s = GAIT.smoothstep(elapsed / self.pose_s)
            q_des = self.q_from + s * (self.q_pose - self.q_from)
            if elapsed >= self.pose_s + 0.5:
                self.gate.start(now, body.q)
                self._go("hold", now)
                off = np.degrees(np.abs(C.unflat(body.q) - self.q_pose)).max()
                self.notice = ("HOLD: torque mode, the joint layer on the IK pose (the ramp "
                               "arrived within %.2f deg).  The gate ramps the cap over %.1f s."
                               "  ENTER to swing (FF %s)."
                               % (off, self.gate.ramp_s, "ON" if self.swing_ff else "OFF"))
            return "position", C.flat(q_des)
        return "torque", self._torque(now, body)

    def _torque(self, now: float, body) -> np.ndarray:
        q4 = C.unflat(body.q)
        qd4 = C.unflat(body.qd)
        self.tau_g = C.flat(TRQ.all_leg_gravity_torque(body.q, body.R)).astype(float)
        tau = self.tau_g.copy()
        self.tau_hold[:] = 0.0
        self.tau_pd[:] = 0.0
        self.tau_ff[:] = 0.0
        self.p_swing[:] = np.nan
        self.v_swing[:] = np.nan
        self.a_swing[:] = np.nan
        self.q_arc[:] = np.nan
        self.kp_eff[:] = np.nan
        self.kd_eff[:] = np.nan
        self.lam_eff[:] = np.nan
        swinging = np.zeros(C.N_LEGS, dtype=bool)
        if self.phase == "swing":
            clock = self.gait.sample(now)
            swinging = ~clock.contact & self.legs
            self.swing_s = np.where(self.legs, clock.swing_s, 0.0)
            # FOUR DOWN, not four at full weight: the arc being finished is
            # what matters in the air (`hw.swing_bench` says why).
            if self.stop_latched and clock.contact.all():
                self._go("hold", now)
                self.notice = ("HOLD: bout %d stopped at a four-foot window.  ENTER swings "
                               "again (FF %s); X ends the run." % (
                                   self.bout, "ON" if self.swing_ff else "OFF"))
                swinging[:] = False
                self.swing_s[:] = 0.0
        else:
            self.swing_s[:] = 0.0
        self.contact = ~swinging
        for i in range(C.N_LEGS):
            sl = slice(3 * i, 3 * i + 3)
            if not swinging[i]:
                # THE JOINT LAYER, as the law's: Kp (q_hold - q) - Kd qd.
                hold = self.kp_joint * (self.q_pose[i] - q4[i]) - self.kd_joint * qd4[i]
                tau[sl] += hold
                self.tau_hold[sl] = hold
                continue
            s = float(self.swing_s[i])
            p, v, a = SWING.swing_reference_pva(self.rest[i], s, self.gait.swing_duration,
                                                height=self.swing_height)
            if self.schedule is None:
                kp, kd = self.kp_swing, self.kd_swing
            else:
                kp, kd, self.lam_eff[i] = self.schedule.gains(body.jac[i])
            self.kp_eff[i], self.kd_eff[i] = kp, kd
            pd = SWING.swing_torque(body, i, p, v, kp=kp, kd=kd)
            tau[sl] += pd
            self.tau_pd[sl] = pd
            if self.swing_ff:
                ff = SWING.swing_feedforward(i, q4[i], v, a, jac=body.jac[i],
                                             inertia=self.m0)
                tau[sl] += ff
                self.tau_ff[sl] = ff
            self.p_swing[i], self.v_swing[i], self.a_swing[i] = p, v, a
            # THE ARC'S OWN IK, for the guard and the log only; no torque reads it.
            qj, _ = SWING.joint_swing_reference(i, self.rest[i], s, self.gait.swing_duration,
                                                q4[i], height=self.swing_height)
            self.q_arc[i] = qj
            dev = np.degrees(np.abs(q4[i] - qj))
            if self.swing_stop_deg > 0.0 and dev.max() > self.swing_stop_deg:
                self.trip = ("swing %s: %s is %.0f deg off the arc's IK (limit %.0f) -- "
                             "the leg has left the arc"
                             % (C.LEGS[i], ("abd", "pitch", "knee")[int(dev.argmax())],
                                dev.max(), self.swing_stop_deg))
        self.tau = tau
        return tau

    # -- the per-swing read-out ------------------------------------------------
    def observe(self, now: float, body, tau_sent) -> list[str]:
        """After the gate: open, feed and close the swing reports.  Lines to print."""
        lines = []
        clip = np.abs(self.tau - tau_sent)
        for i in range(C.N_LEGS):
            sl = slice(3 * i, 3 * i + 3)
            up = self.phase == "swing" and self.swing_s[i] > 0.0
            rep = self.reports.get(i)
            if up:
                if rep is None:
                    rep = self.reports[i] = SwingReport(i, self.bout, self.swing_ff, now,
                                                        self.rest[i])
                rep.add(now, body.x_b[i], self.p_swing[i], self.swing_s[i], self.tau_ff[sl],
                        self.tau_pd[sl], clip[sl].max(), tau_sent[sl])
            elif rep is not None:
                self.counts[i] += 1
                self.done.append(rep)
                lines.append(rep.line(int(self.counts[i])))
                del self.reports[i]
                if (self.swings_wanted and not self.stop_latched
                        and self.counts[self.legs].min() >= self.swings_wanted):
                    self.stop_latched = True
                    lines.append("%d swings landed on every leg -- HOLD at the next "
                                 "four-foot window" % self.swings_wanted)
        return lines

    def status(self, now: float, body) -> str:
        dz = 1e3 * (body.x_b[:, 2] - self.rest[:, 2])
        head = "   %-6s t=%5.1f  FF %-3s  feet dz %s mm" % (
            self.phase, now - self.t_phase, "on" if self.swing_ff else "off",
            np.array2string(dz, precision=1, floatmode="fixed",
                            formatter={"float_kind": lambda v: "%+5.1f" % v}))
        if self.phase == "swing":
            head += "  bout %d cycle %d  swing_s %s%s" % (
                self.bout, self.gait.cycles(now), np.array2string(self.swing_s, precision=2),
                "  stop latched" if self.stop_latched else "")
        return head

    def report(self) -> str:
        """One row per leg: the study's medians, this run's FF state."""
        if not self.done:
            return "  no swing landed"
        out = ["  %-4s %-3s %3s  %11s %6s  %7s  %7s  %8s  %8s  %6s  %6s  %6s  %6s"
               % ("leg", "FF", "n", "apex mm", "s", "lag ms", "rms mm", "h_td mm",
                  "vz_td", "x mm", "tau_ff", "pd", "clip")]
        for leg in range(C.N_LEGS):
            for ff in (False, True):
                mine = [r for r in self.done if r.leg == leg and r.ff == ff]
                if not mine:
                    continue
                med = lambda key: float(np.nanmedian([getattr(r, key) for r in mine]))  # noqa: E731
                out.append("  %-4s %-3s %3d  %5.1f [%4.1f,%4.1f] %4.2f  %5.0f    %5.1f    %+6.1f    "
                           "%+6.2f  %5.1f  %6.2f  %6.2f  %6.2f"
                           % (C.LEGS[leg], "on" if ff else "off", len(mine),
                              1e3 * med("apex"),
                              1e3 * np.nanpercentile([r.apex for r in mine], 10),
                              1e3 * np.nanpercentile([r.apex for r in mine], 90),
                              med("s_apex"), 1e3 * med("lag"), 1e3 * med("rms"),
                              1e3 * med("h_td"), med("vz_td"), 1e3 * med("x_dev"),
                              max(r.ff_peak for r in mine), max(r.pd_peak for r in mine),
                              max(r.clip_peak for r in mine)))
        out.append("  the study (doc/swing_height, RR, MuJoCo through the chain): "
                   "FF off apex 15.0 at s=0.83, lag 55, rms 9.9, h_td +13.0;  "
                   "FF on apex 23.4 at s=0.52, lag 33, rms 3.4, h_td -1.9")
        return "\n".join(out)


class RunLog:
    """Every torque-mode sweep, plus the run's constants, to one npz."""

    def __init__(self, **const) -> None:
        self.rows: list[dict] = []
        self.const = const

    def add(self, **row) -> None:
        self.rows.append(row)

    def save(self, path: str) -> str:
        if not self.rows:
            return "nothing to save"
        cols = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}
        np.savez_compressed(path, **cols, **{k: np.asarray(v) for k, v in self.const.items()})
        return "%d sweeps -> %s" % (len(self.rows), path)


def _print_phase(bench: SwingFF) -> None:
    print("\n>> %-6s : %s" % (bench.phase, PHASE_BLURB[bench.phase]), flush=True)


def run(mb, bench: SwingFF, *, rate_hz: float = RATE_HZ, key=None,
        clock=time.perf_counter, log: "RunLog | None" = None,
        gcq: "RT.QuietGC | None" = None) -> str | None:
    """Drive `bench` on an ARMED `mb` until X or a trip.  Returns the reason.  `hw.swing_bench.run`'s loop: the same
    round robin, latched sweep, gate, monitors and stop lines, with the IMU
    replaced by `TrunkOrientation.level()`."""
    ids = HM.motor_ids()
    n = len(ids)
    unwrappers = CAL.new_unwrappers()
    misses = SAFE.CanMissMonitor(mb)
    readback = SAFE.TorqueReadback(n)
    slot = mb.slot(rate_hz)
    alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * n * slot)
    level = IMU.TrunkOrientation.level()
    guard = RT.SendGuard(mb, ids, GAP_ESTOP_S, SAFE.INPUT_LOST_S, gcq)
    gc_slot = RT.GC_SLOT % n
    link_lost = CANL.bus_errors()

    sweep = 0
    mode, values = "keepalive", None
    q_prev = t_prev = None
    qd_ctrl = np.zeros(n)
    tau_sent = np.zeros(n)
    overruns = 0
    t0 = last_status = bench.t_phase = clock()
    last_phase = bench.phase
    _print_phase(bench)

    deadline = clock() + slot
    k = 0
    while True:
        try:
            mb.poll()
        except link_lost as failure:
            return ("CAN link lost: %s -- the adapter left USB or can0 went down under "
                    "the run" % failure)
        if k == 0:
            now = clock()
            try:
                q, qd_driver = CAL.joint_state(mb, unwrappers)
            except RuntimeError as missing:
                if now - t0 > 1.0:
                    return str(missing)
                k = (k + 1) % n
                guard.sent(0, mb.keepalive(ids[0]), clock())
                deadline = clock() + slot
                continue
            if q_prev is not None and now > t_prev:
                qd_ctrl += alpha_qd * ((q - q_prev) / (now - t_prev) - qd_ctrl)
            q_prev, t_prev = q, now
            body = BSTATE.read(q, qd_ctrl, level)

            pressed = key.get() if key is not None else None
            if pressed in ("x", "X"):
                return "operator X"
            if pressed in ("\r", "\n"):
                print("\n   " + bench.enter(now, q), flush=True)

            mode, values = bench.update(now, body)
            if bench.trip:
                return bench.trip
            notice = bench.take_notice()
            if notice:
                print("\n   " + notice, flush=True)
            if bench.phase != last_phase:
                last_phase = bench.phase
                _print_phase(bench)

            # -- the trips ----------------------------------------------------
            errors = mb.errors()
            latched = [mid for mid, err in errors.items() if err & 0x80]
            if latched:
                return ("input-lost latch (0x80) on CAN %s: a driver heard nothing for "
                        "%.0f ms and went limp" % (latched, 1e3 * SAFE.INPUT_LOST_S))
            temps = np.asarray([mb.rec(mid).temp or 0 for mid in ids])
            reason = bench.gate.estop_reason(q, qd_driver, now, temps=temps,
                                             miss_streaks=misses.update(mb), errors=errors)
            if reason:
                return reason
            tau_meas = np.asarray([mb.torques_nm()[mid] for mid in ids])
            if mode == "torque":
                tau_sent = bench.gate.apply(values, q, now)
                values = tau_sent
                for line in bench.observe(now, body, tau_sent):
                    print("   " + line, flush=True)
                reason = readback.reason(tau_sent, tau_meas, live=True)
                if reason:
                    return reason
                if log is not None:
                    log.add(t=now - t0, phase=bench.phase, bout=bench.bout,
                            q=q, qd=qd_ctrl, x_b=body.x_b,
                            p_swing=bench.p_swing.copy(), v_swing=bench.v_swing.copy(),
                            a_swing=bench.a_swing.copy(), swing_s=bench.swing_s.copy(),
                            contact=bench.contact.copy(), q_arc=bench.q_arc.copy(),
                            kp_eff=bench.kp_eff.copy(), kd_eff=bench.kd_eff.copy(),
                            lam=bench.lam_eff.copy(),
                            tau_req=bench.tau.copy(), tau_cmd=tau_sent.copy(),
                            tau_g=bench.tau_g.copy(), tau_hold=bench.tau_hold.copy(),
                            tau_pd=bench.tau_pd.copy(), tau_ff=bench.tau_ff.copy(),
                            tau_meas=tau_meas,
                            errors=np.asarray([errors.get(mid, 0) or 0 for mid in ids]),
                            temps=temps, cap=bench.gate.cap_now(now))
            else:
                tau_sent = np.zeros(n)

            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                print(bench.status(now, body)
                      + "  |tau|=%.2f  gap=%.1f ms  overrun=%d"
                      % (float(np.abs(tau_sent).max()), guard.worst_ms(), overruns),
                      flush=True)
                if mode == "torque":
                    print(_torque_lines(tau_sent, tau_meas), flush=True)
            sweep += 1

        # -- one frame, to one motor ------------------------------------------
        mid = ids[k]
        sent_at = clock()
        reason = guard.check(k, sent_at)
        if reason:
            return reason
        if (sweep % STATUS_EVERY_SWEEPS == 0
                and k == (sweep // STATUS_EVERY_SWEEPS) % n):
            taken = mb.status1_req(mid)
        elif mode == "position":
            taken = mb.position(mid, float(np.rad2deg(values[k])),
                                max_dps=float(bench.max_dps[k]))
        elif mode == "torque":
            taken = mb.torque(mid, float(values[k]))
        else:
            taken = mb.keepalive(mid)
        guard.sent(k, taken, sent_at)
        if k == gc_slot and gcq is not None:
            gcq.step(sweep)

        k = (k + 1) % n
        overrun = mb.pace(deadline)
        deadline += slot
        if overrun > 3 * slot:
            overruns += 1
            deadline = clock() + slot
    # unreachable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m hw.swing_ff",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--legs", nargs="+", default=list(C.LEGS), choices=list(C.LEGS),
                    help="which legs follow the arc when the clock lifts them")
    ap.add_argument("--swing-ff", action="store_true",
                    help="the inertial feedforward ON for the whole run (off without)")
    ap.add_argument("--swings", type=int, default=0, metavar="N",
                    help="stop a bout once every swinging leg has landed N times; "
                         "0 waits for ENTER")
    ap.add_argument("--height", type=float, default=1e3 * FOLD2_HEIGHT, metavar="MM",
                    help="floor to trunk bottom at FOLD2's hold")
    ap.add_argument("--pose-s", type=float, default=POSE_S,
                    help="the position ramp from the hang to the hold, s")
    ap.add_argument("--log", default=None, metavar="FILE.npz",
                    help="where the run goes; default swing_ff_<on|off>_<time>.npz")

    g = ap.add_argument_group("the clock", "the study's: time alone, no touchdown detection")
    g.add_argument("--period", type=float, default=PERIOD_S, metavar="SECONDS")
    g.add_argument("--duty", type=float, default=DUTY)
    g.add_argument("--contact-ramp", type=float, default=CONTACT_RAMP)

    sw = ap.add_argument_group("the swing", "the walk's law, the study's numbers")
    sw.add_argument("--swing-height", type=float, default=1e3 * SWING_HEIGHT, metavar="MM")
    sw.add_argument("--kp-swing", type=float, nargs=3, default=list(BCFG.KP_SWING_WALK),
                    metavar="N_PER_M")
    sw.add_argument("--kd-swing", type=float, nargs=3, default=list(BCFG.KD_SWING_WALK),
                    metavar="NS_PER_M")
    sw.add_argument("--ff-armature", type=float, default=None, metavar="KG_M2",
                    help="the reflected rotor inertia M0 is built on; default "
                         "params.ARMATURE %.4f, the datasheet's" % P.ARMATURE)
    sw.add_argument("--sched", action="store_true",
                    help="schedule the swing PD on the foot's apparent mass Lambda(q): "
                         "Kp = wn^2 Lambda, Kd = 2 zeta wn Lambda, Lambda from the datasheet "
                         "M0 (not --ff-armature) at every sweep's Jacobian; --kd-swing unused")
    sw.add_argument("--sched-zeta", type=float, nargs="+", default=[0.7], metavar="ZETA",
                    help="damping ratio, one value or one per axis (x y z); 0.7-1")
    sw.add_argument("--sched-wn", type=float, nargs=3, default=None, metavar="RAD_PER_S",
                    help="natural frequency per axis; default is the wn that makes Kp at "
                         "FOLD2's hold equal --kp-swing")
    sw.add_argument("--sched-lam-range", type=float, nargs=2, default=[0.5, 3.0],
                    metavar=("LO", "HI"),
                    help="Lambda is clamped to LO..HI x its value at the hold: Kp's floor "
                         "and the singular pose's cap")
    sw.add_argument("--swing-stop", type=float, default=30.0, metavar="DEG",
                    help="TRIP: a swinging joint this far from the arc's own IK stops "
                         "the run; 0 turns it off")

    j = ap.add_argument_group("the stance legs", "the joint layer, hw.trot's flags")
    j.add_argument("--kp-joint", type=float, default=BCFG.KP_JOINT_HOLD, metavar="NM_PER_RAD")
    j.add_argument("--kd-joint", type=float, default=BCFG.KD_JOINT_HOLD, metavar="NMS_PER_RAD")
    j.add_argument("--kp-joint-abd", type=float, default=BCFG.KP_JOINT_HOLD_ABD,
                   metavar="NM_PER_RAD")
    j.add_argument("--kd-joint-abd", type=float, default=BCFG.KD_JOINT_HOLD_ABD,
                   metavar="NMS_PER_RAD")

    s = ap.add_argument_group("the gate", "the study's")
    s.add_argument("--tau-cap", type=float, default=TAU_CAP,
                   help="torque cap, N*m; the ceiling is the motors' %.0f" % SAFE.TAU_HARD_NM)
    s.add_argument("--tau-slew", type=float, default=TAU_SLEW, metavar="NM_PER_S",
                   help="the gate's |dtau/dt| limit")
    s.add_argument("--overspeed-trip", action="store_true",
                   help="stop on the 7 rad/s joint-speed trip (off, as in the trot)")
    s.add_argument("--rate", type=float, default=RATE_HZ)
    s.add_argument("--bitrate", type=int, default=1_000_000)
    s.add_argument("--arm-timeout", type=float, default=30.0)
    s.add_argument("--unconfirmed", default=None, metavar="WHY")
    args = ap.parse_args(argv)

    log_path = args.log or time.strftime("swing_ff_%s_%%Y%%m%%d-%%H%%M%%S.npz"
                                         % ("on" if args.swing_ff else "off"))
    try:
        gait = GAIT.TrotGait(period=args.period, duty=args.duty, ramp=args.contact_ramp,
                             settle_s=0.0)
    except ValueError as refusal:
        ap.error("the clock: %s" % refusal)
    ids = HM.motor_ids()
    try:
        gate = SAFE.SafetyGate(args.tau_cap, unconfirmed_reason=args.unconfirmed,
                               ceiling=SAFE.TAU_HARD_NM, tau_slew=args.tau_slew,
                               overspeed_trip=args.overspeed_trip)
    except (RuntimeError, ValueError) as refusal:
        print("[swing_ff] REFUSED before opening the bus:\n  %s" % refusal, file=sys.stderr)
        return 2
    m0 = (None if args.ff_armature is None
          else SWING.feedforward_inertia(args.ff_armature))
    schedule = None
    if args.sched:
        try:
            if len(args.sched_zeta) not in (1, 3):
                raise ValueError("--sched-zeta takes 1 value or 3 (x y z)")
            # Reference: FL at FOLD2's hold (the four legs' Lambda diagonals agree);
            # Lambda from the datasheet M0, whatever `--ff-armature` does to the feedforward.
            schedule = SWING.GainSchedule.from_reference(
                HK.foot_jacobian(0, fold2_pose(1e-3 * args.height)[0]),
                zeta=args.sched_zeta[0] if len(args.sched_zeta) == 1 else args.sched_zeta,
                kp=args.kp_swing if args.sched_wn is None else None, wn=args.sched_wn,
                lam_range=tuple(args.sched_lam_range))
        except ValueError as refusal:
            print("[swing_ff] REFUSED before opening the bus:\n  %s" % refusal, file=sys.stderr)
            return 2
    try:
        bench = SwingFF(gate, gait, legs=[C.LEGS.index(name) for name in args.legs],
                        swing_height=1e-3 * args.swing_height, swing_ff=args.swing_ff,
                        kp_swing=args.kp_swing, kd_swing=args.kd_swing,
                        kp_joint=[args.kp_joint_abd, args.kp_joint, args.kp_joint],
                        kd_joint=[args.kd_joint_abd, args.kd_joint, args.kd_joint],
                        ff_inertia=m0, schedule=schedule,
                        height=1e-3 * args.height, pose_s=args.pose_s,
                        swings=args.swings, swing_stop_deg=args.swing_stop)
    except ValueError as refusal:
        print("[swing_ff] REFUSED before opening the bus:\n  %s" % refusal, file=sys.stderr)
        return 2

    # -- the banner ---------------------------------------------------------------
    print("DOG6 swing feedforward A/B on can0 -- HANG THE ROBOT UP FIRST: nothing here holds it")
    print("  CAN ids %s  directions %s  map confirmed: %s" % (ids, HM.directions(),
                                                             CONFIRMED_ON_DOG6))
    print("  no IMU: the trunk is taken as level.  cap %.1f N*m, slew %.0f N*m/s, overspeed "
          "trip %s, swing-stop %s"
          % (args.tau_cap, args.tau_slew, "ON" if args.overspeed_trip else "OFF",
             "OFF" if args.swing_stop <= 0 else "%.0f deg off the arc's IK" % args.swing_stop))
    print("  hold: FOLD2's feet, trunk %.0f mm -- joints abd/pitch/knee %s deg; feet %s mm, "
          "trunk frame"
          % (args.height,
             "  ".join("%s %.0f/%.0f/%.0f" % (C.LEGS[i], *np.degrees(bench.q_pose[i]))
                       for i in range(C.N_LEGS)),
             "  ".join("%s %+.0f/%+.0f/%+.0f" % (C.LEGS[i], *(1e3 * bench.rest[i]))
                       for i in range(C.N_LEGS))))
    print("  stance legs: the joint layer Kp %s Kd %s (abd/pitch/knee) + gravity; a swinging "
          "leg gets none of it" % (bench.kp_joint, bench.kd_joint))
    print("  legs %s   %s" % (" ".join(args.legs), gait))
    demand = SWING.swing_demand(0, bench.rest[0], gait.swing_duration,
                                1e-3 * args.swing_height, bench.q_pose[0])
    print("  swing apex %.0f mm over %.0f ms, Cartesian PD Kp %s Kd %s; knee ~%.1f rad/s, "
          "~%.1f N*m inertial; slew ~%.0f to rise against %.0f"
          % (args.swing_height, 1e3 * gait.swing_duration, bench.kp_swing, bench.kd_swing,
             demand["qd"][2], demand["tau"].max(), demand["slew"], args.tau_slew))
    if schedule is None:
        print("  swing PD: FIXED gains (--sched schedules them on Lambda(q))")
    else:
        # Kp, Kd where the arc takes the foot: the hold and the apex, FL.
        apex_q, _ = SWING.joint_swing_reference(0, bench.rest[0], 0.5, gait.swing_duration,
                                                bench.q_pose[0], height=bench.swing_height)
        kp_h, kd_h, lam_h = schedule.gains(HK.foot_jacobian(0, bench.q_pose[0]))
        kp_a, kd_a, lam_a = schedule.gains(HK.foot_jacobian(0, apex_q))
        print("  swing PD SCHEDULED on Lambda(q): wn %s rad/s zeta %s, Lambda clamped to %s x hold"
              % (np.array2string(schedule.wn, precision=1), np.array2string(schedule.zeta, precision=2),
                 schedule.lam_range))
        print("    hold: Lambda %s kg  Kp %s  Kd %s" % (np.array2string(lam_h, precision=2),
              np.array2string(kp_h, precision=0), np.array2string(kd_h, precision=1)))
        print("    apex: Lambda %s kg  Kp %s  Kd %s" % (np.array2string(lam_a, precision=2),
              np.array2string(kp_a, precision=0), np.array2string(kd_a, precision=1)))
        if list(args.kd_swing) != list(BCFG.KD_SWING_WALK):
            print("    (--kd-swing %s is not used)" % (args.kd_swing,))
    print("  feedforward M0 J^+ (a_ref - Jdot qd_ref): M0 diag %s kg m^2 (armature %s); "
          "%s FOR THIS RUN (--swing-ff); logged as tau_ff"
          % (np.array2string(np.diag(bench.m0), precision=4),
             "%.5f" % (P.ARMATURE if args.ff_armature is None else args.ff_armature),
             "ON" if args.swing_ff else "OFF"))
    print("  log: %s" % log_path)
    print("  the study's prediction (doc/swing_height, RR): FF off apex ~15 mm / h_td ~+13; "
          "FF on apex ~23 / h_td ~-2")
    print("  ENTER: limp -> pose; hold -> swing; swing -> stop at four feet.  X: E-STOP.")

    from .motor import motorbus  # noqa: F401  (the bus module the link opens)
    print("  " + RT.boost())
    try:
        mb = CANL.open_motor_bus(ids, bitrate=args.bitrate, dirs=HM.motor_directions())
    except CANL.LinkError as refusal:
        print("[swing_ff] REFUSED before arming -- CAN: %s" % refusal, file=sys.stderr)
        return 2
    since = CANL.uptime()
    key = KeyPoller()
    if not key.ok:
        print("[swing_ff] stdin is not a terminal: ENTER and X will not work",
              file=sys.stderr)
    log = RunLog(ff_on=bench.swing_ff, q_pose=bench.q_pose, rest=bench.rest,
                 period=gait.period, duty=gait.duty, ramp=gait.ramp,
                 swing_height=bench.swing_height, kp_swing=bench.kp_swing,
                 kd_swing=bench.kd_swing, kp_joint=bench.kp_joint, kd_joint=bench.kd_joint,
                 m0=bench.m0, height=bench.height,
                 sched_on=schedule is not None,
                 sched_wn=np.full(3, np.nan) if schedule is None else schedule.wn,
                 sched_zeta=np.full(3, np.nan) if schedule is None else schedule.zeta,
                 sched_lam_ref=np.full(3, np.nan) if schedule is None else schedule.lam_ref,
                 sched_lam_range=(np.full(2, np.nan) if schedule is None
                                  else np.asarray(schedule.lam_range)),
                 tau_cap=args.tau_cap,
                 tau_slew=args.tau_slew, legs=bench.legs)
    stop = None
    gcq = RT.QuietGC()
    try:
        with mb:
            print("  " + gcq.start())
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[swing_ff] not every motor armed", file=sys.stderr)
                print("[can] %s" % CANL.after_failed_arm(mb), file=sys.stderr)
                return 1
            stop = run(mb, bench, rate_hz=args.rate, key=key, log=log, gcq=gcq)
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        key.restore()
        gc_said = gcq.stop()

    print()
    if gc_said:
        print("[swing_ff] " + gc_said)
    if stop not in (None, "operator X", "Ctrl-C"):
        print("[can] at the stop: %s" % CANL.describe(getattr(mb, "at_stop", None)))
        for line in CANL.dmesg_since(since):
            print("[can] dmesg: " + line)
    print("[swing_ff] swings, FF %s, median [p10, p90] apex:" % ("ON" if bench.swing_ff else "OFF"))
    print(bench.report())
    if gate.started_at is not None:
        print("[swing_ff] peak joint speed, driver / encoder (rad/s; trip %s at %.1f):"
              % ("ON" if gate.overspeed_trip else "OFF", gate.qd_estop))
        for leg in range(C.N_LEGS):
            row = slice(3 * leg, 3 * leg + 3)
            print("  %s  %s" % (C.LEGS[leg], "   ".join(
                "%4.1f / %4.1f" % pair for pair in zip(
                    gate.qd_peak[row], gate.encoder_qd_peak[row]))))
    print("[swing_ff] log: %s" % log.save(log_path))
    if stop:
        print("[swing_ff] stopped: %s.  Motors stopped; the legs hang." % stop)
        return 0 if stop in ("operator X", "Ctrl-C") else 3
    print("[swing_ff] done; motors stopped, the legs hang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
