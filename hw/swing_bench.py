"""The swing leg on its own: the robot HUNG UP, the legs held, ENTER, swing.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.swing_bench --fake --auto 1 --swings 4         the whole path, no robot
    $V -m hw.swing_bench                                    on the robot, in the air
    $V -m hw.swing_bench --swing-ff --swing-height 20 --tau-slew 150 --log bench.npz
    $V -m hw.swing_bench --legs FL --swings 3 --swing-ff    one leg, three swings, stop

HANG THE ROBOT BEFORE YOU START.  Nothing in this file holds it up: there
is no balance law, no height loop, no allocator and no IMU.  Each leg gets
its own Cartesian hold plus its own gravity, and a swinging leg gets
`hw.trot`'s swing law -- `balance.swing`, the same arc, PD, feedforward and
gait clock, through the same `safety.SafetyGate` -- with nothing under the
foot to hide what it does.  That is the experiment: on the floor a swing is
read through the trunk's motion and the other three legs; here the FK of
the swing leg IS the measurement.

    limp    zero torque; the legs hang.  Read q.               ENTER ->
    pose    POSITION mode, a smoothstep to the lift pose over `--pose-s`.
            The legs move to where a standing robot would have them.
    hold    TORQUE mode, entered by itself when the ramp arrives: every
            foot held at its resting site (the FK of the pose it is already
            in, so the handover moves nothing) by the swing PD at zero
            velocity, plus leg gravity.  The gate's 1 s ramp brings the
            cap up from zero, so the legs sag for half a second and come
            back.  In the air that is all it is.                 ENTER ->
    swing   the gait clock runs.  A leg in `--legs` follows the arc while
            the clock says it is off the floor; every other leg keeps the
            hold.  ENTER latches a stop at the next four-foot window and the
            robot is back in hold; `--swings N` does the same after N swings
            have landed.  ENTER in hold swings again.
    X       e-stop, any time.  Motors stop; the legs drop to their hang.

WHAT IT PRINTS, AND WHY IT EXISTS
    One line per swing, the moment the clock says the foot is down:

        swing FL #3  apex 20 -> 15.2 mm  x +1.4/-0.8 (td -0.8)  y 0.3 mm
              vz_td -0.18 m/s  tau_ff 3.4  pd 0.9  clip 2.1 N*m  peak 4.1

    apex     what the arc asked, and what the foot's FK reached above where
             it left.  The whole question of 2026-09-25 -- does the foot
             follow the arc -- is this pair.
    x, y     the largest excursions from the liftoff point, and x at
             touchdown.  The arc has no x or y in it: any x here is the
             swing law's own doing (the Jdot qd term that was missing on
             2026-09-25 put 4 mm there) or the leg's coupling.
    vz_td    the foot's vertical speed on the last swing sweep.  The arc
             ends at zero; what is left is what the slew shaved off the
             braking half, and it is what arrives on the floor.
    tau_ff   the feedforward's peak; pd the PD's peak; clip the most the
             slew was holding back at any sweep; peak the largest torque
             the gate actually sent.

    A swing on the floor that lands hard, drifts in x or does not reach its
    apex has three suspects -- the swing law, the slew, the trunk moving
    under the leg.  Hung up, the third is gone and the other two are on one
    line each.

WHAT IT SHARES WITH THE TROT, EXACTLY
    `balance.swing.swing_reference_pva` (the arc), `swing_torque` (the PD),
    `swing_feedforward`, `joint_swing_reference` / `joint_swing_torque`
    (`--swing joint`), `balance.gait.TrotGait` (the clock, with the same
    period, duty, contact ramp and settle flags), `torque.leg_gravity_torque`,
    and `safety.SafetyGate` with the trot's slew.  The flags are the trot's
    flags with the trot's defaults, so a setting proven here is the setting
    to type into `hw.trot_esti`.  The overspeed TRIP is off as it is in the
    trot (the 2026-09-17 decision -- a 20 mm arc asks the knee for 8 rad/s
    against a 7 rad/s trip written for a stand); the peaks print at exit.

WHAT IT IS NOT
    Not a stand: the "pose" is a position ramp, not the lift, and the hold
    carries no weight because there is none to carry.  Not a trot: no stance
    law, no weight handover, no trunk.  Not a foot-placement test: the arc
    lands where it left.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/swing_bench.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402
from sim import leg_dynamics as LD   # noqa: E402
from sim import params as P          # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402
from . import calibration as CAL     # noqa: E402
from . import hardware_map as HM     # noqa: E402
from . import imu as IMU             # noqa: E402
from . import kinematics as HK       # noqa: E402
from . import safety as SAFE         # noqa: E402
from . import trot as TROT           # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import gait as GAIT    # noqa: E402
from .balance import sequence as SEQ  # noqa: E402
from .balance import state as BSTATE  # noqa: E402
from .balance import swing as SWING  # noqa: E402
from .balance import torque as TRQ   # noqa: E402
from .stand import (GAP_ESTOP_S, QD_FILTER_HZ, RATE_HZ, STATUS_EVERY_SWEEPS,  # noqa: E402
                    STATUS_PERIOD_S, KeyPoller, _torque_lines)

__all__ = ["SwingBench", "SwingReport", "run", "main", "POSE_S", "TAU_CAP"]

#: Seconds for the position-mode smoothstep from the hang to the lift pose.
POSE_S = 2.0

#: The bench's default cap.  A 20 mm / 140 ms arc asks ~3.5 N*m and the legs
#: carry nothing, so 4 is room to swing and not room to do damage.  The
#: ceiling is the motors' own, as the trot's is; `--tau-cap` raises it.
TAU_CAP = 4.0

PHASES = ("limp", "pose", "hold", "swing")
PHASE_BLURB = {
    "limp": "NO TORQUE -- the legs hang.  ENTER ramps them to the lift pose.",
    "pose": "POSITION mode, smoothstep to the lift pose.  Hold follows by itself.",
    "hold": "TORQUE -- each foot held at its site by the swing PD + gravity.  "
            "ENTER swings; X stops.",
    "swing": "the gait clock runs; ENTER latches a stop at the next four-foot "
             "window; X is an E-STOP.",
}


class SwingReport:
    """One leg's swing, from liftoff to the clock's touchdown, measured by FK."""

    def __init__(self, leg: int, t0: float, x0, apex_cmd: float) -> None:
        self.leg = leg
        self.t0 = float(t0)
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.apex_cmd = float(apex_cmd)
        self.z_max = 0.0
        self.x_min = self.x_max = 0.0
        self.y_max = 0.0
        self.ff_peak = self.pd_peak = self.clip_peak = self.tau_peak = 0.0
        self.q0 = None            # joints at liftoff
        self.q_dev = np.zeros(3)  # worst |q - arc IK|, rad, per joint
        self.abd_travel = 0.0     # worst |abd - abd at liftoff|, rad
        self.last = None          # (t, x_b) of the previous sweep
        self.vz = float("nan")
        self.x_td = float("nan")
        self.sweeps = 0

    def add(self, now: float, x_b, ff, pd, clip, tau, q=None, q_arc=None) -> None:
        d = np.asarray(x_b, dtype=float) - self.x0
        if q is not None:
            q = np.asarray(q, dtype=float)
            if self.q0 is None:
                self.q0 = q.copy()
            self.abd_travel = max(self.abd_travel, abs(q[0] - self.q0[0]))
            if q_arc is not None:
                self.q_dev = np.maximum(self.q_dev, np.abs(q - np.asarray(q_arc, float)))
        self.z_max = max(self.z_max, d[2])
        self.x_min = min(self.x_min, d[0])
        self.x_max = max(self.x_max, d[0])
        self.y_max = max(self.y_max, abs(d[1]))
        self.ff_peak = max(self.ff_peak, float(np.abs(ff).max()))
        self.pd_peak = max(self.pd_peak, float(np.abs(pd).max()))
        self.clip_peak = max(self.clip_peak, float(clip))
        self.tau_peak = max(self.tau_peak, float(np.abs(tau).max()))
        if self.last is not None and now > self.last[0]:
            self.vz = (d[2] - (self.last[1][2] - self.x0[2])) / (now - self.last[0])
        self.last = (float(now), np.asarray(x_b, dtype=float).copy())
        self.x_td = d[0]
        self.sweeps += 1

    def line(self, count: int) -> str:
        return ("swing %s #%d  apex %.0f -> %5.1f mm  x %+.1f/%+.1f (td %+.1f)  "
                "y %.1f mm  vz_td %+.2f m/s  tau_ff %.1f  pd %.1f  clip %.1f "
                "N*m  peak %.1f  off the arc abd/pitch/knee %.0f/%.0f/%.0f deg, "
                "abd travel %.0f deg  (%d sweeps)"
                % (C.LEGS[self.leg], count, 1e3 * self.apex_cmd, 1e3 * self.z_max,
                   1e3 * self.x_max, 1e3 * self.x_min, 1e3 * self.x_td,
                   1e3 * self.y_max, self.vz, self.ff_peak, self.pd_peak,
                   self.clip_peak, self.tau_peak, *np.degrees(self.q_dev),
                   np.degrees(self.abd_travel), self.sweeps))


class SwingBench:
    """The four phases and the torque they ask for.  No I/O in here."""

    def __init__(self, gate: SAFE.SafetyGate, gait: GAIT.TrotGait, *,
                 legs, swing: str = "cartesian", swing_height: float,
                 swing_ff: bool, kp_swing, kd_swing, pose_s: float = POSE_S,
                 swings: int = 0, swing_stop_deg: float = 30.0,
                 ff_inertia=None) -> None:
        self.gate = gate
        self.gait = gait
        self.legs = np.zeros(C.N_LEGS, dtype=bool)
        self.legs[list(legs)] = True
        if swing not in SWING.SWING_MODES:
            raise ValueError("swing %r: one of %s" % (swing, SWING.SWING_MODES))
        self.swing = swing
        self.swing_height = float(swing_height)
        self.swing_ff = bool(swing_ff)
        self.ff_inertia = None if ff_inertia is None else np.asarray(ff_inertia, float)
        self.kp_swing = np.asarray(kp_swing, dtype=float)
        self.kd_swing = np.asarray(kd_swing, dtype=float)
        self.pose_s = float(pose_s)
        self.swings_wanted = int(swings)
        #: A TRIP: a swinging joint this far from the arc's own IK ends the run.
        #: The arc needs ~15-30 deg of knee; a joint 30 past it is a leg that
        #: has left the arc, and in the air nothing else stops it.  0 = off.
        self.swing_stop_deg = float(swing_stop_deg)
        self.trip = None
        self.q_arc = np.full((C.N_LEGS, 3), np.nan)   # the arc's IK, swinging legs
        #: The lift pose and its feet: what "pose" ramps to and "hold" holds.
        self.q_pose = np.asarray(BCFG.NOMINAL_POSE, dtype=float).copy()
        self.rest = np.array([HK.foot_position(i, self.q_pose[i])
                              for i in range(C.N_LEGS)])
        self.phase = "limp"
        self.t_phase = 0.0
        self.q_from = None
        self.max_dps = None
        self.tau = np.zeros(C.N_JOINTS)        # requested, before the gate
        self.tau_ff = np.zeros(C.N_JOINTS)
        self.tau_pd = np.zeros(C.N_JOINTS)
        self.p_swing = np.full((C.N_LEGS, 3), np.nan)
        self.swing_s = np.zeros(C.N_LEGS)
        self.contact = np.ones(C.N_LEGS, dtype=bool)
        self.stop_latched = False
        self.notice = None
        self.reports: dict[int, SwingReport] = {}
        self.done: list[SwingReport] = []
        self.counts = np.zeros(C.N_LEGS, dtype=int)
        self._lift_x = np.full((C.N_LEGS, 3), np.nan)
        self._lift_q = np.full((C.N_LEGS, 3), np.nan)

    # -- keys ---------------------------------------------------------------
    def enter(self, now: float, q) -> str:
        if self.phase == "limp":
            self.q_from = np.asarray(q, dtype=float).reshape(C.N_LEGS, 3).copy()
            self.max_dps = SEQ.ramp_motor_dps(self.q_from, self.q_pose, self.pose_s)
            self._go("pose", now)
            return ("POSE: position ramp to the lift pose over %.1f s; the hold "
                    "takes over by itself when it arrives." % self.pose_s)
        if self.phase == "pose":
            return "ENTER ignored: the pose ramp is still running"
        if self.phase == "hold":
            self.gait.reset(now)
            self.stop_latched = False
            self._go("swing", now)
            return ("SWING: %s on %s; the first foot lifts in %.2f s.  ENTER "
                    "stops at a four-foot window, X is an E-STOP."
                    % (self.gait, "/".join(C.LEGS[i] for i in np.flatnonzero(self.legs)),
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
                # ARRIVED.  The hold's target is the FK of the pose the legs
                # are in, so the switch to torque asks for nothing but gravity.
                self.gate.start(now, body.q)
                self._go("hold", now)
                self.notice = ("HOLD: torque mode, each foot at its site.  The gate "
                               "ramps the cap over %.1f s.  ENTER to swing."
                               % self.gate.ramp_s)
            return "position", C.flat(q_des)
        return "torque", self._torque(now, body)

    def _torque(self, now: float, body) -> np.ndarray:
        q4 = C.unflat(body.q)
        tau = C.flat(TRQ.all_leg_gravity_torque(body.q, body.R)).astype(float)
        self.tau_ff[:] = 0.0
        self.tau_pd[:] = 0.0
        self.p_swing[:] = np.nan
        self.q_arc[:] = np.nan
        swinging = np.zeros(C.N_LEGS, dtype=bool)
        if self.phase == "swing":
            clock = self.gait.sample(now)
            swinging = ~clock.contact & self.legs
            self.swing_s = np.where(self.legs, clock.swing_s, 0.0)
            self.contact = ~swinging
            # FOUR DOWN, NOT FOUR AT FULL WEIGHT: at duty 0.72 the contact ramp
            # fills the four-foot window and `full_support` is true for about
            # one sweep a cycle, which a 4 ms loop mostly misses.  In the air
            # the weights are a fiction anyway; the arc being finished is not.
            if self.stop_latched and clock.contact.all():
                self._go("hold", now)
                self.notice = "HOLD: the clock stopped at a four-foot window.  ENTER swings again."
                swinging[:] = False
                self.swing_s[:] = 0.0
                self.contact[:] = True
        else:
            self.swing_s[:] = 0.0
            self.contact[:] = True
        for i in range(C.N_LEGS):
            sl = slice(3 * i, 3 * i + 3)
            if not swinging[i]:
                self._lift_x[i] = np.nan
                self._lift_q[i] = np.nan
                pd = SWING.swing_torque(body, i, self.rest[i], np.zeros(3),
                                        kp=self.kp_swing, kd=self.kd_swing)
                tau[sl] += pd
                self.tau_pd[sl] = pd
                continue
            s = float(self.swing_s[i])
            if self.swing == "joint":
                if not np.isfinite(self._lift_x[i, 0]):
                    self._lift_x[i] = body.x_b[i]
                    self._lift_q[i] = q4[i]
                qj, qdj = SWING.joint_swing_reference(
                    i, self._lift_x[i], s, self.gait.swing_duration, self._lift_q[i],
                    height=self.swing_height)
                p, v, a = SWING.swing_reference_pva(self._lift_x[i], s,
                                                    self.gait.swing_duration,
                                                    height=self.swing_height)
                pd = SWING.joint_swing_torque(body, i, qj, qdj)
            else:
                p, v, a = SWING.swing_reference_pva(self.rest[i], s,
                                                    self.gait.swing_duration,
                                                    height=self.swing_height)
                pd = SWING.swing_torque(body, i, p, v, kp=self.kp_swing,
                                        kd=self.kd_swing)
                # THE ARC'S OWN IK, FOR THE GUARD AND THE REPORT ONLY -- no
                # torque reads it.  191 us a leg, which this loop can afford.
                qj, _ = SWING.joint_swing_reference(
                    i, self.rest[i], s, self.gait.swing_duration, q4[i],
                    height=self.swing_height)
            self.q_arc[i] = qj
            dev = np.degrees(np.abs(q4[i] - qj))
            if self.swing_stop_deg > 0.0 and dev.max() > self.swing_stop_deg:
                self.trip = ("swing %s: %s is %.0f deg off the arc's IK (limit %.0f) "
                             "-- the leg has left the arc"
                             % (C.LEGS[i], ("abd", "pitch", "knee")[int(dev.argmax())],
                                dev.max(), self.swing_stop_deg))
            self.p_swing[i] = p
            tau[sl] += pd
            self.tau_pd[sl] = pd
            if self.swing_ff:
                ff = SWING.swing_feedforward(i, q4[i], v, a, jac=body.jac[i],
                                             inertia=self.ff_inertia)
                tau[sl] += ff
                self.tau_ff[sl] = ff
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
                    rep = self.reports[i] = SwingReport(i, now, body.x_b[i],
                                                        self.swing_height)
                rep.add(now, body.x_b[i], self.tau_ff[sl], self.tau_pd[sl],
                        clip[sl].max(), tau_sent[sl], q=C.unflat(body.q)[i],
                        q_arc=self.q_arc[i])
            elif rep is not None:
                self.counts[i] += 1
                self.done.append(rep)
                lines.append(rep.line(int(self.counts[i])))
                del self.reports[i]
                if (self.swings_wanted and not self.stop_latched
                        and len(self.done) >= self.swings_wanted):
                    self.stop_latched = True
                    lines.append("%d swings landed -- HOLD at the next four-foot window"
                                 % len(self.done))
        return lines

    def status(self, now: float, body) -> str:
        dz = 1e3 * (body.x_b[:, 2] - self.rest[:, 2])
        head = "   %-6s t=%5.1f  feet dz %s mm" % (
            self.phase, now - self.t_phase,
            np.array2string(dz, precision=1, floatmode="fixed",
                            formatter={"float_kind": lambda v: "%+5.1f" % v}))
        if self.phase == "swing":
            head += "  cycle %d  swing_s %s%s" % (
                self.gait.cycles(now),
                np.array2string(self.swing_s, precision=2),
                "  stop latched" if self.stop_latched else "")
        return head

    def report(self) -> str:
        if not self.done:
            return "  no swing landed"
        out = []
        for leg in range(C.N_LEGS):
            mine = [r for r in self.done if r.leg == leg]
            if not mine:
                continue
            out.append("  %s  %d swing%s: apex %.0f -> %.1f mm (worst %.1f)  |x| %.1f mm  "
                       "vz_td %+.2f m/s (worst %+.2f)  tau_ff %.1f  pd %.1f  clip %.1f"
                       % (C.LEGS[leg], len(mine), "" if len(mine) == 1 else "s",
                          1e3 * self.swing_height,
                          1e3 * np.mean([r.z_max for r in mine]),
                          1e3 * min(r.z_max for r in mine),
                          1e3 * max(max(abs(r.x_min), abs(r.x_max)) for r in mine),
                          np.nanmean([r.vz for r in mine]),
                          np.nanmin([r.vz for r in mine]),
                          max(r.ff_peak for r in mine), max(r.pd_peak for r in mine),
                          max(r.clip_peak for r in mine)))
        return "\n".join(out)


class BenchLog:
    """Every torque-mode sweep, to one npz."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)

    def save(self, path: str) -> str:
        if not self.rows:
            return "nothing to save"
        keys = self.rows[0].keys()
        np.savez_compressed(path, **{k: np.array([r[k] for r in self.rows]) for k in keys})
        return "%d sweeps -> %s" % (len(self.rows), path)


def analyse(L, after_s: float = 0.3) -> str:
    """Every swing in a `--log` file, one block each, from the columns alone.

    A swing is a run of sweeps with `swing_s > 0` on one leg.  For each: the
    foot's path from its liftoff point (FK is in the log as `x_b`), the arc's
    apex, the joints against the arc's own IK (`q_arc`), the torque split
    (`tau_ff`, `tau_pd`, what the gate sent, what the drivers measured), the
    slew's clip, and then the `after_s` after the clock's touchdown -- where a
    leg that left the arc shows up as a foot still travelling under the soft
    hold.  Joints print in degrees, abd / pitch / knee.
    """
    t = L["t"]; q = L["q"].reshape(len(t), C.N_LEGS, 3); x_b = L["x_b"]
    swing_s = L["swing_s"]; q_arc = L["q_arc"] if "q_arc" in L.files else None
    tau_ff = L["tau_ff"].reshape(len(t), C.N_LEGS, 3)
    tau_pd = L["tau_pd"].reshape(len(t), C.N_LEGS, 3)
    tau_cmd = L["tau_cmd"].reshape(len(t), C.N_LEGS, 3)
    tau_req = L["tau_req"].reshape(len(t), C.N_LEGS, 3)
    tau_meas = L["tau_meas"].reshape(len(t), C.N_LEGS, 3)
    out = ["%d sweeps, %.1f s of torque mode" % (len(t), t[-1] - t[0])]
    out += _run_health(L, t, tau_cmd, tau_meas, swing_s)
    n = 0
    fits = []                      # (pitch, knee) effective inertia per swing, from tau_meas
    links = np.diag(LD.mass_matrix(0, np.asarray(BCFG.NOMINAL_POSE, float)[0], armature=False))
    for leg in range(C.N_LEGS):
        up = swing_s[:, leg] > 0.0
        edges = np.flatnonzero(np.diff(up.astype(int)))
        starts = [e + 1 for e in edges if up[e + 1]]
        ends = [e + 1 for e in edges if not up[e + 1]]
        for a in starts:
            b = next((e for e in ends if e > a), None)
            if b is None:
                continue
            n += 1
            sl = slice(a, b)
            d = x_b[sl, leg] - x_b[a, leg]
            dq = np.degrees(q[sl, leg] - q[a, leg])
            vz = (x_b[b - 1, leg, 2] - x_b[b - 2, leg, 2]) / max(t[b - 1] - t[b - 2], 1e-3)
            aft = slice(b, min(len(t), b + int(round(after_s / max(np.median(np.diff(t)), 1e-3)))))
            d_aft = x_b[aft, leg] - x_b[a, leg]
            dq_aft = np.degrees(q[aft, leg] - q[a, leg])
            out.append("")
            out.append("swing %d  %s  t %.2f-%.2f s (%d sweeps, %.0f ms)"
                       % (n, C.LEGS[leg], t[a], t[b - 1], b - a, 1e3 * (t[b - 1] - t[a])))
            out.append("  foot   z max %+5.1f mm   x %+5.1f/%+5.1f   y %+5.1f/%+5.1f mm   "
                       "z at touchdown %+5.1f mm, vz %+.2f m/s"
                       % (1e3 * d[:, 2].max(), 1e3 * d[:, 0].max(), 1e3 * d[:, 0].min(),
                          1e3 * d[:, 1].max(), 1e3 * d[:, 1].min(), 1e3 * d[-1, 2], vz))
            out.append("  joints travel from liftoff, abd/pitch/knee: %+.0f/%+.0f/%+.0f .. "
                       "%+.0f/%+.0f/%+.0f deg"
                       % (*dq.min(axis=0), *dq.max(axis=0)))
            if q_arc is not None and np.isfinite(q_arc[sl, leg]).all():
                dev = np.degrees(q[sl, leg] - q_arc[sl, leg])
                worst = np.abs(dev).argmax(axis=0)
                out.append("  off the arc's IK, worst abd/pitch/knee: %+.0f/%+.0f/%+.0f deg "
                           "(knee's worst at s=%.2f)"
                           % (dev[worst[0], 0], dev[worst[1], 1], dev[worst[2], 2],
                              swing_s[a + worst[2], leg]))
            peaks = [np.array2string(np.abs(v).max(axis=0), precision=1)
                     for v in (tau_ff[sl, leg], tau_pd[sl, leg], tau_cmd[sl, leg],
                               tau_meas[sl, leg])]
            out.append("  torque peaks abd/pitch/knee  ff %s  pd %s  sent %s  measured %s  "
                       "slew clip %.1f N*m"
                       % (*peaks, np.abs(tau_req[sl, leg] - tau_cmd[sl, leg]).max()))
            fit = _identify_inertia(t, q, tau_meas, tau_cmd, leg, sl)
            if fit is not None:
                fits.append(fit["meas"])
                out.append("  inertia the leg ANSWERED WITH (tau - g = I qdd + b qd + c), "
                           "pitch / knee: %.4f / %.4f kg m^2 from measured torque, "
                           "%.4f / %.4f from commanded; the model's M0 %.4f / %.4f"
                           % (*fit["meas"], *fit["cmd"], *np.diag(SWING.JOINT_INERTIA_M0)[1:]))
            if aft.stop > aft.start + 2:
                out.append("  %.0f ms after touchdown: foot %+5.1f/%+5.1f/%+5.1f mm off its "
                           "site at the end, farthest %.1f mm; joints %+.0f/%+.0f/%+.0f deg "
                           "from liftoff at the end"
                           % (1e3 * after_s, *(1e3 * d_aft[-1]),
                              1e3 * np.linalg.norm(d_aft, axis=1).max(), *dq_aft[-1]))
    if n == 0:
        out.append("no swing in this log")
    good = np.array([f for f in fits if np.all(np.isfinite(f))])
    if len(good):
        med = np.median(good, axis=0)
        arm = med - links[1:]
        out.append("")
        out.append("THE ARMATURE THAT FITS, over %d swings (median): pitch %.5f, knee %.5f "
                   "kg m^2 against the inherited %.5f; the knee's M0 is 96 %% armature, so "
                   "its fit is the cleaner one.  Effective inertia pitch %.4f knee %.4f "
                   "against the model's %.4f / %.4f (ratio %.2f / %.2f).\n"
                   "  -> try --ff-armature %.5f, and expect the apex to land on the arc's "
                   "number and the x drift to shrink with it."
                   % (len(good), arm[0], arm[1], P.ARMATURE, med[0], med[1],
                      *np.diag(SWING.JOINT_INERTIA_M0)[1:],
                      np.diag(SWING.JOINT_INERTIA_M0)[1] / med[0],
                      np.diag(SWING.JOINT_INERTIA_M0)[2] / med[1], max(arm[1], 0.0)))
    return "\n".join(out)


def _run_health(L, t, tau_cmd, tau_meas, swing_s) -> list[str]:
    """Did the LOOP stall, did a DRIVER quit, did a joint sit on the CAP, and
    where was the clock when the log ends.  The questions behind a stop line."""
    out = ["", "RUN HEALTH"]
    dt = np.diff(t)
    if len(dt):
        worst = int(dt.argmax())
        out.append("  sweep period: median %.1f ms, p99 %.1f ms, worst %.1f ms at t=%.2f s%s"
                   % (1e3 * np.median(dt), 1e3 * np.percentile(dt, 99), 1e3 * dt[worst],
                      t[worst], "  <-- a stall past 50 ms latches the drivers' input-lost "
                      "protection (0x80)" if dt[worst] > 0.05 else ""))
    peak_c = np.abs(tau_cmd).max(axis=0); peak_m = np.abs(tau_meas).max(axis=0)
    out.append("  peak |tau| sent, abd/pitch/knee per leg:     %s"
               % "  ".join("%s %s" % (C.LEGS[i], np.array2string(peak_c[i], precision=1))
                           for i in range(C.N_LEGS)))
    out.append("  peak |tau| measured (driver iq), same order: %s"
               % "  ".join("%s %s" % (C.LEGS[i], np.array2string(peak_m[i], precision=1))
                           for i in range(C.N_LEGS)))
    if "cap" in L.files:
        cap = L["cap"]
        # only once the gate's ramp is mostly up: at cap ~0 every torque "sits on it"
        at_cap = ((np.abs(tau_cmd).reshape(len(t), -1) >= 0.98 * cap[:, None] - 1e-9)
                  & (cap[:, None] >= 0.5 * cap.max()))
        hits = at_cap.sum(axis=0)
        if hits.any():
            names = ["%s.%s" % (C.LEGS[j // 3], ("abd", "pitch", "knee")[j % 3])
                     for j in np.flatnonzero(hits)]
            out.append("  ON THE CAP (%.1f N*m) for %d sweeps in all: %s"
                       % (cap.max(), int(hits.sum()),
                          ", ".join("%s x%d" % (nm, hits[j]) for nm, j in
                                    zip(names, np.flatnonzero(hits)))))
        else:
            out.append("  never on the cap (%.1f N*m)" % cap.max())
    tail = t > t[-1] - 0.25
    if tail.sum() > 5:
        c = np.abs(tau_cmd[tail]).mean(axis=0); m = np.abs(tau_meas[tail]).mean(axis=0)
        quiet = (c > 0.3) & (m < 0.25 * c)
        if quiet.any():
            out.append("  LAST 0.25 s: %s asked for torque and the driver measured "
                       "almost none -- a driver that quit, not a loop that stalled"
                       % ", ".join("%s.%s" % (C.LEGS[i], ("abd", "pitch", "knee")[j])
                                   for i, j in zip(*np.nonzero(quiet))))
        else:
            out.append("  last 0.25 s: measured torque followed the command on every joint")
    if "errors" in L.files:
        err = L["errors"]
        bad = np.flatnonzero(err.any(axis=1))
        if len(bad):
            first = bad[0]
            out.append("  DRIVER ERROR BYTES: first at t=%.2f s: %s%s"
                       % (t[first], " ".join("0x%02x" % e for e in err[first]),
                          "  (0x80 = input lost: that driver saw no frame for 50 ms)"
                          if (err[first] & 0x80).any() else ""))
        else:
            out.append("  no driver error byte in the log")
    up = swing_s[-1] > 0
    out.append("  the log ends %s" % ("MID-SWING of %s at s=%.2f -- the run stopped with a "
                                       "foot in the air"
                                       % ("/".join(C.LEGS[i] for i in np.flatnonzero(up)),
                                          swing_s[-1][up].max())
                                       if up.any() else "with all four feet down"))
    return out


def _identify_inertia(t, q, tau_meas, tau_cmd, leg, sl):
    """Least squares of tau - g on (qdd, qd, 1) over one swing, pitch and knee.

    q is smoothed over five sweeps before it is differenced twice; gravity is
    the leg's own at each pose, trunk level (the bench has no IMU).  Returns
    None if the joints did not move enough to fit (a fake bus, a stalled
    swing), else ``{"meas": (I_pitch, I_knee), "cmd": (...)}``.
    """
    tt = t[sl]
    qq = q[sl, leg]
    if len(tt) < 12:
        return None
    k = np.ones(5) / 5.0
    qs = np.column_stack([np.convolve(qq[:, j], k, mode="same") for j in range(3)])
    qd = np.gradient(qs, tt, axis=0)
    qdd = np.gradient(qd, tt, axis=0)
    g = np.array([TRQ.leg_gravity_torque(leg, qi) for qi in qq])
    inner = slice(3, len(tt) - 3)
    out = {}
    for name, src in (("meas", tau_meas), ("cmd", tau_cmd)):
        y = src[sl, leg] - g
        fit = []
        for j in (1, 2):
            # A FIT NEEDS A MOVING JOINT: 5 deg of travel, and a fitted inertia
            # that is positive and explains the torque -- a fake bus, a stalled
            # swing or encoder dither give none of the three.
            X = np.column_stack([qdd[inner, j], qd[inner, j], np.ones(inner.stop - inner.start)])
            coef, *_ = np.linalg.lstsq(X, y[inner, j], rcond=None)
            resid = y[inner, j] - X @ coef
            r2 = 1.0 - resid.var() / max(y[inner, j].var(), 1e-12)
            ok = (np.ptp(qq[:, j]) > np.radians(5.0) and coef[0] > 0.0 and r2 > 0.5)
            fit.append(float(coef[0]) if ok else float("nan"))
        out[name] = tuple(fit)
    if all(not np.isfinite(v) for v in out["meas"] + out["cmd"]):
        return None
    return out


def _print_phase(bench: SwingBench) -> None:
    print("\n>> %-6s : %s" % (bench.phase, PHASE_BLURB[bench.phase]), flush=True)


def run(mb, bench: SwingBench, *, rate_hz: float = RATE_HZ, key=None,
        auto_s: float | None = None, clock=time.perf_counter,
        log: "BenchLog | None" = None) -> str | None:
    """Drive `bench` on an ARMED `mb` until X or a trip.  Returns the reason.

    `hw.stand.run`'s loop with the stand taken out: the same round robin,
    the same latched sweep, the same gate, monitors and stop lines.  Does
    not stop the motors -- the caller's `with MotorBus(...)` does.
    """
    ids = HM.motor_ids()
    n = len(ids)
    unwrappers = CAL.new_unwrappers()
    misses = SAFE.CanMissMonitor(mb)
    readback = SAFE.TorqueReadback(n)
    slot = mb.slot(rate_hz)
    alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * n * slot)
    level = IMU.TrunkOrientation.level()

    sweep = 0
    mode, values = "keepalive", None
    q_prev = t_prev = None
    qd_ctrl = np.zeros(n)
    tau_sent = np.zeros(n)
    worst_gap = np.zeros(n)
    overruns = 0
    t0 = last_status = bench.t_phase = clock()
    last_phase = bench.phase
    _print_phase(bench)

    deadline = clock() + slot
    k = 0
    while True:
        mb.poll()
        if k == 0:
            now = clock()
            try:
                q, qd_driver = CAL.joint_state(mb, unwrappers)
            except RuntimeError as missing:
                if now - t0 > 1.0:
                    return str(missing)
                k = (k + 1) % n
                mb.keepalive(ids[0])
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
            if auto_s is not None:
                if bench.phase == "limp" and now - bench.t_phase >= auto_s:
                    print("\n   [auto] " + bench.enter(now, q), flush=True)
                elif bench.phase == "hold" and now - bench.t_phase >= auto_s \
                        and not bench.done:
                    print("\n   [auto] " + bench.enter(now, q), flush=True)
                elif bench.phase == "hold" and bench.done \
                        and now - bench.t_phase >= auto_s:
                    return None

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
                return ("input-lost latch (0x80) on CAN %s: a driver heard nothing "
                        "for %.0f ms and went limp" % (latched, 1e3 * SAFE.INPUT_LOST_S))
            temps = np.asarray([mb.rec(mid).temp or 0 for mid in ids])
            reason = bench.gate.estop_reason(q, qd_driver, now, temps=temps,
                                             miss_streaks=misses.update(mb),
                                             errors=errors)
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
                    log.add(t=now - t0, phase=bench.phase, q=q, qd=qd_ctrl,
                            x_b=body.x_b, p_swing=bench.p_swing.copy(),
                            swing_s=bench.swing_s.copy(), contact=bench.contact.copy(),
                            tau_req=bench.tau.copy(), tau_cmd=tau_sent.copy(),
                            tau_ff=bench.tau_ff.copy(), tau_pd=bench.tau_pd.copy(),
                            tau_meas=tau_meas, q_arc=bench.q_arc.copy(),
                            errors=np.asarray([errors.get(mid, 0) or 0 for mid in ids]),
                            temps=temps, cap=bench.gate.cap_now(now))
            else:
                tau_sent = np.zeros(n)

            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                print(bench.status(now, body)
                      + "  |tau|=%.2f  gap=%.1f ms  overrun=%d"
                      % (float(np.abs(tau_sent).max()), 1e3 * worst_gap.max(), overruns),
                      flush=True)
                if mode == "torque":
                    print(_torque_lines(tau_sent, tau_meas), flush=True)
            sweep += 1

        # -- one frame, to one motor ------------------------------------------
        mid = ids[k]
        sent_at = clock()
        previous = mb.rec(mid).last_cmd_t
        if previous is not None:
            gap = sent_at - previous
            worst_gap[k] = max(worst_gap[k], gap)
            if gap > GAP_ESTOP_S:
                return ("CAN %d went %.1f ms without a frame, past the %.0f ms stop "
                        "line" % (mid, 1e3 * gap, 1e3 * GAP_ESTOP_S))
        if (sweep % STATUS_EVERY_SWEEPS == 0
                and k == (sweep // STATUS_EVERY_SWEEPS) % n):
            mb.status1_req(mid)
        elif mode == "position":
            mb.position(mid, float(np.rad2deg(values[k])),
                        max_dps=float(bench.max_dps[k]))
        elif mode == "torque":
            mb.torque(mid, float(values[k]))
        else:
            mb.keepalive(mid)

        k = (k + 1) % n
        overrun = mb.pace(deadline)
        deadline += slot
        if overrun > 3 * slot:
            overruns += 1
            deadline = clock() + slot
    # unreachable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m hw.swing_bench",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tau-cap", type=float, default=TAU_CAP,
                    help="torque cap, N*m; the ceiling is the motors' %.0f" % SAFE.TAU_HARD_NM)
    ap.add_argument("--tau-slew", type=float, default=BCFG.TAU_SLEW_TROT_NM_S,
                    metavar="NM_PER_S", help="the gate's |dtau/dt| limit -- the trot's")
    ap.add_argument("--overspeed-trip", action="store_true",
                    help="stop on the 7 rad/s joint-speed trip (off, as in the trot)")
    ap.add_argument("--pose-s", type=float, default=POSE_S,
                    help="the position ramp from the hang to the lift pose, s")
    ap.add_argument("--rate", type=float, default=RATE_HZ)
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--arm-timeout", type=float, default=30.0)
    ap.add_argument("--unconfirmed", default=None, metavar="WHY")
    ap.add_argument("--fake", action="store_true", help="hw.fake_bus, no dynamics")
    ap.add_argument("--auto", type=float, default=None, metavar="SECONDS",
                    help="press ENTER by itself this long into limp and hold, and "
                         "exit after the swings (--fake only)")
    ap.add_argument("--log", default=None, metavar="FILE.npz")

    sw = ap.add_argument_group("the swing", "hw.trot's flags, hw.trot's defaults")
    sw.add_argument("--legs", nargs="+", default=list(C.LEGS), choices=list(C.LEGS),
                    help="which legs follow the arc when the clock lifts them")
    sw.add_argument("--swings", type=int, default=0, metavar="N",
                    help="stop after N swings have landed; 0 waits for ENTER")
    sw.add_argument("--period", type=float, default=TROT.PERIOD_S, metavar="SECONDS")
    sw.add_argument("--duty", type=float, default=BCFG.DUTY)
    sw.add_argument("--contact-ramp", type=float, default=BCFG.CONTACT_RAMP)
    sw.add_argument("--settle", type=float, default=BCFG.SETTLE_S, metavar="SECONDS")
    sw.add_argument("--settle-every", type=int, default=BCFG.SETTLE_EVERY, metavar="CYCLES")
    sw.add_argument("--swing", choices=SWING.SWING_MODES, default="cartesian")
    sw.add_argument("--swing-height", type=float, default=1e3 * BCFG.SWING_HEIGHT,
                    metavar="MM")
    sw.add_argument("--swing-ff", action="store_true",
                    help="the inertial feedforward, balance.swing.swing_feedforward")
    sw.add_argument("--ff-armature", type=float, default=None, metavar="KG_M2",
                    help="the reflected rotor inertia the feedforward's M0 is built "
                         "on (joint side).  Default params.ARMATURE %.5f -- DOG5's, "
                         "not measured on DOG6; --analyse fits it from a log" % P.ARMATURE)
    sw.add_argument("--kp-swing", type=float, nargs=3, default=list(BCFG.KP_SWING),
                    metavar="N_PER_M")
    sw.add_argument("--kd-swing", type=float, nargs=3, default=list(BCFG.KD_SWING),
                    metavar="NS_PER_M")
    sw.add_argument("--swing-stop", type=float, default=30.0, metavar="DEG",
                    help="TRIP: a swinging joint this far from the arc's own IK "
                         "stops the run; 0 turns it off")
    ap.add_argument("--analyse", default=None, metavar="FILE.npz",
                    help="read a --log file and print every swing: joints against "
                         "the arc's IK, the foot's path, the torque split, the "
                         "300 ms after touchdown.  No bus is opened")
    args = ap.parse_args(argv)
    if args.analyse:
        print(analyse(np.load(args.analyse)))
        return 0
    if args.auto is not None and not args.fake:
        ap.error("--auto is only allowed with --fake")

    try:
        gait = GAIT.TrotGait(period=args.period, duty=args.duty, ramp=args.contact_ramp,
                             settle_s=args.settle, settle_every=args.settle_every)
    except ValueError as refusal:
        ap.error("the gait: %s" % refusal)
    reason = args.unconfirmed or ("hw.fake_bus" if args.fake else None)
    ids = HM.motor_ids()
    try:
        gate = SAFE.SafetyGate(args.tau_cap, unconfirmed_reason=reason,
                               ceiling=SAFE.TAU_HARD_NM, tau_slew=args.tau_slew,
                               overspeed_trip=args.overspeed_trip)
    except (RuntimeError, ValueError) as refusal:
        print("[bench] REFUSED before opening the bus:\n  %s" % refusal, file=sys.stderr)
        return 2
    bench = SwingBench(gate, gait, legs=[C.LEGS.index(name) for name in args.legs],
                       swing=args.swing, swing_height=1e-3 * args.swing_height,
                       swing_ff=args.swing_ff, kp_swing=args.kp_swing,
                       kd_swing=args.kd_swing, pose_s=args.pose_s, swings=args.swings,
                       swing_stop_deg=args.swing_stop,
                       ff_inertia=(None if args.ff_armature is None
                                   else SWING.feedforward_inertia(args.ff_armature)))

    # -- the banner ---------------------------------------------------------------
    print("DOG6 swing bench on %s -- HANG THE ROBOT UP FIRST: nothing here holds it"
          % ("hw.fake_bus" if args.fake else "can0"))
    print("  CAN ids %s  directions %s  map confirmed: %s" % (ids, HM.directions(),
                                                             CONFIRMED_ON_DOG6))
    print("  cap %.1f N*m, slew %.0f N*m/s, overspeed trip %s (peaks print at exit), "
          "swing-stop %s"
          % (args.tau_cap, args.tau_slew, "ON" if args.overspeed_trip else "OFF",
             "OFF" if args.swing_stop <= 0 else "%.0f deg off the arc's IK" % args.swing_stop))
    print("  legs %s   %s" % (" ".join(args.legs), gait))
    demand = SWING.swing_demand(0, bench.rest[0], gait.swing_duration,
                                1e-3 * args.swing_height, bench.q_pose[0])
    print("  swing apex %.0f mm, %s PD Kp %s Kd %s; %.0f ms: knee ~%.1f rad/s, ~%.1f N*m; "
          "slew ~%.0f to rise, ~%.0f to land softly, against %.0f"
          % (args.swing_height, args.swing, np.asarray(args.kp_swing),
             np.asarray(args.kd_swing), 1e3 * gait.swing_duration, demand["qd"][2],
             demand["tau"].max(), demand["slew"], 3.0 * demand["slew"], args.tau_slew))
    if demand["tau"].max() > args.tau_cap:
        print("  WARNING: the arc asks ~%.1f N*m against a %.1f cap -- raise --tau-cap "
              "or lower --swing-height" % (demand["tau"].max(), args.tau_cap))
    m0 = np.diag(SWING.feedforward_inertia(args.ff_armature))
    print("  feedforward %s" % (
        "ON: M0 J^+ (a_ref - Jdot qd_ref), M0 diag %s (armature %s), logged as tau_ff"
        % (np.array2string(m0, precision=4),
           "%.5f measured" % args.ff_armature if args.ff_armature is not None
           else "%.5f INHERITED from DOG5 -- --analyse a log to measure it" % P.ARMATURE)
        if args.swing_ff else "OFF (--swing-ff)"))
    print("  ENTER: limp -> pose; hold -> swing; swing -> stop at four feet.  X: E-STOP.")

    from .motor import motorbus
    if args.fake:
        from .fake_bus import FakeDriverBus
        mb = motorbus.MotorBus(ids, bus=FakeDriverBus(ids=ids), dirs=HM.motor_directions())
    else:
        mb = motorbus.MotorBus(ids, bitrate=args.bitrate, dirs=HM.motor_directions())
    key = KeyPoller()
    if not key.ok and args.auto is None:
        print("[bench] stdin is not a terminal: ENTER and X will not work", file=sys.stderr)
    log = BenchLog() if args.log else None
    stop = None
    try:
        with mb:
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[bench] not every motor armed", file=sys.stderr)
                return 1
            stop = run(mb, bench, rate_hz=args.rate, key=key, auto_s=args.auto, log=log)
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        key.restore()

    print()
    print("[bench] swings:")
    print(bench.report())
    if gate.started_at is not None:
        print("[bench] peak joint speed, driver / encoder (rad/s; trip %s at %.1f):"
              % ("ON" if gate.overspeed_trip else "OFF", gate.qd_estop))
        for leg in range(C.N_LEGS):
            row = slice(3 * leg, 3 * leg + 3)
            print("  %s  %s" % (C.LEGS[leg], "   ".join(
                "%4.1f / %4.1f" % pair for pair in zip(
                    gate.qd_peak[row], gate.encoder_qd_peak[row]))))
    if log is not None:
        print("[bench] log: %s" % log.save(args.log))
    if stop:
        print("[bench] stopped: %s.  Motors stopped; the legs hang." % stop)
        return 0 if stop in ("operator X", "Ctrl-C") else 3
    print("[bench] done; motors stopped, the legs hang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
