"""The legs' gravity torque G(q), hung up: each joint in from both sides, slowly.

The robot HUNG UP, every joint held by a joint PD and brought slowly into the
target from below and from above; the torque it settles at is averaged, and
the mean of the two sides is G.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.gravity_bench --fake --auto 1            the whole path, no robot
    $V -m hw.gravity_bench                            on the robot, in the air: fold2, 160 mm
    $V -m hw.gravity_bench --log gravity.npz          ... and keep every sweep and the table
    $V -m hw.gravity_bench --height 160 140 120       three fold2 holds, one after the other
    $V -m hw.gravity_bench --legs FL --offset 10      one leg, 10 deg out on each side

HANG THE ROBOT BEFORE YOU START, AND HANG IT LEVEL.  Nothing here holds it up,
and G is the torque in the TRUNK's frame, so the trunk's attitude is in it:
at fold2, 160 mm, 5 deg of roll moves every abduction's G by 0.033 N*m (the
model gives it 0.020 level) and 5 deg of pitch moves the pitch's by 0.034,
13 % of it.

    limp     zero torque; the legs hang.  Fold the front legs knee-forward by
             hand, as for `hw.fold2_trot`, and the ramp is a small move.  ENTER ->
    pose     POSITION mode, a smoothstep to the first target over `--pose-s`.
    hold     TORQUE mode, by itself when the ramp arrives: every joint held at
             the target by the PD below, nothing else.  The gate's 1 s ramp
             brings the cap up from zero; the legs settle tau / Kp off the
             target, ~3 deg on the pitch.                                ENTER ->
    measure  per target, from - and then from +: the reference to that side's
             start, a dwell, the SLOW approach into the target, the wait for
             the leg to stop, the average.  The table prints when a target's
             two sides are in.
    done     held at the last target.  ENTER measures again (pass 2, ...).
    X        stop, any time.  Motors stop; the legs drop to their hang.

THE METHOD, the operator's (2026-10-20)
    Every joint is held by a joint PD, and nothing else -- no gravity
    feedforward, no model anywhere in the torque:

        tau = Kp (q_ref - q) + Kd (qd_ref - qd)        `--kp` / `--kd`, abd pitch knee

    The reference brings the whole leg, all three joints at once, slowly into
    the target q* from one side; once the leg has stopped the torque is
    averaged; then the same from the other side:

        from -    q_ref  q* - offset  ->  q*       tau_lo
        from +    q_ref  q* + offset  ->  q*       tau_hi

        G     = (tau_lo + tau_hi) / 2              what holds the leg up at q*
        tau_s = (tau_lo - tau_hi) / 2              the friction band, half-width

    A leg that has stopped is in balance -- the PD's torque against gravity
    and friction -- and friction opposes the way it came, so it stops with
    G + friction coming up in angle and G - friction coming down; the mean
    cancels it.  SLOW (`--speed`, 2 deg/s), so nothing velocity-dependent,
    viscous friction or inertia, is in the reading.  The average starts only
    once every measured joint has stayed within `--settle-deg` (0.05 deg;
    one encoder count is 0.0055) for 0.3 s, and no sooner than `--settle`
    after the reference stopped.

THE SIDES ARE ANGLES, NOT UP AND DOWN
    "from -" is the joint coming up from the smaller angle, on all three
    joints -- the operator's note says it of the knee, whose G is near zero.
    That keeps tau_s positive on every joint whichever way gravity pulls it.
    In up / down terms: at fold2, 160 mm, a + pitch LOWERS the foot on FL and
    RR and LIFTS it on FR and RL, so "from -" is tau_down on FL / RR and
    tau_up on FR / RL, and (tau_up - tau_down) / 2 would print the FL / RR
    friction negative.

WHY THE PD IS HERE AND NOT IN THE DRIVERS' 0xA4 LOOP
    0xA4 is the drivers' own loop and its gains are not ours to read.  If it
    integrates, a leg that stops short has its current wound up until it
    creeps on, and where it ends in the friction band no longer says which
    side it came from.  This PD has no integrator: it stops where its torque
    enters the band and stays there, and the torque it holds is the one it
    commanded.  0xA4 is the `pose` ramp, nothing else.

WHAT PRINTS, one table per target
    q            where the leg stopped, the two sides' mean -- off the target
                 by tau / Kp
    from -, +    tau_lo and tau_hi: the COMMANDED torque, averaged
    G, tau_s     from those -- what a law has to command
    iq           the same from the torque the drivers MEASURED (q-axis
                 current): beside the cmd columns, how well the current loop
                 follows
    model        `balance.torque.leg_gravity_torque` at q, trunk level: the
                 leg gravity every law adds today, from the CAD masses.  ratio
                 is G / model where |model| >= 0.05 N*m -- the pitch, at fold2
    flags        STUCK (the joint came in < 1 deg from that side: friction
                 held it, the reading is the other side's -- raise --offset
                 or --kp), CAP (the gate clipped that joint during an average:
                 the reading is the cap's), moved (it travelled > 0.2 deg
                 during an average); a side that never stopped says so in
                 the header

    `--log FILE.npz` keeps every torque sweep and the tables (`res_*`, axis 1
    from - then from +).

THE GAINS AND THE OVERSPEED TRIP
    Kp 5 / Kd 0.2 by default, the joint layer's (`config.KP_JOINT_HOLD` /
    `KD_JOINT_HOLD`, `_ABD` for abduction): the joint PD every trot and walk
    fly.  The pitch hangs ~3 deg off its target on them, which `q` reports.
    The first run, 2026-10-20, held on `--swing joint`'s 30 / 0.8 instead --
    the set `hw.fold_trot` records failing on the robot -- the operator saw
    "the leg is damping when hold", and it stopped 0.1-0.6 s into its first
    move on the gate's hard overspeed tier, RR.knee's driver at -8.6 rad/s.
    Both overspeed tiers are on, as the gate has them, and the peaks print
    at exit, driver / encoder.

THE TARGET, `--height`
    `posture.FOLD2`'s hold: the feet at its hip-frame sites with the trunk
    `--height` mm above the floor -- `law.ik_reference` at FOLD2's feet,
    seeded with its crouch, the joints `hw.fold2_trot`'s HOLD latches.  160
    mm is that trot's stand.  Several heights measure in turn.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import namedtuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/gravity_bench.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402
from . import calibration as CAL     # noqa: E402
from . import can_link as CANL       # noqa: E402
from . import hardware_map as HM     # noqa: E402
from . import kinematics as HK       # noqa: E402
from . import realtime as RT         # noqa: E402
from . import safety as SAFE         # noqa: E402
from .balance import config as BCFG  # noqa: E402
from .balance import gait as GAIT    # noqa: E402
from .balance import law as BLAW     # noqa: E402
from .balance import posture as POSE  # noqa: E402
from .balance import sequence as SEQ  # noqa: E402
from .balance import torque as TRQ   # noqa: E402
from .fold2_trot import HEIGHT as FOLD2_HEIGHT  # noqa: E402
from .stand import (GAP_ESTOP_S, QD_FILTER_HZ, RATE_HZ, STATUS_EVERY_SWEEPS,  # noqa: E402
                    STATUS_PERIOD_S, KeyPoller, _torque_lines)
from .swing_bench import POSE_S      # noqa: E402

__all__ = ["GravityBench", "GravityLog", "fold2_pose", "result_table", "run", "main",
           "OFFSET_DEG", "SPEED_DPS", "TAU_CAP"]

#: The PD, (abd, pitch, knee): the joint layer's, `config.KP_JOINT_HOLD` /
#: `KD_JOINT_HOLD` and their `_ABD` -- the joint PD every trot and walk fly.
#: The docstring's THE GAINS for why not `--swing joint`'s 30 / 0.8.
KP = np.array([BCFG.KP_JOINT_HOLD_ABD, BCFG.KP_JOINT_HOLD, BCFG.KP_JOINT_HOLD])
KD = np.array([BCFG.KD_JOINT_HOLD_ABD, BCFG.KD_JOINT_HOLD, BCFG.KD_JOINT_HOLD])

#: deg, how far out each side's approach starts, every measured joint at once:
#: the operator's 5-10 deg of pitch.
OFFSET_DEG = 8.0

#: deg/s, the reference INTO the target.  8 deg in 4 s.
SPEED_DPS = 2.0

#: deg/s, the reference's peak between approaches, to the next side's start.
TRANSIT_DPS = 10.0

#: s, the reference parked at a side's start, so the approach sets off from rest.
DWELL_S = 0.5

#: s, the least wait at the target before the leg can count as stopped.
SETTLE_S = 1.0

#: "Stopped": every measured joint within SETTLE_DEG of where it was for QUIET_S.
SETTLE_DEG = 0.05
QUIET_S = 0.3

#: s past `--settle` the bench waits for that.  Then it averages anyway and the
#: table's header says that side never stopped.
SETTLE_WAIT_S = 4.0

#: s, the average.
AVERAGE_S = 1.0

#: N*m.  A leg in the air needs G plus its friction -- the model's G is 0.26 on
#: the fold2 pitch and 0.02 on the rest -- and a PD transient; 2 leaves room
#: for all of it.  A joint ON the cap in an average is flagged CAP.
TAU_CAP = 2.0

#: deg.  A TRIP: a joint this far off its reference.  The PD holds the pitch
#: ~3 deg off it and the cap is 23 deg of PD; past 15 something is stopping
#: the leg.  0 = off; off under --fake, where torque moves nothing.
TRACK_STOP_DEG = 15.0

#: N*m.  Below this |model| the G / model ratio is noise over noise.
RATIO_FLOOR = 0.05

#: deg.  A joint that travelled more than this during an average is flagged.
MOVED_DEG = 0.2

#: deg.  A joint that came in LESS than this in its approach's direction is
#: flagged STUCK: friction held it through the approach, so the torque it
#: settled at is the other side's, and that row's G and tau_s mean nothing.
#: It breaks away after the reference has moved 2 tau_s / Kp -- 4.6 of the
#: 8 deg for 0.2 N*m at Kp 5 -- so a STUCK joint wants --offset or --kp up.
MIN_TRAVEL_DEG = 1.0

PHASES = ("limp", "pose", "hold", "measure", "done")
PHASE_BLURB = {
    "limp": "NO TORQUE -- the legs hang.  ENTER ramps them to the first target.",
    "pose": "POSITION mode, smoothstep to the target.  Hold follows by itself.",
    "hold": "TORQUE -- every joint held at the target by the PD alone.  ENTER "
            "measures; X stops.",
    "measure": "from -, then from +, each target in turn; a table per target.  "
               "X stops (the legs drop).",
    "done": "TORQUE -- held at the last target.  ENTER measures again; X stops "
            "(the legs drop).",
}

#: One step of a pass: `kind` move / dwell / approach / settle / average, for
#: target number `target`, `side` -1 (from -) or +1 (from +).
Step = namedtuple("Step", "kind target side q_from q_to duration")


def fold2_pose(height: float = FOLD2_HEIGHT) -> np.ndarray:
    """(4, 3) rad: `posture.FOLD2`'s hold with the trunk `height` m (floor to
    trunk bottom) up -- `law.ik_reference` at FOLD2's hip-frame feet, seeded
    with its crouch, which is what `hw.fold2_trot`'s HOLD latches (FOLD2's
    stand sites are its crouch's).  ValueError where the IK misses the feet."""
    q = BLAW.ik_reference(height, POSE.FOLD2.q, POSE.FOLD2.foot_xy)
    want = np.zeros((C.N_LEGS, 3))
    want[:, :2] = POSE.FOLD2.foot_xy
    want[:, 2] = -(height + BCFG.TRUNK_BOTTOM_OFFSET - P.FOOT_RADIUS)
    miss = np.abs(np.array([HK.foot_position_hip(i, q[i]) for i in range(C.N_LEGS)])
                  - want).max()
    if not miss < 1e-4:
        raise ValueError("fold2 at %.0f mm: the IK misses the feet by %.1f mm -- out "
                         "of reach" % (1e3 * height, 1e3 * miss))
    return q


class _Average:
    """One side's average: torque commanded and measured, where the leg sat."""

    def __init__(self) -> None:
        n = C.N_JOINTS
        self.n = 0
        self.tau = np.zeros(n)
        self.tau2 = np.zeros(n)
        self.iq = np.zeros(n)
        self.q = np.zeros(n)
        self.q_min = np.full(n, np.inf)
        self.q_max = np.full(n, -np.inf)
        self.cap = np.zeros(n, dtype=bool)

    def add(self, q, tau, iq, at_cap) -> None:
        self.n += 1
        self.tau += tau
        self.tau2 += tau * tau
        self.iq += iq
        self.q += q
        self.q_min = np.minimum(self.q_min, q)
        self.q_max = np.maximum(self.q_max, q)
        self.cap |= at_cap

    def result(self, settled: bool) -> dict:
        n = max(self.n, 1)
        tau = self.tau / n
        return dict(n=self.n, settled=bool(settled), q=self.q / n, tau=tau,
                    std=np.sqrt(np.maximum(self.tau2 / n - tau * tau, 0.0)),
                    iq=self.iq / n, cap=self.cap.copy(),
                    drift=(self.q_max - self.q_min if self.n else np.zeros(C.N_JOINTS)))


class GravityBench:
    """The phases, the plan, the PD and the averages.  No I/O in here."""

    def __init__(self, gate: SAFE.SafetyGate, targets, *, legs=range(C.N_LEGS),
                 offset_deg=OFFSET_DEG, speed_dps: float = SPEED_DPS,
                 transit_dps: float = TRANSIT_DPS, kp=None, kd=None,
                 settle_s: float = SETTLE_S, settle_deg: float = SETTLE_DEG,
                 average_s: float = AVERAGE_S, pose_s: float = POSE_S,
                 track_stop_deg: float = TRACK_STOP_DEG) -> None:
        self.gate = gate
        #: [(label, (12,) rad)], measured in this order.
        self.targets = [(str(label), C.flat(q).astype(float)) for label, q in targets]
        if not self.targets:
            raise ValueError("no target to measure")
        self.legs = np.zeros(C.N_LEGS, dtype=bool)
        self.legs[list(legs)] = True
        if not self.legs.any():
            raise ValueError("no leg to measure")
        #: (12,) the joints measured: all three of every leg in `legs`.  The
        #: other legs hold the target and never leave it.
        self.measured = np.repeat(self.legs, 3)
        offset = np.broadcast_to(np.radians(np.asarray(offset_deg, dtype=float)), (3,))
        if not np.all(offset > 0.0):
            raise ValueError("--offset %s deg: every joint needs a side to come from"
                             % np.degrees(offset))
        self.offset_deg = np.degrees(offset)    # (3,) abd pitch knee
        self.offset = np.where(self.measured, np.tile(offset, C.N_LEGS), 0.0)
        if not (speed_dps > 0.0 and transit_dps > 0.0 and average_s > 0.0
                and settle_s >= 0.0 and settle_deg > 0.0):
            raise ValueError("speeds, the average and --settle-deg must be > 0, "
                             "--settle >= 0")
        self.speed = np.radians(float(speed_dps))
        self.transit = np.radians(float(transit_dps))
        self.kp = np.tile(np.broadcast_to(np.asarray(
            KP if kp is None else kp, dtype=float), (3,)), C.N_LEGS)
        self.kd = np.tile(np.broadcast_to(np.asarray(
            KD if kd is None else kd, dtype=float), (3,)), C.N_LEGS)
        self.settle_s = float(settle_s)
        self.settle_tol = np.radians(float(settle_deg))
        self.average_s = float(average_s)
        self.pose_s = float(pose_s)
        self.track_stop = np.radians(float(track_stop_deg))
        # A SIDE'S START PAST A SOFT LIMIT IS REFUSED HERE, not met by the
        # gate's limit block halfway through an approach.
        for label, q in self.targets:
            for side in (-1, +1):
                start = q + side * self.offset
                out = (start < gate.low) | (start > gate.high)
                if out.any():
                    j = int(np.flatnonzero(out)[0])
                    raise ValueError(
                        "%s: %s from %s starts at %+.1f deg, past its soft limit "
                        "(%+.1f .. %+.1f) -- lower --offset"
                        % (label, HM.JOINT_LABELS[j], "-+"[side > 0],
                           np.degrees(start[j]), np.degrees(gate.low[j]),
                           np.degrees(gate.high[j])))
        self.phase = "limp"
        self.t_phase = 0.0
        self.q_from = None
        self.max_dps = None
        self.q_ref = None                       # (12,) the PD's reference
        self.qd_ref = np.zeros(C.N_JOINTS)
        self.tau = np.zeros(C.N_JOINTS)         # requested, before the gate
        self.steps: list[Step] = []
        self.k = 0
        self.t_step = 0.0
        self.passes = 0
        self.averaging = False                  # this sweep feeds an average
        self.anchor = None
        self.quiet_since = 0.0
        self.side_settled = False
        self._q0 = None                         # where the approach set off from
        self._avg = None
        self._sides: dict = {}
        self.results: list[dict] = []
        self._notices: list[str] = []
        self.trip = None

    # -- the plan -------------------------------------------------------------
    @property
    def approach_s(self) -> float:
        """Every measured joint covers its offset in this, so they arrive together."""
        return float(self.offset.max() / self.speed)

    def _transit_s(self, q_from, q_to) -> float:
        """A smoothstep's peak rate is 1.5 |dq| / T."""
        return max(0.2, 1.5 * float(np.abs(q_to - q_from).max()) / self.transit)

    def target_s(self) -> float:
        """Seconds a target takes if both its sides stop at the first chance."""
        side = DWELL_S + self.approach_s + max(self.settle_s, QUIET_S) + self.average_s
        out = float(self.offset.max())
        return 2.0 * side + sum(max(0.2, 1.5 * d / self.transit) for d in (out, 2.0 * out))

    def _plan(self, now: float) -> None:
        steps = []
        at = self.q_ref.copy()
        for n, (_, q) in enumerate(self.targets):
            for side in (-1, +1):
                start = q + side * self.offset
                steps += [Step("move", n, side, at, start, self._transit_s(at, start)),
                          Step("dwell", n, side, start, start, DWELL_S),
                          Step("approach", n, side, start, q, self.approach_s),
                          Step("settle", n, side, q, q, self.settle_s),
                          Step("average", n, side, q, q, self.average_s)]
                at = q
        self.steps = steps
        self.passes += 1
        self._sides = {}
        self._enter_step(0, now)

    def _enter_step(self, k: int, now: float) -> None:
        self.k = k
        self.t_step = float(now)
        self.anchor = None
        if k < len(self.steps) and self.steps[k].kind == "approach":
            self._q0 = None
        if k < len(self.steps) and self.steps[k].kind == "average":
            self._avg = _Average()

    def step(self):
        """The step running, or None outside a pass."""
        if self.phase != "measure" or self.k >= len(self.steps):
            return None
        return self.steps[self.k]

    # -- keys ---------------------------------------------------------------
    def enter(self, now: float, q) -> str:
        if self.phase == "limp":
            self.q_from = np.asarray(q, dtype=float).copy()
            self.max_dps = SEQ.ramp_motor_dps(self.q_from, self.targets[0][1], self.pose_s)
            self._go("pose", now)
            return ("POSE: position ramp to %s over %.1f s; the PD takes over by "
                    "itself when it arrives." % (self.targets[0][0], self.pose_s))
        if self.phase == "pose":
            return "ENTER ignored: the pose ramp is still running"
        if self.phase in ("hold", "done"):
            self._plan(now)
            self._go("measure", now)
            return ("MEASURE, pass %d: %s, from - then from +, ~%.0f s each.  X stops."
                    % (self.passes, ", ".join(label for label, _ in self.targets),
                       self.target_s()))
        return "ENTER ignored: measuring -- X stops"

    def _go(self, phase: str, now: float) -> None:
        self.phase = phase
        self.t_phase = float(now)

    def take_notices(self) -> list[str]:
        """What happened since the last sweep, oldest first: a table and the
        DONE after it can land in the same sweep."""
        notices, self._notices = self._notices, []
        return notices

    # -- the sweep ------------------------------------------------------------
    def update(self, now: float, q, qd):
        """One sweep.  Returns ``(mode, values)``: keepalive / position / torque."""
        q = np.asarray(q, dtype=float)
        if self.phase == "limp":
            return "keepalive", None
        if self.phase == "pose":
            elapsed = now - self.t_phase
            s = GAIT.smoothstep(elapsed / self.pose_s)
            q_des = self.q_from + s * (self.targets[0][1] - self.q_from)
            if elapsed >= self.pose_s + 0.5:
                # ARRIVED.  The PD's reference is the target the ramp went to,
                # so the switch asks for nothing until gravity pulls a leg off it.
                self.gate.start(now, q)
                self.q_ref = self.targets[0][1].copy()
                self._go("hold", now)
                self._notices.append("HOLD: torque mode, the PD alone at %s.  The gate "
                                     "ramps the cap over %.1f s.  ENTER measures."
                                     % (self.targets[0][0], self.gate.ramp_s))
            return "position", q_des
        if self.phase == "measure":
            self._advance(now, q)
        else:
            self.qd_ref = np.zeros(C.N_JOINTS)
            self.averaging = False
        err = np.abs(q - self.q_ref)
        if (self.track_stop > 0.0 and now - self.gate.started_at > self.gate.ramp_s
                and err.max() > self.track_stop):
            j = int(err.argmax())
            self.trip = ("%s is %.1f deg off its reference (limit %.0f) -- the PD is not "
                         "moving it" % (HM.JOINT_LABELS[j], np.degrees(err[j]),
                                        np.degrees(self.track_stop)))
        # `swing.joint_swing_torque`'s PD on all twelve joints, and nothing else.
        self.tau = self.kp * (self.q_ref - q) + self.kd * (self.qd_ref - np.asarray(qd, float))
        return "torque", self.tau.copy()

    def _advance(self, now: float, q) -> None:
        """Walk the plan to the step `now` is in: `q_ref`, `qd_ref`, `averaging`."""
        self.qd_ref = np.zeros(C.N_JOINTS)
        self.averaging = False
        while self.k < len(self.steps):
            st = self.steps[self.k]
            el = now - self.t_step
            self.q_ref = st.q_to.copy()
            if st.kind == "approach" and self._q0 is None:
                self._q0 = q.copy()
            if st.kind in ("move", "approach") and el < st.duration:
                u = el / st.duration
                d = st.q_to - st.q_from
                if st.kind == "move":            # smoothstep, 3u^2 - 2u^3
                    self.q_ref = st.q_from + GAIT.smoothstep(u) * d
                    self.qd_ref = 6.0 * u * (1.0 - u) * d / st.duration
                else:                            # the slow approach, one speed
                    self.q_ref = st.q_from + u * d
                    self.qd_ref = d / st.duration
                return
            if st.kind == "dwell" and el < st.duration:
                return
            if st.kind == "settle":
                if (self.anchor is None
                        or np.any(np.abs(q - self.anchor)[self.measured] > self.settle_tol)):
                    self.anchor = q.copy()
                    self.quiet_since = float(now)
                self.side_settled = el >= st.duration and now - self.quiet_since >= QUIET_S
                if not self.side_settled and el < st.duration + SETTLE_WAIT_S:
                    return
            if st.kind == "average":
                if el < st.duration:
                    self.averaging = True
                    return
                self._close_side(st)
            self._enter_step(self.k + 1, now)
        self._go("done", now)
        self._notices.append("DONE, pass %d.  Held at %s.  ENTER measures again; X "
                             "stops (the legs drop)." % (self.passes, self.targets[-1][0]))

    def _close_side(self, st: Step) -> None:
        side = self._avg.result(self.side_settled)
        # How far each joint came IN THE APPROACH'S DIRECTION, from where it
        # sat when the approach set off: from - is + travel, from + is -.
        side["travel"] = (side["q"] - self._q0) * -st.side
        self._sides[(st.target, st.side)] = side
        lo = self._sides.get((st.target, -1))
        hi = self._sides.get((st.target, +1))
        if lo is None or hi is None:
            return
        label, q = self.targets[st.target]
        res = {"label": label, "pass": self.passes, "target": q.copy(), "lo": lo, "hi": hi}
        self.results.append(res)
        self._notices.append(result_table(res, self.measured))

    def observe(self, now: float, q, tau_sent, tau_meas) -> None:
        """After the gate: the sweep into the running average, if one is."""
        if not self.averaging:
            return
        tau_sent = np.asarray(tau_sent, dtype=float)
        self._avg.add(np.asarray(q, dtype=float), tau_sent,
                      np.asarray(tau_meas, dtype=float),
                      np.abs(tau_sent) >= 0.98 * self.gate.cap_now(now))

    def status(self, now: float, q) -> str:
        head = "   %-7s t=%5.1f" % (self.phase, now - self.t_phase)
        st = self.step()
        if st is not None:
            head += "  %s from %s: %-8s %4.1f s" % (self.targets[st.target][0],
                                                    "-+"[st.side > 0], st.kind,
                                                    now - self.t_step)
        if self.q_ref is not None:
            err = np.degrees(np.abs(np.asarray(q, dtype=float) - self.q_ref))
            j = int(err.argmax())
            head += "  |q - ref| %.2f deg (%s)" % (err[j], HM.JOINT_LABELS[j])
        return head


def result_table(res: dict, measured) -> str:
    """One target's two sides: G and the friction band per measured joint."""
    lo, hi = res["lo"], res["hi"]
    q = 0.5 * (lo["q"] + hi["q"])
    g = 0.5 * (lo["tau"] + hi["tau"])
    band = 0.5 * (lo["tau"] - hi["tau"])
    g_iq = 0.5 * (lo["iq"] + hi["iq"])
    band_iq = 0.5 * (lo["iq"] - hi["iq"])
    model = C.flat(TRQ.all_leg_gravity_torque(q))
    unsettled = [tag for tag, side in (("-", lo), ("+", hi)) if not side["settled"]]
    out = ["G(q) at %s, pass %d -- N*m, joint frame, %d + %d sweeps averaged%s"
           % (res["label"], res["pass"], lo["n"], hi["n"],
              "  -- NEVER STOPPED from %s: averaged anyway" % " and ".join(unsettled)
              if unsettled else ""),
           "  %-9s %7s   %7s %7s | %7s %6s | %7s %6s | %7s %5s"
           % ("joint", "q deg", "from -", "from +", "G", "tau_s", "G iq", "ts iq",
              "model", "ratio")]
    for j in np.flatnonzero(measured):
        flags = ["STUCK %s (came in %.1f deg)" % (tag, np.degrees(side["travel"][j]))
                 for tag, side in (("-", lo), ("+", hi))
                 if side["travel"][j] < np.radians(MIN_TRAVEL_DEG)]
        flags += ["CAP %s" % tag for tag, side in (("-", lo), ("+", hi)) if side["cap"][j]]
        flags += ["moved %.1f deg %s" % (np.degrees(side["drift"][j]), tag)
                  for tag, side in (("-", lo), ("+", hi))
                  if side["drift"][j] > np.radians(MOVED_DEG)]
        ratio = ("%5.2f" % (g[j] / model[j]) if abs(model[j]) >= RATIO_FLOOR
                 else "%5s" % "-")
        out.append("  %-9s %+7.1f   %+7.3f %+7.3f | %+7.3f %6.3f | %+7.3f %6.3f | "
                   "%+7.3f %s  %s"
                   % (HM.JOINT_LABELS[j], np.degrees(q[j]), lo["tau"][j], hi["tau"][j],
                      g[j], band[j], g_iq[j], band_iq[j], model[j], ratio,
                      ", ".join(flags)))
    return "\n".join(out)


class GravityLog:
    """Every torque-mode sweep, and the tables, to one npz."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)

    def save(self, path: str, bench: GravityBench) -> str:
        if not self.rows:
            return "nothing to save"
        out = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}
        res = bench.results
        if res:
            out["res_label"] = np.array([r["label"] for r in res])
            out["res_pass"] = np.array([r["pass"] for r in res])
            out["res_target"] = np.array([r["target"] for r in res])
            for key in ("q", "tau", "iq", "std", "drift", "cap", "travel"):
                out["res_" + key] = np.array([[r["lo"][key], r["hi"][key]] for r in res])
            out["res_settled"] = np.array([[r["lo"]["settled"], r["hi"]["settled"]]
                                           for r in res])
        out.update(measured=bench.measured, kp=bench.kp, kd=bench.kd,
                   offset=bench.offset, speed=bench.speed, average_s=bench.average_s)
        np.savez_compressed(path, **out)
        return "%d sweeps, %d tables -> %s" % (len(self.rows), len(res), path)


def _print_phase(bench: GravityBench) -> None:
    print("\n>> %-7s : %s" % (bench.phase, PHASE_BLURB[bench.phase]), flush=True)


def run(mb, bench: GravityBench, *, rate_hz: float = RATE_HZ, key=None,
        auto_s: float | None = None, clock=time.perf_counter,
        log: "GravityLog | None" = None,
        gcq: "RT.QuietGC | None" = None) -> str | None:
    """Drive `bench` on an ARMED `mb` until X, a trip, or (`auto_s`) done.

    `hw.swing_bench.run`'s loop: the same round robin, latched sweep, gate,
    monitors, `realtime.SendGuard` and stop lines, the collector in its own
    slot with `gcq`.  Does not stop the motors -- the caller's `with
    MotorBus(...)` does.
    """
    ids = HM.motor_ids()
    n = len(ids)
    unwrappers = CAL.new_unwrappers()
    misses = SAFE.CanMissMonitor(mb)
    readback = SAFE.TorqueReadback(n)
    slot = mb.slot(rate_hz)
    alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * n * slot)
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
            return ("CAN link lost: %s -- the adapter left USB or can0 went "
                    "down under the run" % failure)
        if k == 0:
            now = clock()
            try:
                q, qd_driver = CAL.joint_state(mb, unwrappers)
            except RuntimeError as missing:
                if now - t0 > 1.0:
                    return str(missing)
                k = (k + 1) % n
                # Through the guard like every other frame, or its clock for
                # this motor stops at arm() and the first real slot trips.
                guard.sent(0, mb.keepalive(ids[0]), clock())
                deadline = clock() + slot
                continue
            if q_prev is not None and now > t_prev:
                qd_ctrl += alpha_qd * ((q - q_prev) / (now - t_prev) - qd_ctrl)
            q_prev, t_prev = q, now

            pressed = key.get() if key is not None else None
            if pressed in ("x", "X"):
                return "operator X"
            if pressed in ("\r", "\n"):
                print("\n   " + bench.enter(now, q), flush=True)
            if auto_s is not None and now - bench.t_phase >= auto_s:
                if bench.phase in ("limp", "hold"):
                    print("\n   [auto] " + bench.enter(now, q), flush=True)
                elif bench.phase == "done":
                    return None

            mode, values = bench.update(now, q, qd_ctrl)
            if bench.trip:
                return bench.trip
            for notice in bench.take_notices():
                print("\n" + notice if "\n" in notice else "\n   " + notice, flush=True)
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
                bench.observe(now, q, tau_sent, tau_meas)
                reason = readback.reason(tau_sent, tau_meas, live=True)
                if reason:
                    return reason
                if log is not None:
                    st = bench.step()
                    log.add(t=now - t0, phase=bench.phase,
                            step="" if st is None else st.kind,
                            target=-1 if st is None else st.target,
                            side=0 if st is None else st.side,
                            averaging=bench.averaging, q=q, qd=qd_ctrl,
                            q_ref=bench.q_ref.copy(), qd_ref=bench.qd_ref.copy(),
                            tau_req=bench.tau.copy(), tau_cmd=tau_sent.copy(),
                            tau_meas=tau_meas,
                            errors=np.asarray([errors.get(mid, 0) or 0 for mid in ids]),
                            temps=temps, cap=bench.gate.cap_now(now))
            else:
                tau_sent = np.zeros(n)

            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                print(bench.status(now, q)
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
        prog="python -m hw.gravity_bench",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--height", type=float, nargs="+", default=[1e3 * FOLD2_HEIGHT],
                    metavar="MM",
                    help="the fold2 hold(s) to measure at, floor to trunk bottom, in "
                         "turn; %.0f is hw.fold2_trot's stand" % (1e3 * FOLD2_HEIGHT))
    ap.add_argument("--legs", nargs="+", default=list(C.LEGS), choices=list(C.LEGS),
                    help="the legs measured; the others hold the target")
    ap.add_argument("--offset", type=float, nargs="+", default=[OFFSET_DEG], metavar="DEG",
                    help="how far out each side starts: one value for all three "
                         "joints, or abd pitch knee")
    ap.add_argument("--speed", type=float, default=SPEED_DPS, metavar="DEG_PER_S",
                    help="the approach into the target")
    ap.add_argument("--transit", type=float, default=TRANSIT_DPS, metavar="DEG_PER_S",
                    help="peak speed between approaches")
    ap.add_argument("--settle", type=float, default=SETTLE_S, metavar="SECONDS",
                    help="the least wait at the target before the average")
    ap.add_argument("--settle-deg", type=float, default=SETTLE_DEG, metavar="DEG",
                    help="stopped = every measured joint within this for %.1f s"
                         % QUIET_S)
    ap.add_argument("--average", type=float, default=AVERAGE_S, metavar="SECONDS")
    ap.add_argument("--kp", type=float, nargs=3, default=[float(v) for v in KP],
                    metavar="NM_PER_RAD",
                    help="the PD, abd pitch knee: the joint layer's, what the trots fly")
    ap.add_argument("--kd", type=float, nargs=3, default=[float(v) for v in KD],
                    metavar="NMS_PER_RAD")
    ap.add_argument("--tau-cap", type=float, default=TAU_CAP,
                    help="torque cap, N*m; the ceiling is %.0f" % SAFE.TAU_STAGED_MAX)
    ap.add_argument("--tau-slew", type=float, default=BCFG.TAU_SLEW_TROT_NM_S,
                    metavar="NM_PER_S", help="the gate's |dtau/dt| limit -- the trot's")
    ap.add_argument("--track-stop", type=float, default=TRACK_STOP_DEG, metavar="DEG",
                    help="TRIP: a joint this far off its reference stops the run; "
                         "0 turns it off (off under --fake)")
    ap.add_argument("--pose-s", type=float, default=POSE_S,
                    help="the position ramp from the hang to the first target, s")
    ap.add_argument("--rate", type=float, default=RATE_HZ)
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--arm-timeout", type=float, default=30.0)
    ap.add_argument("--unconfirmed", default=None, metavar="WHY")
    ap.add_argument("--fake", action="store_true",
                    help="hw.fake_bus: no dynamics, so every torque reads ~0")
    ap.add_argument("--auto", type=float, default=None, metavar="SECONDS",
                    help="press ENTER by itself this long into limp and hold, and "
                         "exit when done (--fake only)")
    ap.add_argument("--log", default=None, metavar="FILE.npz")
    args = ap.parse_args(argv)
    if args.auto is not None and not args.fake:
        ap.error("--auto is only allowed with --fake")
    if len(args.offset) not in (1, 3):
        ap.error("--offset takes one value, or three: abd pitch knee")

    try:
        targets = [("fold2 %.0f mm" % h, fold2_pose(1e-3 * h)) for h in args.height]
    except ValueError as refusal:
        print("[gravity] REFUSED before opening the bus:\n  %s" % refusal, file=sys.stderr)
        return 2
    reason = args.unconfirmed or ("hw.fake_bus" if args.fake else None)
    ids = HM.motor_ids()
    try:
        gate = SAFE.SafetyGate(args.tau_cap, unconfirmed_reason=reason,
                               tau_slew=args.tau_slew)
        bench = GravityBench(gate, targets, legs=[C.LEGS.index(name) for name in args.legs],
                             offset_deg=args.offset, speed_dps=args.speed,
                             transit_dps=args.transit, kp=args.kp, kd=args.kd,
                             settle_s=args.settle, settle_deg=args.settle_deg,
                             average_s=args.average, pose_s=args.pose_s,
                             track_stop_deg=0.0 if args.fake else args.track_stop)
    except (RuntimeError, ValueError) as refusal:
        print("[gravity] REFUSED before opening the bus:\n  %s" % refusal, file=sys.stderr)
        return 2

    # -- the banner ---------------------------------------------------------------
    print("DOG6 gravity bench on %s -- HANG THE ROBOT UP, LEVEL: nothing here holds it"
          % ("hw.fake_bus" if args.fake else "can0"))
    print("  CAN ids %s  directions %s  map confirmed: %s" % (ids, HM.directions(),
                                                             CONFIRMED_ON_DOG6))
    print("  cap %.1f N*m, slew %.0f N*m/s, track-stop %s"
          % (args.tau_cap, args.tau_slew,
             "OFF" if bench.track_stop <= 0 else "%.0f deg" % args.track_stop))
    print("  overspeed trip ON, the gate's: driver > %.1f rad/s alone, or > %.1f with the "
          "encoder agreeing" % (gate.qd_estop_hard, gate.qd_estop))
    print("  PD Kp %s N*m/rad  Kd %s N*m*s/rad (abd pitch knee), no gravity feedforward"
          % (np.asarray(args.kp), np.asarray(args.kd)))
    print("  each side %s deg out (abd/pitch/knee), in at %.1f deg/s; stopped = within "
          "%.2f deg for %.1f s, at least %.1f s; %.1f s average.  ~%.0f s a target"
          % ("/".join("%.1f" % v for v in bench.offset_deg), args.speed,
             args.settle_deg, QUIET_S, args.settle, args.average, bench.target_s()))
    print("  legs measured: %s" % " ".join(args.legs))
    for label, q in bench.targets:
        q4 = C.unflat(q)
        g4 = TRQ.all_leg_gravity_torque(q4)
        print("  %s:  %s deg" % (label, "  ".join(
            "%s %+.1f/%+.1f/%+.1f" % (C.LEGS[i], *np.degrees(q4[i])) for i in range(C.N_LEGS))))
        print("  %s   model G %s N*m (trunk level)" % (" " * len(label), "  ".join(
            "%s %+.3f/%+.3f/%+.3f" % (C.LEGS[i], *g4[i]) for i in range(C.N_LEGS))))
    print("  ENTER: limp -> pose; hold -> measure; done -> again.  X: STOP, the legs drop.")

    from .motor import motorbus
    if args.fake:
        from .fake_bus import FakeDriverBus
        mb = motorbus.MotorBus(ids, bus=FakeDriverBus(ids=ids), dirs=HM.motor_directions())
    else:
        # `hw.stand.main`'s order: the process to nice -20, then can0 made
        # clean by `hw.can_link`.  Neither for --fake.
        print("  " + RT.boost())
        try:
            mb = CANL.open_motor_bus(ids, bitrate=args.bitrate,
                                     dirs=HM.motor_directions())
        except CANL.LinkError as refusal:
            print("[gravity] REFUSED before arming -- CAN: %s" % refusal,
                  file=sys.stderr)
            return 2
    since = CANL.uptime()
    key = KeyPoller()
    if not key.ok and args.auto is None:
        print("[gravity] stdin is not a terminal: ENTER and X will not work",
              file=sys.stderr)
    log = GravityLog() if args.log else None
    stop = None
    gcq = RT.QuietGC()
    try:
        with mb:
            print("  " + gcq.start())
            if not mb.arm(rate_hz=args.rate, timeout_s=args.arm_timeout):
                print("[gravity] not every motor armed", file=sys.stderr)
                if not args.fake:
                    print("[can] %s" % CANL.after_failed_arm(mb), file=sys.stderr)
                return 1
            stop = run(mb, bench, rate_hz=args.rate, key=key, auto_s=args.auto,
                       log=log, gcq=gcq)
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        key.restore()
        gc_said = gcq.stop()

    print()
    if gc_said:
        print("[gravity] " + gc_said)
    if not args.fake and stop not in (None, "operator X", "Ctrl-C"):
        print("[can] at the stop: %s"
              % CANL.describe(getattr(mb, "at_stop", None)))
        for line in CANL.dmesg_since(since):
            print("[can] dmesg: " + line)
    if gate.started_at is not None:
        # `hw.stand`'s and `hw.swing_bench`'s: what an overspeed stop needs read
        # beside it -- did the encoder see what the driver's field said.
        print("[gravity] peak joint speed, driver / encoder (rad/s; trip at %.1f driver "
              "alone, %.1f confirmed):" % (gate.qd_estop_hard, gate.qd_estop))
        for leg in range(C.N_LEGS):
            row = slice(3 * leg, 3 * leg + 3)
            print("  %s  %s" % (C.LEGS[leg], "   ".join(
                "%4.1f / %4.1f" % pair for pair in zip(
                    gate.qd_peak[row], gate.encoder_qd_peak[row]))))
    print("[gravity] results:")
    if bench.results:
        for res in bench.results:
            print(result_table(res, bench.measured))
    else:
        print("  no target finished both sides")
    if log is not None:
        print("[gravity] log: %s" % log.save(args.log, bench))
    if stop:
        print("[gravity] stopped: %s.  Motors stopped; the legs hang." % stop)
        return 0 if stop in ("operator X", "Ctrl-C") else 3
    print("[gravity] done; motors stopped, the legs hang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
