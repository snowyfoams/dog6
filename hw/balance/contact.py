"""Touchdown detection, SCHEME A: the swing law's own commanded push on the
floor, gated by the foot's clearance.  No sensor, no new hardware, no new
solve -- every number it reads the law already has.

    ContactDetector.update(now, state, clock, gait) -> GaitSample
        the clock's sample, OVERRIDDEN leg by leg: contact, weight and swing
        progress as the detector believes them.  Call it where the law
        samples the clock, BEFORE the height is taken over the stance feet.
    ContactDetector.observe(leg, force_b, v_ref)
        what the swing law commanded for a swinging (or probing) leg this
        sweep: the foot force, TRUNK frame, and the reference velocity.  Read
        on the NEXT sweep's `update` -- one sweep late, as the estimator is.
    ContactDetector.probe(leg, p, v, a, now) -> (p, v, a)
        the reference of a leg that is PROBING: the arc's end, carried down
        at PROBE_V until the floor answers.
    ContactDetector.stance_phase(now, gait_phase, duty) -> (4,)
        the state estimator's contact phase, with what this knows in it.
    ContactDetector.snapshot() / status(now) / report()

WHY THE DRIVERS' CURRENT CANNOT SEE THE FLOOR, AND WHAT CAN
    Every torque phase runs the drivers in q-axis current control (0xA1).
    The current loop makes iq follow the COMMAND whatever the load: a foot
    that meets the floor changes the leg's motion, not its current.  So the
    floor is not in `tau_meas`; `safety.TorqueReadback` reads that field as
    "did the current loop follow", and `gravity_bench` prints it beside the
    command for the same reason.  Where the floor IS: the encoders.  The foot
    stops, the reference goes on, and the swing impedance's error -- and
    with it the force it commands -- grows.  That force is what this reads.
    In quasi-statics the commanded force is the force on the floor, less
    joint friction (0.1-0.3 N*m a joint, 1-3 N at the foot) and less the
    leg's inertia while it is still moving -- which is why the foot's own
    motion is the second witness below.

HOW A TOUCHDOWN IS DECIDED -- three conditions, all true, STREAK sweeps running
    1. THE LEG IS COMING DOWN: swing progress s >= S_MIN, or the leg is
       probing (below).  The liftoff half of the arc pulls the foot UP off
       the floor and its commanded force points up; nothing there is a
       touchdown, and the test does not run.
    2. THE FLOOR IS WHERE THE FOOT IS, AND WHERE THE REFERENCE IS: the
       foot's clearance above the plane of the feet that are down, from the
       kinematics and the IMU --
           z_clear = z_origin - FOOT_RADIUS + (R x_b)_z
       (`hw.stand.SwingHeight`'s number) -- is within CLEAR_MAX, AND the
       swing law's own reference, measured the same way, has come within
       REF_CLEAR_MAX of that plane or gone below it.  The second half is
       the gate that tells contact from LAG: the floor can only be what
       stops a foot once the reference has reached the floor.  On the robot
       the walk's swing runs 15 mm (median) and 27 mm (p90) off its arc,
       and at Kp_z 400 N/m 27 mm of lag IS 11 N of commanded force -- over
       the threshold below; and in MuJoCo (doc/contact) the foot reaches
       7-9 mm of a 20 mm arc, 33-40 ms behind it, so a foot turning round
       at its low apex under a reference already diving read as stopped
       and pushed, 70 ms before it touched, until this gate was added.
       Where the reference crosses the plane while still descending -- the
       low side of a rolled trunk, whose trunk-frame arc ends below the
       floor -- an early landing is caught; where it never does -- the
       high side -- none is believed, and the probe takes it below.
    3. THE FLOOR IS ANSWERING, by either witness:
         push     the swing law commands a push DOWN on the floor of FZ_TD
                  or more:  fz_push = -[Kp (p_ref - x) + Kd (v_ref - v)]_z.
                  A foot stopped on the floor with the reference still
                  descending at 0.3-0.4 m/s gets 12-16 N from Kd_z 40 at
                  once; the Kp term adds 4 N per 10 mm as the arc goes on
                  below it.
         stopped  the foot has stopped descending (v_z >= -V_STOP, the
                  encoders' 40 Hz-filtered rate through the Jacobian) while
                  the reference still descends (v_ref,z <= -VREF_MIN).  The
                  arc ends with zero speed, so this one fires mid-descent --
                  an early landing -- and the push fires late in the arc and
                  under the probe.
    The streak (STREAK consecutive sweeps, 8 ms) is the debounce.  The
    plane is the two LOWEST planted feet's, so a planted foot that has left
    the floor cannot lift it.

A LANDED FOOT THAT LEAVES THE FLOOR
    A foot the detector has down whose clearance rises past LIFT_MAX (8 mm,
    STREAK sweeps running) while the clock still has it in stance has
    lifted off -- the trunk is tipping away from it, or what it stood on
    gave -- and goes back to PROBING from where it is: out of the
    allocator, its reference its own x/y carried down at PROBE_V until the
    floor answers again.  Found in MuJoCo (doc/contact): a sideways push
    rolled the trunk, the high side's planted foot rose 10 mm with 0 N under
    it and, still "landed", pulled the plane up with it; the other foot on
    that side was then believed landed 8 mm in the air, and the next
    liftoff left the robot on one foot.  Whether a foot that is DOWN is
    still carrying load is the measured torque's to tell (scheme B, J^-T
    (tau_meas - G), which this is written to be joined by); this catches
    the lift-off the kinematics can see.

HOW "STILL AIRBORNE" IS DECIDED, AND WHAT IS DONE ABOUT IT
    The clock's touchdown arrives and none of the above has fired: the leg
    is PROBING.  It stays OUT of the allocator (weight 0, contact False, so
    no force is asked of a foot that is in the air -- which on 2026-10-06
    rolled the trunk +0.9 -> +7.6 deg in 80 ms and ended the walk), its
    height is not averaged into the trunk's, its reference is the arc's END
    carried straight down at PROBE_V (the foot keeps its planned x/y,
    world-fixed when walking), and the leg is PUSHED down along gravity by
    PROBE_PUSH, open loop, beside the impedance -- the clock's "load it
    anyway" made deliberate, without the allocator counting on a foot that
    is not there (MuJoCo found the need: the impedance alone, 16 N at most
    in its first 100 ms, let the trunk drop on the side whose feet were
    still coming down).  The impedance's Kd brakes the foot against the
    reference, so the push settles it at 0.2 + 10/40 = 0.45 m/s, the arc's
    own peak descent; and the push is NOT in what `observe` reads, so it
    cannot call its own touchdown.  On a stopped foot the Kd term alone
    gives Kd_z PROBE_V = 8 N the instant the foot stops, and the Kp term
    grows at Kp_z PROBE_V = 80 N/s.  The floor answers, condition 3 fires,
    the leg lands and is loaded
    over RAMP_S.  Past PROBE_MAX of descent with no answer the leg is
    declared landed anyway (the clock's own assumption, counted as `gave up`)
    so that nothing probes forever; and a probe ends when the four-foot
    window does -- once the other diagonal is halfway through its liftoff
    ramp on the clock the probing foot is declared down, its load ramping
    IN while theirs ramps OUT, so the allocator is never left with one foot
    (the window is 120 ms at duty 0.70; the probe gets 93 of them, 19 mm).

WHAT CHANGES FOR THE LAW, leg by leg (`update`'s sample)
    state     contact   weight                              swing progress
    TRUST     clock's   clock's                             clock's
    SWING     False     0                                   clock's
    LANDED    True      smoothstep((now - t_td)/RAMP_S),    0
                        and the clock's liftoff ramp once
                        the clock has it in stance
    PROBING   False     0                                   1.0 (the arc's end)
    TRUST is a leg the clock has never lifted since the trot started (the
    all-four entry): nothing is known beyond the clock, nothing is changed.
    A LANDED leg's weight is never stepped: it ramps in from the detection
    whether that came early or late, and ramps out on the clock as before.

WHAT IT COSTS
    Four dot products, four scalar velocities, no Jacobian inverse, no new
    allocation: ~20 us a sweep in slot 0.  The probe adds a subtraction.
    The force it reads is the swing law's request BEFORE the safety gate,
    which the gate may slew or clip -- so the request can lead the floor's
    real force by a sweep or two.  For a touchdown that is the right side
    to err on.

THRESHOLDS -- `config.TD_*`, [SIM-TUNED, NOT FLOWN]
    sized on the robot's numbers: 51.1 N of weight (12.8 N a foot on four,
    25.5 on a diagonal), 1-3 N of friction at the foot, Kp_z 400 / Kd_z 40,
    15 / 27 mm of swing tracking lag; the two clearance gates on MuJoCo's
    hovering swing (doc/contact/README.md).  `python -m hw.balance.contact`
    prints them.  The runner's `--td-*` flags move them per run.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/contact.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from . import config as cfg          # noqa: E402
from .gait import GaitSample, smoothstep   # noqa: E402

__all__ = ["ContactDetector", "ContactSnapshot", "TRUST", "SWING", "LANDED",
           "PROBING", "STATE_NAMES"]

#: The per-leg states, as logged (`ContactSnapshot.state`).
TRUST, SWING, LANDED, PROBING = 0, 1, 2, 3
STATE_NAMES = ("trust", "swing", "landed", "probe")


class ContactSnapshot(NamedTuple):
    """One sweep's read-out, for the log and the status line."""

    state: np.ndarray        # (4,) int: TRUST / SWING / LANDED / PROBING
    fz_push: np.ndarray      # (4,) N, the push the swing law commanded; NaN
                             #       for a leg it did not swing
    z_clear: np.ndarray      # (4,) m, each foot above the planted feet's plane
    z_ref: np.ndarray        # (4,) m, the swing reference above it; NaN for a
                             #       leg the law did not swing
    probe: np.ndarray        # (4,) m, how far the probe has carried the
                             #       reference below the arc's end; 0 not probing
    lead: np.ndarray         # (4,) s, the last detection against the clock's
                             #       touchdown: + early, - late; NaN none yet


class ContactDetector:
    """Scheme A.  One per law; `reset` at each trot, `idle` outside one."""

    def __init__(self, *, fz_td: float = cfg.TD_FZ_N,
                 streak: int = cfg.TD_STREAK,
                 s_min: float = cfg.TD_S_MIN,
                 clear_max: float = cfg.TD_CLEAR_MAX,
                 lift_max: float = cfg.TD_LIFT_MAX,
                 ref_clear_max: float = cfg.TD_REF_CLEAR_MAX,
                 v_stop: float = cfg.TD_V_STOP,
                 vref_min: float = cfg.TD_VREF_MIN,
                 probe_v: float = cfg.TD_PROBE_V,
                 probe_max: float = cfg.TD_PROBE_MAX,
                 probe_push: float = cfg.TD_PROBE_PUSH_N,
                 ramp_s: float = cfg.TD_RAMP_S):
        if not fz_td > 0.0:
            raise ValueError("fz_td must be positive, got %r" % fz_td)
        if int(streak) != streak or streak < 1:
            raise ValueError("streak must be a whole number >= 1, got %r"
                             % streak)
        if not 0.0 <= s_min < 1.0:
            raise ValueError("s_min must lie in [0, 1), got %r" % s_min)
        if not clear_max > 0.0:
            raise ValueError("clear_max must be positive, got %r" % clear_max)
        if not lift_max > clear_max:
            raise ValueError("lift_max %r must be above clear_max %r"
                             % (lift_max, clear_max))
        if not probe_v >= 0.0 or not probe_max >= 0.0:
            raise ValueError("probe_v and probe_max must be >= 0")
        if not probe_push >= 0.0:
            raise ValueError("probe_push must be >= 0, got %r" % probe_push)
        if not ramp_s > 0.0:
            raise ValueError("ramp_s must be positive, got %r" % ramp_s)
        self.fz_td = float(fz_td)
        self.streak_n = int(streak)
        self.s_min = float(s_min)
        self.clear_max = float(clear_max)
        self.lift_max = float(lift_max)
        self.ref_clear_max = float(ref_clear_max)
        self.v_stop = float(v_stop)
        self.vref_min = float(vref_min)
        self.probe_v = float(probe_v)
        self.probe_max = float(probe_max)
        #: N, the open-loop push down along gravity the law adds to a
        #: probing leg (`probe_force`); not in what `observe` reads.
        self.probe_push = float(probe_push)
        self.ramp_s = float(ramp_s)
        n = C.N_LEGS
        self.active = False
        self.state = np.zeros(n, dtype=int)
        #: (4,) the EFFECTIVE contact set of the last `update` -- the floor
        #: the next sweep's clearance is measured against.
        self.contact = np.ones(n, dtype=bool)
        self._clock_prev = np.ones(n, dtype=bool)
        self.streak = np.zeros(n, dtype=int)
        self.t_td = np.full(n, np.nan)       # s, when each leg LANDED
        self.t_probe0 = np.full(n, np.nan)   # s, when each probe started
        self.gave_up = np.zeros(n, dtype=bool)
        self.lead = np.full(n, np.nan)
        self.how = [""] * n                  # "push" / "stop" / "limit" / "window"
        #: (4, 3) trunk frame: where a foot that LIFTED OFF was when it did,
        #: the x/y its re-probe keeps; NaN otherwise.
        self._relatch = np.full((n, 3), np.nan)
        self._lift_streak = np.zeros(n, dtype=int)
        self._fz_obs = np.full(n, np.nan)
        self._vref_obs = np.full(n, np.nan)
        self._pref_obs = np.full((n, 3), np.nan)
        self.fz_push = np.full(n, np.nan)
        self.z_clear = np.full(n, np.nan)
        self.z_ref = np.full(n, np.nan)      # the reference above the plane
        self.probe_depth = np.zeros(n)
        # -- the account, for the report ----------------------------------
        self.swings = 0                      # clock liftoffs seen
        self.early: list[float] = []         # s, detections before the clock
        self.late: list[float] = []          # s, after it (probing)
        self.probes = 0
        self.limit = 0                       # probes that hit PROBE_MAX
        self.window = 0                      # probes cut by the two-foot rule
        self.lost = 0                        # lifted again, never detected
        self.lifted = 0                      # landed feet that left the floor
        self.by_push = 0
        self.by_stop = 0
        self.probe_at_td: list[float] = []   # m, depth when a probe landed
        self.trots = 0
        #: s, the clock's stance as last seen by `update` (`stance_phase`).
        self._t_stance = 1.0

    # -- lifecycle ----------------------------------------------------------
    def reset(self, clock: GaitSample) -> None:
        """A trot starts: every leg is the clock's, nothing is known yet."""
        self.active = True
        self.trots += 1
        self.state[:] = TRUST
        self.contact = np.asarray(clock.contact, dtype=bool).copy()
        self._clock_prev = self.contact.copy()
        self.streak[:] = 0
        self.t_td[:] = np.nan
        self.t_probe0[:] = np.nan
        self.gave_up[:] = False
        self.probe_depth[:] = 0.0
        self._relatch[:] = np.nan
        self._lift_streak[:] = 0
        self._fz_obs[:] = np.nan
        self._vref_obs[:] = np.nan
        self._pref_obs[:] = np.nan
        self.fz_push[:] = np.nan
        self.z_clear[:] = np.nan
        self.z_ref[:] = np.nan

    def idle(self) -> None:
        """No trot this sweep (a stand, the HOLD): forget the trot's state.
        The next `update` starts afresh from the clock."""
        if self.active:
            self.active = False
            self.state[:] = TRUST
            self.contact[:] = True
            self.probe_depth[:] = 0.0
            self._relatch[:] = np.nan
            self._lift_streak[:] = 0
            self._fz_obs[:] = np.nan
            self._vref_obs[:] = np.nan
            self._pref_obs[:] = np.nan

    # -- what the law tells it ---------------------------------------------
    def observe(self, leg: int, force_b, v_ref, p_ref) -> None:
        """The swing law's commanded foot force (3,), TRUNK frame -- what
        the leg applies to the foot, so a push on the floor is NEGATIVE z --
        and the reference it was tracking, velocity and position (3,).
        Read next sweep."""
        self._fz_obs[leg] = -float(np.asarray(force_b, dtype=float)[2])
        self._vref_obs[leg] = float(np.asarray(v_ref, dtype=float)[2])
        self._pref_obs[leg] = np.asarray(p_ref, dtype=float).reshape(3)

    @property
    def probing(self) -> np.ndarray:
        """(4,) bool: the legs whose reference is the probe's."""
        return self.state == PROBING

    def depth(self, leg: int, now: float) -> float:
        """m, how far the probe has carried `leg`'s reference below the
        arc's end at `now`: PROBE_V x (time probing), capped at PROBE_MAX."""
        t0 = self.t_probe0[leg]
        if not math.isfinite(t0):
            return 0.0
        return min(self.probe_max, self.probe_v * max(0.0, float(now) - t0))

    def probe(self, leg: int, p, v, a, now: float):
        """The reference of a PROBING leg: `(p, v, a)` -- the arc's end,
        TRUNK frame -- carried down by `depth` at PROBE_V.  No step on the
        sweep the probe starts: the depth is zero there."""
        d = self.depth(leg, now)
        p = np.asarray(p, dtype=float).copy()
        v = np.asarray(v, dtype=float).copy()
        a = np.asarray(a, dtype=float)
        if np.isfinite(self._relatch[leg, 0]):
            # A foot that LIFTED OFF re-probes straight down from where it
            # left: its own x/y, no arc, no foothold.
            p[:2] = self._relatch[leg, :2]
            v[:2] = 0.0
            a = np.zeros(3)
        p[2] -= d
        if d < self.probe_max:
            v[2] -= self.probe_v
        return p, v, a

    def probe_force(self, down_b) -> np.ndarray:
        """(3,) TRUNK frame: the probe's own push on the foot, PROBE_PUSH
        along `down_b`, the unit vector of gravity in the trunk frame
        (`torque.gravity_direction_body(R)`).  The law adds J^T of it to
        every probing leg, open loop, beside the impedance -- and reports
        the impedance's force alone to `observe`, so this push can never
        call its own touchdown."""
        return self.probe_push * np.asarray(down_b, dtype=float)

    # -- the sweep ---------------------------------------------------------
    def update(self, now: float, state, clock: GaitSample, gait) -> GaitSample:
        """The effective sample for this sweep.  `state` is the sweep's
        `state.BodyState` as read -- BEFORE `on_stance`, which should then be
        taken over the sample's contact set -- and `clock` is `gait.sample(now)`.
        """
        now = float(now)
        if not self.active:
            self.reset(clock)
        contact_c = np.asarray(clock.contact, dtype=bool)
        swing_s = np.asarray(clock.swing_s, dtype=float)
        t_swing = float(gait.swing_duration)
        self._t_stance = max(1e-9, float(gait.stance_duration))

        # 1. the floor: the plane of the feet the last sweep had down -- the
        # two LOWEST of them, so a planted foot that has left the floor
        # (less far below the trunk) cannot lift the plane with it.
        zb = state.x_b @ state.R[2]                      # (4,) (R x_b)_z
        planted = self.contact if self.contact.any() else contact_c
        if not planted.any():
            planted = np.ones(C.N_LEGS, dtype=bool)
        z_pl = np.sort(zb[planted])
        z_origin = P.FOOT_RADIUS - float(z_pl[:2].mean())
        z_clear = z_origin - P.FOOT_RADIUS + zb
        # 2. each foot's z speed, trunk frame: J_i qd_i, the z row only.
        qd4 = C.unflat(state.qd)
        v_z = np.einsum("ij,ij->i", state.jac[:, 2, :], qd4)
        # 3. last sweep's swing law, consumed -- and where its reference
        # stood above the same plane (the gate that tells contact from lag).
        fz = self._fz_obs.copy()
        vref = self._vref_obs.copy()
        z_ref = z_origin - P.FOOT_RADIUS + self._pref_obs @ state.R[2]
        self._fz_obs[:] = np.nan
        self._vref_obs[:] = np.nan
        self._pref_obs[:] = np.nan

        # 4. the state machine, leg by leg.
        for i in range(C.N_LEGS):
            c_prev, c = bool(self._clock_prev[i]), bool(contact_c[i])
            if c_prev and not c:                           # the clock lifts it
                if self.state[i] == PROBING:
                    self.lost += 1
                self.state[i] = SWING
                self.swings += 1
                self.streak[i] = 0
                self.t_td[i] = np.nan
                self.t_probe0[i] = np.nan
                self.gave_up[i] = False
                self.how[i] = ""
                self._relatch[i] = np.nan
                self._lift_streak[i] = 0
            elif not c_prev and c:                         # the clock lands it
                if self.state[i] == SWING:
                    self.state[i] = PROBING
                    self.t_probe0[i] = now
                    self.probes += 1
                    self.streak[i] = 0
            self._clock_prev[i] = c
            st = int(self.state[i])
            if st == LANDED and c and not self.gave_up[i]:
                # A LANDED FOOT THAT LEAVES THE FLOOR while the clock has it
                # in stance: back to probing, from where it is.  Not a foot
                # landed by giving up (the limit, the two-foot rule): that
                # one is KNOWN to be in the air, and the clock's assumption
                # was taken deliberately.
                if z_clear[i] > self.lift_max:
                    self._lift_streak[i] += 1
                else:
                    self._lift_streak[i] = 0
                if self._lift_streak[i] >= self.streak_n:
                    self._lift_off(i, now, state.x_b[i])
                    st = PROBING
            else:
                self._lift_streak[i] = 0
            testing = ((st == SWING and swing_s[i] >= self.s_min)
                       or st == PROBING)
            if not testing:
                self.streak[i] = 0
                continue
            near = bool(z_clear[i] <= self.clear_max
                        and np.isfinite(z_ref[i])
                        and z_ref[i] <= self.ref_clear_max)
            push = bool(np.isfinite(fz[i]) and fz[i] >= self.fz_td)
            stopped = bool(np.isfinite(vref[i]) and vref[i] <= -self.vref_min
                           and v_z[i] >= -self.v_stop)
            if near and (push or stopped):
                self.streak[i] += 1
            else:
                self.streak[i] = 0
            if self.streak[i] >= self.streak_n:
                lead = ((1.0 - float(swing_s[i])) * t_swing if st == SWING
                        else -(now - float(self.t_probe0[i])))
                self._land(i, now, lead, "push" if push else "stop")
            elif st == PROBING and self.depth(i, now) >= self.probe_max - 1e-12:
                self._land(i, now, -(now - float(self.t_probe0[i])), "limit")

        # 5. the sample the law acts on.
        sample = self._sample(now, clock, gait)
        if (self.state == PROBING).any():
            # THE TWO-FOOT RULE: a probe ends when the four-foot window
            # does.  Once any OTHER foot the clock has down is halfway
            # through its liftoff ramp (27 ms before the walk's clock lifts
            # it) every probing foot is declared down, so its TD_RAMP_S
            # load ramp IN overlaps theirs OUT and the allocator is never
            # left with one foot.  Halfway, not the ramp's start: at the
            # start the probe would be cut 66 ms after the clock's
            # touchdown on the walk's clock, inside the 65 ms an 8 mm late
            # foot needs to find the floor; halfway leaves it 93 ms.  And,
            # whatever the ramps say, never fewer than two feet in contact.
            down = self._down(clock, gait)
            ending = any(bool(clock.contact[j]) and self.state[j] != PROBING
                         and down[j] < 0.5 for j in range(C.N_LEGS))
            if ending or sample.contact.sum() < 2:
                for i in np.flatnonzero(self.state == PROBING):
                    self._land(i, now, -(now - float(self.t_probe0[i])),
                               "window")
                sample = self._sample(now, clock, gait)
        self.contact = np.asarray(sample.contact, dtype=bool).copy()
        self.fz_push = fz
        self.z_clear = z_clear
        self.z_ref = z_ref
        for i in range(C.N_LEGS):
            self.probe_depth[i] = (self.depth(i, now)
                                   if self.state[i] == PROBING else 0.0)
        return sample

    def _lift_off(self, i: int, now: float, x_b) -> None:
        self.state[i] = PROBING
        self.t_probe0[i] = now
        self.t_td[i] = np.nan
        self._relatch[i] = np.asarray(x_b, dtype=float)
        self._lift_streak[i] = 0
        self.streak[i] = 0
        self.lifted += 1
        self.probes += 1

    def _land(self, i: int, now: float, lead: float, how: str) -> None:
        was = int(self.state[i])
        self.state[i] = LANDED
        self.t_td[i] = now
        self.lead[i] = lead
        self.how[i] = how
        self.streak[i] = 0
        self._lift_streak[i] = 0
        self._relatch[i] = np.nan
        self.gave_up[i] = how in ("limit", "window")
        if was == PROBING:
            self.late.append(-lead)
            self.probe_at_td.append(self.depth(i, now))
            if how == "limit":
                self.limit += 1
            elif how == "window":
                self.window += 1
        else:
            self.early.append(lead)
        if how == "push":
            self.by_push += 1
        elif how == "stop":
            self.by_stop += 1

    @staticmethod
    def _down(clock: GaitSample, gait) -> np.ndarray:
        """(4,) the clock's LIFTOFF ramp alone -- 1 through the stance, a
        smoothstep down over its last `ramp` -- for a landed leg the clock
        has in stance, whose touchdown ramp the detection's replaces."""
        phase = np.asarray(clock.phase, dtype=float)
        s_st = np.clip(phase / float(gait.duty), 0.0, 1.0)
        return smoothstep((1.0 - s_st) / max(float(gait.ramp), 1e-9))

    def _sample(self, now: float, clock: GaitSample, gait) -> GaitSample:
        contact = np.asarray(clock.contact, dtype=bool).copy()
        weight = np.asarray(clock.weight, dtype=float).copy()
        swing_s = np.asarray(clock.swing_s, dtype=float).copy()
        phase = np.asarray(clock.phase, dtype=float)
        down = self._down(clock, gait)
        for i in range(C.N_LEGS):
            st = int(self.state[i])
            if st == SWING or st == PROBING:
                contact[i] = False
                weight[i] = 0.0
                if st == PROBING:
                    swing_s[i] = 1.0
            elif st == LANDED:
                contact[i] = True
                ramp = smoothstep((now - float(self.t_td[i])) / self.ramp_s)
                weight[i] = (min(ramp, float(down[i])) if clock.contact[i]
                             else ramp)
                swing_s[i] = 0.0
        return GaitSample(contact=contact, weight=weight, swing_s=swing_s,
                          full_support=bool(np.all(weight >= 1.0 - 1e-9)),
                          phase=phase)

    # -- the state estimator ----------------------------------------------
    def stance_phase(self, now: float, gait_phase, duty: float) -> np.ndarray:
        """(4,) the filter's contact phase -- 0 touchdown .. 1 liftoff, 0 in
        the air (`state_estimator.adapters.stance_phase`) -- with this in
        it: a PROBING leg is in the air (0), a LANDED one has been down
        since its detection, whatever the clock says."""
        phase = np.asarray(gait_phase, dtype=float).reshape(C.N_LEGS)
        out = np.where(phase < duty, phase / duty, 0.0)
        if not self.active:
            return out
        for i in range(C.N_LEGS):
            st = int(self.state[i])
            if st == PROBING or st == SWING:
                out[i] = 0.0
            elif st == LANDED and math.isfinite(self.t_td[i]):
                since = (float(now) - float(self.t_td[i])) / self._t_stance
                out[i] = max(float(out[i]), min(1.0, since))
        return out

    # -- what the operator reads --------------------------------------------
    def snapshot(self) -> ContactSnapshot:
        return ContactSnapshot(state=self.state.copy(),
                               fz_push=self.fz_push.copy(),
                               z_clear=self.z_clear.copy(),
                               z_ref=self.z_ref.copy(),
                               probe=self.probe_depth.copy(),
                               lead=self.lead.copy())

    def status(self, now: float) -> str:
        """One line, 2 Hz: each leg's state, and the last detection against
        the clock."""
        parts = []
        for i, leg in enumerate(C.LEGS):
            st = int(self.state[i])
            if st == PROBING:
                parts.append("%s probe %.0fmm" % (leg, 1e3 * self.depth(i, now)))
            elif st == LANDED:
                parts.append("%s down %+.0fms%s" % (
                    leg, 1e3 * self.lead[i],
                    "!" if self.gave_up[i] else ""))
            elif st == SWING:
                parts.append("%s swing %.1fmm" % (leg, 1e3 * self.z_clear[i]))
            else:
                parts.append("%s %s" % (leg, STATE_NAMES[st]))
        return ("touchdown A: %s | early %d late %d probe %d (limit %d, "
                "window %d) lifted %d lost %d"
                % ("  ".join(parts), len(self.early), len(self.late),
                   self.probes, self.limit, self.window, self.lifted,
                   self.lost))

    def report(self) -> str:
        def ms(values):
            if not values:
                return "-"
            a = 1e3 * np.asarray(values)
            return "median %+.0f, p90 %+.0f, max %+.0f ms" % (
                np.median(a), np.percentile(a, 90), a.max())
        lines = [
            "  touchdown A     %d swing%s in %d trot%s: detected before the "
            "clock %d, after it (probing) %d, never (lifted again) %d"
            % (self.swings, "" if self.swings == 1 else "s", self.trots,
               "" if self.trots == 1 else "s", len(self.early), len(self.late),
               self.lost),
            "                  early by %s" % ms(self.early),
            "                  late by  %s" % ms(self.late),
            "                  by the push %d, by the stopped foot %d; probes "
            "%d (of them %d a landed foot that left the floor), of which "
            "gave up at the %.0f mm limit %d, cut by the two-foot rule %d"
            % (self.by_push, self.by_stop, self.probes, self.lifted,
               1e3 * self.probe_max, self.limit, self.window),
        ]
        if self.probe_at_td:
            d = 1e3 * np.asarray(self.probe_at_td)
            lines.append("                  probe depth at touchdown median "
                         "%.1f, max %.1f mm" % (np.median(d), d.max()))
        lines.append("                  thresholds: push >= %.1f N or stopped "
                     "(v_z >= %+.2f with v_ref <= %+.2f m/s), the foot within "
                     "%.0f mm of the floor and the reference within %.0f, %d "
                     "sweeps running, from s >= %.2f; probe %.2f m/s to %.0f "
                     "mm pushing %.0f N; load ramp %.0f ms; a landed foot "
                     "past %.0f mm re-probes"
                     % (self.fz_td, -self.v_stop, -self.vref_min,
                        1e3 * self.clear_max, 1e3 * self.ref_clear_max,
                        self.streak_n, self.s_min, self.probe_v,
                        1e3 * self.probe_max, self.probe_push,
                        1e3 * self.ramp_s, 1e3 * self.lift_max))
        return "\n".join(lines)

    def describe(self) -> str:
        return ("touchdown detection (scheme A): the swing law's commanded "
                "push >= %.1f N or a stopped foot, the foot within %.0f mm of "
                "the floor and its reference within %.0f, %d sweeps running, "
                "from swing s >= %.2f; a late foot PROBES down at %.2f m/s to "
                "%.0f mm, pushed %.0f N, out of the allocator until it lands; "
                "a landed foot is loaded over %.0f ms, and re-probes if it "
                "rises past %.0f mm"
                % (self.fz_td, 1e3 * self.clear_max, 1e3 * self.ref_clear_max,
                   self.streak_n, self.s_min, self.probe_v,
                   1e3 * self.probe_max, self.probe_push, 1e3 * self.ramp_s,
                   1e3 * self.lift_max))


def describe() -> str:
    det = ContactDetector()
    kp_z, kd_z = float(cfg.KP_SWING_WALK[2]), float(cfg.KD_SWING_WALK[2])
    return "\n".join([
        "DOG6 touchdown detection, scheme A -- " + det.describe(),
        "  the robot's numbers: weight %.1f N (%.1f a foot on four, %.1f on a "
        "diagonal); walk Kp_z %.0f N/m, Kd_z %.0f N s/m"
        % (cfg.WEIGHT, cfg.WEIGHT / 4.0, cfg.WEIGHT / 2.0, kp_z, kd_z),
        "  a foot stopped mid-descent (reference at 0.4 m/s): Kd alone commands "
        "%.0f N -- over the %.1f N threshold at once" % (kd_z * 0.4, det.fz_td),
        "  a foot stopped under the probe: Kd gives %.0f N at once, Kp adds "
        "%.0f N/s; the threshold in ~%.0f ms"
        % (kd_z * det.probe_v, kp_z * det.probe_v,
           1e3 * max(0.0, det.fz_td - kd_z * det.probe_v) / (kp_z * det.probe_v)),
        "  the swing's tracking lag on the robot, 27 mm p90, is %.0f N of Kp "
        "-- the gates keep it out: the foot within %.0f mm of the floor AND "
        "the reference within %.0f"
        % (kp_z * 0.027, 1e3 * det.clear_max, 1e3 * det.ref_clear_max),
        "  the probe reaches %.0f mm in %.0f ms; the four-foot window at duty "
        "%.2f on a %.1f s clock is %.0f ms (the two-foot rule covers the rest)"
        % (1e3 * det.probe_max, 1e3 * det.probe_max / max(det.probe_v, 1e-9),
           cfg.WALK_DUTY, 0.6, 1e3 * (cfg.WALK_DUTY - 0.5) * 0.6),
    ])


if __name__ == "__main__":
    print(describe())
