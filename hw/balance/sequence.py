"""The phase machine: which law drives the motors this sweep, and what it sends.

    limp -> settle -> crouch -> rise -> hold -> park -> done
                                    ^^^^^^^^^^^^ the only torque phases

    StandSequence.advance(now, q)     step to the next phase (`hw.stand`'s ENTER)
    StandSequence.toggle_trot(now)    enter / leave the trot (`hw.stand`'s T)
    StandSequence.toggle_step(now)    step the feet to `step_to` or back (W)
    StandSequence.update(now, body)   -> (mode, values, trip), once per sweep

THE TROT IS A SUB-STATE OF HOLD, NOT AN EIGHTH PHASE
    DOG5's TROT stage, ported: T from HOLD starts it, T during it LATCHES an
    exit that is taken on the first sweep with all four feet down at full
    weight, and ENTER is refused while it runs -- park from HOLD, never from
    mid-swing.  `phase_name` reads "trot" throughout, so the log slices it and
    the banner names it, while `PHASES` and every stand path stay exactly as
    they were.  Only a sequence built with a `gait` can trot at all.

    THE FOOT STEP IS THE SAME KIND OF SUB-STATE.  W from HOLD runs ONE cycle of
    `step_gait` with `law.BalanceLaw.begin_step`, every swing landing on the
    `step_to` site; `phase_name` reads "step" and HOLD comes back at the first
    four-foot window after the last touchdown.  ENTER in HOLD with the feet
    away from the crouch's sites steps them home FIRST and parks after --
    the park is a joint ramp to the crouch and would drag them.

`mode` is the whole interface to the bus: "keepalive" (values None), "position"
(values: (12,) joint rad, capped by `max_dps`) or "torque" (values: (12,) N*m,
already through `safety.SafetyGate`).  `hw.stand` sends it and does not
interpret it.

WHY THE WHOLE SEQUENCE LIVES IN THIS PACKAGE AND NOT JUST THE LIFT
    Five of the seven phases have no balance law in them at all -- they are the
    drivers' own 0xA4 position loops, closed in the driver with the driver's
    gains.  They are here anyway, because what has to be got right is the SEAM
    between them and the lift, and every part of that seam is a statement about
    the law:

      * the lift is armed from the pose the CROUCH left the robot in, and
        `law.BalanceLaw.arm` latches h0 and the heading from what is MEASURED
        at that instant -- not from `CROUCH_HEIGHT`, because a ramp started
        from a height the robot is not at is a step into k_d,z;
      * the park starts from wherever the LIFT settled, not from Q_CROUCH;
      * the tracking trip changes meaning at the boundary: in position mode it
        compares against the 0xA4 target, in the lift against the IK at the
        commanded height.

    Splitting that across two packages puts a module boundary through the
    middle of one handover.  So `hw.balance` owns WHAT the robot is sent in
    every phase; `hw.stand` owns WHEN -- the CAN slots, the keys, the log, the
    exit report.

    THE DIVIDING LINE IS I/O.  Nothing in this file opens a bus, reads a clock
    of its own or prints, and `now` and `body` both arrive as arguments.  That
    is what lets `--fake` exercise exactly the object the robot runs, and it is
    why the phase machine can be stepped in a test with a hand-built
    `state.BodyState` and no hardware in the process.

WHERE THE BALANCE CONTROLLER ACTUALLY ACTS
    RISE and HOLD, and nowhere else.  `_lift_srb` is a four-line method: call
    `law.BalanceLaw.update`, hand the torque to the gate, keep the output for
    the log.  Everything above it in this file is sequencing.

RISE AND HOLD ARE TWO PHASES AND THE TORQUE PATH IS IDENTICAL ACROSS THEM
    Nothing about the control changes at that edge: same law, same gains, same
    reference -- the quintic arrives at `h_lift` after `rise_s` and `Quintic.at`
    clamps past T, so the reference holds itself.  Physically it is one
    continuous phase and it used to be written as one.

    WHAT THE SPLIT BUYS IS THAT THE TWO CAN BE TOLD APART AFTERWARDS.  The
    rise answers "can it get up"; the hold answers "does the controller
    CORRECT", and that second question is only asked by PUSHING THE ROBOT and
    watching what comes back.  A push has to be sliceable out of the npz, and
    `phase == "hold"` is how; an operator has to be told when it is safe to
    push, and the phase banner is how.  DOG5 split them for the same reason
    and its HOLD banner said, in as many words, "Push the trunk".

    THE EDGE IS A CLOCK, NOT A KEY.  `update` steps rise -> hold the sweep the
    S-curve arrives, exactly as DOG5's `now - t0 >= T_RISE` did.  `advance`
    REFUSES to step out of rise by hand: a phase whose exit is a physical fact
    should not also be exitable early by a keystroke, or a half-finished lift
    becomes a hold that thinks it is at height.

    ONE TRAP LIVES HERE.  The per-leg law's height ramp is a smoothstep over
    time SINCE THE RISE, and `t_phase` resets at the hold boundary -- authored
    against that it would restart at alpha = 0, command CROUCH_HEIGHT at full
    height and drop the robot.  It is authored against `t_lift` for that
    reason.  The SRB law is immune: `arm` latches its own t0.

    `ramp_remaining` still tells `--auto` when the rise has arrived, and is
    zero throughout the hold.

LIMP IS NOT A PHASE THAT DOES NOTHING
    It sends 0xA1 iq=0 keep-alives, which is the only way to hold twelve
    drivers inside their 50 ms input-lost window while commanding no torque at
    all.  It is where a wrong sign or a wrong zero is READ, by hand, before
    anything moves.

    IT IS ALSO WHERE "LEVEL" IS DEFINED.  `law.BalanceLaw.latch_setpoint` is
    called on the limp sweeps and takes the first non-stale reading as the
    attitude setpoint for the whole run -- DOG5's SETPOINT_DYNAMIC, ported.
    Zero torque is the requirement: it is the only phase in which the attitude
    the IMU reports is the attitude the robot RESTS at rather than one the
    loop is enforcing.  So limp is no longer a phase an operator may skip
    past, and a robot being handled through it delays the latch rather than
    corrupting it -- `latch_setpoint` refuses a stale sample and latches once.

WHERE "FORWARD" IS DEFINED, AND IT IS NOT LIMP
    The heading half of the same convention is latched one handover later, on
    the sweep the CROUCH ends and RISE begins: `law.BalanceLaw.arm` takes the
    magnetometer's reading there, `yaw_offset` becomes that number, and
    `state.rezero_yaw` turns the world frame onto it for every sweep after.
    Yaw then reads 0 at the handover and is heading DRIFT from there on.

    THE TWO HALVES DIFFER BECAUSE THE REFERENCES DIFFER.  Level is a fact
    about the world and has to be read where nothing is enforcing it, which is
    limp.  Forward is not a fact about the world -- it is whichever way the
    robot happens to be pointing when it starts pushing -- so it is read where
    the pushing starts, and taking it there is also what makes the yaw error
    exactly zero on the sweep it is taken.  DOG5 split them at the same two
    instants for the same two reasons.
"""
from __future__ import annotations

import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/sequence.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import kinematics as SK     # noqa: E402
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402

from .. import hardware_map as HM    # noqa: E402
from .. import kinematics as HK      # noqa: E402
from .. import safety as SAFE        # noqa: E402
from ..motor import MAX_SPEED_POS    # noqa: E402
from . import config as BCFG         # noqa: E402
from . import law as BLAW            # noqa: E402
from . import posture as POSE        # noqa: E402
from . import state as BSTATE        # noqa: E402

__all__ = ["LAWS", "PHASES", "BLURB", "LIFT_BLURB", "phase_blurb",
           "ramp_motor_dps", "StandSequence",
           "SETTLE_MOTOR_DPS", "SETTLE_TRACK_ESTOP", "TRACK_ESTOP"]

#: The two lift laws, and the whole point of there being two.
#:
#:   srb       the SRB balance controller of `hw.balance`: attitude is
#:             MEASURED, gravity enters once as m*g, and the allocator splits
#:             it across four feet by the geometry the robot actually has.
#:   per-leg   `sim.stand.compliance_torque`: four independent Cartesian
#:             springs, a fixed mg/4 feedforward, and no variable anywhere in
#:             it that names the trunk's orientation.
#:
#: THE SECOND ONE IS KEPT DELIBERATELY.  Without it the first question after a
#: bad run -- "is this worse than what we had?" -- has no answer.  It is also
#: the A/B that says whether a fault is above or below the model: if the
#: PER-LEG law misbehaves too, nothing a controller swap can do will fix it.
LAWS = ("srb", "per-leg")

PHASES = ("limp", "settle", "crouch", "rise", "hold", "park", "done")
BLURB = {
    "limp":   "NO TORQUE -- the attitude setpoint latches here; hold still",
    "settle": "driver position mode, holding the pose it was in",
    "crouch": "driver position mode -> crouch (trunk on the floor)",
    "rise":   "the S-curve up; ends on its own clock, then HOLD",
    "hold":   "AT HEIGHT -- push the trunk and watch it come back",
    "trot":   "TROTTING IN PLACE -- T latches the exit at the next four-foot "
              "window; ENTER is refused until then",
    "step":   "STEPPING THE FEET -- one gait cycle, each diagonal lands on the "
              "new site; HOLD (or PARK) follows at four feet down",
    "park":   "driver position mode -> crouch",
    "done":   "driver position mode, holding crouch.  ENTER exits",
}

#: What the lift phase is actually doing, which depends on `--law`.  The
#: phase banner is the one place an operator reads it, so it says which.
LIFT_BLURB = {
    "srb": "TORQUE, SRB balance controller -- attitude MEASURED, one wrench "
           "allocated across four feet",
    "per-leg": "TORQUE, per-leg Cartesian compliance (THE BASELINE) -- xy "
               "pinned, z lifts, mg/4 each",
}


def phase_blurb(sequence: "StandSequence") -> str:
    """The one line an operator reads when a phase starts.

    The lift's blurb depends on `--law`, and the phase banner is the only
    place that difference is ever stated, so it is stated there.
    """
    if sequence.phase_name in ("rise", "hold", "trot", "step"):
        return "%s -- %s" % (BLURB[sequence.phase_name],
                             LIFT_BLURB[sequence.law])
    return BLURB[sequence.phase_name]


#: 0xA4 speed caps, MOTOR side (1 dps/LSB, 10:1 to the output).  A ramp's cap
#: is its smoothstep peak rate times SPEED_MARGIN, so the driver can follow
#: the reference but a frame error cannot run away at full speed.
SETTLE_MOTOR_DPS = 60.0
MIN_MOTOR_DPS = 60.0
SPEED_MARGIN = 1.5

#: Tracking-error trips in position mode.  Settle's is tight on purpose: it
#: holds the pose it just READ, so any real error there means the driver's
#: position frame and the encoder disagree -- found at 6 deg/s, not at speed.
SETTLE_TRACK_ESTOP = np.deg2rad(5.0)
TRACK_ESTOP = np.deg2rad(15.0)

def ramp_motor_dps(q_from, q_to, seconds: float) -> np.ndarray:
    """(12,) motor-side 0xA4 caps for a smoothstep from `q_from` to `q_to`.

    The smoothstep's peak slope is pi/2, so its peak joint rate is
    ``pi/2 * |dq| / T``.
    """
    peak_rad_s = 0.5 * np.pi * np.abs(C.flat(q_to) - C.flat(q_from)) / seconds
    motor_dps = np.rad2deg(peak_rad_s) * P.GEAR_RATIO * SPEED_MARGIN
    return np.clip(motor_dps, MIN_MOTOR_DPS, MAX_SPEED_POS)


class HoldWatch:
    """What the attitude loop did while the robot was only standing there.

    THE HOLD IS THE EXPERIMENT AND THIS IS ITS INSTRUMENT.  The rise answers
    "can it get up"; the hold answers "does the controller CORRECT", and that
    question is not answered by a height or a torque.  It is answered by three
    numbers: how far a push took the trunk from the setpoint, what moment the
    law asked for while pushing back, and whether it came back.

    MEASURED FROM THE SETPOINT, NOT FROM TRUE LEVEL.  The setpoint is what
    the law is holding the robot at -- see `law.tilt_from_setpoint_deg` -- so
    it is the only origin against which "came back" means anything.  On a
    floor with slope in it the two differ by the slope, permanently, and a
    peak measured from true level would read that slope as a disturbance the
    controller never rejected.

    NO CLOCK OF ITS OWN, like everything else in this file: `now` arrives.
    """

    #: Below this the trunk counts as recovered.  It is not a control
    #: threshold -- nothing reads it but the report -- so it is set where an
    #: operator would stop calling a tilt a tilt, not where the loop stops
    #: correcting.  The measured noise floor at rest is ~0.06 deg.
    SETTLED_DEG = 0.5

    def __init__(self) -> None:
        self.start(0.0)

    def start(self, now: float) -> None:
        self.t0 = float(now)
        self.t_last = float(now)
        self.tilt_deg = 0.0
        #: The live diagnostic, all in the operator's units.  `err` is
        #: measured MINUS setpoint; `moment` is what the law is ASKING the
        #: feet for about x, y, z.  The two together are what answers "is it
        #: not correcting, or is it correcting toward the wrong place":
        #: err != 0 with moment == 0 is a dead loop, err != 0 with moment != 0
        #: is a loop that is trying and not winning, and err == 0 while the
        #: robot is visibly tilted is a WRONG SETPOINT.
        self.rpy_deg = np.full(3, np.nan)
        self.err_deg = np.full(3, np.nan)
        self.moment_nm = np.zeros(3)
        self.h = float("nan")
        self.peak_deg = 0.0
        self.t_peak = None
        self.t_settled = None
        self.peak_moment_nm = 0.0
        self.n = 0

    @property
    def live(self) -> bool:
        return self.n > 0

    def add(self, now: float, body, balance, out) -> None:
        self.n += 1
        self.t_last = float(now)
        self.rpy_deg = np.degrees([body.roll, body.pitch, body.yaw])
        self.err_deg = balance.attitude_error_deg(body)
        self.h = float(body.h)
        self.tilt_deg = balance.tilt_from_setpoint_deg(body)
        if self.tilt_deg > self.peak_deg:
            self.peak_deg = self.tilt_deg
            self.t_peak = float(now)
            # A new peak means the excursion is not over; the recovery clock
            # only starts once the trunk stops going further out.
            self.t_settled = None
        elif (self.t_settled is None and self.t_peak is not None
                and self.tilt_deg < self.SETTLED_DEG):
            self.t_settled = float(now)
        if out is not None:
            self.moment_nm = np.asarray(out.wrench.b_d, dtype=float)[3:6]
            self.peak_moment_nm = max(self.peak_moment_nm,
                                      float(np.abs(self.moment_nm[:2]).max()))

    def recovery_s(self) -> float | None:
        """Peak to settled, in seconds.  None if it has not settled yet."""
        if self.t_peak is None or self.t_settled is None:
            return None
        return self.t_settled - self.t_peak

    def status(self) -> str:
        """What an operator reads while pushing the robot.  Line 1 of two.

        rpy as measured, the error against the setpoint, and the height a
        ruler reaches -- floor to trunk BOTTOM, not the code's trunk-origin
        frame, because the point of printing it is that somebody can check it.
        """
        return ("rpy %+6.2f/%+6.2f/%+7.2f   err %+6.2f/%+6.2f/%+7.2f deg   "
                "h %6.1f mm" % (*self.rpy_deg, *self.err_deg, 1e3 * self.h))

    def moment_status(self) -> str:
        """Line 2: what the law is ASKING FOR, and what the push did."""
        recovered = self.recovery_s()
        return ("M %+5.2f/%+5.2f/%+5.2f N*m   tilt %4.1f now / %4.1f peak   %s"
                % (*self.moment_nm, self.tilt_deg, self.peak_deg,
                   "back in %.1f s" % recovered if recovered is not None
                   else "OUT" if self.tilt_deg >= self.SETTLED_DEG
                   else "settled"))

    def report(self) -> str:
        if not self.live:
            return "  hold            never reached"
        recovered = self.recovery_s()
        return "\n".join([
            "  hold            %d sweeps over %.1f s"
            % (self.n, self.t_last - self.t0),
            "  worst tilt      %.2f deg from the setpoint (%s)"
            % (self.peak_deg,
               # NEVER LEFT and NEVER CAME BACK are opposite results and the
               # first one is the common one: most holds are not pushed.
               # Reporting "never came back" for a robot that never moved
               # reads as a failure of the loop, which is the reverse of what
               # a flat 0.00 deg means.
               "never left it -- no push, or none that moved the trunk"
               if self.peak_deg < self.SETTLED_DEG else
               "recovered under %.1f deg in %.1f s"
               % (self.SETTLED_DEG, recovered) if recovered is not None
               else "NEVER came back under %.1f deg" % self.SETTLED_DEG),
            "  peak att moment %.2f N*m -- what the law asked for to push back"
            % self.peak_moment_nm,
        ])


class StandSequence:
    """The phase machine.  Pure decisions: no bus, no clock of its own.

    `update` is called once per sweep with the measured state and returns
    what every motor should be sent this sweep.  Keeping it free of I/O is what
    lets `--fake` exercise exactly the object the robot runs.
    """

    def __init__(self, gate: SAFE.SafetyGate, *, law: str = "srb",
                 balance: BLAW.BalanceLaw | None = None,
                 crouch: POSE.CrouchPose = POSE.NOMINAL, gait=None,
                 step_to=None, step_gait=None, trot_swings: int = 0,
                 position_trips: bool = True):
        if law not in LAWS:
            raise ValueError("law must be one of %s, got %r" % (LAWS, law))
        # THE PER-LEG BASELINE IS ONLY A BASELINE AT THE NOMINAL CROUCH.
        # `sim.stand.compliance_torque` pins its springs at `sim.stand.FOOT_XY`
        # and takes no posture -- from any other crouch it would spend the lift
        # dragging the feet back to the nominal stance, which is not the law
        # misbehaving and not an A/B of anything.  Refused rather than
        # silently wrong: the SRB law reads the feet it actually has.
        if law == "per-leg" and crouch is not POSE.NOMINAL:
            raise ValueError(
                "the per-leg law pins its Cartesian springs at the NOMINAL "
                "foot xy, so it cannot be run from the %r crouch -- it would "
                "drag the feet to the nominal stance during the lift.  Use "
                "--law srb." % crouch.name)
        self.gate = gate
        self.law = law
        #: The posture CROUCH ramps to, PARK returns to, and the lift starts
        #: from.  `posture.NOMINAL` is what this has always flown; anything
        #: else is a different experiment and the banner says which.
        self.crouch = crouch
        self.balance = balance if balance is not None else BLAW.BalanceLaw()
        self.phase = 0
        self.t_phase = 0.0
        self.q_ref0 = np.zeros((C.N_LEGS, C.N_JOINTS_PER_LEG))
        self.q_des = np.zeros(C.N_JOINTS)
        self.h_cmd = ST.CROUCH_HEIGHT
        self.gravity = np.zeros((C.N_LEGS, C.N_JOINTS_PER_LEG))
        self.tau = np.zeros(C.N_JOINTS)
        self.tau_request = np.zeros(C.N_JOINTS)
        self.tau_peak = 0.0
        self.max_dps = np.full(C.N_JOINTS, SETTLE_MOTOR_DPS)
        #: The last sweep`s measurement and law output, for the status line
        #: and the log.  None until the lift arms.
        self.body = None
        self.out = None
        #: One-shot text for the operator, drained by the runner.  This file
        #: does no I/O -- see the module docstring -- but the setpoint latch
        #: happens HERE, in the phase that owns it, and an operator who is not
        #: told what "level" was defined as cannot read the run.  So the words
        #: are built here and someone else prints them.
        self.notice: str | None = None
        #: When the RISE began.  The per-leg law's height ramp is authored
        #: against it and NOT against `t_phase`, which resets at the HOLD
        #: boundary -- a restart there would send its smoothstep back to
        #: alpha = 0, command CROUCH_HEIGHT at full height, and drop the
        #: robot.  The SRB law is immune by construction: `BalanceLaw.arm`
        #: latches its own t0 and the quintic clamps past T.
        self.t_lift = 0.0
        #: What the attitude loop did while the robot was only standing.
        #: Reset when HOLD is entered, so a push is read against the hold it
        #: happened in and not against the rise that preceded it.
        self.hold = HoldWatch()
        #: The trot clock, `gait.TrotGait`, or None for a sequence that only
        #: stands.  The law reads it; this file only starts and stops it.
        self.gait = gait
        self.trotting = False
        self.trot_exit = False
        self.t_trot = 0.0
        self.trot_runs = 0
        #: The same instrument as the hold's, read against the trot.
        self.trot = HoldWatch()
        #: Swings a T run takes before it latches its own exit; 0 trots until
        #: T again.  1 is `--half-gait`: one diagonal lifts and lands, HOLD.
        self.trot_swings = int(trot_swings)
        self._swings_done = 0
        self._swinging = np.zeros(C.N_LEGS, dtype=bool)
        #: `--half-gait`: each T is the NEXT half cycle, so the diagonal
        #: alternates press by press -- the operator is the clock.
        self._half_next = False
        #: W: where the hold steps the feet to, (4, 2) HIP frame, or None for
        #: an entry point that does not step.  W again, or ENTER to park,
        #: steps them back to the crouch's `foot_xy` first -- the park is a
        #: joint-space ramp to the crouch and would DRAG feet that are
        #: anywhere else.
        self.step_to = (None if step_to is None
                        else np.array(step_to, dtype=float).reshape(C.N_LEGS, 2))
        #: The step's own clock -- slower than a fast trot's, because the
        #: foot also crosses the step distance in one swing.  None: `gait`.
        self.step_gait = step_gait if step_gait is not None else gait
        self.stepping = False
        self.park_after_step = False
        self.step_runs = 0
        self._sweep = 0
        #: `_four_foot_window`'s memory: the previous sweep's smallest contact
        #: weight while all four were down, or None.
        self._exit_min_w = None
        #: The position-mode tracking trips, SETTLE_TRACK_ESTOP and
        #: TRACK_ESTOP.  False turns both OFF -- `hw.stand --no-limits`: a
        #: leg that cannot follow the ramp is then pushed by the driver's own
        #: position loop and nothing here stops it.
        self.position_trips = bool(position_trips)

    @property
    def phase_name(self) -> str:
        if self.stepping:
            return "step"
        return "trot" if self.trotting else PHASES[self.phase]

    @property
    def yaw_offset(self) -> float:
        """rad, the heading this run's world frame is pinned to.

        0.0 until the handover at the end of the CROUCH latches it -- see
        `law.BalanceLaw.arm` and `state.rezero_yaw`.  Every `BodyState` this
        object touches is put in that frame on the way in, on both the
        `advance` and the `update` path, so nothing downstream ever sees the
        magnetometer's own world once torque is live.
        """
        return self.balance.yaw_offset

    def _measured(self, body):
        """`body` in this run's world frame.  A no-op before the handover, and
        a no-op on a body that is already in it -- `state.rezero_yaw`."""
        return body if body is None else BSTATE.rezero_yaw(body,
                                                           self.yaw_offset)

    @property
    def feet_home(self) -> bool:
        """The feet are on the crouch's own sites -- PARK may run."""
        xy = self.balance.foot_xy
        return xy is None or bool(np.allclose(xy, self.crouch.foot_xy,
                                              rtol=0.0, atol=1e-9))

    def toggle_step(self, now: float) -> str:
        """W.  From HOLD, step the feet to `step_to`, or back home."""
        if self.step_to is None or self.step_gait is None:
            return "W ignored: this entry point does not step the feet"
        if self.phase_name != "hold":
            return "W ignored: the feet step from HOLD, not %s" % self.phase_name
        if self.law != "srb":
            return "W ignored: only the SRB law can step"
        target = self.crouch.foot_xy if not self.feet_home else self.step_to
        return self._start_step(now, target, park_after=False)

    def _start_step(self, now: float, target, park_after: bool) -> str:
        # The feet are moving: the old joint target is a stance they are
        # leaving.  Re-latched when the step lands, at the new one.
        self.balance.release_joints()
        self.balance.begin_step(target)
        self.step_gait.reset(now)
        self.stepping = True
        self.park_after_step = park_after
        self.t_phase = float(now)
        self.step_runs += 1
        home = np.allclose(target, self.crouch.foot_xy, rtol=0.0, atol=1e-9)
        return ("STEP: feet %s, one %.2f s gait cycle.  The first foot lifts "
                "in %.2f s.  %sX is an E-STOP."
                % ("back to the %s crouch's sites" % self.crouch.name if home
                   else "to the step sites (hip frame %s mm)"
                   % np.array2string(1e3 * np.asarray(target)[0], precision=0),
                   self.step_gait.period, self.step_gait.entry_s,
                   "PARK follows.  " if park_after else ""))

    def toggle_trot(self, now: float) -> str:
        """T.  Start the trot from HOLD, or latch its exit.  Returns what to
        print -- a refusal or a confirmation, never None."""
        if self.gait is None:
            return "T ignored: this entry point was built without a gait"
        if self.trotting:
            if self.trot_exit:
                return ("T ignored: exit already latched -- HOLD at the next "
                        "four-foot window")
            # LATCHED, NOT TAKEN: switching with a leg mid-swing hands that
            # leg the stance law in mid-air.  `update` takes it.
            self.trot_exit = True
            return "trot exit latched: HOLD at the next four-foot window"
        if self.phase_name != "hold":
            return "T ignored: the trot starts from HOLD, not %s" % self.phase_name
        if self.law != "srb":
            return "T ignored: only the SRB law can trot"
        self.gait.reset(now, half=bool(self.trot_swings) and self._half_next)
        if self.trot_swings:
            self._half_next = not self._half_next
        self.trotting = True
        self.trot_exit = False
        self.t_trot = float(now)
        self.t_phase = float(now)
        self.trot_runs += 1
        self.trot.start(now)
        self._swings_done = 0
        self._swinging[:] = False
        if self.trot_swings:
            lead = "FL/RR" if not self._half_next else "FR/RL"
            return ("HALF GAIT: %s swings, then HOLD by itself; T again swings "
                    "the other diagonal.  The feet lift in %.2f s.  X is an "
                    "E-STOP." % (lead, self.gait.entry_s))
        return ("TROT: %s.  The first foot lifts in %.2f s.  T again returns "
                "to HOLD at a four-foot window; X is an E-STOP."
                % (self.gait, self.gait.entry_s))

    def _four_foot_window(self, gait, now: float) -> bool:
        """Where a trot or a step may be left: all four feet at FULL weight --
        or, when there is no such sweep, the sweep past the window's middle.

        `full_support` wants every weight at 1.  The contact ramp is a
        fraction of STANCE and the four-foot window is (duty - 0.5)/2 of a
        cycle, so at duty 0.80 full weight lasts 60 ms of a 0.5 s cycle and
        this returns True on the first of 15 sweeps; at duty 0.72 the ramp
        fills the window and full weight lasts ONE 4 ms sweep a cycle, which
        a 250 Hz loop mostly misses -- the exit latched by T then rode on for
        cycles (2026-09-25).  So past the peak is the fallback: while all four
        are down the smallest weight rises to the window's middle and falls
        after it, and the first sweep on the way down is the closest thing
        to full support that duty has.  Same instant `full_support` would
        have picked if it existed; one sweep later than it when it does.
        """
        sample = gait.sample(now)
        if sample.full_support or not sample.contact.all():
            self._exit_min_w = None
            return bool(sample.full_support)
        w = float(sample.weight.min())
        past_peak = self._exit_min_w is not None and w < self._exit_min_w - 1e-9
        self._exit_min_w = w
        return past_peak

    def _count_swings(self) -> None:
        """Count diagonal touchdowns; latch the exit after `trot_swings`."""
        if not self.trot_swings or self.out is None or self.out.swing_s is None:
            return
        swinging = np.asarray(self.out.swing_s) > 0.0
        landed = self._swinging & ~swinging
        if landed.any():
            self._swings_done += 1       # a diagonal's two feet land together
        self._swinging = swinging
        if self._swings_done >= self.trot_swings and not self.trot_exit:
            self.trot_exit = True
            self.notice = ("%d swing%s landed -- HOLD at the next four-foot "
                           "window" % (self._swings_done,
                                       "" if self._swings_done == 1 else "s"))

    def take_notice(self) -> str | None:
        """Hand the runner anything waiting to be said, once."""
        notice, self.notice = self.notice, None
        return notice

    @property
    def finished(self) -> bool:
        return self.phase_name == "done"

    def ramp_remaining(self, now: float) -> float:
        """Seconds until the current position ramp has arrived; 0 if none."""
        seconds = {"crouch": ST.RAMP_POSITION, "park": ST.RAMP_POSITION,
                   "rise": ST.RAMP_LIFT}.get(self.phase_name, 0.0)
        return max(0.0, seconds - (now - self.t_phase))

    def advance(self, now: float, q) -> str | None:
        """Enter the next phase.  Returns why it refused, or None."""
        # The runner assigns `self.body` straight from `state.read`, which is
        # the magnetometer's world.  Put it in the run's before anything here
        # reads a heading off it.
        self.body = self._measured(self.body)
        if self.finished:
            return None
        if self.trotting:
            return ("trotting -- press T to latch the exit, then ENTER parks "
                    "from HOLD")
        if self.stepping:
            return "stepping the feet -- HOLD%s follows at four feet down" % (
                " then PARK" if self.park_after_step else "")
        # THE FEET GO HOME BEFORE THE PARK.  The park ramps the joints to the
        # crouch; from any other stance that slides the feet on the floor.
        if (self.phase_name == "hold" and self.law == "srb"
                and self.step_gait is not None and not self.feet_home):
            self.notice = self._start_step(now, self.crouch.foot_xy,
                                           park_after=True)
            return None
        if self.phase_name in ("crouch", "park") and self.ramp_remaining(now) > 0:
            return ("%s ramp still running, %.1f s left"
                    % (self.phase_name, self.ramp_remaining(now)))
        # RISE IS NOT STEPPED BY HAND.  It ends when the S-curve arrives and
        # `update` moves to HOLD by itself -- DOG5 did the same, on a clock and
        # not on a key.  A phase whose exit is a physical fact (the reference
        # got there) should not also be exitable early by a keystroke: that is
        # how a half-finished lift becomes a hold that thinks it is at height.
        if self.phase_name == "rise":
            return ("the rise ends on its own in %.1f s, then HOLD"
                    % self.ramp_remaining(now))
        self.phase += 1
        self.t_phase = now
        # From WHERE THE ROBOT IS, as in sim.stand -- after the lift that is
        # wherever the compliance law settled, not Q_CROUCH.
        self.q_ref0 = C.unflat(q).copy()
        name = self.phase_name
        if name == "park":
            self.balance.release_joints()
        if name == "settle":
            self.max_dps = np.full(C.N_JOINTS, SETTLE_MOTOR_DPS)
        elif name in ("crouch", "park"):
            self.max_dps = ramp_motor_dps(self.q_ref0, self.crouch.q,
                                          ST.RAMP_POSITION)
        elif name == "rise":
            self.t_lift = now
            self.gate.start(now, q=q)
            q4 = C.unflat(q)
            self.gravity = np.stack([SK.leg_gravity_torque(i, q4[i])
                                     for i in range(C.N_LEGS)])
            if self.law == "srb":
                # h0 and the heading are latched HERE, from what is measured
                # at the handover -- see `balance.law.BalanceLaw.arm`.  The
                # crouch has just arrived, so this is the last sweep with the
                # trunk on the floor and the first with torque behind it.
                self.balance.arm(now, self.body)
                # AND THE WORLD TURNS WITH IT, ON THIS SWEEP.  `arm` has just
                # taken the heading; re-taking the body against it now is
                # what makes the yaw error zero on the sweep it was latched
                # rather than one sweep later.  Latch and frame have to move
                # together or the arming sweep sees the whole heading as an
                # attitude error -- the step `arm` exists to avoid.
                self.body = self._measured(self.body)
                self.notice = (
                    "heading zeroed at the handover: %+.1f deg was the "
                    "magnetometer's reading and is now 0 -- world x is where "
                    "the trunk points, and yaw from here is drift off it"
                    % np.degrees(self.yaw_offset))
        elif name == "done":
            self.max_dps = np.full(C.N_JOINTS, SETTLE_MOTOR_DPS)
        return None

    def update(self, now: float, body):
        """One sweep.  Returns ``(mode, values, trip)``.

        `body` is a `balance.state.BodyState` -- the sweep's measurement,
        built by the caller so that `advance` and this method see the same
        one.  `mode` is "keepalive" (values None), "position" (values: (12,)
        joint rad) or "torque" (values: (12,) N*m).  `trip` is a reason to
        stop, or None.
        """
        self._sweep += 1
        self.body = body = self._measured(body)
        q, qd = body.q, body.qd
        name = self.phase_name
        elapsed = now - self.t_phase
        q4 = C.unflat(q)

        if name == "limp":
            self.h_cmd = float("nan")
            # THE ATTITUDE SETPOINT IS LATCHED HERE, AND HERE IS THE ONLY
            # PLACE IT CAN BE.  DOG5 latched it in WAIT for the same reason:
            # limp is the one phase with no torque anywhere, so the attitude
            # the IMU reports is the attitude the robot RESTS at, not one the
            # loop is enforcing.  From this sweep on, "level" for the run means
            # that reading -- see `config.SETPOINT_DYNAMIC` for what that
            # absorbs (mount tilt + floor slope + resting lean) and what it
            # costs (start on a slope, hold that slope).
            #
            # By the crouch it would already be too late: the trunk is on the
            # floor and its lean is the pose's, not the floor's.  The heading
            # half of the same convention is latched later still, at `arm`,
            # and `law.arm` says why the two differ.
            latched = self.balance.latch_setpoint(body)
            if latched is not None:
                drift = self.balance.setpoint_drift_deg()
                self.notice = (
                    "attitude setpoint latched at zero torque: roll %+.2f  "
                    "pitch %+.2f deg  -- THIS is level for the run"
                    % latched)
                if drift > BCFG.SETPOINT_WARN_DEG:
                    self.notice += (
                        "\n   WARNING: %.2f deg from the config statics "
                        "(%+.2f / %+.2f).  Was the robot resting flat, or "
                        "propped against something?  The run will hold THAT "
                        "attitude as level."
                        % (drift, BCFG.SETPOINT_ROLL_DEG,
                           BCFG.SETPOINT_PITCH_DEG))
            return "keepalive", None, None

        # THE RISE ENDS ON A CLOCK, NOT ON A KEY.  When the S-curve has
        # arrived the robot IS at height and the phase name should say so --
        # DOG5 stepped RISE -> HOLD exactly here, on `now - t0 >= T_RISE`.
        # Nothing about the torque changes across this edge: the same law
        # runs, the same gains, the same reference (the quintic clamps past
        # T).  What changes is what the operator and the LOG are told, which
        # is the whole point -- a push has to be sliceable out of the npz, and
        # "phase == hold" is how.
        if name == "rise" and self.ramp_remaining(now) <= 0.0:
            self.phase += 1
            self.t_phase = now
            self.q_ref0 = C.unflat(q).copy()
            self.hold.start(now)
            # THE JOINT TARGET IS FIXED HERE, ONCE: the angles on the sweep
            # the robot reaches HOLD.  Kept through the hold and every trot
            # -- not re-taken at T, which would carry any drift forward.
            self.balance.hold_joints(q)
            name = self.phase_name
            elapsed = 0.0
            self.notice = (
                "at height, closed loop.  PUSH THE TRUNK -- roll/pitch should "
                "come back to the setpoint, and `hold` on the status line is "
                "what it did.  ENTER parks.")

        if name == "trot" and self.trot_exit and self._four_foot_window(self.gait, now):
            self.trotting = False
            self.trot_exit = False
            self.t_phase = now
            self.hold.start(now)
            name = self.phase_name
            if self.trot_swings:
                self.notice = ("back in HOLD -- all four feet down.  T swings "
                               "the %s diagonal, ENTER %s."
                               % ("FL/RR" if self._half_next else "FR/RL",
                                  "parks" if self.feet_home
                                  else "steps the feet back and parks"))
            else:
                self.notice = ("back in HOLD after %d gait cycles -- all four "
                               "feet down.  ENTER %s, T trots again."
                               % (self.gait.cycles(now),
                                  "parks" if self.feet_home
                                  else "steps the feet back and parks"))

        if (name == "step" and self.balance.step_landed
                and self._four_foot_window(self.step_gait, now)):
            self.stepping = False
            self.balance.end_step()
            self.balance.hold_joints(q)
            self.t_phase = now
            self.hold.start(now)
            name = self.phase_name
            if self.park_after_step:
                self.park_after_step = False
                self.notice = "feet home -- parking."
                self.advance(now, q)
                name = self.phase_name
                elapsed = 0.0
            else:
                self.notice = ("feet stepped, all four down -- HOLD.  %s"
                               % ("W steps them back; T trots here; ENTER "
                                  "steps back and parks." if not self.feet_home
                                  else "ENTER parks."))

        if name in ("rise", "hold", "trot", "step"):
            if self.law == "srb":
                result = self._lift_srb(
                    now, body,
                    {"trot": self.gait, "step": self.step_gait}.get(name))
            else:
                # SINCE THE RISE, not since the phase -- see `t_lift`.
                result = self._lift_per_leg(now, now - self.t_lift, q, q4, qd)
            if name == "hold":
                self.hold.add(now, body, self.balance, self.out)
            elif name == "trot":
                self.trot.add(now, self.out.state, self.balance, self.out)
                self._count_swings()
            return result

        self.tau = np.zeros(C.N_JOINTS)
        if name == "settle":
            target = self.q_ref0
            limit = SETTLE_TRACK_ESTOP
        elif name in ("crouch", "park"):
            alpha = ST.smoothstep(elapsed / ST.RAMP_POSITION)
            target = self.q_ref0 + alpha * (self.crouch.q - self.q_ref0)
            limit = TRACK_ESTOP
        else:                                            # done
            target = self.crouch.q
            limit = TRACK_ESTOP
        self.h_cmd = (self.crouch.z_origin if name != "settle"
                      else float("nan"))
        self.q_des = C.flat(target)

        error = np.abs(q - self.q_des)
        if self.position_trips and np.any(error > limit):
            index = int(np.argmax(error))
            trip = ("%s: %s is %.1f deg from its position target (limit %.0f)"
                    % (name, HM.JOINT_LABELS[index], np.rad2deg(error[index]),
                       np.rad2deg(limit)))
            if name == "settle":
                trip += (" -- the driver's position frame disagrees with the "
                         "encoder, or something is pushing the leg")
            return "position", self.q_des, trip
        return "position", self.q_des, None

    # -- the two lift laws -------------------------------------------------
    def _lift_srb(self, now: float, body, gait=None):
        """The SRB balance controller.  Five stages, once, at slot 0.

        The law returns a trip reason; the GATE still shapes the torque
        afterwards, so there is exactly one limiter per quantity.  A trip
        during the lift is a drop, which is why the law reports and this
        method -- and ultimately `run` -- decides.
        """
        self.out = self.balance.update(now, body, clock=time.perf_counter,
                                       gait=gait)
        self.h_cmd = BSTATE.height_to_origin(self.out.command.h)
        self.q_des = self.out.q_ref
        self.gravity = self.balance.gravity
        self.tau_request = self.out.tau
        self.tau = self.gate.apply(self.tau_request, body.q, now)
        self.tau_peak = max(self.tau_peak, float(np.abs(self.tau).max()))
        return "torque", self.tau, self.out.trip

    def _lift_per_leg(self, now: float, elapsed: float, q, q4, qd):
        """`sim.stand.compliance_torque`: THE A/B BASELINE, unchanged.

        Four Cartesian springs in their own hip frames, a fixed mg/4
        feedforward, a raised-cosine height ramp, and the leg gravity term
        refreshed one leg per sweep -- so at any instant the four legs carry
        terms computed 0, 4, 8 and 12 ms ago.  Every one of those is a thing
        the SRB law changes, and leaving them exactly as they were is what
        makes the comparison mean something.

        It reads no IMU, so it trips on nothing an IMU could see.
        """
        alpha = ST.smoothstep(elapsed / ST.RAMP_LIFT)
        self.h_cmd = ST.CROUCH_HEIGHT + alpha * (ST.LIFT_HEIGHT
                                                 - ST.CROUCH_HEIGHT)
        leg = self._sweep % C.N_LEGS
        self.gravity[leg] = SK.leg_gravity_torque(leg, q4[leg])
        request = ST.compliance_torque(q4, C.unflat(qd), self.h_cmd,
                                       kin=HK, gravity=self.gravity)
        self.tau_request = C.flat(request)
        self.tau = self.gate.apply(self.tau_request, q, now)
        self.tau_peak = max(self.tau_peak, float(np.abs(self.tau).max()))
        return "torque", self.tau, None


