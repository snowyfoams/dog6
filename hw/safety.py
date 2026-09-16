"""The gate every torque command goes through, and the trips that stop a run.

Ported from DOG5's shipped hardware gate.  The SHAPE is inherited -- ramp,
cap, limit block, slew, then a separate set of e-stop tests -- because that
shape was arrived at from runs on a real machine.  The NUMBERS are DOG6's
where DOG6 has its own, and are marked where they are still DOG5's.

    1. RAMP        the cap rises from 0 to `tau_cap` over `ramp_s` when the
                   run arms, so an aggressive first tick cannot be a step
    2. CAP         |tau| <= tau_cap, then |tau| <= TAU_HARD unconditionally
    3. LIMIT BLOCK a joint at or past its soft limit cannot be pushed further
                   out; the block is immediate and is re-applied after the
                   slew limiter so residual torque cannot leak through
    4. SLEW        |dtau/dt| <= tau_slew, which is what turns a controller
                   discontinuity into a ramp instead of an impact

and, separately from shaping, :meth:`SafetyGate.estop_reason` answers "should
this run stop right now" from position, speed, temperature, missed CAN replies
and the drivers' own fault bits.

THE TRIPS ARE NOT A SAFETY NET YOU CAN LEAN ON
    DOG5 logged a run where the tilt trip did not fire, the robot ended up on
    its belly at 2.4x body weight, and read perfectly LEVEL while doing so --
    because a robot lying flat is level.  Every gate here can only see what it
    measures.  `tau_cap` starting at 1.0 N*m, and a robot mechanically
    supported for its first runs, are the things actually keeping the machine
    intact.

THE STAGING LADDER, WITH DOG6'S OWN NUMBERS UNDER IT
    Measured through `hw.kinematics`' J^T over a +-0.6 rad envelope around
    Q_STAND, which is the envelope a stand or a slow step actually visits:

        holding the robot up on all four   worst 2.20 N*m   (pitch)
        one trot diagonal, half the weight worst 4.46 N*m   (pitch)
        one leg's own weight, no contact   worst 0.19 N*m   (pitch)

    So TAU_STAGED_MAX = 3.0 is enough to STAND and deliberately not enough to
    TROT.  `params.TAU_MAX_SIM` is 8.0 and is a SIMULATION number; it must
    never be handed to this gate, and :class:`SafetyGate` refuses it.

THE OVERSPEED TRIP HAS TWO TIERS, AND THE LOWER ONE NEEDS TWO WITNESSES
    The driver's speed field and the encoder are separate numbers in the same
    reply.  On DOG5 a sustained trip on the driver field alone was a
    nuisance-trip source (a spurious 5.9 rad/s on a stationary joint), so the
    lower tier fires only when finite-differenced ENCODER position agrees, for
    `QD_ESTOP_STREAK` consecutive checks.  The hard tier fires immediately on
    either.  Call :meth:`estop_reason` (or :meth:`overspeed_reason`) exactly
    once per control decision -- the streak counter counts calls.

NOTHING HERE ARMS ON AN UNMEASURED WIRING MAP, AND THERE ARE TWO TIERS OF THAT
    EMPTY       `hardware_map` has rows with no CAN id or no direction.  There
                are no joint coordinates to shape a torque in, so the gate
                raises `MapIncomplete` and there is no override.  Nothing you
                could pass would make the signs exist.
    UNCONFIRMED  the map is filled in but `hw.CONFIRMED_ON_DOG6` is still
                False -- the rows have not been re-driven and watched.  The
                gate raises, and `unconfirmed_reason=` overrides it, because
                commanding a small torque IS part of confirming.

    Shaping a torque perfectly and sending it to the wrong motor is exactly
    the failure `hardware_map` exists to prevent, so a gate that quietly
    worked on either would make both flags decorative.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/safety.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import params as P          # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402

from .calibration import LIMIT_ESTOP_MARGIN, soft_limits   # noqa: E402
from .hardware_map import (          # noqa: E402
    JOINT_LABELS, MapIncomplete, N_JOINTS, is_complete, motor_ids, unassigned,
    unconfirmed,
)

__all__ = ["SafetyGate", "CanMissMonitor", "TorqueReadback", "TAU_HARD_NM",
           "TAU_START_MAX", "TAU_STAGED_MAX", "QD_ESTOP", "QD_ESTOP_HARD",
           "TEMP_ESTOP_C", "MISS_ESTOP", "TAU_BLACKOUT_FRAC",
           "TAU_BLACKOUT_STREAK", "describe"]

# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------
#: Absolute ceiling, above any tau_cap.  The 0xA1 iq field saturates at
#: 2048 LSB / 206.04 = `params.TAU_SATURATION` = 9.94 N*m anyway, so nothing
#: above that is even sendable; this sits below it so the gate, not the
#: protocol, is what clips.
TAU_HARD_NM = 9.0

#: `tau_cap` for a first run on a supported robot.  There is a reason it is 1:
#: it cannot hold the robot up (a four-leg stand needs 2.2), so a controller
#: that is wrong cannot push hard enough to matter.  [INHERITED FROM DOG5]
TAU_START_MAX = 1.0

#: The ceiling the staging ladder climbs to.  Enough to stand (2.20 measured),
#: not enough to trot (4.46 measured).  Past this you are not staging any
#: more, and the next number should come from a log, not from this file.
TAU_STAGED_MAX = 3.0

#: [INHERITED FROM DOG5 -- NOT MEASURED ON DOG6]  DOG5's slew and ramp.  DOG6
#: has the same motors and a lighter chassis, so these are conservative here
#: rather than optimistic, which is the right direction to inherit in.
DEFAULT_TAU_SLEW_NM_S = 5.0
DEFAULT_TORQUE_RAMP_S = 1.0

#: `params.QD_ESTOP` is the sustained tier; the hard tier is one rad/s above
#: it, as on DOG5.
QD_ESTOP = P.QD_ESTOP                   # 7.0 rad/s, needs both witnesses
QD_ESTOP_HARD = QD_ESTOP + 1.0          # rad/s, immediate, either witness
QD_ESTOP_STREAK = 3                     # consecutive confirmed checks
QD_ESTOP_ENCODER_CONFIRM_RATIO = 0.5    # encoder must exceed this * QD_ESTOP

TEMP_ESTOP_C = 80                       # [INHERITED FROM DOG5]
MISS_ESTOP = 20                         # consecutive missed CAN replies

#: The drivers' input-signal-lost protection on DOG6: a motor that hears no
#: frame for this long latches error 0x80 and goes limp.  Per the operator,
#: 2026-09-15.  Every loop that commands a motor has to re-send well inside it.
INPUT_LOST_S = 0.050


class SafetyGate:
    """Torque shaping (ramp, cap, limit block, slew) plus the e-stop tests."""

    def __init__(self, tau_cap: float = TAU_START_MAX, *,
                 tau_hard: float = TAU_HARD_NM,
                 tau_slew: float = DEFAULT_TAU_SLEW_NM_S,
                 ramp_s: float = DEFAULT_TORQUE_RAMP_S,
                 qd_estop: float = QD_ESTOP,
                 qd_estop_hard: float = QD_ESTOP_HARD,
                 limits=None,
                 unconfirmed_reason: str | None = None,
                 ceiling: float = TAU_STAGED_MAX,
                 overspeed_trip: bool = True):
        # Tier 1: no signs exist.  No override, because there is nothing to
        # override -- `soft_limits` below is in joint coordinates and there is
        # no map from a joint to a motor to shape a torque through.
        if not is_complete():
            raise MapIncomplete(
                "a torque gate needs joint coordinates, and %d of %d rows "
                "of hw/hardware_map.py are empty (%s).\n"
                "  DOG6's twelve were measured on 2026-09-15, so an empty row "
                "means an edited table or a\n  replaced driver -- and eleven "
                "signs out of twelve is not a map.\n"
                "  Re-measure that row: `python -m hw.bringup spin --id <n>`."
                % (len(unassigned()), N_JOINTS, ", ".join(unassigned())))
        # Tier 2: the signs exist but nobody has watched them move.
        if not CONFIRMED_ON_DOG6 and not unconfirmed_reason:
            raise RuntimeError(
                "hw.CONFIRMED_ON_DOG6 is False: %d of %d joints are filled in "
                "but have not been re-driven in joint coordinates and watched "
                "(%s).\n"
                "  Confirm them: `python -m hw.bringup check --joint <label> "
                "--go`.\n"
                "  To command torque DURING that check, pass "
                "unconfirmed_reason='...' and keep tau_cap small."
                % (len(unconfirmed()), N_JOINTS, ", ".join(unconfirmed())))
        if tau_cap > tau_hard:
            raise ValueError(f"tau_cap {tau_cap} exceeds the hard limit "
                             f"{tau_hard} N*m")
        # `ceiling` IS A DELIBERATE ARGUMENT, NOT A FLAG.  The stand keeps the
        # staging ceiling; `hw.fold_trot` raises it to TAU_HARD_NM on the
        # operator's decision of 2026-09-16 (a diagonal pair in the fold stance
        # needs 3.3 N*m offline).  tau_hard still bounds it from above.
        if tau_cap > ceiling:
            raise ValueError(
                f"tau_cap {tau_cap} is above the staging ceiling "
                f"{ceiling} N*m.  If you meant params.TAU_MAX_SIM "
                f"({P.TAU_MAX_SIM}), that is a SIMULATION number and does not "
                "belong on hardware.  Raise TAU_STAGED_MAX deliberately, from "
                "a log, one step at a time.")
        self.tau_cap = float(tau_cap)
        self.tau_hard = float(tau_hard)
        self.tau_slew = float(tau_slew)
        self.ramp_s = float(ramp_s)
        self.qd_estop = float(qd_estop)
        self.qd_estop_hard = float(qd_estop_hard)
        self.unconfirmed_reason = unconfirmed_reason
        #: False is DOG5's trot runner (`trot_hw.TorqueGate.overspeed_reason`):
        #: both tiers still measure and keep their peaks for the exit report,
        #: and neither stops the run.  `hw.fold_trot` passes it -- see there.
        self.overspeed_trip = bool(overspeed_trip)
        self.low, self.high = soft_limits() if limits is None else limits

        self.started_at = None
        self.last_time = None
        self.previous_tau = np.zeros(N_JOINTS)
        self.qd_streaks = np.zeros(N_JOINTS, dtype=int)
        self.qd_peak = np.zeros(N_JOINTS)
        self.encoder_qd = np.zeros(N_JOINTS)
        self.encoder_qd_peak = np.zeros(N_JOINTS)
        self._overspeed_last_q = None
        self._overspeed_last_time = None

    # -- lifecycle -------------------------------------------------------
    def start(self, now: float, q=None) -> None:
        """Arm the gate at `now`; the ramp starts here.  `q` seeds the
        encoder-velocity witness so the first check is not a step."""
        self.started_at = float(now)
        self.last_time = float(now)
        if q is not None:
            self._overspeed_last_q = np.asarray(q, dtype=float).copy()
            self._overspeed_last_time = float(now)

    def cap_now(self, now: float) -> float:
        """The cap in force at `now` -- `tau_cap` scaled by the ramp."""
        if self.started_at is None:
            raise RuntimeError("SafetyGate.start() has not been called")
        fraction = np.clip((now - self.started_at) / self.ramp_s, 0.0, 1.0)
        return self.tau_cap * float(fraction)

    # -- shaping ---------------------------------------------------------
    def apply(self, tau, q, now: float) -> np.ndarray:
        """Shape a requested (12,) torque into the one to actually send."""
        cap = self.cap_now(now)
        q = np.asarray(q, dtype=float)
        limited = np.clip(np.asarray(tau, dtype=float), -cap, cap)
        limited = np.clip(limited, -self.tau_hard, self.tau_hard)
        limited = np.where((q >= self.high) & (limited > 0.0), 0.0, limited)
        limited = np.where((q <= self.low) & (limited < 0.0), 0.0, limited)

        dt = np.clip(now - self.last_time, 1.0e-4, 0.05)
        max_change = self.tau_slew * dt
        output = self.previous_tau + np.clip(
            limited - self.previous_tau, -max_change, max_change
        )
        # A directional limit block is immediate, even where the slew limiter
        # would otherwise leave residual torque pointing further out of bounds.
        output = np.where((q >= self.high) & (output > 0.0), 0.0, output)
        output = np.where((q <= self.low) & (output < 0.0), 0.0, output)
        self.previous_tau = output
        self.last_time = float(now)
        return output

    # -- trips -----------------------------------------------------------
    def overspeed_reason(self, qd, q, now: float):
        """Two-tier overspeed trip.  One call == one check; see the module
        docstring.  Returns a reason string, or None."""
        qd = np.asarray(qd, dtype=float)
        q = np.asarray(q, dtype=float)
        speed = np.abs(qd)
        self.qd_peak = np.maximum(self.qd_peak, speed)

        if self._overspeed_last_q is None:
            confirmed = np.zeros(N_JOINTS, dtype=bool)
            self.encoder_qd.fill(0.0)
        else:
            dt = float(now) - self._overspeed_last_time
            if np.isfinite(dt) and dt > 0.0:
                self.encoder_qd = (q - self._overspeed_last_q) / dt
                limit = QD_ESTOP_ENCODER_CONFIRM_RATIO * self.qd_estop
                confirmed = np.abs(self.encoder_qd) > limit
                self.encoder_qd_peak = np.maximum(
                    self.encoder_qd_peak, np.abs(self.encoder_qd)
                )
            else:
                self.encoder_qd.fill(0.0)
                confirmed = np.zeros(N_JOINTS, dtype=bool)
        self._overspeed_last_q = q.copy()
        self._overspeed_last_time = float(now)
        if not self.overspeed_trip:
            return None

        if np.any(speed > self.qd_estop_hard):
            index = int(np.argmax(speed))
            return (f"overspeed {JOINT_LABELS[index]}: {qd[index]:+.1f} rad/s "
                    f"over the {self.qd_estop_hard:.1f} rad/s hard limit")

        self.qd_streaks = np.where(
            (speed > self.qd_estop) & confirmed, self.qd_streaks + 1, 0
        )
        tripped = self.qd_streaks >= QD_ESTOP_STREAK
        if np.any(tripped):
            index = int(np.argmax(np.where(tripped, speed, 0.0)))
            return (f"confirmed overspeed {JOINT_LABELS[index]}: driver "
                    f"{qd[index]:+.1f} rad/s, encoder "
                    f"{self.encoder_qd[index]:+.1f} rad/s for "
                    f"{int(self.qd_streaks[index])} consecutive checks over "
                    f"the {self.qd_estop:.1f} rad/s driver limit")
        return None

    def estop_reason(self, q, qd, now: float, *, temps=None,
                     miss_streaks=None, errors=None,
                     enforce_position_limits: bool = True):
        """Should this run stop?  Returns a reason string, or None.

        `temps`, `miss_streaks` and `errors` are the hardware-only witnesses
        and may be omitted (the simulator has none of them).
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            return "invalid encoder or velocity state"

        if enforce_position_limits:
            outside = ((q < self.low - LIMIT_ESTOP_MARGIN)
                       | (q > self.high + LIMIT_ESTOP_MARGIN))
            if np.any(outside):
                index = int(np.flatnonzero(outside)[0])
                return (f"joint limit exceeded: {JOINT_LABELS[index]}="
                        f"{q[index]:+.2f} rad")

        overspeed = self.overspeed_reason(qd, q, now)
        if overspeed:
            return overspeed

        if temps is not None:
            temps = np.asarray(temps)
            if np.any(temps > TEMP_ESTOP_C):
                index = int(np.argmax(temps))
                return (f"overtemp CAN {motor_ids()[index]}: "
                        f"{int(temps[index])} C")

        if miss_streaks is not None:
            miss_streaks = np.asarray(miss_streaks)
            if np.any(miss_streaks >= MISS_ESTOP):
                index = int(np.argmax(miss_streaks))
                return (f"CAN {motor_ids()[index]} missed "
                        f"{int(miss_streaks[index])} consecutive replies")

        if errors:
            # bit 7 (0x80) is the input-signal-lost latch, which the runner
            # recovers over CAN; bits 0-6 are real faults and stop the run.
            hard = {mid: err for mid, err in errors.items() if err & 0x7F}
            if hard:
                detail = ", ".join(f"CAN {mid}=0x{err:02x}"
                                   for mid, err in hard.items())
                return f"motor fault: {detail}"
        return None


#: The measured-torque witness.  A motor is "not producing" when the drivers
#: report q-axis current under this fraction of what was commanded, and only
#: when something substantial was commanded (`TAU_BLACKOUT_FLOOR`) -- near
#: zero the ratio is meaningless.  [INHERITED FROM DOG5, where a blackout read
#: exactly zero measured torque with the command pinned at the cap]
TAU_BLACKOUT_FRAC = 0.25
TAU_BLACKOUT_FLOOR = 0.30               # N*m commanded, below which no test
TAU_BLACKOUT_STREAK = 50                # sweeps -> 0.2 s at 250 Hz


class TorqueReadback:
    """Commanded torque against the q-axis current the drivers report back.

    THE TRIP NOTHING ELSE CAN STAND IN FOR.  Every other witness in this file
    asks the drivers how they are, and a driver in a brown-out answers
    cheerfully: on DOG5 a blackout produced ZERO measured torque with the
    command pinned at the cap, CAN still replying, no fault bit, no
    over-speed, no over-temperature, no missed frame.  The only number that
    disagreed was the one in the reply nobody was reading.

    IT IS A STREAK AND IT HAS A FLOOR, for the same reason the overspeed trip
    has two witnesses.  The measured current lags the command through the
    current loop and the gearbox, so an instantaneous ratio is noisy during
    every transient; and near zero commanded torque the ratio is not a
    quantity at all.  So the test only runs on motors being asked for real
    torque, and only fires when a majority of them have been silent together
    for `TAU_BLACKOUT_STREAK` consecutive checks -- one limp motor is a
    mechanical fault, and all twelve at once is the power rail.

    Call `reason` exactly once per control decision; the streak counts calls.
    """

    def __init__(self, n_joints: int = N_JOINTS, *,
                 fraction: float = TAU_BLACKOUT_FRAC,
                 floor: float = TAU_BLACKOUT_FLOOR,
                 streak: int = TAU_BLACKOUT_STREAK):
        self.fraction = float(fraction)
        self.floor = float(floor)
        self.streak_limit = int(streak)
        self.streak = 0
        self.peak = np.zeros(n_joints)
        self.total = np.zeros(n_joints)
        self.samples = 0

    def reason(self, tau_cmd, tau_meas, *, live: bool = True):
        """A trip reason, or None.  `live` is False outside torque mode.

        The peak and the MEAN per motor accumulate regardless of `live`, so
        the exit report can say what the motors actually produced across the
        whole run -- which is the end-to-end evidence that commanded torque
        became real current, and the only such evidence there is until the
        torque constant is calibrated.
        """
        tau_cmd = np.asarray(tau_cmd, dtype=float)
        tau_meas = np.asarray(tau_meas, dtype=float)
        self.peak = np.maximum(self.peak, np.abs(tau_meas))
        self.total += np.abs(tau_meas)
        self.samples += 1
        if not live:
            self.streak = 0
            return None

        asked = np.abs(tau_cmd) > self.floor
        if not asked.any():
            self.streak = 0
            return None
        silent = asked & (np.abs(tau_meas)
                          < self.fraction * np.abs(tau_cmd))
        if silent.sum() <= asked.sum() // 2:
            self.streak = 0
            return None
        self.streak += 1
        if self.streak < self.streak_limit:
            return None
        index = int(np.argmax(np.where(silent, np.abs(tau_cmd), 0.0)))
        return ("%d of %d loaded motors reported under %.0f %% of their "
                "commanded torque for %d consecutive sweeps (worst %s: "
                "asked %.2f, measured %.2f N*m) -- the drivers are answering "
                "but not producing"
                % (int(silent.sum()), int(asked.sum()), 100 * self.fraction,
                   self.streak, JOINT_LABELS[index], tau_cmd[index],
                   tau_meas[index]))

    def report(self) -> str:
        if self.samples == 0:
            return "  torque readback no samples"
        mean = self.total / self.samples
        return ("  torque readback  peak %.2f N*m, mean %.2f N*m over %d "
                "sweeps (worst motor %s)"
                % (self.peak.max(), mean.max(), self.samples,
                   JOINT_LABELS[int(np.argmax(self.peak))]))


class CanMissMonitor:
    """Consecutive-missed-reply streak per joint, from a `MotorBus`'s counters."""

    def __init__(self, mb):
        self._ids = motor_ids()
        self.previous = np.asarray([mb.rec(mid).missed for mid in self._ids])
        self.streaks = np.zeros(N_JOINTS, dtype=int)

    def update(self, mb) -> np.ndarray:
        current = np.asarray([mb.rec(mid).missed for mid in self._ids])
        added = current - self.previous
        self.streaks = np.where(added > 0, self.streaks + added, 0)
        self.previous = current
        return self.streaks.copy()


def describe() -> str:
    return "\n".join([
        "DOG6 safety gate",
        "  tau ladder      start %.1f  ->  staged ceiling %.1f  ->  hard %.1f"
        "  (iq saturates at %.2f)"
        % (TAU_START_MAX, TAU_STAGED_MAX, TAU_HARD_NM, P.TAU_SATURATION),
        "  measured need   stand on four 2.20   trot diagonal 4.46   leg's own"
        " weight 0.19 N*m",
        "                  so %.1f stands and does NOT trot, by design"
        % TAU_STAGED_MAX,
        "  slew / ramp     %.1f N*m/s, %.1f s   [INHERITED FROM DOG5]"
        % (DEFAULT_TAU_SLEW_NM_S, DEFAULT_TORQUE_RAMP_S),
        "  overspeed       %.1f rad/s sustained (two witnesses, %d checks), "
        "%.1f immediate" % (QD_ESTOP, QD_ESTOP_STREAK, QD_ESTOP_HARD),
        "  overtemp        %d C          CAN miss  %d consecutive"
        % (TEMP_ESTOP_C, MISS_ESTOP),
        "  arming          %s"
        % ("permitted" if CONFIRMED_ON_DOG6 else
           "REFUSED -- hardware_map is EMPTY (%d rows), no joint coordinates "
           "exist" % len(unassigned()) if not is_complete() else
           "REFUSED -- hw.CONFIRMED_ON_DOG6 is False (%d joints unconfirmed)"
           % len(unconfirmed())),
        "",
        "  params.TAU_MAX_SIM is %.1f and is a SIMULATION number.  This gate "
        "refuses it." % P.TAU_MAX_SIM,
    ])


if __name__ == "__main__":
    print(describe())
