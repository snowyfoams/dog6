"""Every number the hardware MPC path adds.  The controller's own numbers are
NOT here -- they stay in `sim.cmpc.config`, so the robot runs the constants
the simulation was gated with.

    python -m hw.cmpc.config          print the lot, with its provenance

Tags, as in `hw.balance.config`:

    [SIM]        `sim.cmpc.config`'s value, read rather than copied
    [HW]         a fact about the robot or its loop, from `hw.stand`/`hw.safety`
    [DERIVED]    computed here from the two above
    [UNTUNED]    a threshold with no hardware provenance yet
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/cmpc/config.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.cmpc"

from sim import params as P                  # noqa: E402
from sim import stand as ST                  # noqa: E402
from sim.cmpc import config as SIMCFG        # noqa: E402

from .. import stand as HS                   # noqa: E402
from ..balance import config as BCFG         # noqa: E402

# ===========================================================================
# the gait
# ===========================================================================
#: What `--gait` may be.  `stand` is four feet down always and is the default;
#: `trot` is `sim.cmpc.gait`'s timetable, run when asked for (no gate and no
#: refusal since 2026-11-10, on request).
GAITS = ("stand", "trot")
DEFAULT_GAIT = "stand"

# ===========================================================================
# the solve, inside the CAN schedule
# ===========================================================================
#: The MPC period on the robot.  The simulator's by default.  A TROT must not
#: go below 40 Hz -- `sim.cmpc.config.MPC_HZ` has the measured subharmonic --
#: but a four-foot STAND has no diagonal pendulum mode to excite, so a Pi
#: that cannot afford 40 may run the stand gait at 20.  Measured, then set:
#: the exit report prints the solve's p50/p95/max every run.  [SIM]
MPC_HZ = float(SIMCFG.MPC_HZ)
MPC_HZ_MIN_TROT = 40.0

#: THE SOLVE RUNS AT SLOT 0 AND DELAYS EVERY MOTOR BEHIND IT.  `hw.stand.run`
#: stops the run when any motor's command gap passes `GAP_ESTOP_S` (25 ms,
#: half the drivers' 50 ms input-lost window).  A solve of S seconds pushes
#: the worst gap to about S + two sweeps (8 ms, when the status frame falls on
#: that motor), so S must stay well under 17 ms.  12 ms leaves 5 ms for the
#: operating system, and the loop re-anchors rather than catching up.
#:
#: The desktop solves in 4.3 ms median; the Pi 4.7 ms (fake bus, stand gait,
#: 2026-11-10).  A REPORTING line, not a trip (2026-11-10, on request): an
#: over-budget solve is counted in the exit report, and the CAN gap stop line
#: in `hw.stand.run` is what stops a loop that really stalls.
#: [DERIVED from hw.stand.GAP_ESTOP_S]
SOLVE_BUDGET_S = 0.012
assert SOLVE_BUDGET_S + 2 * (1.0 / HS.RATE_HZ) < HS.GAP_ESTOP_S

#: Statuses whose forces are used.  Any other status keeps the LAST GOOD
#: forces (`run.HardwareController`) -- the zero-order hold the method assumes
#: anyway -- where the QP itself returns zero.  No trip (2026-11-10).
QP_OK_STATUSES = ("solved", "solved inaccurate")

# ===========================================================================
# the state estimate
# ===========================================================================
#: Low-pass on the kinematic trunk velocity.  The encoder difference is
#: already filtered at `hw.stand.QD_FILTER_HZ` (40 Hz); this sits below it so
#: the MPC's velocity rows see the trunk and not the quantisation.  [UNTUNED]
VELOCITY_FILTER_HZ = 20.0

#: Pitch beyond which the Euler yaw-rate conversion is not evaluated and the
#: previous yaw rate is held.  Far outside the tilt trip; a guard, not a limit.
YAW_RATE_PITCH_LIMIT_RAD = float(np.deg2rad(60.0))

# ===========================================================================
# the height command
# ===========================================================================
#: Where the MPC starts: the trunk-origin height MEASURED at the handover,
#: not a constant -- the same latch `hw.balance.law.BalanceLaw.arm` makes and
#: for the same reason (a step into a differentiated term).  `--mpc-height`
#: then ramps from there with the balance package's C2 quintic.
#:
#: Bounds on what may be commanded, trunk-ORIGIN frame.  The floor keeps the
#: trunk box off the ground with margin; the ceiling is the CAD's drawn
#: stance, which `sim.cmpc` was gated at.  [DERIVED]
Z_MIN = float(ST.CROUCH_HEIGHT + 0.040)                   # 0.075 m
Z_MAX = float(P.STAND_HEIGHT)                             # 0.1925 m
#: One press of + or - on the keyboard during the MPC phase.
Z_NUDGE_M = 0.005
#: Seconds a nudge or a `--mpc-height` ramp takes.  The quintic's peak rate is
#: 1.875 dh/T, so 5 mm in 1 s is 9 mm/s.
Z_RAMP_S = 1.0

# ===========================================================================
# the MPC phase on a tilt.  It adds no trips to the stand's but NaN
# (2026-11-10, on request: the height, solve-budget and QP-failure trips,
# each a limp drop from the lift height, are gone)
# ===========================================================================
#: THE FALL HOLD, IN PLACE OF THE TILT RUN-STOP, 2026-10-06, on request ("set
#: the cmpc tilt stop is 45 and also if bigger 45, no stop, just hold the
#: joint angle").  The stop was the balance package's 12 deg, and a stop here
#: is `MotorBus.close()`: the robot dropped limp from the lift height.  Now
#: nothing on a tilt stops the MPC phase.  Past this `BodyState.tilt_deg`
#: (max |roll|, |pitch|) the MPC stops for the rest of the run and the drivers
#: hold every joint at the angle it had on that sweep -- `run.HardwareCmpc`,
#: phase "fall" -- until ENTER parks or X.  `--fall-hold`; 0 turns it off,
#: and then nothing acts on a tilt.  `cloud`'s trots hold past the same 45.
#:
#: IN THE DRIVERS' POSITION LOOP (0xA4), NOT A TORQUE PD: it is how settle,
#: crouch, park and done hold the joints, and the hold runs at settle's speed
#: cap, `hw.stand.SETTLE_MOTOR_DPS`.
FALL_HOLD_DEG = 45.0

#: Velocity commands the operator may give in the TROT gait.  `sim.cmpc`'s
#: ceilings are 0.6 m/s and 90 deg/s and were reached in simulation at 8 N*m;
#: nothing on the staging ladder has earned those.  [UNTUNED]
V_MAX_HW = 0.2                                            # m/s
YAW_RATE_MAX_HW = float(np.deg2rad(30.0))                 # rad/s

#: IMU staleness is HELD, not tripped, as in `hw.balance` -- the controller
#: keeps the last R and the estimator the last attitude.  The threshold is
#: the balance package's.  [HW]
IMU_MAX_AGE_S = float(BCFG.IMU_MAX_AGE_S)


def describe() -> str:
    return "\n".join([
        "DOG6 hardware MPC configuration (the controller's own numbers are "
        "sim.cmpc.config's)",
        "  gait            %s by default; --gait trot runs, no gate"
        % DEFAULT_GAIT,
        "  MPC             %.0f Hz (sim's); trot not below %.0f, stand may go "
        "lower" % (MPC_HZ, MPC_HZ_MIN_TROT),
        "  solve budget    %.0f ms, counted, no trip; CAN stop line %.0f ms"
        % (1e3 * SOLVE_BUDGET_S, 1e3 * HS.GAP_ESTOP_S),
        "  QP status       %s used; any other keeps the last good forces"
        % " / ".join(QP_OK_STATUSES),
        "  estimator       velocity low-pass %.0f Hz, yaw from the gyro"
        % VELOCITY_FILTER_HZ,
        "  height          latched at handover; commandable %.0f..%.0f mm "
        "origin (%.0f..%.0f mm floor to trunk bottom)"
        % (1e3 * Z_MIN, 1e3 * Z_MAX,
           1e3 * (Z_MIN - BCFG.TRUNK_BOTTOM_OFFSET),
           1e3 * (Z_MAX - BCFG.TRUNK_BOTTOM_OFFSET)),
        "  tilt            no trip; past %.0f deg the fall hold, every joint "
        "held where it was (0xA4)" % FALL_HOLD_DEG,
        "  trips           NaN only; IMU age %.0f ms holds"
        % (1e3 * IMU_MAX_AGE_S),
        "  trot command    |v| <= %.2f m/s, |yaw rate| <= %.0f deg/s"
        % (V_MAX_HW, np.rad2deg(YAW_RATE_MAX_HW)),
    ])


if __name__ == "__main__":
    print(describe())
