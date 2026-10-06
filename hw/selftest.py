"""Gate the hardware layer with no robot, no adapter and no simulator.

    python -m hw.selftest

WHAT THIS CAN AND CANNOT TELL YOU
    CAN     that the ported code speaks the LK protocol correctly, that the
            direction convention is applied consistently in BOTH directions,
            that the encoder unwrap survives a wrap, that the safety gate
            shapes and trips the way it claims, that the IMU frame maths is
            self-consistent -- and that every path needing joint coordinates
            REFUSES while `hardware_map` is empty.  All of that is code, and
            code can be checked on a laptop.

    CANNOT  which motor is which joint, or which way it turns.  Those are
            physical facts about an assembled robot.  Nothing here can supply
            them and nothing here pretends to.

    So a green run means "the plumbing is sound", never "it is safe to power
    the robot".  That distinction is why this file exists separately from
    `sim.selftest`, which gates the kinematics against MuJoCo.

THE MAP IS EMPTY, SO THE PROTOCOL SECTION SUPPLIES ITS OWN
    `hardware_map` has no CAN ids and no directions -- see there for why
    DOG5's are not borrowed.  Section 5 therefore installs a SYNTHETIC map
    with deliberately awkward ids and mixed signs, and drives the real code
    through it: `motor_ids`, `motor_directions`, `MotorBus`, `joint_state`,
    `set_zero_all`, `SafetyGate`.  The synthetic table lives for the length of
    a `with` block and is never importable, so nothing can accidentally run on
    it -- but every line it exercises is a line the real map will run through.

    Section 3 checks the other half: that those same paths REFUSE on the real,
    empty map, with `MapIncomplete` and no override.

THE DIRECTION ROUND-TRIP IS THE CHECK WORTH HAVING
    `MotorBus.position` applies a joint's direction on the way out;
    `MotorBus.encoders_deg` does NOT apply it on the way back, while
    `speeds_dps` and `torques_nm` DO.  Three conventions in one class, all
    correct, all easy to mix up -- and a mix-up is a joint that reads the
    mirror of where it is.  Section 5 drives all twelve through a real 0xA4
    and reads them back through `hw.calibration`, which is the only
    combination a controller actually uses.
"""
from __future__ import annotations

import contextlib
import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/selftest.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from . import CONFIRMED_ON_DOG6      # noqa: E402
from . import calibration as CAL     # noqa: E402
from . import hardware_map as HM     # noqa: E402
from . import imu as IMU             # noqa: E402
from . import kinematics as K        # noqa: E402
from . import safety as SAFE         # noqa: E402
from .hardware_map import MapIncomplete   # noqa: E402

_FAILURES: list[str] = []
_PASSES = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _PASSES
    if ok:
        _PASSES += 1
        print("  ok    %-62s %s" % (label, detail))
    else:
        _FAILURES.append(label)
        print("  FAIL  %-62s %s" % (label, detail))


def close(label: str, a, b, tol: float, unit: str = "") -> None:
    worst = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    check(label, worst <= tol, "worst %.3g%s (tol %.3g)" % (worst, unit, tol))


def raises(label: str, fn, exc=Exception, detail: str = "") -> None:
    try:
        fn()
    except exc:
        check(label, True, detail)
    except Exception as other:       # noqa: BLE001
        check(label, False, f"raised {type(other).__name__} instead")
    else:
        check(label, False, "did not raise")


# ---------------------------------------------------------------------------
#: A map that is NOT DOG6's and is not DOG5's either.  Ids ascend in sevens
#: and the signs alternate in a pattern no real harness would produce, both on
#: purpose: if this table ever leaks out of its `with` block, nothing about it
#: looks plausible enough to be mistaken for a measurement.
_SYNTHETIC_IDS = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
_SYNTHETIC_DIRS = [+1, -1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1]


@contextlib.contextmanager
def _synthetic_map():
    """Install `_SYNTHETIC_IDS`/`_SYNTHETIC_DIRS` for the length of a block.

    Patches `hardware_map.HARDWARE_JOINTS` in place.  Everything downstream --
    `motor_ids`, `directions`, `is_complete`, `hw.calibration`, `hw.safety` --
    reaches the table through a FUNCTION that resolves it at call time, which
    is exactly why they are functions.  A module that imported the tuple by
    name would not see this, and would silently keep testing the empty map.
    """
    original = HM.HARDWARE_JOINTS
    HM.HARDWARE_JOINTS = tuple(
        HM.HardwareJoint(row.leg, row.joint, row.model_name, can_id=can_id,
                         direction=direction, confirmed=True,
                         note="hw.selftest synthetic map, not a measurement")
        for row, can_id, direction in zip(original, _SYNTHETIC_IDS,
                                          _SYNTHETIC_DIRS))
    try:
        HM.validate()
        yield
    finally:
        HM.HARDWARE_JOINTS = original


# ===========================================================================
def main() -> int:
    print("DOG6 hardware self-test  (no robot, no adapter, no MuJoCo)\n")

    # -- 1. the tables ------------------------------------------------------
    print("tables")
    HM.validate()
    CAL.validate()
    check("hardware_map and calibration validate on an EMPTY map", True,
          "%d rows, %d wired" % (HM.N_JOINTS, len(HM.assigned())))
    check("the map is in coordinates.JOINT_NAMES order",
          tuple(j.model_name for j in HM.HARDWARE_JOINTS) == C.JOINT_NAMES)
    check("no row carries a borrowed CAN id or direction",
          all(j.can_id is None and j.direction is None
              for j in HM.HARDWARE_JOINTS),
          "DOG5's table is deliberately absent")
    check("the module flag cannot be True on an incomplete map",
          not (CONFIRMED_ON_DOG6 and not HM.is_complete()),
          "%d/%d wired, %d/%d confirmed, CONFIRMED_ON_DOG6=%s"
          % (len(HM.assigned()), HM.N_JOINTS, HM.confirmed_count(),
             HM.N_JOINTS, CONFIRMED_ON_DOG6))
    check("Q_STAND sits inside params.JOINT_LIMITS with room",
          bool(np.all(np.abs(C.flat(C.Q_STAND))
                      < np.tile(P.JOINT_LIMITS[:, 1], C.N_LEGS) - 0.3)),
          "%.3f rad of margin"
          % float(np.min(np.tile(P.JOINT_LIMITS[:, 1], C.N_LEGS)
                         - np.abs(C.flat(C.Q_STAND)))))

    # `validate` has to catch the ways a hand-edited table goes wrong, because
    # a hand-edited table is the only kind this project will ever have.
    def _with_rows(rows):
        original = HM.HARDWARE_JOINTS
        HM.HARDWARE_JOINTS = rows
        try:
            HM.validate()
        finally:
            HM.HARDWARE_JOINTS = original

    base = HM.HARDWARE_JOINTS
    raises("validate rejects two joints claiming one CAN id",
           lambda: _with_rows(
               (base[0].__class__(base[0].leg, base[0].joint,
                                  base[0].model_name, can_id=9, direction=+1),
                base[1].__class__(base[1].leg, base[1].joint,
                                  base[1].model_name, can_id=9, direction=-1))
               + base[2:]), ValueError)
    raises("...a direction that is not +1 or -1",
           lambda: _with_rows(
               (base[0].__class__(base[0].leg, base[0].joint,
                                  base[0].model_name, can_id=9, direction=0),)
               + base[1:]), ValueError)
    raises("...and a row marked confirmed with nothing measured in it",
           lambda: _with_rows(
               (base[0].__class__(base[0].leg, base[0].joint,
                                  base[0].model_name, confirmed=True),)
               + base[1:]), ValueError)

    # -- 2. what survives an empty map -------------------------------------
    print("\nthe motor frame, which needs no map")
    check("the conversion has NO gearbox division",
          abs(CAL.ENCODER_GAIN - 360.0 / 65535.0) < 1e-15,
          "ENCODER_GAIN = %.9f = 360/65535, the OUTPUT shaft" % CAL.ENCODER_GAIN)
    unwrap = CAL.EncoderUnwrap()
    degrees = [unwrap.update(r) for r in
               (100, 50, 10, 65530, 65500, 65530, 10, 50, 100)]
    check("EncoderUnwrap is continuous across a wrap, both ways",
          float(np.max(np.abs(np.diff(degrees)))) < 1.0,
          "largest step %.4f deg over a track that crosses zero twice"
          % float(np.max(np.abs(np.diff(degrees)))))
    first = CAL.EncoderUnwrap().update(65535 - 100)
    check("...and centres the FIRST sample into [-180, 180)",
          -180.0 <= first < 180.0, "raw 65435 -> %.3f deg" % first)
    raises("...and refuses a value outside uint16",
           lambda: CAL.EncoderUnwrap().update(70000), ValueError)
    low, high = CAL.soft_limits()
    check("soft_limits works with no map -- it is a convention, not wiring",
          low.shape == (HM.N_JOINTS,) and bool(np.all(high > low)),
          "abd +-%.2f  pitch +-%.2f  knee +-%.2f rad"
          % (P.ABD_LIM, P.PITCH_LIM, P.KNEE_LIM))

    # -- 3. what REFUSES on an empty map -----------------------------------
    print("\nthe refusals, which are the point of the empty map")
    probe = np.linspace(-2.5, 2.5, HM.N_JOINTS)
    raises("motoroutput_deg -> joint_rad refuses",
           lambda: CAL.motoroutput_deg_to_joint_rad(probe), MapIncomplete,
           "MapIncomplete naming the %d rows still to measure"
           % len(HM.unassigned()))
    raises("...and joint_rad -> motoroutput_deg refuses",
           lambda: CAL.joint_rad_to_motoroutput_deg(probe), MapIncomplete)
    raises("hardware_map.motor_ids() refuses", HM.motor_ids, MapIncomplete)
    raises("hardware_map.motor_directions() refuses", HM.motor_directions,
           MapIncomplete)
    raises("SafetyGate refuses to construct", SAFE.SafetyGate, MapIncomplete)
    raises("...and unconfirmed_reason does NOT override that tier",
           lambda: SAFE.SafetyGate(0.5, unconfirmed_reason="a good reason"),
           MapIncomplete, "there are no signs for a reason to override")
    raises("FakeDriverBus refuses to invent a default id list",
           lambda: __import__("hw.fake_bus", fromlist=["x"]).FakeDriverBus(),
           MapIncomplete)
    check("hw.require_confirmed() refuses",
          _refuses(lambda: __import__("hw", fromlist=["x"]).require_confirmed()))

    # -- 4. the safety gate, under a synthetic map -------------------------
    print("\nthe safety gate")
    check("the staged ceiling stands but does not trot",
          2.20 < SAFE.TAU_STAGED_MAX < 4.46,
          "%.1f N*m, against 2.20 measured to stand and 4.46 to trot"
          % SAFE.TAU_STAGED_MAX)
    check("the hard ceiling is under the iq saturation",
          SAFE.TAU_HARD_NM < P.TAU_SATURATION,
          "%.1f < %.2f N*m" % (SAFE.TAU_HARD_NM, P.TAU_SATURATION))

    with _synthetic_map():
        raises("refuses params.TAU_MAX_SIM (%.1f), which is a SIM number"
               % P.TAU_MAX_SIM,
               lambda: SAFE.SafetyGate(P.TAU_MAX_SIM,
                                       unconfirmed_reason="test"), ValueError)
        gate = SAFE.SafetyGate(2.0, unconfirmed_reason="hw.selftest")
        gate.start(0.0, q=np.zeros(HM.N_JOINTS))
        q0 = np.zeros(HM.N_JOINTS)
        demanded = np.full(HM.N_JOINTS, 5.0)
        check("the ramp starts at zero", abs(gate.cap_now(0.0)) < 1e-12)
        close("...and reaches tau_cap after ramp_s",
              gate.cap_now(gate.ramp_s), gate.tau_cap, 1e-12, " N m")
        out = gate.apply(demanded, q0, 0.004)
        check("the slew limiter turns a 5 N*m step into a ramp",
              float(np.max(np.abs(out)))
              <= SAFE.DEFAULT_TAU_SLEW_NM_S * 0.004 + 1e-12,
              "first tick %.4f N*m" % float(np.max(np.abs(out))))
        t = 0.004
        for _ in range(2000):
            t += 0.004
            out = gate.apply(demanded, q0, t)
        close("...and settles at the cap, never above it",
              float(np.max(np.abs(out))), gate.tau_cap, 1e-9, " N m")

        at_high = high.copy()
        g2 = SAFE.SafetyGate(2.0, unconfirmed_reason="hw.selftest")
        g2.start(0.0, q=at_high)
        for i in range(500):
            pushing = g2.apply(np.full(HM.N_JOINTS, +5.0), at_high, i * 0.004)
        check("a joint at its high limit cannot be pushed further out",
              float(np.max(pushing)) <= 0.0,
              "worst %+.3g N m" % float(np.max(pushing)))
        g3 = SAFE.SafetyGate(2.0, unconfirmed_reason="hw.selftest")
        g3.start(0.0, q=at_high)
        for i in range(500):
            pulling = g3.apply(np.full(HM.N_JOINTS, -5.0), at_high, i * 0.004)
        check("...but can still be pulled back in",
              float(np.min(pulling)) < -1.0, "%.3f N m" % float(np.min(pulling)))

        g4 = SAFE.SafetyGate(1.0, unconfirmed_reason="hw.selftest")
        g4.start(0.0, q=np.zeros(HM.N_JOINTS))
        fast = np.zeros(HM.N_JOINTS)
        fast[3] = SAFE.QD_ESTOP_HARD + 1.0
        check("the hard overspeed tier fires immediately, one witness",
              bool(g4.overspeed_reason(fast, np.zeros(HM.N_JOINTS), 0.004)))
        g5 = SAFE.SafetyGate(1.0, unconfirmed_reason="hw.selftest")
        g5.start(0.0, q=np.zeros(HM.N_JOINTS))
        lying = np.zeros(HM.N_JOINTS)
        lying[3] = SAFE.QD_ESTOP + 0.5       # driver says fast, encoder still
        fired = [g5.overspeed_reason(lying, np.zeros(HM.N_JOINTS),
                                     0.004 * (i + 1))
                 for i in range(SAFE.QD_ESTOP_STREAK + 2)]
        check("the sustained tier does NOT fire on the driver field alone",
              not any(fired),
              "%d checks against a stationary encoder" % len(fired))
        g6 = SAFE.SafetyGate(1.0, unconfirmed_reason="hw.selftest")
        g6.start(0.0, q=np.zeros(HM.N_JOINTS))
        moving = np.zeros(HM.N_JOINTS)
        reasons = []
        for i in range(SAFE.QD_ESTOP_STREAK + 1):
            moving[3] += lying[3] * 0.004    # encoder agrees this time
            reasons.append(g6.overspeed_reason(lying, moving.copy(),
                                               0.004 * (i + 1)))
        check("...and DOES once the encoder confirms, after the streak",
              any(reasons), "fired on check %d of %d"
              % (next((i + 1 for i, r in enumerate(reasons) if r), -1),
                 len(reasons)))
        zeros = np.zeros(HM.N_JOINTS)
        check("an over-temperature trips",
              bool(g4.estop_reason(zeros, zeros, 1.0,
                                   temps=np.full(HM.N_JOINTS,
                                                 SAFE.TEMP_ESTOP_C + 5))))
        check("a hard driver fault trips, the 0x80 input-lost latch does not",
              bool(g4.estop_reason(zeros, zeros, 2.0,
                                   errors={_SYNTHETIC_IDS[0]: 0x02}))
              and not g4.estop_reason(zeros, zeros, 3.0,
                                      errors={_SYNTHETIC_IDS[0]: 0x80}),
              "0x80 is recovered over CAN by the arming ladder")

    # -- 5. the protocol path, against fake drivers ------------------------
    print("\nthe protocol path, against hw.fake_bus under a synthetic map")
    from .fake_bus import FakeDriverBus
    from .motor import motorbus

    with _synthetic_map():
        ids = HM.motor_ids()
        check("motor_ids / motor_directions come back once the map is filled",
              ids == _SYNTHETIC_IDS
              and HM.motor_directions() == dict(zip(_SYNTHETIC_IDS,
                                                    _SYNTHETIC_DIRS)),
              "ids %s" % ids)
        close("calibration.joint_directions matches the table",
              CAL.joint_directions(), _SYNTHETIC_DIRS, 0.0)

        with motorbus.MotorBus(ids, bus=FakeDriverBus(ids=ids),
                               dirs=HM.motor_directions()) as mb:
            armed = mb.arm(rate_hz=1000.0, verbose=False)
            check("arming clears the input-lost latch on all twelve over CAN",
                  armed, "no power cycle, 0x9B -> 0x88 ladder")
            mb.status1()
            mb.status2()
            mb.poll()
            check("every motor replied to 0x9A",
                  all(v is not None for v in mb.errors().values()))

            unwrappers = CAL.new_unwrappers()
            q, qd = CAL.joint_state(mb, unwrappers)
            check("calibration.joint_state returns (12,) q and qd",
                  q.shape == (HM.N_JOINTS,) and qd.shape == (HM.N_JOINTS,))

            # THE DIRECTION ROUND-TRIP.  A distinct target to every joint
            # through 0xA4, read back through the calibration -- the only path
            # a controller uses.  With six of the twelve signs negative, any
            # convention mismatch inside MotorBus.position /
            # encoders_deg / motoroutput_deg_to_joint_rad shows up as a
            # mirrored joint rather than cancelling out.
            target = np.linspace(-0.4, 0.4, HM.N_JOINTS)
            target_deg = np.rad2deg(target)
            slot = mb.slot(1000.0)
            deadline = time.perf_counter() + slot
            for step in range(4000):
                index = step % len(ids)
                mb.poll()
                mb.position(ids[index], float(target_deg[index]), max_dps=2000.0)
                mb.pace(deadline)
                deadline += slot
            mb.poll()
            reached, _ = CAL.joint_state(mb, unwrappers)
            close("every joint round-trips 0xA4 -> encoder -> joint_rad, "
                  "signs and all", reached, target, 1e-3, " rad")
            check("...and six of the twelve signs are negative, so it could "
                  "not have cancelled",
                  sum(1 for d in _SYNTHETIC_DIRS if d < 0) == 6)

            offsets = mb.set_zero_all(timeout=1.0, rate_hz=1000.0)
            check("0x19 set-zero acks on all twelve",
                  all(v is not None for v in offsets.values()),
                  "%d offsets written"
                  % sum(v is not None for v in offsets.values()))
            raises("calibration.set_zero_all refuses without confirm=True",
                   lambda: CAL.set_zero_all(mb), RuntimeError)

    check("the synthetic map is gone again afterwards",
          not HM.is_complete() and len(HM.assigned()) == 0,
          "the real table is still empty, as it should be")

    # -- 6. the IMU frame maths --------------------------------------------
    print("\nthe IMU frame maths (no device)")
    check("SENSOR_TO_FLU is Rx(180 deg) and is its own inverse",
          np.allclose(IMU.SENSOR_TO_FLU @ IMU.SENSOR_TO_FLU, np.eye(3))
          and np.allclose(np.linalg.det(IMU.SENSOR_TO_FLU), 1.0),
          "diag%s" % (tuple(int(v) for v in np.diag(IMU.SENSOR_TO_FLU)),))
    close("SENSOR_TO_TRUNK == R_BODY_IMU @ SENSOR_TO_FLU",
          IMU.SENSOR_TO_TRUNK, C.R_BODY_IMU @ IMU.SENSOR_TO_FLU, 1e-15)
    check("R_BODY_IMU is still the identity PLACEHOLDER",
          bool(np.allclose(C.R_BODY_IMU, np.eye(3))),
          "the board is not mounted; sim.selftest gates it against the MJCF")
    roll, pitch, yaw, wx, wy, wz = IMU.sensor_to_trunk(5.0, 3.0, 20.0,
                                                       (0.1, 0.2, 0.3))
    check("NED -> FLU negates pitch, heading and the y/z rates",
          (roll, pitch, yaw) == (5.0, -3.0, -20.0)
          and np.allclose((wx, wy, wz), np.rad2deg((0.1, -0.2, -0.3))),
          "roll %+.1f pitch %+.1f yaw %+.1f deg" % (roll, pitch, yaw))
    check("wrap_deg lands on (-180, 180]",
          IMU.wrap_deg(180.0) == 180.0 and IMU.wrap_deg(-180.0) == 180.0
          and IMU.wrap_deg(190.0) == -170.0)

    # -- 7. the yardstick the bring-up reads directions against ------------
    print("\nhw.kinematics, which is what makes a measured direction readable")
    rows = HM.direction_check_table(0.1)
    check("the direction-check table needs no map at all",
          len(rows) == HM.N_JOINTS
          and all(abs(r[1]) + abs(r[2]) + abs(r[3]) > 1.0 for r in rows),
          "every joint moves its foot at least 1 mm per 0.1 rad")
    check("...and every joint has an unambiguous dominant axis to watch",
          all(max(abs(v) for v in r[1:]) > 1.5 * sorted(
              (abs(v) for v in r[1:]))[-2] for r in rows),
          "the largest component beats the next by 1.5x or more")
    stance = K.all_foot_positions(C.Q_STAND)
    close("...at the stance the kinematics still agrees with sim's Q_STAND",
          np.abs(stance), np.abs(stance[0]), 1e-12, " m")

    # -- 8. hw.stand's wave phases, against a robot that follows -----------
    print("\nhw.stand --wave: sim.wave's five phases through the phase machine")
    _wave_phases()

    # ----------------------------------------------------------------------
    print("\n%d checks, %d failed" % (_PASSES + len(_FAILURES), len(_FAILURES)))
    print("The stand's balance controller is gated separately, and this file "
          "does not run it:\n    python -m hw.balance.selftest")
    if _FAILURES:
        for name in _FAILURES:
            print("  FAILED: %s" % name)
        return 1
    print("the hardware plumbing is sound.  The map is EMPTY -- see "
          "hw/README.md, then `python -m hw.bringup plan`.")
    return 0


def _refuses(fn) -> bool:
    try:
        fn()
    except Exception:                # noqa: BLE001
        return True
    return False


def _wave_phases() -> None:
    """Drive `hw.stand.HardwareStand` with `--wave FL` through shift, raise,
    wave, lower and unshift on a SYNTHETIC robot that sits exactly on the IK
    of every target -- the same fixture `hw.balance.selftest` uses for the
    lift.  No bus: `update` is pure, which is why it can be done at all.

    What this proves: the phase list is spliced right, the latches happen at
    the right entries, a perfectly tracking robot trips nothing, every
    torque is inside the gate's cap, and each of the three trips fires on
    the state that should fire it.  What it cannot: that the robot stays up.
    `python -m sim.wave --headless` is that.
    """
    from sim import stand as ST
    from sim import wave as WV
    from . import stand as HST
    from .balance import state as BSTATE
    from .balance import config as BCFG

    level = IMU.TrunkOrientation.level()

    def body_at(q, R=None):
        if R is None:
            orientation = level
        else:
            roll, pitch, yaw = C.zyx_from_rot(R)
            orientation = IMU.TrunkOrientation(R=R, omega_b=np.zeros(3),
                                               roll=roll, pitch=pitch, yaw=yaw,
                                               age_s=0.0)
        return BSTATE.read(C.flat(q), np.zeros(C.N_JOINTS), orientation)

    with _synthetic_map():
        gate = SAFE.SafetyGate(2.0, unconfirmed_reason="hw.selftest")
        stand = HST.HardwareStand(gate, law="per-leg", wave="FL")
        check("--wave splices the five phases between lift and park",
              stand.phases == ("limp", "settle", "crouch", "lift") + WV.WAVE_PHASES
              + ("park", "done"), " ".join(stand.phases))
        check("...and without it the list is the plain six",
              HST.phases_for(None) == HST.PHASES)

        # walk to the lift, as a run does: settle, crouch, lift
        q_lift = WV._POSE_LIFT
        now = 0.0
        stand.t_phase = now
        for expect in ("settle", "crouch", "lift"):
            now += 10.0                              # every ramp has arrived
            stand.advance(now, C.flat(ST.Q_CROUCH if expect != "lift" else q_lift))
            stand.update(now, body_at(C.unflat(ST.Q_CROUCH)))
        check("the walk reaches lift", stand.phase_name == "lift")

        # the guard: a shift from the crouch is refused, and finally
        refused = stand.advance(now + 10.0, C.flat(ST.Q_CROUCH))
        check("shift is REFUSED while the trunk is still on the floor",
              bool(refused) and stand.refusal_final and stand.phase_name == "lift",
              refused or "")
        # ...and allowed from the standing pose, latching from it
        now += 10.0
        refused = stand.advance(now, C.flat(q_lift))
        check("...and entered from the standing pose", refused is None
              and stand.phase_name == "shift")
        check("shift latches the feet from the MEASURED pose",
              np.allclose(stand.wave.z0, K.all_foot_positions(q_lift)[:, 2])
              and np.linalg.norm(stand.wave.shift) > 0.03,
              "z0 %.4f m, shift %s mm" % (stand.wave.z0.mean(),
                                          np.array2string(1e3 * stand.wave.shift,
                                                          precision=1)))

        # a robot that sits on the IK of every target, phase by phase
        trips, modes, peak = [], set(), 0.0
        q = q_lift.copy()
        dt = 0.004
        for phase in WV.WAVE_PHASES:
            if stand.phase_name != phase:
                refused = stand.advance(now, C.flat(q))
                trips.append(refused)
            t0 = now
            while now - t0 <= stand.wave.duration(phase) + 0.1:
                p_des, _, _ = stand.wave.targets(phase, now - t0)
                q = K.all_leg_ik(p_des, q_seed=q)
                mode, values, trip = stand.update(now, body_at(q))
                modes.add(mode)
                trips.append(trip)
                peak = max(peak, float(np.abs(values).max()))
                now += dt
        check("a perfectly tracking robot trips nowhere through all five",
              all(t is None for t in trips),
              next((t for t in trips if t), "") or "shift raise wave lower unshift")
        check("...every wave phase is a TORQUE phase", modes == {"torque"})
        check("...and every torque is inside the gate's cap",
              peak <= gate.tau_cap + 1e-9, "peak %.2f N*m" % peak)
        check("the paw's own floor spot was latched at raise",
              np.allclose(stand.wave.p_floor[2], stand.wave.z0[0], atol=1e-6))
        check("the status line carries the three-foot height, not the paw",
              abs(stand.wave.h_stance - ST.LIFT_HEIGHT) < 2e-3
              and "h3" in stand.wave.status(),
              "h3 %.4f m" % stand.wave.h_stance)

        # the three trips, each on the state that should fire it
        refused = stand.advance(now, C.flat(q))
        check("after unshift the next phase is park", refused is None
              and stand.phase_name == "park")
        gate2 = SAFE.SafetyGate(2.0, unconfirmed_reason="hw.selftest")
        s2 = HST.HardwareStand(gate2, law="per-leg", wave="FL")
        s2.phases = ("wave",) + s2.phases[1:]
        s2.t_phase = 0.0
        gate2.start(0.0, q=C.flat(q_lift))
        s2.wave.enter("shift", q_lift)
        s2.wave.enter("raise", q_lift)
        p_des, _, _ = s2.wave.targets("wave", 1.0)
        q_wave = K.all_leg_ik(p_des, q_seed=q_lift)
        _, _, trip = s2.update(1.0, body_at(q_wave))
        check("on the IK of the wave targets: no trip", trip is None, trip or "")
        tilted = body_at(q_wave, R=C.rot_x(np.deg2rad(BCFG.TILT_STOP_DEG + 3)))
        _, _, trip = s2.update(1.0, tilted)
        check("the tilt trip fires on three feet", "tilt" in (trip or ""), trip or "")
        far = p_des.copy()
        far[0] += (0.0, 0.0, -0.08)                  # the paw 80 mm low
        _, _, trip = s2.update(1.0, body_at(K.all_leg_ik(far, q_seed=q_wave)))
        check("the paw trip fires when the paw is far from its reference",
              "paw" in (trip or ""), trip or "")
        off = q_wave.copy()
        off[1, 0] += np.deg2rad(30.0)               # FR abduction 30 deg off
        _, _, trip = s2.update(1.0, body_at(off))
        check("the joint trip fires on a stance leg off the IK",
              "from the IK" in (trip or ""), trip or "")


if __name__ == "__main__":
    sys.exit(main())
