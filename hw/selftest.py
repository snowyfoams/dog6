"""Gate the hardware layer with no robot, no adapter and no simulator.

    python -m hw.selftest

WHAT THIS CAN AND CANNOT TELL YOU
    CAN     that the ported code speaks the LK protocol correctly, that the
            direction convention is applied consistently in BOTH directions,
            that the encoder unwrap survives a wrap, that the safety gate
            shapes and trips the way it claims, that the IMU frame maths is
            self-consistent, and that the torque gate refuses a map with a
            hole in it.  All of that is code, and code can be checked on a
            laptop.

    CANNOT  which motor is which joint, or which way it turns.  Those are
            physical facts about an assembled robot.  DOG6's were established
            on 2026-09-15 by `hw.bringup`, on the robot, and they live in
            `hardware_map` -- not here.  Nothing in this file can re-derive
            them and nothing here pretends to.

    So a green run means "the plumbing is sound", never "it is safe to power
    the robot".  That distinction is why this file exists separately from
    `sim.selftest`, which gates the kinematics against MuJoCo.

THE PROTOCOL SECTION SUPPLIES ITS OWN MAP, AND STILL SHOULD
    Section 4 installs a SYNTHETIC map with deliberately awkward ids and mixed
    signs and drives the real code through it: `motor_ids`,
    `motor_directions`, `MotorBus`, `joint_state`, `set_zero_all`,
    `SafetyGate`.  Running it against DOG6's own table instead would prove
    LESS, not more -- ids 1..12 in ascending order are exactly the arrangement
    in which an off-by-one indexes the right value by accident.  The synthetic
    table lives for the length of a `with` block and is never importable.

THE DIRECTION ROUND-TRIP IS THE CHECK WORTH HAVING
    `MotorBus.position` applies a joint's direction on the way out;
    `MotorBus.encoders_deg` does NOT apply it on the way back, while
    `speeds_dps` and `torques_nm` DO.  Three conventions in one class, all
    correct, all easy to mix up -- and a mix-up is a joint that reads the
    mirror of where it is.  Section 4 drives all twelve through a real 0xA4
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

#: DOG6's own bring-up, 2026-09-15, restated here ON PURPOSE.  This is a
#: SECOND COPY of a measurement, which anywhere else in this project would be
#: the bug -- but a copy that is only ever COMPARED is the opposite of a stale
#: one.  Nothing reads these to drive a motor; they exist so an edit to
#: `hardware_map` nobody meant turns this file red rather than passing quietly.
#: Re-wire the robot and both change together, in one commit.
_MEASURED_IDS = frozenset(range(1, 13))
_MEASURED_NEGATIVE_IDS = frozenset({3, 6, 7, 9, 10, 12})


@contextlib.contextmanager
def _synthetic_map():
    """Install `_SYNTHETIC_IDS`/`_SYNTHETIC_DIRS` for the length of a block.

    Patches `hardware_map.HARDWARE_JOINTS` in place.  Everything downstream --
    `motor_ids`, `directions`, `is_complete`, `hw.calibration`, `hw.safety` --
    reaches the table through a FUNCTION that resolves it at call time, which
    is exactly why they are functions.  A module that imported the tuple by
    name would not see this, and would silently keep testing DOG6's own.
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


@contextlib.contextmanager
def _holed_map():
    """Blank ONE row for the length of a block, to prove the guard still fires.

    `hw.safety` refuses a map it cannot read every sign from, and that refusal
    is a SAFETY GATE, not a note about how far the bring-up has got: a driver
    that is replaced or re-flashed puts its row back to None, and the gate is
    what stops a torque law running on the eleven signs that are left.  DOG6's
    table is complete, so the hole has to be made here.

    One row, not twelve -- a single hole is the case that actually happens and
    the harder one to catch.
    """
    original = HM.HARDWARE_JOINTS
    HM.HARDWARE_JOINTS = (
        (HM.HardwareJoint(original[0].leg, original[0].joint,
                          original[0].model_name),) + original[1:])
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
    real_table = HM.HARDWARE_JOINTS   # to prove every patch below is undone
    HM.validate()
    CAL.validate()
    check("hardware_map and calibration validate", True,
          "%d rows, %d wired, %d confirmed"
          % (HM.N_JOINTS, len(HM.assigned()), HM.confirmed_count()))
    check("the map is in coordinates.JOINT_NAMES order",
          tuple(j.model_name for j in HM.HARDWARE_JOINTS) == C.JOINT_NAMES)
    check("every row is a measurement read off DOG6",
          HM.is_complete() and HM.confirmed_count() == HM.N_JOINTS,
          "2026-09-15, `bringup spin` then `bringup check`, one motor at a time")
    # A TRIPWIRE, NOT A DERIVATION.  Nothing reads these to drive a motor --
    # see `_MEASURED_IDS`.  An unintended edit to `hardware_map` turns this
    # red instead of passing quietly.
    check("...and it is still the record from that day",
          {j.can_id for j in HM.HARDWARE_JOINTS} == _MEASURED_IDS
          and {j.can_id for j in HM.HARDWARE_JOINTS if j.direction < 0}
          == _MEASURED_NEGATIVE_IDS,
          "ids 1..12, -1 on %s"
          % ", ".join(str(i) for i in sorted(_MEASURED_NEGATIVE_IDS)))
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

    # -- 2. the motor's own frame, which needs no map ----------------------
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

    # -- 3. the safety gate, under a synthetic map -------------------------
    print("\nthe safety gate")
    # The torque gate still refuses a map with a HOLE in it, and that is a
    # safety gate rather than a statement about DOG6: it is what stands
    # between a re-flashed driver and a torque law running on a sign nobody
    # measured.  DOG6's own table is complete, so the state has to be made.
    with _holed_map():
        raises("a torque gate refuses a map with a hole in it",
               SAFE.SafetyGate, MapIncomplete,
               "one row blanked; `MapIncomplete` names it")
        raises("...and unconfirmed_reason does NOT override that tier",
               lambda: SAFE.SafetyGate(0.5, unconfirmed_reason="a good reason"),
               MapIncomplete, "there are no signs for a reason to override")
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

    # -- 4. the protocol path, against fake drivers ------------------------
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
          HM.HARDWARE_JOINTS is real_table
          and HM.confirmed_count() == HM.N_JOINTS,
          "DOG6's measured table is back, %d/%d wired and confirmed"
          % (len(HM.assigned()), HM.N_JOINTS))

    # -- 5. the IMU frame maths --------------------------------------------
    print("\nthe IMU frame maths (no device)")
    check("SENSOR_TO_FLU is Rx(180 deg) and is its own inverse",
          np.allclose(IMU.SENSOR_TO_FLU @ IMU.SENSOR_TO_FLU, np.eye(3))
          and np.allclose(np.linalg.det(IMU.SENSOR_TO_FLU), 1.0),
          "diag%s" % (tuple(int(v) for v in np.diag(IMU.SENSOR_TO_FLU)),))
    close("SENSOR_TO_TRUNK == R_BODY_IMU @ SENSOR_TO_FLU",
          IMU.SENSOR_TO_TRUNK, C.R_BODY_IMU @ IMU.SENSOR_TO_FLU, 1e-15)
    check("R_BODY_IMU is the identity AND that is a measurement",
          bool(np.allclose(C.R_BODY_IMU, np.eye(3)))
          and bool(C.R_BODY_IMU_MEASURED),
          "board mounted aligned with the trunk, as DOG5; sim.selftest gates "
          "it against the MJCF's imu site quat")
    roll, pitch, yaw, wx, wy, wz = IMU.sensor_to_trunk(5.0, 3.0, 20.0,
                                                       (0.1, 0.2, 0.3))
    check("NED -> FLU negates pitch, heading and the y/z rates",
          (roll, pitch, yaw) == (5.0, -3.0, -20.0)
          and np.allclose((wx, wy, wz), np.rad2deg((0.1, -0.2, -0.3))),
          "roll %+.1f pitch %+.1f yaw %+.1f deg" % (roll, pitch, yaw))
    check("wrap_deg lands on (-180, 180]",
          IMU.wrap_deg(180.0) == 180.0 and IMU.wrap_deg(-180.0) == 180.0
          and IMU.wrap_deg(190.0) == -170.0)

    # -- 6. the yardstick the bring-up read directions against -------------
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

    # ----------------------------------------------------------------------
    print("\n%d checks, %d failed" % (_PASSES + len(_FAILURES), len(_FAILURES)))
    print("The stand's balance controller is gated separately, and this file "
          "does not run it:\n    python -m hw.balance.selftest")
    if _FAILURES:
        for name in _FAILURES:
            print("  FAILED: %s" % name)
        return 1
    print("the hardware plumbing is sound, over the map measured on DOG6 on "
          "2026-09-15.\nThat is still not 'safe to power': section 4 ran "
          "against `fake_bus`, which obeys whatever\nmap it is handed.  "
          "`python -m hw.bringup plan` for where the robot is.")
    return 0


def _refuses(fn) -> bool:
    try:
        fn()
    except Exception:                # noqa: BLE001
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
