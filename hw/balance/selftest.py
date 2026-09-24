"""Gate the balance controller with no robot, no IMU and no simulator.

    python -m hw.balance.selftest

WHAT THIS CAN AND CANNOT TELL YOU
    CAN     that the frame conventions round-trip, that the two world/body
            crossings are the ones written in the design note, that the S-curve
            is C2 and its CoM conversion is consistent with the measurement
            that will be differenced against it, that the allocator reproduces
            the wrench it was asked for and stays inside the cone, that the
            closed-form leg gravity agrees with the chain walk over random
            poses AND random orientations, that the torque sign agrees with
            the law this one replaces, and that every trip fires on the state
            that should fire it.

    CANNOT   whether R_BODY_IMU is right (it is an identity placeholder and
            the board is not mounted), whether any gain is a good gain, or
            whether the floor has the friction the cone assumes.

    So a green run means the law computes what it says it computes.  It never
    says the robot will stand.

THE THREE CHECKS THAT ARE WORTH THE WHOLE FILE
    Every one of them is a sign or a frame that is EXACTLY ZERO ERROR AT THE
    IDENTITY, so a level bench test cannot see it:

      * the missing R^T in stage 5, which grows linearly with tilt
      * the world/body attitude-error pair, which coincides at R_d = I
      * the trunk-rotation term in zdot, which vanishes at omega = 0

    All three are checked AT A TILT, never at the identity.
"""
from __future__ import annotations

import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/selftest.py`
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import kinematics as SK     # noqa: E402
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402

from .. import imu as IMU            # noqa: E402
from .. import kinematics as HK      # noqa: E402
from .. import fold_stand as FS      # noqa: E402
from .. import safety as SAFE        # noqa: E402
from . import allocation as ALLOC    # noqa: E402
from . import config as cfg          # noqa: E402
from . import controller as CTRL     # noqa: E402
from . import law as LAW             # noqa: E402
from . import reference as REF       # noqa: E402
from . import posture as POSE        # noqa: E402
from . import sequence as SEQ        # noqa: E402
from . import state as STATE         # noqa: E402
from . import torque as TRQ          # noqa: E402

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
    worst = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    check(label, worst <= tol, "worst %.3g%s (tol %.3g)" % (worst, unit, tol))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def pose_at(h: float, foot_xy=None, q_seed=None) -> np.ndarray:
    """(4, 3) joints with the feet at `foot_xy` and the trunk bottom at `h`.

    `foot_xy` None means the nominal `sim.stand.FOOT_XY`; a `CrouchPose` fills
    it to build states in a posture the lift was not designed for.

    `hw.kinematics`' CLOSED-FORM inverse, not `sim.stand.pose_for_height`'s
    damped least squares.  The difference is 9 nm of residual, which is
    nothing to the robot and everything to a fixture: a pose that is 9 nm off
    the commanded height makes the PD ask for 8.6 uN, and a check on "zero
    error gives exactly mg" then fails on the SOLVER rather than on the law.
    """
    height = STATE.height_to_origin(h)
    if foot_xy is None:
        targets = ST.foot_targets(height)
    else:
        targets = np.zeros((C.N_LEGS, 3))
        targets[:, :2] = np.asarray(foot_xy, dtype=float)
        targets[:, 2] = -(height - P.FOOT_RADIUS)
    return HK.all_leg_ik(targets,
                         q_seed=ST.Q_CROUCH if q_seed is None else q_seed)


def state_at(h: float, R=None, omega_b=None, qd=None, age_s: float = 0.0,
             foot_xy=None, q_seed=None, srb=None):
    """A `BodyState` at height `h` and orientation `R`, feet planted."""
    R = np.eye(3) if R is None else R
    roll, pitch, yaw = C.zyx_from_rot(R)
    orientation = IMU.TrunkOrientation(
        R=R, omega_b=np.zeros(3) if omega_b is None else np.asarray(omega_b),
        roll=roll, pitch=pitch, yaw=yaw, age_s=age_s)
    q = C.flat(pose_at(h, foot_xy=foot_xy, q_seed=q_seed))
    return STATE.read(q, np.zeros(C.N_JOINTS) if qd is None else qd,
                      orientation, srb=srb)


def main() -> int:
    rng = np.random.default_rng(20260915)
    print("=" * 78)
    print("hw.balance selftest -- no robot, no IMU, no simulator")
    print("=" * 78)

    # =====================================================================
    print("\n1. the rotation conventions (sim.coordinates)")
    # =====================================================================
    for _ in range(50):
        rpy = rng.uniform(-1.2, 1.2, size=3)
        R = C.rot_zyx(*rpy)
        back = C.zyx_from_rot(R)
        if np.max(np.abs(np.asarray(back) - rpy)) > 1e-12:
            break
    else:
        check("rot_zyx and zyx_from_rot round-trip over 50 poses", True,
              "to 1e-12")
    check("R is orthonormal with det +1",
          abs(np.linalg.det(C.rot_zyx(0.3, -0.2, 1.1)) - 1.0) < 1e-12
          and np.allclose(C.rot_zyx(0.3, -0.2, 1.1).T
                          @ C.rot_zyx(0.3, -0.2, 1.1), np.eye(3)))

    # The FLU signs, which are what a physical tilt test has to reproduce.
    right = np.array([0.0, -1.0, 0.0])          # a point on the robot's right
    nose = np.array([1.0, 0.0, 0.0])
    check("roll > 0 puts the RIGHT side DOWN",
          float((C.rot_zyx(np.deg2rad(10), 0, 0) @ right)[2]) < 0,
          "z of the right-hand point goes negative")
    check("pitch > 0 puts the NOSE DOWN",
          float((C.rot_zyx(0, np.deg2rad(10), 0) @ nose)[2]) < 0)
    check("yaw > 0 swings the nose LEFT",
          float((C.rot_zyx(0, 0, np.deg2rad(10)) @ nose)[1]) > 0)

    # log_so3 against a known axis-angle, and at both awkward ends.
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    for angle in (1e-9, 0.4, np.pi - 1e-8):
        k = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
        got = C.log_so3(R)
        ok = np.max(np.abs(np.abs(got) - np.abs(angle * axis))) < 1e-6
        check("log_so3 recovers axis*angle at %.4g rad" % angle, ok,
              "|%s|" % np.array2string(got, precision=4))
    close("log_so3(I) is exactly zero", C.log_so3(np.eye(3)), np.zeros(3), 0.0)

    # =====================================================================
    print("\n2. hw.imu: R and omega^b in SI")
    # =====================================================================
    check("R_BODY_IMU is still the identity PLACEHOLDER",
          np.allclose(C.R_BODY_IMU, np.eye(3)),
          "the board is not mounted; nothing below its chain is verified")
    close("trunk_rotation == rot_zyx while the mounting is identity",
          IMU.trunk_rotation(0.2, -0.1, 0.5), C.rot_zyx(0.2, -0.1, 0.5), 1e-15)
    # Eq (Rb): the mounting rotation enters the ATTITUDE on the right.
    mount = C.rot_z(0.3)
    saved = C.R_BODY_IMU
    try:
        C.R_BODY_IMU = mount
        close("...and R = Rzyx @ R_m^T once it is not",
              IMU.trunk_rotation(0.2, -0.1, 0.5),
              C.rot_zyx(0.2, -0.1, 0.5) @ mount.T, 1e-15)
    finally:
        C.R_BODY_IMU = saved
    check("SENSOR_TO_TRUNK left-multiplies the gyro (eq wb)",
          np.allclose(IMU.SENSOR_TO_TRUNK,
                      C.R_BODY_IMU @ IMU.SENSOR_TO_FLU))
    roll, pitch, yaw, wx, wy, wz = IMU.sensor_to_trunk(10.0, 20.0, 30.0,
                                                       (1.0, 2.0, 3.0))
    check("NED -> FLU negates pitch, heading and the y/z rates",
          (roll, pitch, yaw) == (10.0, -20.0, -30.0)
          and abs(wx - np.degrees(1.0)) < 1e-9
          and abs(wy + np.degrees(2.0)) < 1e-9
          and abs(wz + np.degrees(3.0)) < 1e-9)
    orientation = IMU.TrunkOrientation(C.rot_x(0.5), np.array([1.0, 0.0, 0.0]),
                                       0.5, 0.0, 0.0, 0.001)
    close("omega^w = R omega^b", orientation.omega_w,
          C.rot_x(0.5) @ np.array([1.0, 0.0, 0.0]), 1e-15)
    check("staleness is reported against cfg.IMU_MAX_AGE_S",
          not orientation.is_stale(cfg.IMU_MAX_AGE_S)
          and IMU.TrunkOrientation(np.eye(3), np.zeros(3), 0, 0, 0,
                                   0.2).is_stale(cfg.IMU_MAX_AGE_S))

    # =====================================================================
    print("\n3. the S-curve reference")
    # =====================================================================
    ramp = REF.Quintic.ramp(0.0, 1.0, 1.0)
    close("Quintic.ramp is exactly (0, 0, 0, 10, -15, 6)",
          ramp.c, np.array([0.0, 0.0, 0.0, 10.0, -15.0, 6.0]), 1e-12)
    ends = [ramp.at(0.0), ramp.at(1.0)]
    check("C2 at both ends: hdot and hddot are zero there",
          max(abs(ends[0].hdot), abs(ends[0].hddot),
              abs(ends[1].hdot), abs(ends[1].hddot)) < 1e-12,
          "the raised cosine is only C1 -- its hddot STEPS")
    check("peak hdot is 1.875 dh/T (cosine: 1.571)",
          abs(ramp.peak_velocity - REF.PEAK_VELOCITY_FACTOR) < 1e-3,
          "%.4f" % ramp.peak_velocity)
    check("peak hddot is 5.7735 dh/T^2 (cosine: 4.935)",
          abs(ramp.peak_acceleration - REF.PEAK_ACCELERATION_FACTOR) < 1e-3,
          "%.4f" % ramp.peak_acceleration)
    check("it is HELD past T, not extrapolated",
          abs(ramp.at(50.0).h - 1.0) < 1e-12 and abs(ramp.at(50.0).hdot) < 1e-12)

    # the analytic derivative, against a finite difference of the position
    dt = 1e-6
    worst = max(abs((ramp.at(t + dt).h - ramp.at(t - dt).h) / (2 * dt)
                    - ramp.at(t).hdot) for t in np.linspace(0.05, 0.95, 40))
    check("hdot IS the derivative of h, not a finite difference", worst < 1e-6,
          "worst %.2e" % worst)

    # re-planning keeps C2 across the join
    stand = REF.Quintic.ramp(cfg.H_CROUCH, cfg.H_LIFT, cfg.T_RISE)
    mid = stand.at(1.2)
    replanned = REF.Quintic.between(mid.h, mid.hdot, mid.hddot,
                                    cfg.H_LIFT, 0.0, 0.0, 2.0)
    start = replanned.at(0.0)
    check("a re-plan starts from the current (h, hdot, hddot)",
          max(abs(start.h - mid.h), abs(start.hdot - mid.hdot),
              abs(start.hddot - mid.hddot)) < 1e-12,
          "restarting s would put a velocity STEP mid-lift")

    # the CoM conversion, and the cancellation the pinned CoM buys
    R = C.rot_y(np.deg2rad(7.0))
    command = REF.HeightCommand(0.08, 0.05, 0.0)
    com = REF.com_command(command, R)
    body = state_at(0.08, R=R)
    close("p_c,z,d - p_c,z is EXACTLY the height error at a pinned CoM",
          com.p_cz - body.p_cz, command.h - body.h, 1e-12, " m")
    check("the conversion coefficient is 1, not 0.677",
          abs(com.p_cz_dot - command.hdot) < 1e-15,
          "because both sides use the same constant c^b")

    # =====================================================================
    print("\n4. state: the two crossings, and the height chain")
    # =====================================================================
    level = state_at(cfg.H_LIFT)
    close("h at the lift pose is the commanded lift height",
          level.h, cfg.H_LIFT, 1e-9, " m")
    close("h == sim.stand.height_from_fk - TRUNK_BOTTOM_OFFSET at R = I",
          level.h,
          ST.height_from_fk(C.unflat(level.q)) - cfg.TRUNK_BOTTOM_OFFSET,
          1e-12, " m")
    close("p_c,z = z_origin + (R c^b)_z",
          level.p_cz, level.z_origin + cfg.COM_BODY[2], 1e-15, " m")
    check("z_origin - h is the constant TRUNK_BOX_HALF_z",
          abs((level.z_origin - level.h) - cfg.TRUNK_BOTTOM_OFFSET) < 1e-15,
          "%.5f m -- DOG5's 38 mm lesson" % cfg.TRUNK_BOTTOM_OFFSET)

    # the world/body crossing for the moment arms
    tilt = C.rot_zyx(np.deg2rad(6.0), np.deg2rad(-4.0), np.deg2rad(20.0))
    tilted = state_at(cfg.H_LIFT, R=tilt)
    close("r_i^w = R (x_i^b - c^b)", tilted.r_w,
          (tilted.x_b - cfg.COM_BODY) @ tilt.T, 1e-15, " m")
    check("the feet are NOT square in the world once the trunk is yawed",
          np.max(np.abs(tilted.r_w[:, :2] - (tilted.x_b - cfg.COM_BODY)[:, :2]))
          > 1e-3,
          "which is what a body-frame grasp map would miss")

    # rotating the foot vector before averaging: the thing sim.stand cannot do
    pitched = state_at(cfg.H_LIFT, R=C.rot_y(np.deg2rad(10.0)))
    naive = ST.height_from_fk(C.unflat(pitched.q)) - cfg.TRUNK_BOTTOM_OFFSET
    check("a tilted trunk's h differs from the level-trunk formula",
          abs(pitched.h - naive) > 1e-3,
          "%.1f mm at 10 deg -- sim.stand.height_from_fk assumes level"
          % (1e3 * abs(pitched.h - naive)))

    # The rotation term in zdot.  At a LEVEL square stance it is identically
    # zero and that is not luck: (omega x x)_z summed over four feet is
    # (omega x sum_i x_i)_z, the mean foot is on the trunk's own z axis, and
    # the z component of a cross product with a vertical vector vanishes.
    # Tilt the trunk and the mean foot leaves that axis, and the term appears.
    level_spin = state_at(cfg.H_LIFT, omega_b=np.array([0.0, 0.5, 0.0]))
    check("at a LEVEL square stance the rotation term cancels by symmetry",
          abs(level_spin.zdot_origin) < 1e-12,
          "which is why it is so easy to leave out and never notice")
    tilt_spin = state_at(cfg.H_LIFT, R=C.rot_y(np.deg2rad(10.0)),
                         omega_b=np.array([0.0, 0.5, 0.0]))
    naive_rate = float(-np.einsum("kij,kj->ki", tilt_spin.jac,
                                  C.unflat(tilt_spin.qd))[:, 2].mean())
    check("...and does NOT once the trunk is tilted", 
          abs(tilt_spin.zdot_origin - naive_rate) > 1e-4,
          "%.4f m/s at 0.5 rad/s with qd = 0, where the leg term gives 0"
          % tilt_spin.zdot_origin)
    still = state_at(cfg.H_LIFT)
    close("...and is zero when nothing is moving at all",
          still.zdot_origin, 0.0, 1e-12)

    # =====================================================================
    print("\n5. the balance controller")
    # =====================================================================
    gains = CTRL.BalanceGains()
    cmd = REF.com_command(REF.HeightCommand(cfg.H_LIFT, 0.0, 0.0), np.eye(3))
    perfect = CTRL.balance_wrench(level, cmd, np.eye(3), gains)
    close("zero error, level: b_d is (0, 0, mg, 0, 0, 0)",
          perfect.b_d, np.array([0, 0, cfg.WEIGHT, 0, 0, 0]), 1e-9)
    check("gravity enters ONCE, in the force row -- not mg/4 per foot",
          abs(perfect.b_d[2] - cfg.MASS * P.GRAVITY) < 1e-9)

    rolled = state_at(cfg.H_LIFT, R=C.rot_x(np.deg2rad(5.0)))
    wrench = CTRL.balance_wrench(rolled,
                                 REF.com_command(
                                     REF.HeightCommand(cfg.H_LIFT, 0.0, 0.0),
                                     rolled.R), np.eye(3), gains)
    check("a POSITIVE roll asks for a NEGATIVE Mx (push the right side up)",
          wrench.b_d[3] < 0, "Mx %+.3f N*m at +5 deg" % wrench.b_d[3])
    close("the attitude error is log(R_d R^T)^vee",
          wrench.e_R, C.log_so3(np.eye(3) @ rolled.R.T), 1e-15)
    check("...which at +5 deg roll is -5 deg about x",
          abs(np.degrees(wrench.e_R[0]) + 5.0) < 1e-6,
          "%+.4f deg" % np.degrees(wrench.e_R[0]))

    pitched_state = state_at(cfg.H_LIFT, R=C.rot_y(np.deg2rad(8.0)))
    wrench_p = CTRL.balance_wrench(
        pitched_state, REF.com_command(REF.HeightCommand(cfg.H_LIFT, 0, 0),
                                       pitched_state.R), np.eye(3), gains)
    check("a POSITIVE pitch (nose down) asks for a NEGATIVE My",
          wrench_p.b_d[4] < 0, "My %+.3f N*m at +8 deg" % wrench_p.b_d[4])
    close("I_G is R I^b R^T", wrench_p.inertia_w,
          pitched_state.R @ cfg.INERTIA_BODY @ pitched_state.R.T, 1e-15)

    # The world and body attitude errors: e^w = R e^b exactly, so they
    # COINCIDE at R = I and a level bench test cannot tell them apart.  The
    # law is world throughout because that is the frame the grasp map and the
    # cone are built in.
    R_des_yawed = C.rot_z(np.deg2rad(30.0))
    e_world = C.log_so3(R_des_yawed @ rolled.R.T)
    e_body = C.log_so3(rolled.R.T @ R_des_yawed)
    close("e_R^world == R e_R^body", e_world, rolled.R @ e_body, 1e-12)
    check("...so they differ off the identity, and only there",
          np.max(np.abs(e_world - e_body)) > 1e-3,
          "%.4f rad apart at this attitude" % np.max(np.abs(e_world - e_body)))

    # the horizontal channels are structurally off
    check("K_p,p and K_d,p have zero x and y entries",
          gains.kp_pos[0] == 0.0 and gains.kp_pos[1] == 0.0
          and gains.kd_pos[0] == 0.0 and gains.kd_pos[1] == 0.0,
          "nothing on DOG6 measures p_c,x or p_c,y")
    # THE YAW SPRING IS ON SINCE 2026-09-24 and this is the arithmetic that
    # says it is allowed to be: what it asks at a few degrees of drift has to
    # fit inside the tangential budget of TWO feet, because a trot is where
    # the heading actually moves.  `config.KP_YAW` has the decision.
    yaw_nm_per_deg = np.radians(1.0) * cfg.INERTIA_BODY[2, 2] * gains.kp_att[2]
    diagonal_nm = 0.5 * cfg.MU * cfg.WEIGHT * cfg.FOOT_RADIUS_XY
    check("the yaw spring holds the LATCHED heading, and the damper is on",
          gains.kp_att[2] > 0.0 and gains.kd_att[2] > 0.0,
          "%.3f N*m/deg, zeta %.2f" % (yaw_nm_per_deg,
                                       gains.kd_att[2]
                                       / (2 * np.sqrt(gains.kp_att[2]))))
    check("...and 7 deg of drift stays inside a DIAGONAL's yaw capacity",
          7.0 * yaw_nm_per_deg < diagonal_nm,
          "%.2f N*m asked against %.2f available on two feet"
          % (7.0 * yaw_nm_per_deg, diagonal_nm))
    check("...and it is the SLOWEST attitude axis, the one measurement that "
          "can lie",
          gains.kp_att[2] < gains.kp_att[0] and gains.kp_att[2] < gains.kp_att[1],
          "yaw %.2f Hz against roll/pitch %.2f"
          % (np.sqrt(gains.kp_att[2]) / (2 * np.pi),
             np.sqrt(gains.kp_att[0]) / (2 * np.pi)))

    ablated = CTRL.BalanceGains().ablate_attitude()
    w_ab = CTRL.balance_wrench(rolled, cmd, np.eye(3), ablated)
    close("the ablation produces no moment at all", w_ab.b_d[3:], np.zeros(3),
          1e-15)
    rolled_cmd = REF.com_command(REF.HeightCommand(cfg.H_LIFT, 0.0, 0.0),
                                 rolled.R)
    held = CTRL.balance_wrench(rolled, rolled_cmd, np.eye(3), gains,
                               hold_attitude=True)
    close("a stale IMU freezes the moment, not the force",
          held.b_d[3:], np.zeros(3), 1e-15)
    check("...and the height loop keeps running through it",
          abs(held.b_d[2] - wrench.b_d[2]) < 1e-9,
          "%.4f N either way" % held.b_d[2])

    # =====================================================================
    print("\n6. the allocator")
    # =====================================================================
    square = state_at(cfg.H_LIFT)
    b_hold = np.array([0.0, 0.0, cfg.WEIGHT, 0.0, 0.0, 0.0])
    result = ALLOC.allocate(square.r_w, b_hold)
    close("a square stance holding mg splits it evenly", result.fz,
          np.full(4, cfg.WEIGHT / 4), 0.02, " N")
    close("...and the residual is zero when nothing clips",
          result.residual, np.zeros(6), 1e-4)
    check("nothing clipped", not result.clipped.any())

    A = ALLOC.grasp_map(square.r_w)
    close("A f reproduces b_d exactly (the grasp map is consistent)",
          A @ result.f_w.reshape(-1), b_hold, 1e-4)
    check("A is (6, 12) with identity force rows",
          A.shape == (6, 12) and np.allclose(A[:3, :3], np.eye(3)))
    close("the moment block is the skew of r_w",
          A[3:, 3:6], HK.hat(square.r_w[1]), 1e-15)

    normal = (A * (1.0 / ALLOC.weight_vector())) @ A.T
    normal[np.diag_indices(6)] += cfg.LAMBDA
    try:
        np.linalg.cholesky(normal)
        check("the 6x6 normal matrix is positive definite", True,
              "cond %.1f" % np.linalg.cond(normal))
    except np.linalg.LinAlgError:
        check("the 6x6 normal matrix is positive definite", False)

    # a moment the stance cannot supply -> the cone bites, and it is reported
    b_huge = np.array([0.0, 0.0, cfg.WEIGHT, 40.0, 0.0, 0.0])
    hard = ALLOC.allocate(square.r_w, b_huge)
    check("an impossible moment clips and the residual reports it",
          hard.clipped.any() and hard.residual_moment > 1.0,
          "residual %.1f N / %.2f N*m" % (hard.residual_force,
                                          hard.residual_moment))
    check("every foot stays inside the hard cone after projection",
          bool(np.all(np.linalg.norm(hard.f_w[:, :2], axis=1)
                      <= cfg.MU * hard.f_w[:, 2] + 1e-9)))
    check("...and inside the normal-force box, f_z >= FZ_MIN > 0",
          bool(np.all(hard.f_w[:, 2] >= cfg.FZ_MIN - 1e-9)
               and np.all(hard.f_w[:, 2] <= cfg.FZ_MAX + 1e-9)),
          "the unilateral constraint the per-leg law does not have")

    # a big lateral force: the tangential weight should keep it out of the cone
    b_side = np.array([6.0, 0.0, cfg.WEIGHT, 0.0, 0.0, 0.0])
    sideways = ALLOC.allocate(square.r_w, b_side)
    ratio = (np.linalg.norm(sideways.f_unclipped[:, :2], axis=1)
             / sideways.f_unclipped[:, 2]).max()
    check("the tangential weight keeps the raw solution inside the cone",
          ratio < cfg.MU, "worst |f_t|/f_z %.3f against mu %.2f"
                          % (ratio, cfg.MU))

    monitor = ALLOC.ResidualMonitor()
    fired = [monitor.reason(hard) for _ in range(cfg.RESIDUAL_STREAK)]
    check("the residual trip needs a STREAK, not one sweep",
          fired[0] is None and fired[-1] is not None,
          "%d sweeps" % cfg.RESIDUAL_STREAK)
    check("...and one good sweep resets it",
          ALLOC.ResidualMonitor().reason(result) is None)

    # =====================================================================
    print("\n7. stage 5: the sign, the rotation, and the closed form")
    # =====================================================================
    worst_R = worst_level = 0.0
    for _ in range(300):
        q4 = rng.uniform(-2.5, 2.5, size=(4, 3))
        R = C.rot_zyx(*rng.uniform(-np.pi, np.pi, size=3))
        walk = np.stack([TRQ._gravity_walk(i, q4[i], R) for i in range(4)])
        worst_R = max(worst_R, np.abs(walk
                                      - TRQ.all_leg_gravity_torque(q4, R)).max())
        sim = np.stack([SK.leg_gravity_torque(i, q4[i]) for i in range(4)])
        worst_level = max(worst_level,
                          np.abs(sim - TRQ.all_leg_gravity_torque(q4)).max())
    check("closed-form leg gravity == the chain walk, 300 random poses AND "
          "orientations", worst_R < 1e-12, "worst %.2e N*m" % worst_R)
    check("...and == sim.kinematics.leg_gravity_torque at R = I",
          worst_level < 1e-12, "worst %.2e N*m" % worst_level)

    # THE SIGN CHECK: at a level stand this must agree with the law it replaces.
    q4 = pose_at(cfg.H_LIFT)
    f_even = np.tile([0.0, 0.0, cfg.WEIGHT / 4.0], (4, 1))
    tau_srb = TRQ.stance_torque(level, f_even,
                               gravity=np.zeros((4, 3)))
    tau_old = np.stack([HK.foot_jacobian(i, q4[i]).T
                        @ np.array([0.0, 0.0, -cfg.WEIGHT / 4.0])
                        for i in range(4)])
    close("-J^T R^T f^w at f_w = +mg/4 == the per-leg law's -mg/4 feedforward",
          tau_srb, C.flat(tau_old), 1e-12, " N*m")
    check("...so a stance leg is commanded to PUSH DOWN on the floor",
          float(tau_srb[2]) * float(C.flat(tau_old)[2]) > 0)

    # THE ROTATION CHECK: drop R^T and the error grows with tilt, from zero.
    errors = []
    for degrees in (0.0, 5.0, 10.0, 20.0):
        R = C.rot_y(np.deg2rad(degrees))
        body = state_at(cfg.H_LIFT, R=R)
        f_w = np.tile([0.0, 0.0, cfg.WEIGHT / 4.0], (4, 1))
        correct = TRQ.stance_torque(body, f_w, gravity=np.zeros((4, 3)))
        dropped = C.flat(np.stack([-(body.jac[i].T @ f_w[i]) for i in range(4)]))
        errors.append(float(np.abs(correct - dropped).max()))
    check("a missing R^T is EXACTLY zero at R = I", errors[0] < 1e-15)
    check("...and grows with tilt, so no level bench test can see it",
          errors[1] < errors[2] < errors[3],
          "0 / %.3f / %.3f / %.3f N*m at 0, 5, 10, 20 deg" % tuple(errors[1:]))

    # =====================================================================
    print("\n8. the law, end to end")
    # =====================================================================
    balance = LAW.BalanceLaw()
    crouch = state_at(cfg.H_CROUCH)
    balance.arm(0.0, crouch)
    check("h0 is LATCHED from the measurement, not from CROUCH_HEIGHT",
          abs(balance.h0 - crouch.h) < 1e-12,
          "starting a ramp where the robot is not is a step input")
    close("the ramp starts at h0 with zero velocity",
          [balance.ramp.at(0.0).h, balance.ramp.at(0.0).hdot],
          [crouch.h, 0.0], 1e-12)

    out = balance.update(0.0, crouch)
    check("at the crouch, armed and tracking: no trip", out.trip is None,
          "|tau| max %.2f N*m" % np.abs(out.tau).max())
    close("...and the wrench is just the weight", out.wrench.b_d[:3],
          [0, 0, cfg.WEIGHT], 1e-6)

    # follow the ramp with a robot that tracks it perfectly
    trips = []
    for t in np.linspace(0.0, cfg.T_RISE, 41):
        h = balance.ramp.at(t)
        body = state_at(h.h)
        trips.append(balance.update(t, body).trip)
    check("a perfectly tracking lift trips nowhere", all(t is None for t in trips))
    check("the peak torque over that lift is under the staging cap",
          balance.tau_peak < 3.0, "%.2f N*m against TAU_STAGED_MAX 3.0"
                                  % balance.tau_peak)

    # each trip, on the state that should fire it
    over = LAW.BalanceLaw()
    over.arm(0.0, crouch)
    tipped = state_at(cfg.H_CROUCH, R=C.rot_x(np.deg2rad(cfg.TILT_STOP_DEG + 3)))
    check("the tilt trip fires", "tilt" in (over.update(0.0, tipped).trip or ""),
          over.update(0.0, tipped).trip)

    bent = LAW.BalanceLaw()
    bent.arm(0.0, crouch)
    wrong = state_at(cfg.H_CROUCH + 0.09)      # 90 mm from the commanded height
    check("the tracking trip fires on a leg that is nowhere near the IK",
          "tracking" in (bent.update(0.0, wrong).trip or ""),
          bent.update(0.0, wrong).trip)

    stale = LAW.BalanceLaw()
    stale.arm(0.0, crouch)
    old = state_at(cfg.H_CROUCH, R=C.rot_x(np.deg2rad(5.0)),
                   age_s=10 * cfg.IMU_MAX_AGE_S)
    result = stale.update(0.0, old)
    check("a stale IMU HOLDS rather than tripping", result.trip is None
          and result.imu_held and np.allclose(result.wrench.b_d[3:], 0.0),
          "a trip during the lift is a drop")

    # the timing budget
    timing = LAW.BalanceLaw()
    timing.arm(0.0, crouch)
    body = state_at(0.05)
    for _ in range(200):
        timing.update(0.5, body, clock=time.perf_counter)
    samples = np.asarray(timing.timing.samples)
    slot_us = 1e6 / (250.0 * 12)
    check("the whole law fits in one 333 us CAN slot on this machine",
          float(np.percentile(samples, 95)) * 1e6 < slot_us,
          "%s   (slot %.0f us)" % (timing.timing.summary(), slot_us))

    # =====================================================================
    print("\n9. the attitude setpoint: DOG5's SETPOINT_DYNAMIC, ported")
    # =====================================================================
    # The convention is "world := body here".  Yaw always had it -- psi_0 is
    # latched when torque arms -- and these gate the other two axes, which
    # DOG5 added on 2026-08-28 and DOG6 was missing.
    tilted = state_at(cfg.H_CROUCH, R=C.rot_x(np.deg2rad(3.0)))

    fresh = LAW.BalanceLaw()
    close("before the latch the setpoint IS the config statics",
          [np.degrees(fresh.sp_roll), np.degrees(fresh.sp_pitch)],
          [cfg.SETPOINT_ROLL_DEG, cfg.SETPOINT_PITCH_DEG], 1e-12, "deg")
    latched = fresh.latch_setpoint(tilted)
    check("the latch takes the reading it is handed",
          latched is not None and abs(latched[0] - 3.0) < 1e-9,
          "roll %+.2f  pitch %+.2f deg" % latched)
    check("a SECOND latch is refused -- once per run, in limp only",
          fresh.latch_setpoint(state_at(cfg.H_CROUCH,
                                        R=C.rot_x(np.deg2rad(9.0)))) is None
          and abs(np.degrees(fresh.sp_roll) - 3.0) < 1e-9,
          "level has a truth to return to; re-latching would drag it")

    stale_sp = LAW.BalanceLaw()
    check("a STALE sample is refused, the statics stand",
          stale_sp.latch_setpoint(
              state_at(cfg.H_CROUCH, R=C.rot_x(np.deg2rad(3.0)),
                       age_s=10 * cfg.IMU_MAX_AGE_S)) is None
          and not stale_sp.setpoint_latched,
          "a bad setpoint is a constant the loop fights all run")

    # THE PROPERTY THE WHOLE CONVENTION EXISTS FOR: at the attitude it was
    # latched at, the attitude error is EXACTLY zero, so arming cannot step
    # the wrench.  DOG5's reason for latching yaw at arm, now on all three.
    armed_sp = LAW.BalanceLaw()
    armed_sp.latch_setpoint(tilted)
    armed_sp.arm(0.0, tilted)
    close("R_des == the R it was latched at", armed_sp.R_des, tilted.R, 1e-12)
    close("...so e_R is exactly zero there",
          C.log_so3(tilted.R.T @ armed_sp.R_des), np.zeros(3), 1e-12, "rad")
    close("...and the attitude half of the wrench is zero with it",
          CTRL.balance_wrench(tilted, REF.com_command(
              armed_sp.ramp.at(0.0), tilted.R), armed_sp.R_des,
              armed_sp.gains).b_d[3:], np.zeros(3), 1e-9, "N*m")

    check("the tilt trip measures FROM the setpoint, not from true level",
          abs(armed_sp.tilt_from_setpoint_deg(tilted)) < 1e-9
          and abs(tilted.tilt_deg - 3.0) < 1e-9,
          "state.tilt_deg still reports %.1f deg for the log" % tilted.tilt_deg)
    check("...so the stop is reached %.0f deg from where the run began"
          % cfg.TILT_STOP_DEG,
          abs(armed_sp.tilt_from_setpoint_deg(
              state_at(cfg.H_CROUCH,
                       R=C.rot_x(np.deg2rad(3.0 + cfg.TILT_STOP_DEG))))
              - cfg.TILT_STOP_DEG) < 1e-9)

    close("level_attitude is latched_attitude at a zero setpoint",
          CTRL.level_attitude(0.7), CTRL.latched_attitude(0.0, 0.0, 0.7), 1e-15)

    _dynamic = cfg.SETPOINT_DYNAMIC
    try:
        cfg.SETPOINT_DYNAMIC = False
        check("SETPOINT_DYNAMIC off refuses the latch outright",
              LAW.BalanceLaw().latch_setpoint(tilted) is None,
              "the config statics are then the whole story, as DOG5 ran")
    finally:
        cfg.SETPOINT_DYNAMIC = _dynamic

    # =====================================================================
    print("\n9b. the heading: DOG5's yaw lock, as a turn of the world frame")
    # =====================================================================
    psi = np.deg2rad(37.0)
    heading = state_at(cfg.H_CROUCH, R=C.rot_zyx(np.deg2rad(3.0), 0.0, psi),
                       omega_b=[0.11, -0.07, 0.23])
    zeroed = STATE.rezero_yaw(heading, psi)
    check("rezero_yaw puts the latched heading at exactly zero",
          abs(zeroed.yaw) < 1e-12 and abs(zeroed.yaw_offset - psi) < 1e-12,
          "%+.1f deg in, %+.3g deg out" % (np.degrees(heading.yaw),
                                           np.degrees(zeroed.yaw)))
    close("...and world x IS the trunk's nose: R's first column",
          zeroed.R @ np.array([1.0, 0.0, 0.0]),
          C.rot_zyx(np.deg2rad(3.0), 0.0, 0.0) @ np.array([1.0, 0.0, 0.0]),
          1e-12)
    close("...roll and pitch are untouched -- Rz changes neither",
          [zeroed.roll, zeroed.pitch], [heading.roll, heading.pitch], 1e-15,
          " rad")
    close("...and so is the whole height chain: (Rz v)_z == v_z",
          [zeroed.h, zeroed.p_cz, zeroed.zdot_origin, zeroed.p_cz_dot],
          [heading.h, heading.p_cz, heading.zdot_origin, heading.p_cz_dot],
          0.0, " m")
    close("the two world-frame vectors turn WITH the frame, exactly",
          np.vstack([zeroed.omega_w, zeroed.r_w]),
          np.vstack([C.rot_z(-psi) @ heading.omega_w,
                     heading.r_w @ C.rot_z(-psi).T]), 1e-15)
    check("a second rezero to the same heading is a no-op",
          STATE.rezero_yaw(zeroed, psi) is zeroed,
          "the state carries the frame it is in, so advance and update may "
          "both call it")

    # THE BUG THIS REPLACES, WRITTEN OUT.  log(R_des R^T) does not separate:
    # a heading left inside R does not come out of the log map on the yaw
    # axis alone, so it is NOT something KP_YAW = 0 can switch off.
    roll_err = np.deg2rad(4.0)
    R_off = C.rot_zyx(roll_err, 0.0, psi)          # 4 deg of roll, 37 of yaw
    leak = C.log_so3(C.rot_zyx(0.0, 0.0, psi) @ R_off.T)
    check("a heading inside R leaks roll into PITCH through the log map",
          abs(leak[1]) > 0.1 * abs(leak[0]) and abs(leak[0]) < 0.99 * roll_err,
          "4 deg roll at %.0f deg heading -> %+.2f roll, %+.2f PITCH deg"
          % (np.degrees(psi), *np.degrees(leak[:2])))
    clean = C.log_so3(np.eye(3) @ STATE.rezero_yaw(
        state_at(cfg.H_CROUCH, R=R_off), psi).R.T)
    check("...and with the frame turned instead, the leak is GONE",
          abs(clean[1]) < 1e-12 and abs(abs(clean[0]) - roll_err) < 1e-12,
          "the same 4 deg reads %+.2f roll, %+.2g pitch deg"
          % (np.degrees(clean[0]), np.degrees(clean[1])))

    # arm: the latch and the turn are one instant.
    armed_yaw = LAW.BalanceLaw()
    armed_yaw.latch_setpoint(heading)
    armed_yaw.arm(0.0, heading)
    check("arm latches the heading and sets sp_yaw to EXACTLY zero",
          abs(armed_yaw.yaw_offset - psi) < 1e-12 and armed_yaw.sp_yaw == 0.0,
          "yaw_offset %+.1f deg" % np.degrees(armed_yaw.yaw_offset))
    at_arm = STATE.rezero_yaw(heading, armed_yaw.yaw_offset)
    close("...so e_R is exactly zero on the arming sweep, at ANY heading",
          C.log_so3(armed_yaw.R_des @ at_arm.R.T), np.zeros(3), 1e-12, " rad")
    # The same at rest, where the rate term is not also in the answer: the
    # ARM cannot step the wrench, which is the property the latch exists for.
    still = state_at(cfg.H_CROUCH, R=C.rot_zyx(np.deg2rad(3.0), 0.0, psi))
    still_law = LAW.BalanceLaw()
    still_law.latch_setpoint(still)
    still_law.arm(0.0, still)
    still = STATE.rezero_yaw(still, still_law.yaw_offset)
    close("...and the attitude half of the wrench with it",
          CTRL.balance_wrench(still, REF.com_command(
              still_law.ramp.at(0.0), still.R), still_law.R_des,
              still_law.gains).b_d[3:], np.zeros(3), 1e-9, " N*m")
    check("yaw error then reads DRIFT off the latch, not the magnetometer",
          abs(armed_yaw.attitude_error_deg(
              STATE.rezero_yaw(
                  state_at(cfg.H_CROUCH,
                           R=C.rot_zyx(0.0, 0.0, psi + np.deg2rad(6.0))),
                  armed_yaw.yaw_offset))[2] - 6.0) < 1e-9,
          "6 deg off the latched heading reads +6.0 deg")
    # arm reads the RAW heading, so it lands on the same offset whether the
    # state it is handed has been through rezero_yaw already or not.
    rearmed = LAW.BalanceLaw()
    rearmed.arm(0.0, at_arm)
    close("arm recovers the raw heading from an already-turned state",
          rearmed.yaw_offset, armed_yaw.yaw_offset, 1e-12, " rad")

    # -- and through the phase machine, which is where it actually fires ---
    faced = state_at(cfg.H_CROUCH, R=C.rot_zyx(0.0, 0.0, psi))
    seq_y = SEQ.StandSequence(SAFE.SafetyGate(3.0), law="srb")
    t_y = 0.0
    seq_y.update(t_y, faced)                     # limp
    check("through LIMP, SETTLE and CROUCH the world is the magnetometer's",
          seq_y.yaw_offset == 0.0 and abs(seq_y.body.yaw - psi) < 1e-12,
          "nothing is being held yet, so there is nothing to pin it to")
    for wait_s in (0.0, ST.RAMP_POSITION + 1.0, 0.0):
        assert seq_y.advance(t_y, faced.q) is None
        seq_y.update(t_y, faced)
        t_y += wait_s
        seq_y.update(t_y, faced)
    assert seq_y.phase_name == "rise", seq_y.phase_name
    check("the CROUCH -> RISE handover latches it and zeroes the yaw",
          abs(seq_y.yaw_offset - psi) < 1e-12
          and abs(seq_y.body.yaw) < 1e-12,
          "offset %+.1f deg, yaw now %+.3g deg"
          % (np.degrees(seq_y.yaw_offset), np.degrees(seq_y.body.yaw)))
    check("...on the ARMING sweep, not the one after",
          abs(seq_y.balance.attitude_error_deg(seq_y.body)[2]) < 1e-12,
          "the yaw error is zero the instant the heading is taken")
    check("...and the operator is told, with the reading that was taken",
          "%+.1f" % np.degrees(psi) in (seq_y.notice or ""),
          (seq_y.notice or "").splitlines()[0])
    seq_y.update(t_y, state_at(cfg.H_CROUCH,
                               R=C.rot_zyx(0.0, 0.0, psi + np.deg2rad(6.0))))
    check("a later sweep off the magnetometer's world is turned too",
          abs(np.degrees(seq_y.body.yaw) - 6.0) < 1e-9,
          "6 deg of heading drift reads +6.0 deg, not %+.0f"
          % np.degrees(psi + np.deg2rad(6.0)))

    # =====================================================================
    print("\n10. RISE and HOLD: the phase the controller is WATCHED in")
    # =====================================================================
    def _to_rise(law: str):
        """A fresh sequence sitting at the START of the rise.  -> (seq, t)."""
        seq = SEQ.StandSequence(SAFE.SafetyGate(3.0), law=law)
        crouch = state_at(cfg.H_CROUCH)
        t = 0.0
        seq.update(t, crouch)                    # limp: latches the setpoint
        # limp -> settle -> crouch -> rise.  Only the CROUCH ramp needs the
        # clock moved on; `advance` refuses to leave it while it is running.
        for wait_s in (0.0, ST.RAMP_POSITION + 1.0, 0.0):
            refused = seq.advance(t, crouch.q)
            assert refused is None, refused
            seq.update(t, crouch)
            t += wait_s
            seq.update(t, crouch)
        assert seq.phase_name == "rise", seq.phase_name
        return seq, t

    def _to_hold(law: str):
        """...and on up to the top of it.  -> (seq, t)."""
        seq, t = _to_rise(law)
        while seq.phase_name == "rise":
            t += 0.004
            h = (seq.balance.ramp.at(t - seq.balance.t0).h if law == "srb"
                 else cfg.H_LIFT)
            seq.update(t, state_at(h))
        return seq, t

    check("PHASES has rise then hold, between crouch and park",
          SEQ.PHASES == ("limp", "settle", "crouch", "rise", "hold",
                         "park", "done"), str(SEQ.PHASES))

    srb, t_srb = _to_hold("srb")
    check("the rise ends ON THE CLOCK, with no keypress",
          srb.phase_name == "hold",
          "arrived %.2f s after the crouch handover" % srb.balance.rise_s)

    # THE TRAP THE SPLIT CREATED.  `t_phase` resets at the boundary, and the
    # per-leg law's smoothstep is authored against time since the RISE.  Were
    # it authored against `t_phase` its alpha would restart at 0 here and
    # command CROUCH_HEIGHT at full height -- a drop, at the worst moment.
    legged, t_leg = _to_hold("per-leg")
    h_before = legged.h_cmd
    legged.update(t_leg + 0.004, state_at(cfg.H_LIFT))
    close("per-leg: h_cmd does NOT restart across the rise -> hold edge",
          legged.h_cmd, h_before, 1e-9, " m")
    check("...and it is at the lift height, not the crouch",
          abs(legged.h_cmd - ST.LIFT_HEIGHT) < 1e-9,
          "%.1f mm, crouch would be %.1f"
          % (1e3 * legged.h_cmd, 1e3 * ST.CROUCH_HEIGHT))

    # REFUSED MID-RISE, and the sequence must still BE in the rise when the
    # refusal is asked for -- a check that passes because it never got there
    # is worse than no check.
    mid, t_mid = _to_rise("srb")
    mid.update(t_mid + 0.5, state_at(cfg.H_CROUCH))
    refusal = mid.advance(t_mid + 0.5, state_at(cfg.H_CROUCH).q)
    check("ENTER is REFUSED out of the rise -- its exit is a physical fact",
          mid.phase_name == "rise" and isinstance(refusal, str),
          "still in %s; %r" % (mid.phase_name, refusal))

    # -- THE PUSH.  The whole reason the phase exists. --------------------
    def _push(seq, t, roll_deg, seconds):
        out = None
        for _ in range(int(seconds / 0.004)):
            t += 0.004
            seq.update(t, state_at(cfg.H_LIFT, R=C.rot_x(np.deg2rad(roll_deg))))
            out = seq.out
        return t, out

    t_srb, quiet = _push(srb, t_srb, 0.0, 0.2)
    close("standing at the setpoint, the law asks for NO attitude moment",
          quiet.wrench.b_d[3:], np.zeros(3), 1e-9, "N*m")

    t_srb, pushed = _push(srb, t_srb, 6.0, 0.4)
    check("rolled +6 deg, the moment OPPOSES it (this is the correction)",
          pushed.wrench.b_d[3] < 0.0,
          "Mx %+.3f N*m against roll +6.0 deg" % pushed.wrench.b_d[3])
    check("...and it is roll alone -- no phantom pitch or yaw",
          abs(pushed.wrench.b_d[4]) < 1e-6 and abs(pushed.wrench.b_d[5]) < 1e-6,
          "My %+.1e  Mz %+.1e N*m" % (pushed.wrench.b_d[4],
                                      pushed.wrench.b_d[5]))

    t_srb, _ = _push(srb, t_srb, 0.0, 1.0)       # released
    check("HoldWatch caught the excursion", srb.hold.peak_deg > 5.9,
          "peak %.2f deg from the setpoint" % srb.hold.peak_deg)
    check("...and that it came back", srb.hold.recovery_s() is not None
          and srb.hold.tilt_deg < SEQ.HoldWatch.SETTLED_DEG,
          "back under %.1f deg in %.2f s"
          % (SEQ.HoldWatch.SETTLED_DEG, srb.hold.recovery_s() or -1))
    check("...and recorded the moment it took",
          srb.hold.peak_moment_nm > 0.0,
          "%.3f N*m peak" % srb.hold.peak_moment_nm)

    quiet_watch = SEQ.HoldWatch()
    check("a hold that never moved reports no recovery rather than 0 s",
          quiet_watch.recovery_s() is None and not quiet_watch.live)

    # -- the diagnostic the hold print exists to carry --------------------
    # err != 0 with M == 0 is a DEAD loop; err != 0 with M != 0 is a loop
    # trying and not winning; err == 0 under a visible tilt is a WRONG
    # SETPOINT.  The print has to be able to say which.
    nose_up = state_at(cfg.H_LIFT, R=C.rot_y(np.deg2rad(-2.0)))
    err = srb.balance.attitude_error_deg(nose_up)
    close("attitude_error_deg is measured MINUS setpoint, per axis",
          err[:2],
          [np.degrees(nose_up.roll - srb.balance.sp_roll),
           np.degrees(nose_up.pitch - srb.balance.sp_pitch)], 1e-12, " deg")
    check("...and nose-UP reads NEGATIVE pitch, as the frame says",
          err[1] < -1.9, "%+.2f deg at 2 deg nose up" % err[1])
    check("yaw error is NaN before arm, not a false zero",
          not np.isfinite(LAW.BalanceLaw().attitude_error_deg(nose_up)[2]),
          "there is no heading reference until torque is live")

    srb.hold.add(1.0, nose_up, srb.balance,
                 srb.balance.update(1.0, nose_up))
    check("the hold line carries rpy, the error and the ruler's height",
          "rpy" in srb.hold.status() and "err" in srb.hold.status()
          and abs(srb.hold.h - nose_up.h) < 1e-12,
          srb.hold.status())
    check("...and the moment the law is ASKING for, which says it is alive",
          abs(srb.hold.moment_nm[1]) > 0.5,
          "M_y %+.2f N*m against %+.2f deg of pitch error"
          % (srb.hold.moment_nm[1], err[1]))

    # -- the tilt stop is a PARAMETER, and still a trip -------------------
    check("tilt_stop_deg defaults to the config value",
          LAW.BalanceLaw().tilt_stop_deg == cfg.TILT_STOP_DEG,
          "%.0f deg" % cfg.TILT_STOP_DEG)
    tipped = state_at(cfg.H_CROUCH, R=C.rot_x(np.deg2rad(20.0)))
    strict = LAW.BalanceLaw(); strict.arm(0.0, crouch)
    loose = LAW.BalanceLaw(tilt_stop_deg=45.0); loose.arm(0.0, crouch)
    check("20 deg trips the 12 deg stop and NOT a 45 deg one",
          "tilt" in (strict.update(0.0, tipped).trip or "")
          and "tilt" not in (loose.update(0.0, tipped).trip or ""),
          "raising it is what lets a steady-state tilt be READ")
    check("...but 50 deg still trips the raised one -- it is a trip, not off",
          "tilt" in (loose.update(
              0.0, state_at(cfg.H_CROUCH,
                            R=C.rot_x(np.deg2rad(50.0)))).trip or ""))
    check("hw.fold_stand raises it deliberately, and says so",
          FS.TILT_STOP_DEG == 45.0 and FS.TILT_STOP_DEG > cfg.TILT_STOP_DEG,
          "%.0f deg against the %.0f deg default"
          % (FS.TILT_STOP_DEG, cfg.TILT_STOP_DEG))

    # -- the TRACKING stop is a switch, and switching it changes no torque --
    check("track_stop_deg defaults to config.TRACK_STOP_RAD",
          abs(LAW.BalanceLaw().track_stop_deg
              - np.rad2deg(cfg.TRACK_STOP_RAD)) < 1e-12,
          "%.0f deg" % np.rad2deg(cfg.TRACK_STOP_RAD))
    astray = state_at(cfg.H_CROUCH + 0.09)       # 90 mm off the commanded h
    on = LAW.BalanceLaw(); on.arm(0.0, crouch)
    off = LAW.BalanceLaw(track_stop_deg=0.0); off.arm(0.0, crouch)
    out_on, out_off = on.update(0.0, astray), off.update(0.0, astray)
    check("a leg far from the IK trips it ON and does NOT trip it OFF",
          "tracking" in (out_on.trip or "")
          and "tracking" not in (out_off.trip or ""),
          "on: %r" % out_on.trip)
    # THE CLAIM THE SWITCH RESTS ON: the IK feeds the trip and the log only.
    close("...and the torque is IDENTICAL either way -- a monitor, not a law",
          out_off.tau, out_on.tau, 0.0, " N*m")
    close("...and q_ref is still computed and logged when it is off",
          out_off.q_ref, out_on.q_ref, 0.0, " rad")
    check("hw.fold_stand switches it off, deliberately",
          FS.TRACK_STOP_DEG == 0.0,
          "it fires at 40 mm of ramp lag against a 55 mm rise from the fold")

    # =====================================================================
    print("\n11. the FOLD posture: a crouch the lift was not designed for")
    # =====================================================================
    raw = C.unflat(POSE.FOLD_CAPTURED_Q)
    hip_raw = SK.hip_to_foot_stance(raw)
    check("the raw capture is NOT symmetric and NOT one height",
          abs(hip_raw[0, 0] - hip_raw[1, 0]) > 1e-3
          or np.ptp(hip_raw[:, 2]) > 1e-3,
          "front x differ %.1f mm, foot z spans %.1f mm"
          % (1e3 * abs(hip_raw[0, 0] - hip_raw[1, 0]),
             1e3 * np.ptp(hip_raw[:, 2])))

    close("regularised: the two front feet mirror exactly",
          POSE.FOLD.foot_xy[0], POSE.FOLD.foot_xy[1] * [1, -1], 1e-15, " m")
    close("...and the two rear feet",
          POSE.FOLD.foot_xy[2], POSE.FOLD.foot_xy[3] * [1, -1], 1e-15, " m")
    check("...and the JOINTS come out mirrored, which is the point of doing "
          "it in foot space",
          np.abs(POSE.FOLD.q[1] + POSE.FOLD.q[0]).max() < 1e-12
          and np.abs(POSE.FOLD.q[3] + POSE.FOLD.q[2]).max() < 1e-12,
          "worst %.1e rad" % max(np.abs(POSE.FOLD.q[1] + POSE.FOLD.q[0]).max(),
                                 np.abs(POSE.FOLD.q[3] + POSE.FOLD.q[2]).max()))
    x_fold = SK.all_foot_positions(POSE.FOLD.q)
    close("...and all four feet are at ONE trunk-frame height",
          x_fold[:, 2], np.full(C.N_LEGS, x_fold[0, 2]), 1e-12, " m")
    check("regularising moved the height by under a millimetre",
          abs(POSE.FOLD.z_origin
              - (P.FOOT_RADIUS - SK.all_foot_positions(raw)[:, 2].mean())) < 1e-3,
          "h %.2f mm, from a capture at %.2f"
          % (1e3 * POSE.FOLD.h,
             1e3 * (P.FOOT_RADIUS - SK.all_foot_positions(raw)[:, 2].mean()
                    - cfg.TRUNK_BOTTOM_OFFSET)))

    check("the fold crouch does NOT start on the floor",
          POSE.FOLD.h > 0.05 and abs(POSE.NOMINAL.h) < 1e-9,
          "fold h %.1f mm against nominal %.1f" % (1e3 * POSE.FOLD.h,
                                                   1e3 * POSE.NOMINAL.h))
    check("every leg is inside its reach at the LIFT height",
          float(np.max(POSE.FOLD.reach_used)) < 0.9,
          "worst %.3f of LEG_REACH at the crouch"
          % float(np.max(POSE.FOLD.reach_used)))

    # WHY foot_xy HAD TO BECOME AN ARGUMENT.
    close("the reference at the fold's own height IS the fold pose",
          LAW.ik_reference(POSE.FOLD.h, POSE.FOLD.q, POSE.FOLD.foot_xy),
          POSE.FOLD.q, 1e-12, " rad")
    nominal_ref = LAW.ik_reference(POSE.FOLD.h, POSE.FOLD.q)
    check("...while the NOMINAL pin commands a different pose",
          np.abs(C.flat(nominal_ref) - C.flat(POSE.FOLD.q)).max()
          > np.deg2rad(10.0),
          "%.1f deg (the stop is %.0f) -- two postures, not a failure"
          % (np.degrees(np.abs(C.flat(nominal_ref)
                               - C.flat(POSE.FOLD.q)).max()),
             np.rad2deg(cfg.TRACK_STOP_RAD)))

    def _lift(pose):
        """Torque and trips over a perfectly tracked lift from `pose`."""
        law = LAW.BalanceLaw(foot_xy=pose.foot_xy, dynamic_setpoint=False,
                             srb=pose.srb)
        at = lambda h: state_at(h, foot_xy=pose.foot_xy, q_seed=pose.q,
                                srb=pose.srb)
        law.arm(0.0, at(pose.h))
        trips, taus, fz = [], [], None
        for t in np.linspace(0.0, cfg.T_RISE, 61):
            out = law.update(t, at(law.ramp.at(t).h))
            trips.append(out.trip)
            taus.append(float(np.abs(out.tau).max()))
            fz = out.allocation.fz
        return [x for x in trips if x], max(taus), fz

    bad, tau_fold, fz_fold = _lift(POSE.FOLD)
    check("THE FOLD LIFTS: no trip over the whole rise",
          not bad, bad[0] if bad else "rise + allocator clean")
    check("...and it needs MORE torque than the nominal crouch",
          tau_fold > _lift(POSE.NOMINAL)[1],
          "%.2f N*m against %.2f -- folded legs, worse leverage"
          % (tau_fold, _lift(POSE.NOMINAL)[1]))
    check("...more than TAU_START_MAX, so the default cap CANNOT lift it",
          tau_fold > 1.0, "%.2f N*m against a 1.0 N*m start cap" % tau_fold)
    check("...and under TAU_STAGED_MAX, so --tau-cap 3.0 can",
          tau_fold < 3.0, "%.2f N*m against the 3.0 N*m staged ceiling"
          % tau_fold)
    check("front and rear share the weight: rear folded as the front",
          abs(fz_fold[2:].sum() - fz_fold[:2].sum()) < 0.02 * fz_fold.sum(),
          "front %.0f%% / rear %.0f%%, CoM over the support centroid"
          % (100 * fz_fold[:2].sum() / fz_fold.sum(),
             100 * fz_fold[2:].sum() / fz_fold.sum()))

    # -- the SRB model is PINNED PER POSTURE, and still a constant ---------
    check("the nominal posture flies config.SRB ITSELF, not a copy",
          POSE.NOMINAL.srb is cfg.SRB,
          "the path hw.stand has always flown stays bit-identical")
    check("the fold posture derives its own, and it DIFFERS",
          POSE.FOLD.srb is not cfg.SRB
          and abs(POSE.FOLD.srb.inertia_body[1, 1]
                  - cfg.SRB.inertia_body[1, 1]) > 0.01
          and abs(POSE.FOLD.srb.com_body[2] - cfg.COM_BODY[2]) > 1e-3,
          "I_yy %.4f against %.4f, c^b z %+.1f mm against %+.1f"
          % (POSE.FOLD.srb.inertia_body[1, 1], cfg.SRB.inertia_body[1, 1],
             1e3 * POSE.FOLD.srb.com_body[2], 1e3 * cfg.COM_BODY[2]))
    check("...and it is still a CONSTANT -- same object every read",
          POSE.FOLD.srb is POSE.FOLD.srb,
          "pinned, not evaluated per sweep")
    check("the arrays cannot be written through",
          not POSE.FOLD.srb.com_body.flags.writeable)

    # The first fold run logged 0.67 N*m of steady pitch moment: the old
    # rear-tucked fold put c^b 13.8 mm back.  Rear folded as the front
    # (2026-09-17), the CoM is back over the trunk origin.
    dx = float(cfg.COM_BODY[0] - POSE.FOLD.srb.com_body[0])
    phantom = cfg.WEIGHT * abs(dx)
    check("the refolded rear puts c^b x back on the nominal: no pitch bias",
          phantom < 0.05,
          "%.1f mm x %.1f N = %.2f N*m" % (1e3 * abs(dx), cfg.WEIGHT, phantom))

    # THE CANCELLATION com_command's docstring depends on: the reference and
    # the measurement must convert with the SAME c^b or the z error is biased.
    for pose in (POSE.NOMINAL, POSE.FOLD):
        here = state_at(cfg.H_LIFT, foot_xy=pose.foot_xy, q_seed=pose.q,
                        srb=pose.srb)
        cmd = REF.com_command(REF.HeightCommand(cfg.H_LIFT, 0.0, 0.0),
                              here.R, pose.srb)
        close("%s: at the commanded height the CoM z error is ZERO"
              % pose.name, cmd.p_cz, here.p_cz, 1e-9, " m")
    mixed = REF.com_command(REF.HeightCommand(cfg.H_LIFT, 0.0, 0.0),
                            np.eye(3), cfg.SRB)
    fold_here = state_at(cfg.H_LIFT, foot_xy=POSE.FOLD.foot_xy,
                         q_seed=POSE.FOLD.q, srb=POSE.FOLD.srb)
    check("...and MIXING the two models puts a bias straight into it",
          abs(mixed.p_cz - fold_here.p_cz) > 1e-3,
          "%.1f mm of phantom height error -- why they travel as one object"
          % (1e3 * abs(mixed.p_cz - fold_here.p_cz)))

    bad_fix, tau_fix, _ = _lift(POSE.FOLD)
    check("the fold still lifts with the corrected model",
          not bad_fix, bad_fix[0] if bad_fix else "rise clean, %.2f N*m peak"
          % tau_fix)

    # -- ROLL BY MOMENT: the second fold run's roll that did not come back --
    ixx = POSE.FOLD.srb.inertia_body[0, 0]
    iyy = POSE.FOLD.srb.inertia_body[1, 1]
    default = CTRL.BalanceGains()
    check("equal kp_att is NOT equal stiffness on this robot",
          iyy * default.kp_att[1] > 3.0 * ixx * default.kp_att[0],
          "fold: roll %.2f N*m/rad against pitch %.2f"
          % (ixx * default.kp_att[0], iyy * default.kp_att[1]))
    rolled = CTRL.BalanceGains()
    rolled.kp_att[0], rolled.kd_att[0] = FS.ROLL_GAINS
    close("fold_stand's roll kp puts roll at DOG5's 10 N*m/rad",
          rolled.kp_att[0] * ixx, 10.0, 0.05, " N*m/rad")
    close("...and pitch and yaw are exactly as they were",
          [rolled.kp_att[1], rolled.kd_att[1], rolled.kp_att[2],
           rolled.kd_att[2]],
          [default.kp_att[1], default.kd_att[1], default.kp_att[2],
           default.kd_att[2]], 0.0, "")

    def _roll_push(gains):
        """The REAL law in the fold stance, rolled 5.6 deg -> commanded Mx."""
        law = LAW.BalanceLaw(gains=gains, foot_xy=POSE.FOLD.foot_xy,
                             dynamic_setpoint=False, srb=POSE.FOLD.srb)
        at = lambda R=None: state_at(cfg.H_LIFT, R=R, foot_xy=POSE.FOLD.foot_xy,
                                     q_seed=POSE.FOLD.q, srb=POSE.FOLD.srb)
        law.arm(0.0, at())
        law.ramp = REF.Quintic.ramp(cfg.H_LIFT, cfg.H_LIFT, 1.0)
        out = law.update(1.0, at(C.rot_x(np.deg2rad(5.6))))
        return out
    weak, stiff = _roll_push(CTRL.BalanceGains()), _roll_push(rolled)
    check("the same 5.6 deg roll now asks for ~3x the restoring moment",
          stiff.wrench.b_d[3] < 2.5 * weak.wrench.b_d[3] < 0.0,
          "Mx %+.3f -> %+.3f N*m (the run logged -0.14 with the old gains)"
          % (weak.wrench.b_d[3], stiff.wrench.b_d[3]))
    # The residual is NOT zero and is not meant to be: LAMBDA's Tikhonov
    # damping leaves ~1e-4 N*m at any gain.  What matters is that it stays at
    # that floor rather than growing with the bigger ask.
    check("...and the allocator delivers it -- no trip, residual at the "
          "lambda floor",
          stiff.trip is None and stiff.allocation.residual_moment < 1e-3,
          "%.1e N*m (sustained trip %.2f); fz %s N"
          % (stiff.allocation.residual_moment, cfg.RESIDUAL_MOMENT_NM,
             np.array2string(stiff.allocation.fz, precision=1)))
    stiff_lift = LAW.BalanceLaw(gains=rolled, foot_xy=POSE.FOLD.foot_xy,
                                dynamic_setpoint=False, srb=POSE.FOLD.srb)
    at_h = lambda h: state_at(h, foot_xy=POSE.FOLD.foot_xy,
                              q_seed=POSE.FOLD.q, srb=POSE.FOLD.srb)
    stiff_lift.arm(0.0, at_h(POSE.FOLD.h))
    trips = [stiff_lift.update(t, at_h(stiff_lift.ramp.at(t).h)).trip
             for t in np.linspace(0.0, cfg.T_RISE, 61)]
    check("the fold still lifts clean with the stiffer roll",
          not any(trips), "peak %.2f N*m" % stiff_lift.tau_peak)

    # THE LATENCY THE CHOICE OF DAMPING WAS MADE ON.  Sampled double
    # integrator, ZOH at the 4 ms sweep, `d` sweeps of pure delay, PD on state.
    def _max_latency_ms(kp, kd, T=0.004):
        for d in range(1, 60):
            A = np.array([[1, T], [0, 1]]); B = np.array([[T * T / 2], [T]])
            F = np.zeros((2 + d, 2 + d)); F[:2, :2] = A; F[:2, 2:3] = B
            for i in range(2, 1 + d):
                F[i, i + 1] = 1.0
            F[1 + d, :2] = -np.array([kp, kd])
            if np.abs(np.linalg.eigvals(F)).max() >= 1.0:
                return 4 * (d - 1)
        return 999
    lat = _max_latency_ms(rolled.kp_att[0], rolled.kd_att[0])
    check("kd %.0f is the roll damping with the MOST latency margin"
          % rolled.kd_att[0],
          lat >= max(_max_latency_ms(rolled.kp_att[0], kd)
                     for kd in (6, 17, 29, 35)),
          "stable to %d ms; kd 29 gets %d, kd 6 gets %d"
          % (lat, _max_latency_ms(rolled.kp_att[0], 29.0),
             _max_latency_ms(rolled.kp_att[0], 6.0)))
    check("...and that margin is UNDER the IMU freeze, which is why the hold "
          "line prints the age",
          lat < 1e3 * cfg.IMU_MAX_AGE_S,
          "%d ms stable against a %.0f ms freeze" % (lat, 1e3 * cfg.IMU_MAX_AGE_S))

    try:
        SEQ.StandSequence(SAFE.SafetyGate(3.0), law="per-leg", crouch=POSE.FOLD)
        refused = None
    except ValueError as why:
        refused = str(why)
    check("per-leg is REFUSED from a non-nominal crouch, not silently wrong",
          refused is not None,
          "it pins its springs at the nominal foot xy and has no posture")

    # =====================================================================
    print("\n12. the trot in place (gait, swing, the trot half of the law)")
    from .. import fold_trot as FT
    from . import gait as GAIT
    from . import swing as SWING

    # -- the clock ------------------------------------------------------
    gait = GAIT.TrotGait()
    gait.reset(0.0)
    fl, fr, rl, rr = (C.LEGS.index(n) for n in ("FL", "FR", "RL", "RR"))
    t_end = gait.cycle_start(3 * gait.settle_every)
    ts = np.arange(0.0, t_end, 0.002)
    samples = [gait.sample(t) for t in ts]
    contact = np.array([s.contact for s in samples])
    weight = np.array([s.weight for s in samples])
    check("the diagonals pair: FL with RR, FR with RL, at every instant",
          bool(np.all(contact[:, fl] == contact[:, rr])
               and np.all(contact[:, fr] == contact[:, rl])))
    check("at least two feet are planted at every instant",
          bool(np.all(contact.sum(axis=1) >= 2)),
          "counts seen %s" % sorted(set(contact.sum(axis=1).tolist())))
    check("a swinging foot carries exactly zero weight",
          bool(np.all(weight[~contact] == 0.0)))
    check("the weight never steps (the handover is a ramp)",
          float(np.abs(np.diff(weight, axis=0)).max()) < 0.03,
          "worst change %.4f per 2 ms"
          % float(np.abs(np.diff(weight, axis=0)).max()))
    check("reset starts the clock at FULL four-foot weight -- entering moves "
          "nothing", bool(np.all(weight[0] == 1.0)) and gait.full_support(0.0))
    first_lift = ts[np.argmax(~contact.all(axis=1))]
    close("...and the first foot lifts entry_s later", first_lift,
          gait.entry_s, 0.0021, " s")
    settle = np.array([gait.settling(t) for t in ts])
    check("every settle is four feet at full weight",
          settle.any() and bool(np.all(weight[settle] >= 1.0 - 1e-9)),
          "%.2f s of settle in %.1f s" % (settle.mean() * t_end, t_end))
    lifts = [leg for _, leg in sorted(
        (i, leg) for leg in (fl, fr)
        for i in np.flatnonzero(contact[:-1, leg] & ~contact[1:, leg]))]
    check("one gait: the diagonals strictly take turns, no lead flip",
          len(lifts) >= 4 and all(a != b for a, b in zip(lifts, lifts[1:])),
          " ".join("FL/RR" if leg == fl else "FR/RL" for leg in lifts))
    try:
        GAIT.TrotGait(duty=0.5)
        refused = False
    except ValueError:
        refused = True
    check("duty 0.5 is refused: no window to hand the load across in",
          refused)

    # -- the allocator's trot path ----------------------------------------
    fold = POSE.FOLD
    at_fold = state_at(cfg.H_LIFT, foot_xy=fold.foot_xy, q_seed=fold.q,
                       srb=fold.srb)
    b_stand = np.array([0.0, 0.0, cfg.WEIGHT, 0.12, -0.3, 0.0])
    a_none = ALLOC.allocate(at_fold.r_w, b_stand)
    a_ones = ALLOC.allocate(at_fold.r_w, b_stand, contact=np.ones(4))
    close("contact all ones is the stand's allocation (nothing clips)",
          a_ones.f_w, a_none.f_w, 1e-9, " N")
    diag = np.array([1.0, 0.0, 0.0, 1.0])
    a_diag = ALLOC.allocate(at_fold.r_w, b_stand, contact=diag)
    check("a swinging foot gets EXACTLY zero force -- not fz_min, not 1e-6",
          bool(np.all(a_diag.f_w[[fr, rl]] == 0.0)),
          "fz %s N" % np.array2string(a_diag.fz, precision=2))
    # To the allocator's LAMBDA floor, not to 1e-6: the Tikhonov damping
    # leaves a few hundredths of a newton in the force rows on two feet.
    close("...and the diagonal still carries the robot's weight",
          a_diag.fz.sum(), cfg.WEIGHT, 0.05, " N")
    ramp_w = np.array([1.0, 0.002, 0.002, 1.0])
    a_ramp = ALLOC.allocate(at_fold.r_w, b_stand, contact=ramp_w)
    check("a foot ramping out is bounded by its weight (fz <= w fz_max)",
          bool(np.all(a_ramp.fz <= ramp_w * cfg.FZ_MAX + 1e-9)),
          "fz %s N" % np.array2string(a_ramp.fz, precision=2))

    # -- the height over the stance feet ----------------------------------
    close("on_stance over all four feet is read() exactly",
          [STATE.on_stance(at_fold, np.ones(4, bool), fold.srb).p_cz,
           STATE.on_stance(at_fold, np.ones(4, bool), fold.srb).h],
          [at_fold.p_cz, at_fold.h], 1e-12, " m")
    rest = SWING.rest_feet_b(cfg.H_LIFT, fold.foot_xy)
    close("rest_feet_b is where the IK put the feet",
          rest, at_fold.x_b, 1e-9, " m")
    q_up = C.unflat(at_fold.q).copy()
    apex = rest[fr] + np.array([0.0, 0.0, cfg.SWING_HEIGHT])
    hip = apex.copy()
    hip[:2] -= P.HIP_OFFSET[fr][:2]
    q_up[fr] = HK.leg_ik(fr, hip, q_seed=q_up[fr])
    lifted = STATE.read(C.flat(q_up), np.zeros(12),
                        IMU.TrunkOrientation.level(), srb=fold.srb)
    stance_fr = np.array([True, False, True, True])
    check("a foot at its apex reads the trunk LOW over all four feet...",
          lifted.h < cfg.H_LIFT - 0.009,
          "h %.1f mm against %.1f" % (1e3 * lifted.h, 1e3 * cfg.H_LIFT))
    close("...and exactly right over the stance feet",
          STATE.on_stance(lifted, stance_fr, fold.srb).h, cfg.H_LIFT, 1e-9,
          " m")

    # -- the swing arc and law --------------------------------------------
    p0, v0 = SWING.swing_reference(rest[fl], 0.0, gait.swing_duration)
    pm, _ = SWING.swing_reference(rest[fl], 0.5, gait.swing_duration)
    p1, v1 = SWING.swing_reference(rest[fl], 1.0, gait.swing_duration)
    close("the arc leaves and lands on the resting site, at rest",
          [*p0, *p1, *v0, *v1], [*rest[fl], *rest[fl], 0, 0, 0, 0, 0, 0],
          1e-12)
    close("...straight up: apex SWING_HEIGHT above it, no x or y",
          pm - rest[fl], [0.0, 0.0, cfg.SWING_HEIGHT], 1e-12, " m")
    close("the swing impedance is zero on the reference, at rest",
          SWING.swing_torque(at_fold, fl, rest[fl], np.zeros(3)),
          np.zeros(3), 1e-12, " N*m")

    # THE FOLD'S JOINT SWING: the same arc through the IK, abd held.
    q_fold = C.unflat(at_fold.q)
    worst_abd = worst_xy = worst_z = worst_qd = 0.0
    for leg in range(C.N_LEGS):
        for s in np.linspace(0.0, 1.0, 41)[:-1]:
            qj, qdj = SWING.joint_swing_reference(
                leg, rest[leg], s, FT.PERIOD_S * (1.0 - cfg.DUTY), q_fold[leg])
            pj, _ = SWING.swing_reference(rest[leg], s,
                                          FT.PERIOD_S * (1.0 - cfg.DUTY))
            xj = HK.foot_position(leg, qj)
            worst_abd = max(worst_abd, abs(qj[0] - q_fold[leg][0]))
            worst_xy = max(worst_xy, float(np.abs(xj[:2] - rest[leg][:2]).max()))
            worst_z = max(worst_z, abs(xj[2] - pj[2]))
            worst_qd = max(worst_qd, abs(qdj[0]))
    check("joint swing: abd is HELD at the resting IK angle, every leg",
          worst_abd < 1e-12 and worst_qd == 0.0,
          "worst %.1e rad, abd rate %.1e" % (worst_abd, worst_qd))
    check("...and the foot goes up in z: no more than 2 mm of x or y",
          worst_xy < 2e-3 and worst_z < 1e-3,
          "x/y %.2f mm, z off the arc %.2f mm" % (1e3 * worst_xy,
                                                1e3 * worst_z))
    close("the joint swing is zero torque on the reference, at rest",
          SWING.joint_swing_torque(at_fold, fl, *SWING.joint_swing_reference(
              fl, rest[fl], 0.0, 0.4, q_fold[fl])),
          np.zeros(3), 1e-9, " N*m")
    try:
        LAW.BalanceLaw(swing="joint").begin_step(fold.foot_xy)
        refused = False
    except ValueError:
        refused = True
    check("the joint swing refuses a foot step: it only lifts in z", refused)

    # THE LIFTOFF LATCH: the 2026-09-17 hardware kick.  A foot 8 mm off the
    # pinned rest site must start its arc where it IS -- zero PD at liftoff.
    latch_law = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                               srb=fold.srb, track_stop_deg=0.0,
                               swing="joint")
    latch_law.arm(0.0, at_fold)
    latch_law.ramp = REF.Quintic.ramp(cfg.H_LIFT, cfg.H_LIFT, 1.0)
    off_q = C.unflat(at_fold.q).copy()
    lat_gait = GAIT.TrotGait(period=FT.PERIOD_S)
    lat_gait.reset(0.0)
    t_lift = next(t for t in np.arange(0.0, FT.PERIOD_S, 0.004)
                  if not lat_gait.sample(t).contact.all())
    lifter = int(np.flatnonzero(~lat_gait.sample(t_lift).contact)[0])
    hip_off = rest[lifter] - P.HIP_OFFSET[lifter] + [0.004, 0.0, 0.008]
    off_q[lifter] = HK.leg_ik(lifter, hip_off, q_seed=off_q[lifter])
    off_state = STATE.read(C.flat(off_q), np.zeros(12),
                           IMU.TrunkOrientation.level(), srb=fold.srb)
    with_swing = latch_law.update(t_lift, off_state, gait=lat_gait)
    s0 = float(with_swing.swing_s[lifter])
    q_l, qd_l = SWING.joint_swing_reference(
        lifter, off_state.x_b[lifter], s0, lat_gait.swing_duration,
        off_q[lifter])
    kick = SWING.joint_swing_torque(off_state, lifter, q_l, qd_l)
    check("joint swing, foot 8 mm off its site: the arc starts ON the foot",
          np.allclose(latch_law._lift_x[lifter], off_state.x_b[lifter],
                      atol=1e-12)
          and float(np.abs(kick).max()) < 0.2,
          "PD at liftoff %s N*m (unlatched: 8 mm = ~4 N*m on the knee)"
          % np.array2string(kick, precision=3))
    check("hw.fold_trot's swing lands >= 300 ms, where MuJoCo says it tracks",
          FT.PERIOD_S * (1.0 - cfg.DUTY) >= 0.30,
          "%.0f ms of swing" % (1e3 * FT.PERIOD_S * (1.0 - cfg.DUTY)))

    # A tracked trot through two settle blocks: the law in each stance, the
    # swing legs where the arc says, the stance legs at the IK, the trunk
    # level.  The residual moment is what separates the two stances.
    def _tracked_trot(pose, gains, period=None, **law_kw):
        at_pose = state_at(cfg.H_LIFT, foot_xy=pose.foot_xy, q_seed=pose.q,
                           srb=pose.srb)
        pose_rest = SWING.rest_feet_b(cfg.H_LIFT, pose.foot_xy)
        law = LAW.BalanceLaw(gains=gains, foot_xy=pose.foot_xy, srb=pose.srb,
                             dynamic_setpoint=False, track_stop_deg=25.0,
                             **law_kw)
        law.arm(0.0, at_pose)
        law.ramp = REF.Quintic.ramp(cfg.H_LIFT, cfg.H_LIFT, 1.0)
        clock = GAIT.TrotGait() if period is None else GAIT.TrotGait(
            period=period)
        clock.reset(0.0)
        q_stance = C.unflat(at_pose.q)
        trips, taus, fz_sum, heights, moments = set(), [], [], [], []
        for t in np.arange(0.0, clock.cycle_start(2 * clock.settle_every),
                           0.004):
            sample = clock.sample(t)
            q_t = q_stance.copy()
            for leg in np.flatnonzero(~sample.contact):
                p_arc, _ = SWING.swing_reference(
                    pose_rest[leg], sample.swing_s[leg], clock.swing_duration)
                hip = p_arc.copy()
                hip[:2] -= P.HIP_OFFSET[leg][:2]
                q_t[leg] = HK.leg_ik(leg, hip, q_seed=q_stance[leg])
            body = STATE.read(C.flat(q_t), np.zeros(12),
                              IMU.TrunkOrientation.level(), srb=pose.srb)
            out = law.update(t, body, gait=clock)
            if out.trip:
                trips.add(out.trip)
            taus.append(out.tau)
            fz_sum.append(out.allocation.fz.sum())
            heights.append(out.state.h)
            moments.append(out.allocation.residual_moment)
        return trips, np.array(taus), fz_sum, heights, max(moments)

    fold_gains = CTRL.BalanceGains()
    fold_gains.kp_att[0], fold_gains.kd_att[0] = FS.ROLL_GAINS
    runs = {"fold": _tracked_trot(fold, fold_gains,
                                  period=FT.PERIOD_S,
                                  tilt_stop_deg=FS.TILT_STOP_DEG,
                                  swing="joint"),
            "nominal": _tracked_trot(POSE.NOMINAL, CTRL.BalanceGains())}
    for name, (trips, taus, fz_sum, heights, moment) in runs.items():
        step = float(np.abs(np.diff(taus, axis=0)).max())
        check("%s: two settle blocks of trot, no trip, every torque finite"
              % name, not trips and bool(np.all(np.isfinite(taus))),
              "; ".join(sorted(trips))[:60])
        check("%s: peak torque inside the trot's 9 N*m cap" % name,
              float(np.abs(taus).max()) < FT.TAU_CAP,
              "%.2f N*m" % np.abs(taus).max())
        # 0.1 N: the lambda floor, plus -- in the nominal run -- the small
        # attitude moment the config setpoint asks of a level fixture trunk,
        # which a diagonal pair cannot make either.
        close("%s: the stance feet carry the weight (lambda floor)" % name,
              fz_sum, cfg.WEIGHT, 0.1, " N")
        close("%s: the height the law acts on ignores the swing apex" % name,
              heights, cfg.H_LIFT, 1e-9, " m")
        check("%s: no handover step the trot slew cannot follow in 2 sweeps"
              % name, step < 2 * cfg.TAU_SLEW_TROT_NM_S * 0.004,
              "worst %.3f N*m in 4 ms (%.0f N*m/s)" % (step, step / 0.004))
    # The nominal's residual is not zero only because the fixture trunk is
    # level and the setpoint is the config statics (-0.29 / +0.12 deg): the
    # law asks for ~0.04 N*m of trim.  The fold's too, since the rear was
    # refolded as the front (2026-09-17); before, the CoM sat 17.2 mm off.
    check("both stances' diagonals pass through the CoM",
          max(runs["nominal"][4], runs["fold"][4]) < 0.2,
          "worst residual moment on two feet %.3f vs %.3f N*m"
          % (runs["nominal"][4], runs["fold"][4]))

    # The residual trip: suppressed on two feet, live on four.  A monitor that
    # fires on the first sweep with ANY residual makes the two cases differ
    # only in which sweep the clock is on.
    res_law = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                             srb=fold.srb, track_stop_deg=0.0)
    res_law.arm(0.0, at_fold)
    res_law.residual = ALLOC.ResidualMonitor(force_n=0.0, moment_nm=0.0,
                                             streak=1)
    two = next(t for t in ts if gait.sample(t).contact.sum() == 2)
    two_out = res_law.update(two, at_fold, gait=gait)
    check("on two feet the residual trip does not count the geometry",
          two_out.trip is None and two_out.allocation.residual_moment > 1e-3,
          "residual %.2f N*m, no trip" % two_out.allocation.residual_moment)
    four_out = res_law.update(0.0, at_fold, gait=gait)
    check("...and on four feet the same monitor still trips",
          four_out.trip is not None and "residual" in four_out.trip,
          "residual %.1e N*m" % four_out.allocation.residual_moment)

    # -- the phase machine --------------------------------------------------
    seq = SEQ.StandSequence(SAFE.SafetyGate(3.0), crouch=fold,
                            balance=LAW.BalanceLaw(
                                foot_xy=fold.foot_xy, dynamic_setpoint=False,
                                srb=fold.srb, track_stop_deg=0.0),
                            gait=GAIT.TrotGait())
    refusal = seq.toggle_trot(0.0)
    check("T is refused outside HOLD", "ignored" in refusal, refusal[:50])
    seq.phase = SEQ.PHASES.index("hold")
    seq.body = at_fold
    seq.gate.start(0.0, q=at_fold.q)
    seq.balance.arm(0.0, at_fold)
    seq.toggle_trot(1.0)
    check("T from HOLD trots, and the phase NAME says so",
          seq.trotting and seq.phase_name == "trot")
    check("ENTER is refused while trotting",
          isinstance(seq.advance(1.1, at_fold.q), str))
    seq.toggle_trot(1.0 + seq.gait.entry_s + 0.05)       # latched mid-swing
    t_latch = 1.0 + seq.gait.entry_s + 0.05
    mid = seq.update(t_latch, at_fold)
    check("a latched exit is NOT taken mid-swing",
          seq.trotting and mid[0] == "torque")
    t = t_latch
    while seq.trotting and t < t_latch + 2.0:
        t += 0.004
        seq.update(t, at_fold)
    check("...it is taken at the next four-foot window, back to HOLD",
          seq.phase_name == "hold" and seq.gait.full_support(t),
          "%.2f s after the latch" % (t - t_latch))
    half = SEQ.StandSequence(SAFE.SafetyGate(3.0), crouch=fold,
                             balance=LAW.BalanceLaw(
                                 foot_xy=fold.foot_xy, dynamic_setpoint=False,
                                 srb=fold.srb, track_stop_deg=0.0),
                             gait=GAIT.TrotGait(), trot_swings=1)
    half.phase = SEQ.PHASES.index("hold")
    half.body = at_fold
    half.gate.start(0.0, q=at_fold.q)
    half.balance.arm(0.0, at_fold)
    presses, t = [], 1.0
    for _ in range(3):
        t0 = t
        half.toggle_trot(t0)
        lifted, lift_wt = set(), 1.0
        while half.trotting and t < t0 + 2.0 * half.gait.period:
            t += 0.004
            half.update(t, at_fold)
            if half.trotting and half.out.swing_s is not None:
                lifted |= set(np.flatnonzero(half.out.swing_s > 0.0).tolist())
        presses.append((tuple(sorted(C.LEGS[i] for i in lifted)),
                        half.phase_name == "hold" and half.gait.full_support(t)
                        and t - t0 < half.gait.period, t - t0))
        t += 0.5                                         # the operator waits
    check("--half-gait: each T is ONE diagonal's swing, then HOLD by itself",
          all(ok for _, ok, _ in presses),
          "; ".join("%s %.2f s" % ("/".join(legs), dt)
                    for legs, _, dt in presses))
    check("...and the diagonal alternates press by press",
          [legs for legs, _, _ in presses]
          == [("FR", "RL"), ("FL", "RR"), ("FR", "RL")],
          " -> ".join("/".join(legs) for legs, _, _ in presses))
    # The sequence: latched ON REACHING HOLD, kept through T and the exit.
    js = SEQ.StandSequence(SAFE.SafetyGate(3.0), crouch=fold,
                           balance=LAW.BalanceLaw(
                               foot_xy=fold.foot_xy, dynamic_setpoint=False,
                               srb=fold.srb, track_stop_deg=0.0),
                           gait=GAIT.TrotGait(), trot_swings=1)
    js.phase = SEQ.PHASES.index("rise")
    js.body = at_fold
    js.gate.start(0.0, q=at_fold.q)
    js.balance.arm(0.0, at_fold)
    js.t_phase = -10.0                               # the rise has arrived
    none_before = js.balance.q_hold is None
    js.update(0.0, at_fold)
    latched = js.balance.q_hold
    check("the joint target is latched on the sweep the robot reaches HOLD",
          none_before and js.phase_name == "hold" and latched is not None
          and np.array_equal(latched, at_fold.q))
    js.toggle_trot(0.1)
    t = 0.1
    while js.trotting and t < 3.0:
        t += 0.004
        js.update(t, at_fold)
    check("...and T neither re-latches nor releases it: the same angles "
          "after the trot", js.phase_name == "hold"
          and js.balance.q_hold is latched)
    jl = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                        srb=fold.srb, track_stop_deg=0.0,
                        kp_joint=cfg.KP_JOINT_HOLD,
                        kd_joint=cfg.KD_JOINT_HOLD)
    jl.arm(0.0, at_fold)
    jg = GAIT.TrotGait()
    jg.reset(0.0)
    t_two = next(t for t in np.arange(0.0, jg.period, 0.002)
                 if jg.sample(t).contact.sum() == 2)
    swing_now = ~jg.sample(t_two).contact
    base = jl.update(t_two, at_fold, gait=jg).tau
    dq = np.full(C.N_JOINTS, 0.05)
    jl.hold_joints(at_fold.q - dq)            # the hold was 0.05 rad lower
    with_layer = jl.update(t_two, at_fold, gait=jg).tau
    expect = C.unflat(-jl.kp_joint * dq).copy()
    expect[swing_now] = 0.0
    close("joint layer: stance legs get Kp (q_hold - q), swing legs no spring",
          with_layer - base, C.flat(expect), 1e-9, " N*m")
    jl.release_joints()
    close("...and released, the law is exactly the one without it",
          jl.update(t_two, at_fold, gait=jg).tau, base, 1e-12, " N*m")
    try:
        SAFE.SafetyGate(FT.TAU_CAP)
        default_refuses = False
    except ValueError:
        default_refuses = True
    knee = list(C.LEGS).index("RR") * 3 + 2
    def _knee_trip(trip_on):
        gate = SAFE.SafetyGate(3.0, overspeed_trip=trip_on)
        q_run, qd_run = np.zeros(12), np.zeros(12)
        qd_run[knee] = 7.5
        gate.start(0.0, q=q_run)
        reason = None
        for i in range(1, 6):
            q_run = q_run.copy()
            q_run[knee] += 7.4 * 0.004
            reason = reason or gate.estop_reason(q_run, qd_run, 0.004 * i)
        return reason, gate.qd_peak[knee]
    on, _ = _knee_trip(True)
    off, peak = _knee_trip(False)
    check("the 2026-09-16 knee speed trips the stand's gate, not the trot's",
          on is not None and off is None and peak == 7.5,
          "trot gate kept the %.1f rad/s peak" % peak)
    check("the 9 N*m cap needs the ceiling raised explicitly",
          default_refuses and SAFE.SafetyGate(
              FT.TAU_CAP, ceiling=FT.TAU_CAP).tau_cap == SAFE.TAU_HARD_NM)

    # -- the hold's foot step (hw.trot's W) --------------------------------
    from .. import trot as TR
    wide = TR.CROUCH                     # the trot's crouch
    home_rest = SWING.rest_feet_b(cfg.H_LIFT, wide.foot_xy)
    hip_rest = SWING.rest_feet_b(cfg.H_LIFT, TR.STEP_FOOT_XY)
    step_gait = GAIT.TrotGait(period=TR.STEP_PERIOD_S)
    dur = step_gait.swing_duration
    p0, v0 = SWING.swing_reference(home_rest[fl], 0.0, dur, land_b=hip_rest[fl])
    p1, v1 = SWING.swing_reference(home_rest[fl], 1.0, dur, land_b=hip_rest[fl])
    close("a step arc leaves the old site and lands on the new one, at rest",
          [*p0, *p1, *v0, *v1], [*home_rest[fl], *hip_rest[fl], 0, 0, 0, 0,
                                 0, 0], 1e-12)

    def _tracked_step(law, gait, t_start, q_start):
        """The law stepping, legs where it says: stance at the IK of the
        site each leg stands on, swing at last sweep's arc point."""
        trips, taus, fz_sum = set(), [], []
        p_prev = np.full((C.N_LEGS, 3), np.nan)
        q_now = q_start
        t = t_start
        while t < t_start + 3.0 * gait.period:
            stance = SWING.rest_feet_b(cfg.H_LIFT, law.foot_xy)
            q_t = q_now.copy()
            for leg in range(C.N_LEGS):
                p_leg = p_prev[leg] if np.all(np.isfinite(p_prev[leg])) \
                    else stance[leg]
                hip = p_leg.copy()
                hip[:2] -= P.HIP_OFFSET[leg][:2]
                q_t[leg] = HK.leg_ik(leg, hip, q_seed=q_now[leg])
            q_now = q_t
            body = STATE.read(C.flat(q_t), np.zeros(12),
                              IMU.TrunkOrientation.level(), srb=wide.srb)
            out = law.update(t, body, gait=gait)
            if out.trip:
                trips.add(out.trip)
            taus.append(out.tau)
            fz_sum.append(out.allocation.fz.sum())
            p_prev = out.p_swing
            if law.step_landed and gait.full_support(t):
                break
            t += 0.004
        return t, q_now, trips, np.array(taus), fz_sum

    at_wide = state_at(cfg.H_LIFT, foot_xy=wide.foot_xy, q_seed=wide.q,
                       srb=wide.srb)
    step_law = LAW.BalanceLaw(foot_xy=wide.foot_xy, srb=wide.srb,
                              dynamic_setpoint=False, track_stop_deg=25.0)
    step_law.arm(0.0, at_wide)
    step_law.ramp = REF.Quintic.ramp(cfg.H_LIFT, cfg.H_LIFT, 1.0)
    check("the law COPIES the posture's foot_xy -- a step must not move it",
          step_law.foot_xy is not wide.foot_xy)
    step_law.begin_step(TR.STEP_FOOT_XY)
    step_gait.reset(0.0)
    t_in, q_in, trips, taus, fz_sum = _tracked_step(
        step_law, step_gait, 0.0, C.unflat(at_wide.q))
    check("W: every foot lands under its hip inside one step cycle, no trip",
          step_law.step_landed and not trips
          and t_in < step_gait.period + step_gait.entry_s,
          "%.2f s; %s" % (t_in, "; ".join(sorted(trips))[:60]))
    check("...peak torque inside the trot's 9 N*m cap",
          float(np.abs(taus).max()) < TR.TAU_CAP,
          "%.2f N*m" % np.abs(taus).max())
    close("...and the stance feet carry the weight throughout (lambda floor)",
          fz_sum, cfg.WEIGHT, 0.1, " N")
    close("...and the feet ARE under the hips: q is the IK there",
          C.flat(q_in), C.flat(LAW.ik_reference(cfg.H_LIFT, q_in,
                                                TR.STEP_FOOT_XY)),
          1e-6, " rad")
    check("the trot's crouch posture itself is untouched by the step",
          np.allclose(wide.foot_xy, POSE.POSTURES[wide.name].foot_xy)
          and step_law.foot_xy is not wide.foot_xy)
    step_law.end_step()
    step_law.begin_step(wide.foot_xy)
    step_gait.reset(t_in + 0.5)
    t_back, _, trips, taus, _ = _tracked_step(step_law, step_gait,
                                              t_in + 0.5, q_in)
    check("...and back to the crouch's sites the same way, no trip",
          step_law.step_landed and not trips
          and np.allclose(step_law.foot_xy, wide.foot_xy),
          "peak %.2f N*m" % np.abs(taus).max())

    seq = SEQ.StandSequence(SAFE.SafetyGate(3.0), crouch=wide,
                            balance=LAW.BalanceLaw(
                                foot_xy=wide.foot_xy, dynamic_setpoint=False,
                                srb=wide.srb, track_stop_deg=0.0),
                            gait=GAIT.TrotGait(),
                            step_to=TR.STEP_FOOT_XY,
                            step_gait=GAIT.TrotGait(period=TR.STEP_PERIOD_S))
    refusal = seq.toggle_step(0.0)
    check("W is refused outside HOLD", "ignored" in refusal, refusal[:50])
    seq.phase = SEQ.PHASES.index("hold")
    seq.body = at_wide
    seq.gate.start(0.0, q=at_wide.q)
    seq.balance.arm(0.0, at_wide)
    seq.toggle_step(1.0)
    check("W from HOLD steps, and the phase NAME says so",
          seq.stepping and seq.phase_name == "step")
    check("ENTER and T are refused while stepping",
          isinstance(seq.advance(1.1, at_wide.q), str)
          and "ignored" in seq.toggle_trot(1.1))
    # Fixture: the robot is not simulated here, the legs are moved by hand.
    seq.balance.foot_xy[:] = TR.STEP_FOOT_XY
    seq.balance.end_step()
    seq.stepping = False
    check("ENTER in HOLD with the feet away does NOT park -- it steps home",
          seq.advance(2.0, at_wide.q) is None and seq.phase_name == "step"
          and seq.park_after_step)
    t = 2.0
    while seq.phase_name == "step" and t < 5.0:
        t += 0.004
        seq.update(t, at_wide)
    check("...and parks by itself once the feet are home and all four down",
          seq.phase_name == "park" and seq.feet_home,
          "%s after %.2f s" % (seq.phase_name, t - 2.0))

    # =====================================================================
    print("\n" + "=" * 78)
    if _FAILURES:
        print("FAILED %d of %d" % (len(_FAILURES), len(_FAILURES) + _PASSES))
        for label in _FAILURES:
            print("  - %s" % label)
        return 1
    print("all %d checks passed" % _PASSES)
    print("This says the law computes what it says it computes.  It does NOT")
    print("say the robot will stand: R_BODY_IMU is an identity placeholder,")
    print("every gain is [UNTUNED], and nothing here has seen a floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
