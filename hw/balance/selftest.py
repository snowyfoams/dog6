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
from dataclasses import replace

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

    # each trip, on the state that should fire it (no tilt trip: deleted
    # 2026-10-05, section 15 has what a tilt does now)
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

    check("the fall hold measures FROM the setpoint, not from true level",
          abs(armed_sp.tilt_from_setpoint_deg(tilted)) < 1e-9
          and abs(tilted.tilt_deg - 3.0) < 1e-9,
          "state.tilt_deg still reports %.1f deg for the log" % tilted.tilt_deg)
    check("...so it is reached %.0f deg from where the run began"
          % cfg.FALL_HOLD_DEG,
          abs(armed_sp.tilt_from_setpoint_deg(
              state_at(cfg.H_CROUCH,
                       R=C.rot_x(np.deg2rad(3.0 + cfg.FALL_HOLD_DEG))))
              - cfg.FALL_HOLD_DEG) < 1e-9)

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

    # PARALLEL, 2026-09-25: each rear foot is where the regularised front
    # leg's shape, hung from the rear pitch hinge, would put it -- then,
    # 2026-09-28, the rear FOLD_REAR_FORWARD and the front FOLD_FRONT_FORWARD
    # further forward in trunk x.  Every knee stays behind its hip.
    # 2026-10-01, then each foot turned about its abduction axis to
    # FOLD_ABD.  The pitch hinge is ON that axis, so the turn keeps the
    # hinge-to-foot x and the foot's distance from the axis, and nothing else.
    # 2026-10-02, then every knee opened FOLD_KNEE_OPEN, abd and pitch kept:
    # these checks run on the fold with the knee shut again, `q_shut`, so
    # they say FOLD is that fold with ONLY the knee moved.
    q_shut = POSE.FOLD.q.copy()
    q_shut[:, 2] += np.sign(q_shut[:, 2]) * POSE.FOLD_KNEE_OPEN
    z_shut = P.FOOT_RADIUS - SK.all_foot_positions(q_shut)[:, 2].mean()

    def _frames(leg, q=q_shut):
        foot, anchors, *_ = SK.leg_frames(leg, q[leg])
        return foot, anchors[1], anchors[2]      # foot, pitch hinge, knee

    def _x_and_radius(v):
        return [v[0], np.hypot(v[1], v[2])]
    folded = POSE.regularise(POSE.FOLD_CAPTURED_Q)[:2] - P.HIP_TO_PITCH[:2]
    for name, legs, ahead in (("front", (0, 1), POSE.FOLD_FRONT_FORWARD),
                              ("rear", (2, 3), POSE.FOLD_REAR_FORWARD)):
        close("%s feet: the captured front fold from their own pitch hinge, "
              "%.0f mm ahead" % (name, 1e3 * ahead),
              [*_x_and_radius(_frames(legs[0])[0] - _frames(legs[0])[1]),
               *_x_and_radius(_frames(legs[1])[0] - _frames(legs[1])[1])],
              [*_x_and_radius(folded[0] + [ahead, 0.0, 0.0]),
               *_x_and_radius(folded[1] + [ahead, 0.0, 0.0])], 1e-12, " m")
    close("...turned about the abduction axis to a round %.0f deg"
          % np.degrees(POSE.FOLD_ABD),
          np.abs(POSE.FOLD.q[:, 0]), np.full(C.N_LEGS, POSE.FOLD_ABD),
          1e-12, " rad")
    check("...and every knee is still BEHIND its pitch hinge",
          all(_frames(leg)[2][0] < _frames(leg)[1][0] for leg in range(4)),
          "knee x - hinge x %s mm" % np.array2string(
              1e3 * np.array([_frames(leg)[2][0] - _frames(leg)[1][0]
                              for leg in range(4)]), precision=1))
    check("regularising moved the height by under a millimetre",
          abs(z_shut
              - (P.FOOT_RADIUS - SK.all_foot_positions(raw)[:, 2].mean())) < 1e-3,
          "h %.2f mm knee shut, from a capture at %.2f"
          % (1e3 * (z_shut - cfg.TRUNK_BOTTOM_OFFSET),
             1e3 * (P.FOOT_RADIUS - SK.all_foot_positions(raw)[:, 2].mean()
                    - cfg.TRUNK_BOTTOM_OFFSET)))

    # THE FEET ON THE FLOOR, 2026-10-02: the MG5010 knee housing is 31.7 mm
    # in radius about the knee axis (the meshes).  With the knee shut it
    # reached 20 mm below the floor plane and the robot sat on it, feet up.
    knee_r = 0.0317

    def _housing_above_floor(q, z_origin):
        return np.array([z_origin + _frames(leg, q)[2][2] - knee_r
                         for leg in range(C.N_LEGS)])
    shut_gap = _housing_above_floor(q_shut, z_shut)
    open_gap = _housing_above_floor(POSE.FOLD.q, POSE.FOLD.z_origin)
    check("FOLD's knees open %.0f deg, abd and pitch kept, and the knee "
          "motors clear the floor with the feet planted"
          % np.degrees(POSE.FOLD_KNEE_OPEN),
          shut_gap.max() < 0.0 and open_gap.min() > 0.005,
          "housing %.1f mm above the floor, %.1f with the knee shut"
          % (1e3 * open_gap.min(), 1e3 * shut_gap.max()))

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
    check("the rear feet carry MORE of the weight, as the polygon says",
          fz_fold[2:].sum() > fz_fold[:2].sum(),
          "front %.0f%% / rear %.0f%%, CoM sits behind the support centroid"
          % (100 * fz_fold[:2].sum() / fz_fold.sum(),
             100 * fz_fold[2:].sum() / fz_fold.sum()))

    # THE KNEES-OUT FOLD, 2026-10-01 (`hw.fold2_trot`): the front legs the
    # rear legs mirrored fore-aft, every knee motor outboard of its hip, every
    # foot FOLD2_FOOT_OUT outboard of its pitch hinge in the crouch AND at the
    # stand -- there is no slanted rise, so the two are one set of feet.
    from .. import fold2_trot as F2
    f2 = POSE.FOLD2
    close("FOLD2: the front feet are the rear feet MIRRORED, x negated",
          f2.foot_xy[:2], f2.foot_xy[2:] * [-1.0, 1.0], 1e-15, " m")
    close("...and so are the joints: abd kept, pitch and knee negated",
          f2.q[:2], f2.q[2:] * [1.0, -1.0, -1.0], 1e-12, " rad")
    close("...left and right mirror on each axle",
          [*f2.foot_xy[0], *f2.foot_xy[2]],
          [*(f2.foot_xy[1] * [1, -1]), *(f2.foot_xy[3] * [1, -1])], 1e-15,
          " m")
    q_stand2 = LAW.ik_reference(F2.HEIGHT, f2.q, f2.foot_xy)
    outboard, foot_out = [], []
    for q4 in (f2.q, q_stand2):
        for leg in range(C.N_LEGS):
            foot, anchors, *_ = SK.leg_frames(leg, q4[leg])
            out = 1.0 if leg < 2 else -1.0               # +x front, -x rear
            outboard.append(out * (anchors[2][0] - anchors[1][0]))
            foot_out.append(out * (foot[0] - anchors[1][0]))
    check("...every knee OUTBOARD of its pitch hinge, crouch and %.0f mm "
          "stand" % (1e3 * F2.HEIGHT), min(outboard) > 0.02,
          "knee outboard by %.0f..%.0f mm" % (1e3 * min(outboard),
                                             1e3 * max(outboard)))
    close("...every foot FOLD2_FOOT_OUT outboard of it, both", foot_out,
          np.full(2 * C.N_LEGS, POSE.FOLD2_FOOT_OUT), 1e-12, " m")
    close("...abd a round %.0f deg, at FOLD's crouch height with the knee "
          "shut" % np.degrees(POSE.FOLD_ABD),
          [*np.abs(f2.q[:, 0]), f2.z_origin],
          [*np.full(C.N_LEGS, POSE.FOLD_ABD), z_shut], 1e-12)
    check("...inside JOINT_LIMITS, crouch and stand",
          P.within_limits(f2.q) and P.within_limits(q_stand2),
          "knee %.1f / %.1f deg against %.1f"
          % (np.degrees(np.abs(f2.q[:, 2]).max()),
             np.degrees(np.abs(q_stand2[:, 2]).max()),
             np.degrees(P.KNEE_LIM)))
    close("the reference at FOLD2's own height IS the FOLD2 pose",
          LAW.ik_reference(f2.h, f2.q, f2.foot_xy), f2.q, 1e-12, " rad")
    close("the mirror puts the CoM on BOTH trot diagonals at the stand",
          f2.srb_at(F2.HEIGHT).com_body[:2], [0.0, 0.0], 1e-4, " m")
    bad2, tau_fold2, fz_fold2 = _lift(f2)
    check("THE KNEES-OUT FOLD LIFTS: no trip over the whole rise",
          not bad2, bad2[0] if bad2 else
          "rise + allocator clean, %.2f N*m peak" % tau_fold2)
    close("...and front and rear carry the weight EQUALLY",
          fz_fold2[:2].sum() / fz_fold2.sum(), 0.5, 0.005)
    opts = F2.stand_options()
    check("hw.fold2_trot flies FOLD2 at %.0f mm, tracked, NO slanted rise"
          % (1e3 * F2.HEIGHT),
          opts["crouch"] is f2 and opts["rise_track"]
          and opts.get("stand_xy") is None and F2.HEIGHT == 0.160
          and opts["height"] == F2.HEIGHT)
    check("...on hw.fold_trot's clock and apex, hw.trot's cap",
          F2.PERIOD_S == 0.6 and F2.SWING_HEIGHT == 0.020
          and opts["swing_height"] == F2.SWING_HEIGHT
          and opts["gait"].period == F2.PERIOD_S
          and opts["tau_cap"] == F2.TAU_CAP == SAFE.TAU_HARD_NM,
          "%.1f s, %.0f mm, %.1f N*m" % (F2.PERIOD_S, 1e3 * F2.SWING_HEIGHT,
                                         F2.TAU_CAP))

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

    # The first fold run logged 0.67 N*m of steady pitch moment: the
    # rear-tucked capture put c^b 13.8 mm back, and the law was pinned at the
    # NOMINAL c^b.  The parallel fold (2026-09-25) puts it back again, 10.7
    # mm, and the rear feet forward (2026-09-28) 8.5, so the per-posture model
    # is what keeps that moment out of the law.  The front feet forward too
    # make it 6.2 mm.
    dx = float(cfg.COM_BODY[0] - POSE.FOLD.srb.com_body[0])
    phantom = cfg.WEIGHT * abs(dx)
    check("pinning the NOMINAL c^b in the fold stance is a phantom moment",
          POSE.FOLD.srb.com_body[0] < 0.0 and 0.3 < phantom < 1.0,
          "c^b x %+.1f mm: %.1f mm x %.1f N = %.2f N*m"
          % (1e3 * POSE.FOLD.srb.com_body[0], 1e3 * abs(dx), cfg.WEIGHT,
             phantom))

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

    def _roll_push(gains, alloc="qp"):
        """The REAL law in the fold stance, rolled 5.6 deg -> commanded Mx."""
        law = LAW.BalanceLaw(gains=gains, foot_xy=POSE.FOLD.foot_xy,
                             dynamic_setpoint=False, srb=POSE.FOLD.srb,
                             alloc=alloc)
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
    # The residual is NOT zero and is not meant to be: each allocator's own
    # regularisation leaves a floor -- the QP's alpha (the default, 1e-4: the
    # soft cone, ~1 % of the ask) and the least squares' LAMBDA (Tikhonov,
    # ~1e-4 N*m at any gain).  What matters is that it stays at that floor,
    # two orders under the sustained trip, rather than growing past it.
    stiff_wls = _roll_push(rolled, alloc="wls")
    check("...and the allocator delivers it -- no trip; the QP within 2 % of "
          "the ask (alpha), the least squares at the lambda floor",
          stiff.trip is None and stiff_wls.trip is None
          and stiff.allocation.residual_moment
          < 0.02 * abs(stiff.wrench.b_d[3])
          and stiff_wls.allocation.residual_moment < 1e-3,
          "QP %.1e, WLS %.1e N*m (sustained trip %.2f); fz %s N"
          % (stiff.allocation.residual_moment,
             stiff_wls.allocation.residual_moment, cfg.RESIDUAL_MOMENT_NM,
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
    from .. import trot as TROT
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
          lifted.h < cfg.H_LIFT - 0.9 * cfg.SWING_HEIGHT / 4,
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

    # THE JOINT SWING (`--swing joint`): the same arc through the IK, abd
    # held.  The fold trot's until 2026-09-28; checked here at the 2.0 s
    # clock it was sized for, which is no longer `hw.fold_trot`'s.
    joint_period = 2.0
    q_fold = C.unflat(at_fold.q)
    worst_abd = worst_xy = worst_z = worst_qd = 0.0
    for leg in range(C.N_LEGS):
        for s in np.linspace(0.0, 1.0, 41)[:-1]:
            qj, qdj = SWING.joint_swing_reference(
                leg, rest[leg], s, joint_period * (1.0 - cfg.DUTY), q_fold[leg])
            pj, _ = SWING.swing_reference(rest[leg], s,
                                          joint_period * (1.0 - cfg.DUTY))
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
    lat_gait = GAIT.TrotGait(period=joint_period)
    lat_gait.reset(0.0)
    t_lift = next(t for t in np.arange(0.0, joint_period, 0.004)
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
    check("hw.fold_trot: hw.trot's cap, the operator's clock and apex "
          "(2026-10-02)",
          FT.TAU_CAP == TROT.TAU_CAP and FT.PERIOD_S == 0.6
          and FT.SWING_HEIGHT == 0.020,
          "%.1f s, %.0f ms of swing, %.0f mm apex"
          % (FT.PERIOD_S, 1e3 * FT.PERIOD_S * (1.0 - cfg.DUTY),
             1e3 * FT.SWING_HEIGHT))

    # A tracked trot through two settle blocks: the law in each stance, the
    # swing legs where the arc says, the stance legs at the IK, the trunk
    # level.  The residual moment is what separates the two stances.
    def _tracked_trot(pose, gains, period=None,
                      swing_height=cfg.SWING_HEIGHT, **law_kw):
        at_pose = state_at(cfg.H_LIFT, foot_xy=pose.foot_xy, q_seed=pose.q,
                           srb=pose.srb)
        pose_rest = SWING.rest_feet_b(cfg.H_LIFT, pose.foot_xy)
        law = LAW.BalanceLaw(gains=gains, foot_xy=pose.foot_xy, srb=pose.srb,
                             dynamic_setpoint=False, track_stop_deg=25.0,
                             swing_height=swing_height, **law_kw)
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
                    pose_rest[leg], sample.swing_s[leg], clock.swing_duration,
                    swing_height)
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
                                  swing_height=FT.SWING_HEIGHT),
            "fold2": _tracked_trot(POSE.FOLD2, fold_gains,
                                   period=F2.PERIOD_S,
                                   swing_height=F2.SWING_HEIGHT),
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
    # law asks for ~0.04 N*m of trim.  The parallel fold's is the geometry,
    # below.
    check("the nominal's diagonals pass through the CoM",
          runs["nominal"][4] < 0.2,
          "worst residual moment on two feet %.3f N*m" % runs["nominal"][4])
    check("...and the knees-out fold's, by its mirror",
          runs["fold2"][4] < 0.2,
          "worst residual moment on two feet %.3f N*m" % runs["fold2"][4])

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
          two_out.trip is None and two_out.allocation.residual_moment > 0.5,
          "residual %.2f N*m, no trip" % two_out.allocation.residual_moment)
    # THE GEOMETRY IS THE PARALLEL FOLD'S (2026-09-25) WITH THE REAR FEET
    # 20 mm AND THE FRONT 20 mm FORWARD (2026-09-28): front feet 101 mm
    # further out than the rear, and c^b 6.2 mm back, put the CoM d = 20.4
    # mm behind BOTH diagonal
    # support lines -- W d of moment no diagonal pair can make, its pitch
    # part nose-up on both, its roll part changing sign.  Exactly parallel it
    # was 14.8 mm; the rear-tucked capture had 17.2; the rear mirrored as the
    # front, 2026-09-17 to 09-25, had none.
    on = np.flatnonzero(gait.sample(two).contact)
    run = at_fold.x_b[on[1], :2] - at_fold.x_b[on[0], :2]
    arm = fold.srb.com_body[:2] - at_fold.x_b[on[0], :2]
    d = abs(run[0] * arm[1] - run[1] * arm[0]) / np.linalg.norm(run)
    close("...and that residual IS W d, the CoM %.1f mm off the diagonal"
          % (1e3 * d), two_out.allocation.residual_moment, cfg.WEIGHT * d,
          0.05, " N*m")
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
    moving = replace(at_fold, qd=np.full(C.N_JOINTS, 2.0))
    jn = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                        srb=fold.srb, track_stop_deg=0.0,
                        kp_joint=cfg.KP_JOINT_HOLD, kd_joint=0.0)
    jn.arm(0.0, at_fold)
    jn.hold_joints(at_fold.q - dq)
    expect = C.unflat(-cfg.KD_JOINT_HOLD * moving.qd).copy()
    expect[swing_now] = 0.0
    close("...and the damper: -Kd qd on stance legs, NONE on swing legs "
          "(2026-10-01)",
          jl.update(t_two, moving, gait=jg).tau
          - jn.update(t_two, moving, gait=jg).tau, C.flat(expect), 1e-9, " N*m")
    jl.release_joints()
    close("...and released, the law is exactly the one without it",
          jl.update(t_two, at_fold, gait=jg).tau, base, 1e-12, " N*m")
    # THE RISE TRACKED, 2026-10-01: before HOLD the layer's target is the IK
    # reference at the commanded height; after HOLD nothing changes.
    rt = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                        srb=fold.srb, track_stop_deg=0.0,
                        kp_joint=cfg.KP_JOINT_HOLD, kd_joint=cfg.KD_JOINT_HOLD,
                        rise_track=True)
    rt.arm(0.0, at_fold)
    rt.ramp = LAW.REF.Quintic.ramp(at_fold.h, at_fold.h + 0.040, 1.0)
    jl.ramp = rt.ramp                      # the same 40 mm, untracked
    out_rt = rt.update(1.0, at_fold)
    out_jl = jl.update(1.0, at_fold)
    expect = (cfg.KP_JOINT_HOLD * (out_rt.q_ref - at_fold.q)
              - cfg.KD_JOINT_HOLD * at_fold.qd)
    close("rise tracked: before HOLD the joint layer holds the legs on the "
          "IK reference at the commanded height", out_rt.tau - out_jl.tau,
          expect, 1e-9, " N*m")
    check("...which is a real pull: the reference 40 mm up is %.1f deg from "
          "the crouch" % np.degrees(np.abs(out_rt.q_ref - at_fold.q)).max(),
          np.degrees(np.abs(out_rt.q_ref - at_fold.q)).max() > 5.0)
    # ABD'S OWN GAINS, 2026-10-02: once a pose is latched, the abd column
    # flies kp_joint_abd / kd_joint_abd; pitch and knee, and the tracked
    # rise, keep kp_joint / kd_joint.  40 / 1.0 is a test value, not a gain.
    def _layer(**kw):
        law = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                             srb=fold.srb, track_stop_deg=0.0, **kw)
        law.arm(0.0, at_fold)
        return law
    j_off = _layer()
    j_abd = _layer(kp_joint=cfg.KP_JOINT_HOLD, kd_joint=cfg.KD_JOINT_HOLD,
                   kp_joint_abd=40.0, kd_joint_abd=1.0)
    j_off.hold_joints(at_fold.q - dq)
    j_abd.hold_joints(at_fold.q - dq)
    kp_col = np.array([40.0, cfg.KP_JOINT_HOLD, cfg.KP_JOINT_HOLD])
    kd_col = np.array([1.0, cfg.KD_JOINT_HOLD, cfg.KD_JOINT_HOLD])
    expect = -kp_col * C.unflat(dq) - kd_col * C.unflat(moving.qd)
    expect[swing_now] = 0.0
    close("abd's own gains once HOLD latched: abd Kp/Kd its own, pitch and "
          "knee Kp/Kd the layer's, swing legs none",
          j_abd.update(t_two, moving, gait=jg).tau
          - j_off.update(t_two, moving, gait=jg).tau, C.flat(expect), 1e-9,
          " N*m")
    r_abd = _layer(kp_joint=cfg.KP_JOINT_HOLD, kd_joint=cfg.KD_JOINT_HOLD,
                   kp_joint_abd=40.0, kd_joint_abd=1.0, rise_track=True)
    r_abd.ramp = rt.ramp
    close("...and the tracked rise ignores them: the rise is the layer's "
          "Kp/Kd on all three joints", r_abd.update(1.0, at_fold).tau,
          out_rt.tau, 1e-12, " N*m")
    check("...and the defaults are the layer as it was",
          cfg.KP_JOINT_HOLD_ABD == cfg.KP_JOINT_HOLD
          and cfg.KD_JOINT_HOLD_ABD == cfg.KD_JOINT_HOLD,
          "abd %.1f / %.2f" % (cfg.KP_JOINT_HOLD_ABD, cfg.KD_JOINT_HOLD_ABD))
    # THE SWING LEVELED IN ROLL, 2026-10-02 (`swing_roll`): the arc about the
    # trunk origin with the roll from level removed.  The stance latch stays
    # HOLD's -- a per-touchdown re-latch was tried the same day and removed.
    def _sr(on):
        law = _layer(kp_joint=cfg.KP_JOINT_HOLD, kd_joint=cfg.KD_JOINT_HOLD,
                     swing_roll=on)
        law.hold_joints(at_fold.q)
        return law

    def _gait():
        g = GAIT.TrotGait()
        g.reset(0.0)
        return g
    sp_roll = _sr(False).sp_roll
    at_level = state_at(cfg.H_LIFT, R=C.rot_zyx(sp_roll, 0.0, 0.0),
                        foot_xy=fold.foot_xy, q_seed=fold.q, srb=fold.srb)
    close("swing_roll at a LEVEL trunk (the run's setpoint, %+.2f deg) is the "
          "swing as it was" % np.degrees(sp_roll),
          _sr(True).update(t_two, at_level, gait=_gait()).tau,
          _sr(False).update(t_two, at_level, gait=_gait()).tau, 1e-12, " N*m")
    roll3 = np.radians(3.0)
    rolled = state_at(cfg.H_LIFT, R=C.rot_zyx(sp_roll + roll3, 0.0, 0.0),
                      foot_xy=fold.foot_xy, q_seed=fold.q, srb=fold.srb)
    sr, g = _sr(True), _gait()
    out = sr.update(t_two, rolled, gait=g)
    s_two = g.sample(t_two)
    rest = SWING.rest_feet_b(sr.ramp.at(t_two - sr.t0).h, sr.foot_xy)
    arc = np.array([SWING.swing_reference(rest[i], float(s_two.swing_s[i]),
                                          g.swing_duration,
                                          height=sr.swing_height)[0]
                    for i in np.flatnonzero(swing_now)])
    close("...rolled 3 deg from level, the swing target is the level arc "
          "turned by the roll, about the trunk origin",
          (C.rot_x(roll3) @ out.p_swing[swing_now].T).T, arc, 1e-12, " m")
    p_a, v_a, _ = SWING.roll_level(rest[0], np.zeros(3), np.zeros(3), roll3,
                                   0.7)
    p_b, _, _ = SWING.roll_level(rest[0], np.zeros(3), np.zeros(3),
                                 roll3 + 0.7e-6, 0.7)
    close("...and its velocity carries the roll rate (0.7 rad/s against a "
          "finite difference)", v_a, (p_b - p_a) / 1e-6, 1e-6, " m/s")
    tl, gl = _sr(True), _gait()
    latched, base = tl.q_hold, tl.q_hold.copy()
    landings, prev = 0, gl.sample(0.0).contact
    for t_k in np.arange(0.0, 2.0 * gl.period, 0.004):
        now_c = gl.sample(t_k).contact
        landings += int((~prev & now_c).sum())
        prev = now_c
        tl.update(t_k, rolled, gait=gl)
    check("...and the stance latch does NOT follow it: HOLD's, level, the "
          "same array, through %d touchdowns at the rolled trunk" % landings,
          landings >= 4 and tl.q_hold is latched
          and np.array_equal(tl.q_hold, base))
    # THE KNEE SWING, 2026-10-02 (`--swing knee`, swing.py): the shin alone,
    # in the knee frame, toward the CoM -- at FOLD2's hold, for
    # `hw.fold2_trot`.
    f2k = POSE.FOLD2
    q_f2 = LAW.ik_reference(F2.HEIGHT, f2k.q, f2k.foot_xy)
    close("knee bump: 0, 0, 0 at liftoff and touchdown; 1 with zero slope "
          "at the apex", [SWING.knee_bump(s) for s in (0.0, 0.5, 1.0)],
          [[0.0, 0.0, 0.0], [1.0, 0.0, -24.0], [0.0, 0.0, 0.0]], 1e-12)
    amps = np.array([SWING.knee_swing_amplitude(i, q_f2[i], F2.SWING_HEIGHT)
                     for i in range(C.N_LEGS)])
    check("...FOLD2: every knee turns ITS OWN way, the sign of its own angle, "
          "by the same |A|",
          np.array_equal(np.sign(amps), np.sign(q_f2[:, 2]))
          and np.ptp(np.abs(amps)) < 1e-12,
          "%s deg (FL FR RL RR)" % np.array2string(np.degrees(amps),
                                                   precision=2))
    apex_d, rest_err, held = [], [], []
    for i in range(C.N_LEGS):
        f0 = SK.foot_position(i, q_f2[i])
        q_a = SWING.knee_swing_reference(q_f2[i], amps[i], 0.5, 0.24)[0]
        apex_d.append(SK.foot_position(i, q_a) - f0)
        end = SWING.knee_swing_reference(q_f2[i], amps[i], 1.0, 0.24)
        rest_err.append(np.r_[end[0] - q_f2[i], end[1], end[2]])
        held.append(max(np.abs(SWING.knee_swing_reference(
            q_f2[i], amps[i], s, 0.24)[0][:2] - q_f2[i][:2]).max()
            for s in np.linspace(0.0, 1.0, 11)))
    apex_d = np.array(apex_d)
    close("...at the apex every foot is the apex height up, trunk z",
          apex_d[:, 2], np.full(C.N_LEGS, F2.SWING_HEIGHT), 1e-12, " m")
    check("...and has moved TOWARD the CoM, the same distance on all four",
          all(np.sign(apex_d[i, 0]) == -np.sign(SK.foot_position(i, q_f2[i])[0])
              for i in range(C.N_LEGS))
          and np.ptp(np.abs(apex_d[:, 0])) < 1e-12,
          "%.1f mm in" % (1e3 * abs(apex_d[0, 0])))
    close("...MIRRORED: FR is FL in -y, RL is FL in -x, RR is both",
          apex_d[1:], [apex_d[0] * [1, -1, 1], apex_d[0] * [-1, 1, 1],
                       apex_d[0] * [-1, -1, 1]], 1e-12, " m")
    close("...abd and pitch never move; it lands where it lifted, qd and qdd "
          "zero", [*held, *np.ravel(rest_err)], np.zeros(C.N_LEGS * 10), 1e-12)
    q_fs = LAW.ik_reference(FT.HEIGHT, POSE.FOLD.q, FT.STAND_XY)
    refused = []
    for i in range(C.N_LEGS):
        try:
            SWING.knee_swing_amplitude(i, q_fs[i], F2.SWING_HEIGHT)
            refused.append(False)
        except ValueError:
            refused.append(True)
    check("...REFUSED where the shin leans outboard: hw.fold_trot's front "
          "legs, not its rear", refused == [True, True, False, False],
          "refused %s" % refused)
    d_06 = SWING.knee_swing_demand(0, q_f2[0], 0.12, F2.SWING_HEIGHT)
    d_12 = SWING.knee_swing_demand(0, q_f2[0], 0.24, F2.SWING_HEIGHT)
    check("...its torque rate outruns the trot's slew at 120 ms of swing "
          "(0.6 s) and fits at 240 (1.2 s)",
          d_06["slew"] > cfg.TAU_SLEW_TROT_NM_S > d_12["slew"],
          "%.0f / %.0f N*m/s against %.0f" % (d_06["slew"], d_12["slew"],
                                              cfg.TAU_SLEW_TROT_NM_S))
    st_f2 = state_at(F2.HEIGHT, foot_xy=f2k.foot_xy, q_seed=f2k.q,
                     srb=f2k.srb)

    def _knee_law(kp, kd):
        law = LAW.BalanceLaw(foot_xy=f2k.foot_xy, dynamic_setpoint=False,
                             srb=f2k.srb, track_stop_deg=0.0,
                             h_lift=F2.HEIGHT, swing="knee",
                             swing_height=F2.SWING_HEIGHT,
                             kp_swing_knee=np.full(3, kp),
                             kd_swing_knee=np.full(3, kd))
        law.arm(0.0, st_f2)
        return law
    k_on, k_off, g_k = _knee_law(30.0, 0.8), _knee_law(0.0, 0.0), _gait()
    t_k = 0.0
    while True:
        smp = g_k.sample(t_k)
        o_on = k_on.update(t_k, st_f2, gait=g_k)
        o_off = k_off.update(t_k, st_f2, gait=g_k)
        if (smp.swing_s >= 0.5).any() or t_k > g_k.period:
            break
        t_k += 0.004
    legs = np.flatnonzero(smp.swing_s > 0)
    expect = np.zeros(C.N_JOINTS)
    feet = []
    for i in legs:
        q_i = C.unflat(st_f2.q)[i]
        q_r, qd_r, _ = SWING.knee_swing_reference(
            q_i, SWING.knee_swing_amplitude(i, q_i, F2.SWING_HEIGHT),
            float(smp.swing_s[i]), g_k.swing_duration)
        expect[3 * i:3 * i + 3] = 30.0 * (q_r - q_i) + 0.8 * qd_r
        feet.append(HK.foot_position(i, q_r))
    close("the law, --swing knee: the swing legs get the knee PD on the bump, "
          "abd and pitch held at liftoff; nothing else changes",
          o_on.tau - o_off.tau, expect, 1e-9, " N*m")
    close("...its p_swing is the bump's foot (%s, s %.2f)"
          % ("/".join(np.array(C.LEGS)[legs]), float(smp.swing_s[legs[0]])),
          o_on.p_swing[legs], np.array(feet), 1e-12, " m")
    try:
        k_on.begin_step(f2k.foot_xy)
        stepped = True
    except ValueError:
        stepped = False
    check("...and a W foot step is refused: it lands where it lifted",
          not stepped)
    rt.hold_joints(at_fold.q)
    jl.hold_joints(at_fold.q)
    close("...and once HOLD has latched, tracked and untracked are the same law",
          rt.update(1.0, at_fold).tau, jl.update(1.0, at_fold).tau, 1e-12, " N*m")
    jl.release_joints()
    # THE PIN FOLLOWS --height, 2026-10-01 (posture.CrouchPose.srb_at).
    check("srb_at(H_LIFT) is the posture's own pinned model, the same object",
          fold.srb_at(cfg.H_LIFT) is fold.srb
          and POSE.NOMINAL.srb_at(cfg.H_LIFT) is cfg.SRB)
    p115 = np.zeros((C.N_LEGS, 3))
    p115[:, :2] = fold.foot_xy
    p115[:, 2] = -(0.115 + cfg.TRUNK_BOTTOM_OFFSET - P.FOOT_RADIUS)
    com115, _ = SK.body_inertia(HK.all_leg_ik(p115, q_seed=fold.q))
    close("srb_at(115 mm) is the whole-robot CoM at THAT hold pose",
          fold.srb_at(0.115).com_body, com115, 1e-12, " m")
    check("...which for the fold is %.1f mm behind the 145 pin -- the pitch "
          "bias --height used to carry"
          % (1e3 * (fold.srb.com_body[0] - fold.srb_at(0.115).com_body[0])),
          fold.srb.com_body[0] - fold.srb_at(0.115).com_body[0] > 0.003
          and fold.srb_at(0.115) is fold.srb_at(0.115)
          and abs(POSE.NOMINAL.srb_at(0.115).com_body[0]) < 1e-6)
    # THE SLANTED RISE, 2026-10-01 (law.BalanceLaw.stand_xy).
    stand_xy = fold.foot_xy.copy()
    stand_xy[:, 0] = P.HIP_TO_PITCH[:, 0] - 0.035
    try:
        LAW.BalanceLaw(foot_xy=fold.foot_xy, srb=fold.srb, stand_xy=stand_xy,
                       kp_joint=1.0, kd_joint=0.1)
        refused = False
    except ValueError:
        refused = True
    check("stand_xy without rise_track is refused", refused)
    sl = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                        srb=fold.srb_at(0.160, stand_xy), track_stop_deg=0.0,
                        h_lift=0.160, rise_s=3.0, kp_joint=cfg.KP_JOINT_HOLD,
                        kd_joint=cfg.KD_JOINT_HOLD, rise_track=True,
                        stand_xy=stand_xy, slide_from_h=0.110)
    at_crouch = state_at(fold.h, foot_xy=fold.foot_xy, q_seed=fold.q,
                       srb=fold.srb)   # the fold crouch, 60 mm
    sl.arm(0.0, at_crouch)
    sl.update(0.1, at_crouch)
    check("slanted rise: below slide_from_h the sites are still the crouch's",
          np.allclose(sl.foot_xy, fold.foot_xy, atol=1e-12)
          and sl.ramp.at(0.1).h < 0.110)
    # the sweep where the commanded height is halfway between 110 and 160
    t_mid = next(t for t in np.arange(0.0, 3.0, 0.001) if sl.ramp.at(t).h >= 0.135)
    sl.update(t_mid, at_crouch)
    s_mid = (sl.ramp.at(t_mid).h - 0.110) / 0.050
    expect_xy = fold.foot_xy + ST.smoothstep(s_mid) * (stand_xy - fold.foot_xy)
    close("...halfway up the slide the sites are the smoothstep between, in height",
          sl.foot_xy, expect_xy, 1e-9, " m")
    out_arr = sl.update(3.0, at_crouch)
    check("...at the ramp's arrival they are EXACTLY the stand's, and q_ref is "
          "the IK there",
          np.array_equal(sl.foot_xy, stand_xy)
          and np.allclose(out_arr.q_ref, C.flat(LAW.ik_reference(
              0.160, C.unflat(at_crouch.q), stand_xy)), atol=1e-12))
    q_hold_ref = sl.hold_reference(at_crouch.q)
    check("hold_reference is that IK, not the measured pose",
          np.allclose(q_hold_ref, out_arr.q_ref, atol=1e-12)
          and not np.allclose(q_hold_ref, at_crouch.q, atol=1e-3))
    sl.hold_joints(q_hold_ref)
    sl.foot_xy[0, 0] += 0.01                   # a step would move it; the slide must not touch it now
    sl.update(3.5, at_crouch)
    check("...and once a pose is latched the slide leaves foot_xy alone",
          abs(sl.foot_xy[0, 0] - stand_xy[0, 0] - 0.01) < 1e-12)
    sl.foot_xy[0, 0] -= 0.01
    seq_sl = SEQ.StandSequence(SAFE.SafetyGate(3.0, unconfirmed_reason="selftest",
                                              limits=(np.full(C.N_JOINTS, -np.inf),
                                                      np.full(C.N_JOINTS, np.inf))),
                               law="srb", balance=sl, crouch=fold, gait=GAIT.TrotGait())
    check("sequence: with a slanted rise 'home' is the stand's sites, and the feet are home there",
          np.array_equal(seq_sl.home_xy, stand_xy) and seq_sl.feet_home)
    plain = LAW.BalanceLaw(foot_xy=fold.foot_xy, srb=fold.srb, track_stop_deg=0.0)
    seq_pl = SEQ.StandSequence(SAFE.SafetyGate(3.0, unconfirmed_reason="selftest",
                                              limits=(np.full(C.N_JOINTS, -np.inf),
                                                      np.full(C.N_JOINTS, np.inf))),
                               law="srb", balance=plain, crouch=fold)
    check("...and without one it is the crouch's, as before",
          np.array_equal(seq_pl.home_xy, fold.foot_xy) and seq_pl.feet_home)
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

    # PER-LEG SWING GAINS, 2026-10-05, on request: one leg whose swing is
    # weak tuned alone (`--kp-swing-rl` etc.).  (3,) is every leg's; a
    # (4, 3) table is a row per leg, and only that leg's swing feels it.
    kp_rows = np.tile(cfg.KP_SWING, (C.N_LEGS, 1))
    kd_rows = np.tile(cfg.KD_SWING, (C.N_LEGS, 1))
    kp_rows[rl] = [10.0, 10.0, 600.0]
    kd_rows[rl] = [5.0, 5.0, 60.0]
    up10 = rest + np.array([0.0, 0.0, 0.010])
    v_up = np.array([0.0, 0.0, 0.3])
    close("swing gains per leg: a (4, 3) table is each leg's own row in the "
          "impedance",
          [SWING.swing_torque(at_fold, i, up10[i], v_up, kp=kp_rows,
                              kd=kd_rows) for i in range(C.N_LEGS)],
          [SWING.swing_torque(at_fold, i, up10[i], v_up, kp=kp_rows[i],
                              kd=kd_rows[i]) for i in range(C.N_LEGS)],
          0.0, " N*m")

    def _leg_law(kp, kd):
        law = LAW.BalanceLaw(foot_xy=fold.foot_xy, dynamic_setpoint=False,
                             srb=fold.srb, track_stop_deg=0.0,
                             kp_swing=kp, kd_swing=kd)
        law.arm(0.0, at_fold)
        return law
    l_shared = _leg_law(cfg.KP_SWING, cfg.KD_SWING)
    l_rows = _leg_law(kp_rows, kd_rows)
    l_tiled = _leg_law(np.tile(cfg.KP_SWING, (C.N_LEGS, 1)),
                       np.tile(cfg.KD_SWING, (C.N_LEGS, 1)))
    g_leg = _gait()
    rl_up = rl_down = others = tiled = 0.0
    for t_l in np.arange(0.0, 2.0 * g_leg.period, 0.004):
        o_s = l_shared.update(t_l, at_fold, gait=g_leg)
        d = C.unflat(l_rows.update(t_l, at_fold, gait=g_leg).tau - o_s.tau)
        tiled = max(tiled, float(np.abs(
            l_tiled.update(t_l, at_fold, gait=g_leg).tau - o_s.tau).max()))
        if not g_leg.sample(t_l).contact[rl]:       # the law's own test
            rl_up = max(rl_up, float(np.abs(d[rl]).max()))
        else:
            rl_down = max(rl_down, float(np.abs(d[rl]).max()))
        others = max(others, float(np.abs(np.delete(d, rl, axis=0)).max()))
    check("the law with RL's own row: RL's swing torque moves, RL in stance "
          "and the other three legs do not",
          rl_up > 0.05 and rl_down == 0.0 and others == 0.0,
          "RL swinging up to %.2f N*m apart; down %.1e, others %.1e"
          % (rl_up, rl_down, others))
    check("...every row the shared gains is the shared law, bit for bit",
          tiled == 0.0, "%.1e N*m" % tiled)
    try:
        LAW.BalanceLaw(kp_swing=np.zeros((3, C.N_LEGS)))
        refused = False
    except ValueError:
        refused = True
    check("...a table of the wrong shape is refused when the law is built",
          refused)
    import argparse
    import contextlib
    import io
    from .. import stand as STAND
    kp_cli, kd_cli, own_cli = STAND.swing_leg_gains(argparse.Namespace(
        kp_swing=list(cfg.KP_SWING), kd_swing=list(cfg.KD_SWING),
        kp_swing_rl=[10.0, 10.0, 600.0], kd_swing_rr=[5.0, 5.0, 60.0]))
    check("the flags: --kp-swing in every row, --kp-swing-rl in RL's, "
          "--kd-swing-rr in RR's",
          own_cli == ["RL", "RR"]
          and np.array_equal(kp_cli[rl], [10.0, 10.0, 600.0])
          and np.array_equal(np.delete(kp_cli, rl, axis=0),
                             np.tile(cfg.KP_SWING, (C.N_LEGS - 1, 1)))
          and np.array_equal(kd_cli[rr], [5.0, 5.0, 60.0])
          and np.array_equal(np.delete(kd_cli, rr, axis=0),
                             np.tile(cfg.KD_SWING, (C.N_LEGS - 1, 1))),
          "own %s" % own_cli)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            FT.main(["--fake", "--swing", "joint",
                     "--kp-swing-rl", "10", "10", "600"])
        exit_code = None
    except SystemExit as refusal_exit:
        exit_code = refusal_exit.code
    check("hw.fold_trot takes --kp-swing-rl, and refuses it on the joint "
          "swing before the bus", exit_code == 2
          and "the Cartesian swing's" in err.getvalue(), "exit %r" % exit_code)

    # =====================================================================
    print("\n13. the state estimator in the hold's and the trot's x/y "
          "(hw.fully_trot)")
    # =====================================================================
    def _xy_gains():
        g = CTRL.BalanceGains()
        g.kp_pos[:2], g.kd_pos[:2] = cfg.KP_XY_EST, cfg.KD_XY_EST
        return g

    # -- the controller: the x/y rows are the z row's PD, or exactly zero ---
    cmd = REF.com_command(REF.HeightCommand(cfg.H_LIFT, 0.0, 0.0), at_fold.R,
                          fold.srb)
    plain = CTRL.balance_wrench(at_fold, cmd, np.eye(3), CTRL.BalanceGains(),
                                srb=fold.srb)
    no_xy = CTRL.balance_wrench(at_fold, cmd, np.eye(3), _xy_gains(),
                                srb=fold.srb)
    check("x/y gains and NO x/y measurement: the wrench is bit-identical",
          np.array_equal(plain.b_d, no_xy.b_d) and not no_xy.b_d[:2].any())
    e_xy, v_xy = np.array([0.004, -0.002]), np.array([0.010, 0.020])
    fed = CTRL.balance_wrench(at_fold, cmd, np.eye(3), _xy_gains(),
                              srb=fold.srb, xy=CTRL.XyFeedback(e_xy, v_xy))
    close("given one, the x/y rows are m (kp e - kd v), the z row's PD",
          fed.b_d[:2], cfg.MASS * (cfg.KP_XY_EST * e_xy - cfg.KD_XY_EST * v_xy),
          1e-12, " N")
    check("...and no other row of the wrench moves",
          np.array_equal(fed.b_d[2:], no_xy.b_d[2:]))
    far = CTRL.balance_wrench(at_fold, cmd, np.eye(3), _xy_gains(),
                              srb=fold.srb, xy=CTRL.XyFeedback(
                                  np.array([0.3, 0.4]), np.zeros(2)))
    close("the authority clamp: |a_xy| stops at XY_ACC_MAX, direction kept",
          far.acc_lin[:2], cfg.XY_ACC_MAX * np.array([0.6, 0.8]), 1e-12,
          " m/s^2")

    # -- the law: only est_xy, only xy_hold, only a usable estimate ---------
    def _xy_law(est_xy=True, state=at_fold):
        law = LAW.BalanceLaw(gains=_xy_gains(), foot_xy=fold.foot_xy,
                             dynamic_setpoint=False, srb=fold.srb,
                             track_stop_deg=0.0, est_xy=est_xy)
        law.arm(0.0, state)
        law.ramp = REF.Quintic.ramp(cfg.H_LIFT, cfg.H_LIFT, 1.0)
        return law
    tg = GAIT.TrotGait()
    tg.reset(0.0)
    p0 = np.array([0.012, -0.003, 0.2])
    com_xy = fold.srb.com_body[:2]                 # R = I at this fixture

    def _est(t, p=p0, v=np.zeros(3), yaw=0.0, acc_age=0.0):
        return LAW.TrunkEstimate(t=t, p_w=np.asarray(p, float),
                                 v_w=np.asarray(v, float), yaw_offset=yaw,
                                 acc_age_s=acc_age)
    off, bare = _xy_law(est_xy=False), _xy_law(est_xy=False)
    off.feed_estimate(_est(0.996, p=p0 + [0.05, 0.05, 0.0]))
    a_out = off.update(1.0, at_fold, gait=tg, xy_hold=True)
    b_out = bare.update(1.0, at_fold, gait=tg, xy_hold=True)
    check("without est_xy a fed estimate moves NO torque -- every other trot",
          np.array_equal(a_out.tau, b_out.tau) and not a_out.wrench.b_d[:2].any()
          and not a_out.xy_on)
    check("...and is still on the output, for the log",
          a_out.estimate is not None and abs(a_out.est_age_s - 0.004) < 1e-12)
    hold_law = _xy_law()
    hold_law.feed_estimate(_est(0.996, p=p0 + [0.05, 0.05, 0.0]))
    h_out = hold_law.update(1.0, at_fold)
    check("with est_xy, a sweep without xy_hold (the rise, the W step) never "
          "reads it: rows zero, no latch",
          not h_out.wrench.b_d[:2].any() and not h_out.xy_on
          and hold_law.xy_des is None)

    trot_law = _xy_law()
    trot_law.feed_estimate(_est(0.996))
    first = trot_law.update(1.0, at_fold, gait=tg, xy_hold=True)
    close("the trot's first sweep LATCHES the CoM x/y the estimate gives",
          trot_law.xy_des, p0[:2] + com_xy, 1e-15, " m")
    check("...so closing the loop asks for no force on that sweep",
          first.xy_on and not first.wrench.b_d[:2].any())
    trot_law.feed_estimate(_est(1.0, p=p0 + [0.010, 0.0, 0.0]))
    ahead = trot_law.update(1.004, at_fold, gait=tg, xy_hold=True)
    close("the trunk 10 mm AHEAD of the latch is pushed back: -m kp 0.010",
          ahead.wrench.b_d[:2], [-cfg.MASS * cfg.KP_XY_EST * 0.010, 0.0],
          1e-9, " N")
    trot_law.feed_estimate(_est(1.004, v=[0.0, 0.05, 0.0]))
    moving = trot_law.update(1.008, at_fold, gait=tg, xy_hold=True)
    close("a 50 mm/s drift to the left is damped: -m kd 0.05 in y",
          moving.wrench.b_d[:2], [0.0, -cfg.MASS * cfg.KD_XY_EST * 0.05],
          1e-9, " N")

    refusals = (("an estimate 30 ms old", _est(0.982), "old"),
                ("one from the FUTURE", _est(1.020), "old"),
                ("one in another world frame", _est(1.008, yaw=0.3), "world"),
                ("one made on a 120 ms old 0x40", _est(1.008, acc_age=0.12),
                 "accelerometer"),
                ("a non-finite one", _est(1.008, p=[np.nan, 0.0, 0.2]),
                 "finite"))
    for label, est, word in refusals:
        before = trot_law.xy_refused
        trot_law.feed_estimate(est)
        r_out = trot_law.update(1.012, at_fold, gait=tg, xy_hold=True)
        check("REFUSED: %s -- x/y rows zero, the flown law, counted" % label,
              not r_out.xy_on and not r_out.wrench.b_d[:2].any()
              and r_out.trip is None and trot_law.xy_refused == before + 1
              and word in trot_law.xy_refusal, trot_law.xy_refusal)
    stale_imu = state_at(cfg.H_LIFT, age_s=2.0 * cfg.IMU_MAX_AGE_S,
                         foot_xy=fold.foot_xy, q_seed=fold.q, srb=fold.srb)
    trot_law.feed_estimate(_est(1.008))
    s_out = trot_law.update(1.012, stale_imu, gait=tg, xy_hold=True)
    check("REFUSED: a fresh estimate on a STALE attitude -- rotated by that R",
          not s_out.xy_on and s_out.imu_held and not s_out.wrench.b_d[:2].any()
          and "attitude" in trot_law.xy_refusal, trot_law.xy_refusal)
    check("a feed of None keeps the last estimate rather than clearing it",
          (trot_law.feed_estimate(None) or True)
          and trot_law.estimate is not None)

    # THE CoM, NOT THE ORIGIN: a trunk turning in place moves its CoM by
    # omega x R c^b, and that is the velocity the damper must see.
    tilt_R = C.rot_x(np.deg2rad(4.0))
    w_b = np.array([0.0, 0.6, 0.0])                 # pitching at 0.6 rad/s
    at_tilt = state_at(cfg.H_LIFT, R=tilt_R, omega_b=w_b,
                       foot_xy=fold.foot_xy, q_seed=fold.q, srb=fold.srb)
    tilt_law = _xy_law(state=at_tilt)
    tilt_law.feed_estimate(_est(0.996))
    tilt_out = tilt_law.update(1.0, at_tilt, gait=tg, xy_hold=True)
    com_w = tilt_R @ fold.srb.com_body
    v_com = np.cross(at_tilt.omega_w, com_w)[:2]
    close("latched at a tilt on the CoM: p_w + R c^b",
          tilt_law.xy_des, p0[:2] + com_w[:2], 1e-15, " m")
    close("...and a still ORIGIN under a turning trunk is a moving CoM, damped",
          tilt_out.wrench.b_d[:2],
          -cfg.MASS * cfg.KD_XY_EST * v_com, 1e-12, " N")

    # -- the sequence: a fresh latch per HOLD and per trot -----------------
    xseq = SEQ.StandSequence(SAFE.SafetyGate(3.0), crouch=fold,
                             balance=LAW.BalanceLaw(
                                 gains=_xy_gains(), foot_xy=fold.foot_xy,
                                 dynamic_setpoint=False, srb=fold.srb,
                                 track_stop_deg=0.0, est_xy=True),
                             gait=GAIT.TrotGait(),
                             step_to=TR.STEP_FOOT_XY,
                             step_gait=GAIT.TrotGait(period=TR.STEP_PERIOD_S))
    xseq.phase = SEQ.PHASES.index("hold")
    xseq.body = at_fold
    xseq.gate.start(0.0, q=at_fold.q)
    xseq.balance.arm(0.0, at_fold)
    xseq.feed_estimate(_est(0.996, p=p0 + [0.05, 0.0, 0.0]))
    xseq.update(1.0, at_fold)
    check("sequence: HOLD latches the x/y where the filter has it and drives",
          xseq.out.xy_on and xseq.balance.xy_des is not None
          and not xseq.out.wrench.b_d[:2].any())
    close("...the CoM's x/y on its first usable sweep: p_w + R c^b",
          xseq.balance.xy_des, p0[:2] + [0.05, 0.0] + com_xy, 1e-15, " m")
    xseq.feed_estimate(_est(1.0, p=p0 + [0.06, 0.0, 0.0]))
    xseq.update(1.004, at_fold)
    check("...and pushes back as the filter's x moves off it",
          xseq.out.xy_on and xseq.out.wrench.b_d[0] < 0.0)
    xseq.balance.xy_des = np.array([9.0, 9.0])         # a stale latch
    xseq.toggle_trot(1.0)
    check("...T releases any old latch before the trot's first sweep",
          xseq.balance.xy_des is None)
    t = 1.004
    xseq.feed_estimate(_est(t))
    xseq.update(t + 0.004, at_fold)
    check("...and the trot's first sweep latches a fresh one and drives x/y",
          xseq.out.xy_on and xseq.balance.xy_des is not None
          and not xseq.out.wrench.b_d[:2].any())
    xseq.toggle_trot(t + 0.004)
    t += 0.004
    xseq.balance.xy_des = np.array([9.0, 9.0])         # the trot's, marked
    while xseq.trotting and t < 4.0:
        xseq.feed_estimate(_est(t))
        t += 0.004
        xseq.update(t, at_fold)
    check("...and the exit releases it: HOLD again, latched afresh where the "
          "trot ended, not the trot's",
          xseq.phase_name == "hold" and xseq.out.xy_on
          and np.allclose(xseq.balance.xy_des, p0[:2] + com_xy, 0.0, 1e-15)
          and not xseq.out.wrench.b_d[:2].any(), "%.2f s" % (t - 1.0))
    xseq.toggle_step(t)
    xseq.feed_estimate(_est(t))
    xseq.update(t + 0.004, at_fold)
    check("W from HOLD releases the hold's latch, and the step never reads it",
          xseq.phase_name == "step" and xseq.balance.xy_des is None
          and not xseq.out.xy_on)
    report = xseq.balance.report()
    check("the exit report counts what the loop drove and refused",
          "x/y on the estimator" in report and "refused" in report)

    # =====================================================================
    print("\n14. walking (hw.fold_walk): keys, reference, QP, footstep, "
          "swing, one reference")
    # =====================================================================
    from . import footstep as WFOOT
    from . import keys as WKEYS
    from . import qp as WQP
    from . import swing_control as WSC
    from . import trajectory as WTRAJ
    from . import walk as WWALK
    from .. import fold_walk as FW

    # -- keys.py: sim.cmpc.run's mapping ------------------------------------
    keys = WKEYS.WalkKeys()
    one = {}
    for ch in "wsadqe":
        keys.stop()
        keys.press(ch)
        one[ch] = (keys.command.vx, keys.command.vy, keys.command.yaw_rate)
    v1, y1 = cfg.WALK_V_STEP, cfg.WALK_YAW_STEP
    close("keys: W/S x, A/D y, Q/E yaw -- one step each, nothing else moves",
          [one[ch] for ch in "wsadqe"],
          [(v1, 0, 0), (-v1, 0, 0), (0, v1, 0), (0, -v1, 0), (0, 0, y1),
           (0, 0, -y1)], 1e-15)
    keys.stop()
    for _ in range(50):
        keys.press("W"), keys.press("a"), keys.press("E")
    close("...upper case too, accumulating into the hardware box and no "
          "further", [keys.command.vx, keys.command.vy, keys.command.yaw_rate],
          [cfg.WALK_VX_MAX, cfg.WALK_VY_MAX, -cfg.WALK_YAW_RATE_MAX], 1e-12)
    wide = WKEYS.WalkKeys(v_step=0.1, vx_max=1.0)   # 0.1 is not a binary
    for ch in "wwwsss":                               # fraction: 0.1 x 3 - 0.1
        wide.press(ch)                                # x 3 is 2.8e-17 in floats
    check("...three W and three S are exactly zero, not 1e-17",
          wide.command.vx == 0.0)
    keys.press("q")
    keys.press(" ")
    check("...SPACE stops all three",
          keys.command.vx == keys.command.vy == keys.command.yaw_rate == 0.0)
    check("...T, X, ENTER stay hw.stand's: not owned, and press says None",
          not any(keys.owns(ch) for ch in ("t", "T", "x", "X", "\n", "\r",
                                           None))
          and keys.press("t") is None)

    # -- trajectory.py: cMPC's generator, slewed, clipped, leashed ----------
    zero = WTRAJ.Command(vx=0.0, vy=0.0, yaw_rate=0.0)
    wref = WTRAJ.WalkReference()
    p_a, yaw_a = np.array([0.3, -0.2]), np.radians(90.0)
    wref.reset(p_a, yaw_a)
    s0 = wref.update(0.0, zero)
    check("reference: reset ON the robot -- the first sweep has no error",
          np.array_equal(s0.p, p_a) and s0.yaw == yaw_a and not s0.v.any()
          and not s0.a.any())
    # Inside the keys' box, whatever it is (60 % of doc/walk's since
    # 2026-10-05): this is the generator's arithmetic, not the box.
    fwd = WTRAJ.Command(vx=cfg.WALK_VX_MAX, vy=0.0, yaw_rate=0.0)
    run = [wref.update(0.004 * k, fwd) for k in range(1, 151)]
    k_half = int(round(0.5 * fwd.vx / cfg.WALK_ACC_MAX / 0.004))
    close("...a key's step is a ramp at WALK_ACC_MAX, and its rate is fed "
          "forward as a", [run[k_half - 1].command.vx,
                           float(np.linalg.norm(run[k_half - 1].a))],
          [cfg.WALK_ACC_MAX * 0.004 * k_half, cfg.WALK_ACC_MAX], 1e-12)
    close("...turned by the REFERENCE heading: body x at 90 deg is world +y",
          run[-1].v, [0.0, fwd.vx], 1e-12, " m/s")
    dist = 0.004 * sum(r.command.vx for r in run)
    close("...and integrated: the position is the slewed velocity's sum",
          run[-1].p - p_a, [0.0, dist], 1e-12, " m")
    wref.reset(np.zeros(2), 0.0)
    turn = WTRAJ.Command(vx=0.10, vy=0.0, yaw_rate=np.radians(20.0))
    s_t = [wref.update(0.004 * k, turn) for k in range(0, 501)][-1]
    close("...turning at 20 deg/s, a = r x v: the velocity carried round "
          "with the heading", s_t.a,
          s_t.yaw_rate * np.array([-s_t.v[1], s_t.v[0]]), 1e-12, " m/s^2")
    wref.reset(np.zeros(2), 0.0)
    fast = WTRAJ.Command(vx=0.50, vy=0.0, yaw_rate=0.0)
    for k in range(0, 751):
        s_l = wref.update(0.004 * k, fast, np.zeros(2), 0.0)
    check("...0.5 m/s asked: clipped to the hardware box before cMPC sees it",
          s_l.command.vx == cfg.WALK_VX_MAX, "%.2f m/s" % s_l.command.vx)
    close("...and a robot that does not follow is not chased: the leash "
          "holds the reference WALK_LEASH ahead", np.linalg.norm(s_l.p),
          cfg.WALK_LEASH, 1e-12, " m")
    check("...and says so, and counts", s_l.leashed and wref.leash_sweeps > 0)
    wref.reset(np.zeros(2), 2.0 * np.pi + 0.5)
    s_y = wref.update(0.0, zero, None, 0.5 + np.radians(30.0))
    close("...the yaw leash compares on the CIRCLE and keeps the reference's "
          "own turn", s_y.yaw,
          2.0 * np.pi + 0.5 + np.radians(30.0) - cfg.WALK_YAW_LEASH, 1e-12,
          " rad")
    wref.reset(np.zeros(2), 0.0)
    wref.update(0.0, zero)
    wref.command = fwd                              # past the slew
    s_late = wref.update(1.0, fwd)                  # a sweep 1 s late
    close("...a sweep 1 s late integrates DT_MAX and no more", s_late.p[0],
          fwd.vx * WTRAJ.DT_MAX, 1e-15, " m")

    # -- qp.py: the cone inside the problem ---------------------------------
    plain = LAW.BalanceLaw(foot_xy=fold.foot_xy, srb=fold.srb)
    check("QP is the law's DEFAULT allocator (2026-10-04, on request): the "
          "cone inside the problem; alloc='wls' keeps the least squares",
          plain.alloc == "qp" and isinstance(plain.qp, WQP.QpAllocator)
          and LAW.BalanceLaw(foot_xy=fold.foot_xy, srb=fold.srb,
                             alloc="wls").qp is None)
    qpa = WQP.QpAllocator()
    trim = np.array([0.0, 0.0, cfg.WEIGHT, 0.3, -0.2, 0.05])
    a_q = qpa.allocate(at_fold.r_w, trim)
    a_l = ALLOC.allocate(at_fold.r_w, trim, mu=cfg.MU)
    close("QP: with every face slack it IS the least squares, to alpha's "
          "regularisation (0.02 N of 57.7)", a_q.f_w, a_l.f_w, 2e-2, " N")
    check("...found on the fast path, one 12x12 solve, no iteration",
          qpa.last_iter == 0 and not a_q.clipped.any())
    a_d = qpa.allocate(at_fold.r_w, trim,
                       contact=np.array([1.0, 0.0, 0.0, 1.0]))
    check("...a swinging foot is not in the problem: exactly zero force",
          not a_d.f_w[[1, 2]].any())
    a_r = qpa.allocate(at_fold.r_w, trim,
                       contact=np.array([1.0, 0.5 * WQP.W_PLANTED_MIN, 1.0,
                                         1.0]))
    check("...nor one ramped below W_PLANTED_MIN, the box that collapses "
          "onto fz = 0", not a_r.f_w[1].any())
    push = np.array([0.0, 27.0, cfg.WEIGHT, 0.0, 0.0, 0.0])
    a_q = qpa.allocate(at_fold.r_w, push)
    a_l = ALLOC.allocate(at_fold.r_w, push, mu=cfg.MU)
    f = a_q.f_w
    check("...a 27 N push puts feet on the cone: every foot on or inside "
          "its pyramid and box",
          qpa.last_iter > 0
          and bool(np.all(np.abs(f[:, :2]) <= cfg.MU * f[:, 2:3] + 1e-9))
          and bool(np.all(f[:, 2] >= cfg.FZ_MIN - 1e-9))
          and bool(np.all(f[:, 2] <= cfg.FZ_MAX + 1e-9)),
          "%d iterations" % qpa.last_iter)
    check("...and delivers more of the wrench than the least squares' clip",
          np.linalg.norm(a_q.residual) < np.linalg.norm(a_l.residual),
          "|residual| %.3f against %.3f" % (np.linalg.norm(a_q.residual),
                                            np.linalg.norm(a_l.residual)))
    # A CERTIFICATE, NOT A COMPARISON: x feasible, and the gradient a
    # non-negative combination of the working rows -- KKT, which for a
    # convex QP is optimality.
    stat_w = lam_w = viol_w = 0.0
    qpa = WQP.QpAllocator(max_iter=60)       # the solver, not the sweep's cap
    capped0 = qpa.capped
    hardest = None
    for k in range(400):
        r_k = at_fold.r_w + rng.normal(0.0, 0.01, (4, 3))
        b_k = (np.array([0.0, 0.0, cfg.WEIGHT, 0.0, 0.0, 0.0])
               + rng.normal(0.0, 1.0, 6) * [12.0, 12.0, 10.0, 1.5, 1.5, 0.5])
        c_k = (None, np.array([1.0, 0.0, 0.0, 1.0]),
               np.array([0.0, 1.0, 1.0, 0.0]), rng.random(4))[k % 4]
        out_k = qpa.allocate(r_k, b_k, contact=c_k)
        w_k = np.ones(4) if c_k is None else c_k
        legs = np.flatnonzero(w_k >= WQP.W_PLANTED_MIN)
        H, g, _A, Cm, dvec, _lo, _hi = qpa._build(r_k, b_k, legs, w_k)
        x = out_k.f_w[legs].reshape(-1)
        slot = {int(leg): j for j, leg in enumerate(legs)}
        work = [WQP.ROWS_PER_FOOT * slot[leg] + row
                for leg, row in sorted(qpa.active)]
        grad = H @ x + g
        if work:
            lam = np.linalg.lstsq(Cm[work].T, -grad, rcond=None)[0]
            grad = grad + Cm[work].T @ lam
            lam_w = min(lam_w, float(lam.min()))
        stat_w = max(stat_w, float(np.abs(grad).max()))
        viol_w = max(viol_w, float((Cm @ x - dvec).max()))
        if hardest is None or qpa.last_iter > hardest[0]:
            hardest = (qpa.last_iter, r_k, b_k, c_k)
    check("...KKT on 400 random wrenches, stances and ramps: stationary, "
          "multipliers >= 0, feasible, converged",
          stat_w < 1e-6 and lam_w > -1e-6 and viol_w <= 1e-9
          and qpa.capped == capped0,
          "stationarity %.1e, min lambda %.1e, violation %.1e, worst %d it"
          % (stat_w, lam_w, viol_w, qpa.iter_max_seen))
    # THE CAP (config.QP_MAX_ITER) IS THE SWEEP'S WORST-CASE TIME: stopped
    # early on the hardest of those, the force is still inside the cone and
    # no worse than where the active set started -- every step descends.
    _, r_h, b_h, c_h = hardest
    w_h = np.ones(4) if c_h is None else c_h
    legs_h = np.flatnonzero(w_h >= WQP.W_PLANTED_MIN)
    H, g, _A, Cm, dvec, lo_h, hi_h = qpa._build(r_h, b_h, legs_h, w_h)
    start = WQP.project_feasible(np.linalg.solve(H, -g).reshape(-1, 3), lo_h,
                                 hi_h, cfg.MU).reshape(-1)
    short = WQP.QpAllocator(max_iter=2)
    x_s = short.allocate(r_h, b_h, contact=c_h).f_w[legs_h].reshape(-1)

    def _obj(x):
        return 0.5 * x @ H @ x + g @ x
    check("...and a solve stopped at its cap is still inside the cone, and "
          "no worse than its projected start",
          not short.last_converged and short.capped == 1
          and float((Cm @ x_s - dvec).max()) <= 1e-9
          and _obj(x_s) <= _obj(start) + 1e-12,
          "the %d-iteration case stopped at 2: cost %.4f against %.4f at the "
          "start, cap in the robot's sweeps %d" % (hardest[0], _obj(x_s),
                                                   _obj(start),
                                                   cfg.QP_MAX_ITER))

    # -- footstep.py: eq (33), the arc in the world -------------------------
    sites = SWING.rest_feet_b(cfg.H_LIFT, fold.foot_xy)[:, :2]
    t_st, t_sw = 0.42, 0.18                          # duty 0.70 at 0.6 s
    fp = WFOOT.FootstepPlanner(sites, t_stance=t_st, t_swing=t_sw,
                               height=0.02, kv=cfg.STEP_KV,
                               step_max=cfg.STEP_MAX_XY)
    v_w = np.array([0.10, 0.0])
    tr = WFOOT.TrunkXY(p=np.array([1.0, 2.0]), v=v_w, yaw=0.0, yaw_rate=0.0,
                       measured=True)
    rs = WTRAJ.RefSample(p=np.zeros(2), v=v_w, a=np.zeros(2), yaw=0.0,
                         yaw_rate=0.0, command=zero, leashed=False)
    land = fp.foothold(0, 0.25, tr, rs)
    close("footstep: eq (33) -- the site where the trunk will be at "
          "touchdown, plus v T_st / 2", land,
          tr.p + v_w * 0.75 * t_sw + sites[0] + 0.5 * t_st * v_w, 1e-15, " m")
    quick = tr._replace(v=np.array([0.15, 0.0]))
    close("...a trunk 0.05 m/s FASTER than the reference steps kv x 0.05 "
          "further, Raibert's term",
          fp.foothold(0, 0.25, quick, rs) - land,
          [(0.5 * t_st + cfg.STEP_KV) * 0.05, 0.0], 1e-15, " m")
    n_cl = fp.clamped
    wild = fp.foothold(0, 0.25, tr._replace(v=np.array([2.0, -2.0])),
                       rs._replace(v=np.zeros(2)))
    close("...a wild velocity is clamped to STEP_MAX_XY about the site",
          np.abs(wild - tr.p - sites[0]), cfg.STEP_MAX_XY, 1e-12, " m")
    check("...and counted", fp.clamped == n_cl + 1)
    r_t = 0.5
    spun = fp.foothold(2, 0.25, tr._replace(v=np.zeros(2), yaw=0.3,
                                            yaw_rate=r_t),
                       rs._replace(v=np.zeros(2), yaw_rate=r_t))
    ang = 0.3 + r_t * 0.75 * t_sw + 0.5 * r_t * t_st
    close("...turning in place: the site turned to mid-stance (pYawCorrected)",
          spun, tr.p + C.rot_z(ang)[:2, :2] @ sites[2], 1e-15, " m")
    fp.release(0)
    x_lift = np.array([sites[0, 0] - 0.02, sites[0, 1] + 0.004, -0.165])
    a0 = fp.plan(0, 0.0, tr, rs, x_lift, -0.165)
    close("...the arc starts where the foot IS, latched at liftoff",
          a0.p, x_lift, 1e-15, " m")
    close("...at GROUND speed: zero in the world is -v in the trunk",
          a0.v[:2], -v_w, 1e-15, " m/s")
    tr_end = tr._replace(p=tr.p + v_w * t_sw)
    a1 = fp.plan(0, 1.0, tr_end, rs, x_lift, -0.165)
    close("...and lands on the foothold, at ground speed, at rest height",
          np.r_[tr_end.p + a1.p[:2], a1.v[:2], a1.p[2]],
          np.r_[a1.land_w, -v_w, -0.165], 1e-12)
    close("...the apex is swing_height above rest at mid-swing",
          fp.plan(0, 0.5, tr._replace(p=tr.p + 0.5 * v_w * t_sw), rs,
                  x_lift, -0.165).p[2], -0.165 + 0.02, 1e-15, " m")
    # The rates against finite differences, along two paths on which the
    # foothold stands still: a straight walk at the reference speed, and a
    # turn in place.  -omega x p_b is in v_b, Coriolis and centripetal in a_b.
    fd_v = fd_a = 0.0
    for v_k, r_k in ((np.array([0.12, -0.04]), 0.0),
                     (np.zeros(2), 0.6)):
        fpk = WFOOT.FootstepPlanner(sites, t_stance=t_st, t_swing=t_sw,
                                    height=0.02, kv=cfg.STEP_KV,
                                    step_max=cfg.STEP_MAX_XY)
        p0k, y0k = np.array([0.4, -0.1]), 0.2
        rsk = WTRAJ.RefSample(p=np.zeros(2), v=v_k, a=np.zeros(2), yaw=0.0,
                              yaw_rate=r_k, command=zero, leashed=False)

        def _at(t, fpk=fpk, v_k=v_k, r_k=r_k, p0k=p0k, y0k=y0k, rsk=rsk):
            trk = WFOOT.TrunkXY(p=p0k + v_k * t, v=v_k, yaw=y0k + r_k * t,
                                yaw_rate=r_k, measured=True)
            return fpk.plan(1, 0.1 + t / t_sw, trk, rsk, x_lift, -0.165)
        _at(-0.1 * t_sw)                              # latch the liftoff
        for t in (0.02, 0.05, 0.09, 0.13):
            h = 1e-4
            mid, lo, hi = _at(t), _at(t - h), _at(t + h)
            fd_v = max(fd_v, float(np.abs((hi.p - lo.p) / (2 * h)
                                          - mid.v).max()))
            fd_a = max(fd_a, float(np.abs((hi.p - 2 * mid.p + lo.p) / h ** 2
                                          - mid.a).max()))
    check("...v and a are the derivatives of p in the TRUNK frame, walking "
          "and turning (finite differences)", fd_v < 1e-5 and fd_a < 1e-3,
          "worst %.1e m/s, %.1e m/s^2" % (fd_v, fd_a))
    from sim.cmpc.swing import SwingTrajectory
    arc_w = 0.0
    for s in np.linspace(0.0, 1.0, 23):
        mine = fp.plan(0, float(s), tr, rs, x_lift, -0.165)
        lift = np.r_[mine.lift_w, 0.0]
        ref_xy = SwingTrajectory(lift, np.r_[mine.land_w, 0.0], height=0.0,
                                 duration=t_sw).at(float(s))
        ref_z = SwingTrajectory(np.r_[0.0, 0.0, -0.165],
                                np.r_[0.0, 0.0, -0.165], height=0.02,
                                duration=t_sw).at(float(s))
        arc_w = max(arc_w, float(np.abs(tr.p + mine.p[:2]
                                        - ref_xy[0][:2]).max()),
                    abs(mine.p[2] - ref_z[0][2]), abs(mine.a[2] - ref_z[2][2]))
    check("...and its scalar arcs ARE sim.cmpc.swing.SwingTrajectory's",
          arc_w < 1e-12, "worst %.1e" % arc_w)
    # AT REST (v_ref 0, r 0) the arc is hw.fold_trot's in-place target, in
    # the TRUNK frame from the foot to the site; the filter is not in it
    # (footstep.py, 2026-10-05).
    rest_rs = rs._replace(v=np.zeros(2), yaw_rate=0.0)
    still = WFOOT.TrunkXY(p=np.zeros(2), v=np.zeros(2), yaw=0.0,
                          yaw_rate=0.0, measured=True)
    noisy = WFOOT.TrunkXY(p=np.array([0.03, -0.01]),
                          v=np.array([0.30, -0.20]), yaw=0.2, yaw_rate=0.5,
                          measured=True)

    def _fp(**kw):
        return WFOOT.FootstepPlanner(sites, t_stance=t_st, t_swing=t_sw,
                                     height=0.02, kv=cfg.STEP_KV,
                                     step_max=cfg.STEP_MAX_XY, **kw)
    fa, fb = _fp(), _fp()
    dev = 0.0
    for s in np.linspace(0.0, 1.0, 21):
        ra = fa.plan(0, float(s), still, rest_rs, x_lift, -0.165)
        rb = fb.plan(0, float(s), noisy, rest_rs, x_lift, -0.165)
        dev = max(dev, float(np.abs(np.r_[ra.p - rb.p, ra.v - rb.v,
                                          ra.a - rb.a]).max()))
    check("footstep AT REST (v_ref 0, r 0): the swing is the same whatever "
          "the filter says -- 32 mm off, 0.36 m/s, 11 deg turned, turning",
          dev == 0.0 and bool(fa.rest_swing[0] and fb.rest_swing[0]),
          "worst %.1e" % dev)
    r0 = fa.plan(0, 0.0, still, rest_rs, x_lift, -0.165)
    r1 = fa.plan(0, 1.0, still, rest_rs, x_lift, -0.165)
    close("...it starts on the foot, lands on the site at rest height, and "
          "is at zero trunk-frame speed at both ends",
          np.r_[r0.p, r1.p, r0.v[:2], r1.v[:2]],
          np.r_[x_lift, sites[0], -0.165, np.zeros(4)], 1e-15)
    fd_rv = fd_ra = 0.0
    for s in (0.2, 0.4, 0.7, 0.9):
        hh = 1e-4
        lo, mid, hi = (fa.plan(0, s + d / t_sw, still, rest_rs, x_lift,
                               -0.165) for d in (-hh, 0.0, hh))
        fd_rv = max(fd_rv, float(np.abs((hi.p - lo.p) / (2 * hh)
                                        - mid.v).max()))
        fd_ra = max(fd_ra, float(np.abs((hi.p - 2 * mid.p + lo.p) / hh ** 2
                                        - mid.a).max()))
    check("...v and a are its derivatives (finite differences)",
          fd_rv < 1e-5 and fd_ra < 1e-3,
          "worst %.1e m/s, %.1e m/s^2" % (fd_rv, fd_ra))
    fc = _fp()
    fc.plan(0, 0.0, still, rest_rs, x_lift, -0.165)
    late = fc.plan(0, 1.0, tr, rs, x_lift, -0.165)
    fc.release(0)
    fc.plan(0, 0.0, tr, rs, x_lift, -0.165)
    check("...decided at liftoff: a command that arrives mid-swing does not "
          "move its landing off the site, and the next swing is placed",
          bool(np.allclose(late.p[:2], sites[0], 0.0, 1e-15))
          and not fc.rest_swing[0]
          and (fc.swings_at_rest, fc.swings_placed) == (1, 1),
          "lands %s mm off the site" % np.round(1e3 * (late.p[:2]
                                                     - sites[0]), 3))
    fo = _fp(place_at_rest=True)
    slow = noisy._replace(v=np.array([0.10, -0.05]))
    fo.plan(0, 0.0, slow, rest_rs, x_lift, -0.165)
    old = fo.plan(0, 1.0, slow, rest_rs, x_lift, -0.165)
    close("...place_at_rest: the placement at rest as flown before, the site "
          "plus (T_st / 2 + STEP_KV) v_hat",
          old.land_w, slow.p + C.rot_z(0.2)[:2, :2] @ sites[0]
          + (0.5 * t_st + cfg.STEP_KV) * slow.v, 1e-15, " m")

    # -- swing_control.py ---------------------------------------------------
    sw_state = replace(at_fold, qd=np.full(C.N_JOINTS, 1.5))
    qd0 = C.unflat(sw_state.qd)[0]
    v0 = sw_state.jac[0] @ qd0
    on_arc = WFOOT.SwingRef(p=sw_state.x_b[0].copy(), v=v0,
                            a=np.array([0.5, -0.3, 4.0]), land_w=None,
                            lift_w=None)
    tau_o, ff_o = WSC.swing_osc_torque(sw_state, 0, on_arc)
    close("swing (osc): ON the arc, the torque is the feedforward alone",
          tau_o, ff_o, 1e-12, " N*m")
    e_p, e_v = np.array([0.01, -0.005, 0.02]), np.array([0.1, 0.0, -0.2])
    off = on_arc._replace(p=on_arc.p + e_p, v=on_arc.v + e_v)
    tau_e, ff_e = WSC.swing_osc_torque(sw_state, 0, off)
    wn, zt = cfg.WN_SWING_OSC, cfg.ZETA_SWING_OSC
    close("...off it, J M0^-1 (tau - tau_ff) is the commanded acceleration "
          "wn^2 e + 2 zeta wn e_dot",
          sw_state.jac[0] @ np.linalg.solve(SWING.JOINT_INERTIA_M0,
                                            tau_e - ff_e),
          wn * wn * e_p + 2.0 * zt * wn * e_v, 1e-9, " m/s^2")
    tau_i, ff_i = WSC.swing_control_torque(sw_state, 0, off)
    close("swing (impedance): J^T (Kp e + Kd e_dot), no feedforward unless "
          "asked", np.r_[tau_i, ff_i],
          np.r_[sw_state.jac[0].T @ (cfg.KP_SWING_WALK * e_p
                                     + cfg.KD_SWING_WALK * e_v),
                np.zeros(3)], 1e-12, " N*m")

    # -- walk.py: one reference, every target from it -----------------------
    def _walk_law(swing_law="osc", alloc="qp", **plan_kw):
        plan = WWALK.WalkPlan(swing_law=swing_law, **plan_kw)
        law = LAW.BalanceLaw(gains=_xy_gains(), foot_xy=fold.foot_xy.copy(),
                             dynamic_setpoint=False, srb=fold.srb,
                             track_stop_deg=0.0, est_xy=True,
                             kp_joint=cfg.KP_JOINT_HOLD,
                             kd_joint=cfg.KD_JOINT_HOLD, alloc=alloc,
                             walk=plan)
        law.arm(0.0, at_fold)
        law.ramp = REF.Quintic.ramp(cfg.H_LIFT, cfg.H_LIFT, 1.0)
        law.hold_joints(at_fold.q)
        return law, plan
    try:
        LAW.BalanceLaw(foot_xy=fold.foot_xy, srb=fold.srb, swing="joint",
                       walk=WWALK.WalkPlan())
        refused = False
    except ValueError:
        refused = True
    check("walk: refused on a swing that cannot place a foot in x/y "
          "(--swing joint)", refused)
    wl, wp = _walk_law()
    wl.feed_estimate(_est(0.996))
    out_h = wl.update(1.0, at_fold, xy_hold=True)
    check("...attached and NOT engaged before the trot: the HOLD is the "
          "flown one (no reference on the output)",
          not wp.engaged and out_h.ref is None)
    tg = GAIT.TrotGait()
    tg.reset(1.004)
    wl.feed_estimate(_est(1.0))
    out_w = wl.update(1.004, at_fold, gait=tg, xy_hold=True)
    check("...engaged on the trot's first sweep, the four anchors on the "
          "HOLD sites", wp.engaged and out_w.ref is not None
          and bool(np.isfinite(wp.anchor_w).all()) and wp.steps == 0)
    q_e, qd_e = wp.hold_targets(wl, at_fold, out_w.ref)
    close("...on that sweep the stance targets ARE q_hold: no step in the "
          "joint layer", q_e, C.unflat(wl.q_hold), 1e-9, " rad")
    check("...and the x/y rows close on zero error at zero rate: no step "
          "in the wrench", out_w.xy_on
          and float(np.abs(out_w.wrench.b_d[:2]).max()) < 1e-9
          and not qd_e.any())
    # A TRUNK EXACTLY ON THE REFERENCE GETS NO JOINT TORQUE -- position and
    # rate, the rate built by differencing the reference's own motion, not
    # from walk.py's formula.
    ref_m = out_w.ref._replace(p=out_w.ref.p + [0.012, -0.006],
                               yaw=out_w.ref.yaw + np.radians(5.0),
                               v=np.array([0.10, 0.02]), yaw_rate=0.2)

    def _feet(ref_k):
        p_org, _ = wp._origin_ref(wl, ref_k)
        Rk = C.rot_z(float(ref_k.yaw))[:2, :2]
        xy = (wp.anchor_w - p_org[None, :]) @ Rk
        return np.c_[xy, wp.z_rest]
    dt_m = 1e-6
    x_m = _feet(ref_m)
    xd_m = (_feet(ref_m._replace(p=ref_m.p + ref_m.v * dt_m,
                                 yaw=ref_m.yaw + ref_m.yaw_rate * dt_m))
            - _feet(ref_m._replace(p=ref_m.p - ref_m.v * dt_m,
                                   yaw=ref_m.yaw - ref_m.yaw_rate * dt_m))
            ) / (2.0 * dt_m)
    q_m = np.array([HK.leg_ik(i, x_m[i] - np.asarray(P.HIP_OFFSET[i]),
                              q_seed=C.unflat(at_fold.q)[i])
                    for i in range(C.N_LEGS)])
    R_m = C.rot_z(float(ref_m.yaw))
    o_m = IMU.TrunkOrientation(R=R_m, omega_b=np.array([0.0, 0.0, 0.2]),
                               roll=0.0, pitch=0.0, yaw=float(ref_m.yaw),
                               age_s=0.0)
    st_q = STATE.read(C.flat(q_m), np.zeros(C.N_JOINTS), o_m, srb=fold.srb)
    qd_m = np.array([np.linalg.solve(st_q.jac[i], xd_m[i])
                     for i in range(C.N_LEGS)])
    st_m = STATE.read(C.flat(q_m), C.flat(qd_m), o_m, srb=fold.srb)
    q_t, qd_t = wp.hold_targets(wl, st_m, ref_m)
    close("...a trunk exactly ON the reference, 12 mm and 5 deg on, gets no "
          "joint spring", q_t, q_m, 1e-9, " rad")
    close("...and walking it at the reference's v and r, no joint damper "
          "(target rate against a difference of the reference's motion)",
          qd_t, qd_m, 1e-6, " rad/s")
    both = np.array([True, False, False, True])
    q_p, qd_p = wp.hold_targets(wl, st_m, ref_m, both)
    check("...and a swinging leg costs no IK: its rows are q and zero",
          np.array_equal(q_p[[1, 2]], C.unflat(st_m.q)[[1, 2]])
          and not qd_p[[1, 2]].any()
          and np.allclose(q_p[[0, 3]], q_t[[0, 3]], 0.0, 1e-15))
    far = ref_m._replace(p=ref_m.p + [0.08, 0.0])
    q_f, _ = wp.hold_targets(wl, st_m, far)
    reach = max(float(np.linalg.norm(
        HK.foot_position(i, q_f[i])[:2] - st_m.x_b[i, :2]))
        for i in range(C.N_LEGS))
    close("...a reference 80 mm off pulls each leg's target only "
          "HOLD_XY_ERR_MAX away", reach, cfg.HOLD_XY_ERR_MAX, 1e-9, " m")
    before = wp.anchor_w.copy()
    wp._planted = both.copy()
    est_td = _est(1.0, p=[0.05, 0.02, 0.2])
    wp.contacts(wl, at_fold, out_w.ref, np.ones(4, dtype=bool), est_td)
    feet_w = (est_td.p_w[None, :] + at_fold.x_b @ at_fold.R.T)[:, :2]
    close("...a foot that lands is anchored where the filter puts it, "
          "p_hat + R x_b", wp.anchor_w[[1, 2]], feet_w[[1, 2]], 1e-15, " m")
    check("...and the feet that stayed down keep theirs",
          np.array_equal(wp.anchor_w[[0, 3]], before[[0, 3]]))
    # The placement reads the filter's x/y velocity LOW-PASSED (2026-10-05).
    lp = WWALK.WalkPlan(v_filter_hz=1.0)
    tau_lp = 1.0 / (2.0 * np.pi)
    v_stp = np.array([0.10, -0.05, 0.0])
    lp._low_pass(_est(0.0, v=v_stp))             # starts the clock, at rest
    t_k = 0.0
    while t_k < tau_lp - 1e-12:
        t_k = min(tau_lp, t_k + 0.004)
        lp._low_pass(_est(t_k, v=v_stp))
    once = lp.v_lp.copy()
    lp._low_pass(_est(t_k, v=5.0 * v_stp))       # the same estimate again
    close("walk: the placement's x/y velocity is the filter's low-passed at "
          "v_filter_hz -- 63 % of a step after one time constant, and an "
          "estimate read twice counts once",
          np.r_[once, lp.v_lp - once],
          np.r_[(1.0 - np.exp(-1.0)) * v_stp[:2], 0.0, 0.0], 1e-12, " m/s")
    tx_lp = lp.trunk_xy(wl, at_fold, out_w.ref, _est(1.0, v=v_stp))
    tx_raw = WWALK.WalkPlan(v_filter_hz=0.0).trunk_xy(wl, at_fold, out_w.ref,
                                                      _est(1.0, v=v_stp))
    check("...the planner reads that; v_filter_hz 0 reads the estimate raw",
          np.array_equal(tx_lp.v, lp.v_lp)
          and np.array_equal(tx_raw.v, v_stp[:2]))
    # Two cycles through the law at zero command on the static fixture.
    wl2, wp2 = _walk_law()
    tg2 = GAIT.TrotGait()
    tg2.reset(1.0)
    t = 1.0
    trips2, moved, swing_f, fin = set(), 0.0, 0.0, True
    p_first = None
    while t < 1.0 + 2.0 * tg2.period:
        wl2.feed_estimate(_est(t - 0.004))
        o2 = wl2.update(t, at_fold, gait=tg2, xy_hold=True)
        if o2.trip:
            trips2.add(o2.trip)
        if p_first is None:
            p_first = o2.ref.p.copy()
        moved = max(moved, float(np.abs(o2.ref.p - p_first).max()),
                    abs(o2.ref.yaw - out_w.ref.yaw))
        off_legs = o2.contact <= 0.0
        if off_legs.any():
            swing_f = max(swing_f,
                          float(np.abs(o2.allocation.f_w[off_legs]).max()))
        fin &= bool(np.all(np.isfinite(o2.tau)))
        t += 0.004
    check("...two cycles at zero command: the reference stands still, no "
          "trip, every torque finite", moved == 0.0 and not trips2 and fin,
          "; ".join(sorted(trips2))[:60])
    check("...the QP gives every swinging foot exactly zero force",
          swing_f == 0.0)
    check("...and every one of its swings flew AT REST: onto the site, "
          "trunk frame, no velocity term",
          wp2.planner.swings_at_rest > 0 and wp2.planner.swings_placed == 0,
          "%d at rest, %d placed" % (wp2.planner.swings_at_rest,
                                     wp2.planner.swings_placed))
    rep = wl2.report()
    check("...and the exit report has the QP and the walk",
          "qp allocator" in rep and "touchdowns anchored" in rep)
    # hw.fold_walk builds the plan the way doc/walk flew it.
    import argparse
    opts = FW.walk_options()
    clock = opts["gait"]
    check("hw.fold_walk: hw.fold_trot's stand and trot, the walking hook, "
          "the QP", opts["alloc"] == "qp"
          and isinstance(opts["hook"], FW.WalkHook)
          and opts["crouch"] is POSE.FOLD)
    check("...on the walk's clock: hw.fold_trot's period at WALK_DUTY, the "
          "contact ramp inside its four-foot window",
          clock.period == FT.PERIOD_S and clock.duty == cfg.WALK_DUTY
          and clock.ramp == cfg.WALK_CONTACT_RAMP
          and cfg.WALK_CONTACT_RAMP < (cfg.WALK_DUTY - 0.5)
          / (2.0 * cfg.WALK_DUTY),
          "%.2f s, duty %.2f, %.0f ms of swing, ramp %.3f"
          % (clock.period, clock.duty, 1e3 * clock.swing_duration,
             clock.ramp))
    ap_w = argparse.ArgumentParser()
    opts["hook"].add_arguments(ap_w)
    a_w = ap_w.parse_args([])
    a_w.ff_armature, a_w.swing_ff = None, False
    opts["hook"].configure(a_w)
    built = opts["hook"].law_kwargs(a_w)["walk"]
    check("...and the plan its flags build is the walk doc/walk flew: the "
          "task-space swing at WN_SWING_OSC",
          built.swing_law == cfg.WALK_SWING_LAW == "osc"
          and np.array_equal(built.wn_swing, cfg.WN_SWING_OSC)
          and built.zeta_swing == cfg.ZETA_SWING_OSC
          and built.reference.vx_max == cfg.WALK_VX_MAX,
          "wn %s rad/s" % np.array2string(built.wn_swing, precision=0))
    a_r = ap_w.parse_args(["--place-at-rest"])
    a_r.ff_armature, a_r.swing_ff = None, False
    opts["hook"].configure(a_r)
    check("...at zero command its swings land on their sites, trunk frame "
          "(the default); --place-at-rest is the placement as flown before",
          not built.place_at_rest and cfg.WALK_PLACE_AT_REST is False
          and opts["hook"].law_kwargs(a_r)["walk"].place_at_rest)
    a_f = ap_w.parse_args(["--v-filter-hz", "0"])
    a_f.ff_armature, a_f.swing_ff = None, False
    opts["hook"].configure(a_f)
    check("...its placement velocity is low-passed at WALK_V_FILTER_HZ; "
          "--v-filter-hz 0 reads it raw",
          built.v_filter_hz == cfg.WALK_V_FILTER_HZ > 0.0
          and opts["hook"].law_kwargs(a_f)["walk"].v_filter_hz == 0.0)
    # ONE LEG'S OWN IMPEDANCE, 2026-10-05, on request (`--kp-swing-walk-rl`
    # etc.): a (4, 3) table in the plan, and only that leg's swing feels it.
    walk_kp = np.tile(cfg.KP_SWING_WALK, (C.N_LEGS, 1))
    walk_kd = np.tile(cfg.KD_SWING_WALK, (C.N_LEGS, 1))
    walk_kp[rl] = [150.0, 150.0, 600.0]
    walk_kd[rl] = [6.0, 6.0, 60.0]
    a_l = ap_w.parse_args(["--swing-law", "impedance",
                           "--kp-swing-walk-rl", "150", "150", "600",
                           "--kd-swing-walk-rl", "6", "6", "60"])
    a_l.ff_armature, a_l.swing_ff = None, False
    opts["hook"].configure(a_l)
    plan_l = opts["hook"].law_kwargs(a_l)["walk"]
    check("--kp/--kd-swing-walk-rl: RL's own row in the walk's impedance, "
          "--kp/--kd-swing-walk in the other three",
          plan_l.swing_law == "impedance"
          and np.array_equal(plan_l.kp_swing, walk_kp)
          and np.array_equal(plan_l.kd_swing, walk_kd)
          and opts["hook"].own_swing == ["RL"],
          "own %s" % opts["hook"].own_swing)
    a_o = ap_w.parse_args(["--kp-swing-walk-rl", "150", "150", "600"])
    a_o.ff_armature, a_o.swing_ff = None, False
    try:
        opts["hook"].configure(a_o)
        refused = False
    except ValueError:
        refused = True
    check("...refused on the osc law, which has no Kp/Kd to take", refused)
    err_w = io.StringIO()
    try:
        with contextlib.redirect_stderr(err_w):
            FW.main(["--fake", "--kp-swing-walk-rl", "150", "150", "600"])
        exit_w = None
    except SystemExit as refusal_exit:
        exit_w = refusal_exit.code
    check("...hw.fold_walk at its default osc swing refuses them before the "
          "bus", exit_w == 2 and "the impedance's gains" in err_w.getvalue(),
          "exit %r" % exit_w)
    sw_ref = argparse.Namespace(p=rest[rl] + [0.0, 0.0, 0.010],
                                v=np.array([0.0, 0.0, 0.3]), a=np.zeros(3))
    close("...the impedance takes each leg's own row",
          WSC.swing_control_torque(at_fold, rl, sw_ref, kp=walk_kp,
                                   kd=walk_kd)[0],
          WSC.swing_control_torque(at_fold, rl, sw_ref, kp=walk_kp[rl],
                                   kd=walk_kd[rl])[0], 0.0, " N*m")
    wl_s, _ = _walk_law(swing_law="impedance")
    wl_r, _ = _walk_law(swing_law="impedance", kp_swing=walk_kp,
                        kd_swing=walk_kd)
    tg_l = GAIT.TrotGait()
    tg_l.reset(1.0)
    t = 1.0
    rl_up = rl_down = others = 0.0
    while t < 1.0 + 2.0 * tg_l.period:
        wl_s.feed_estimate(_est(t - 0.004))
        wl_r.feed_estimate(_est(t - 0.004))
        o_s = wl_s.update(t, at_fold, gait=tg_l, xy_hold=True)
        d = C.unflat(wl_r.update(t, at_fold, gait=tg_l, xy_hold=True).tau
                     - o_s.tau)
        if not tg_l.sample(t).contact[rl]:       # the law's own test
            rl_up = max(rl_up, float(np.abs(d[rl]).max()))
        else:
            rl_down = max(rl_down, float(np.abs(d[rl]).max()))
        others = max(others, float(np.abs(np.delete(d, rl, axis=0)).max()))
        t += 0.004
    check("...the walking law with RL's own row: RL's swing torque moves, RL "
          "in stance and the other three legs do not",
          rl_up > 0.05 and rl_down == 0.0 and others == 0.0,
          "RL swinging up to %.2f N*m apart; down %.1e, others %.1e"
          % (rl_up, rl_down, others))
    # hw.fold2_walk: the same walk over hw.fold2_trot (fold_walk.walking).
    from .. import fold2_trot as F2T
    from .. import fold2_walk as F2W
    t2 = F2T.stand_options()
    o2 = FW.walking(t2)
    c2 = o2["gait"]
    check("hw.fold2_walk: hw.fold2_trot's stand -- FOLD2, its straight rise, "
          "its hold -- walking: the hook, the QP, the walk's clock",
          o2["crouch"] is POSE.FOLD2 and "stand_xy" not in o2
          and o2["height"] == F2T.HEIGHT == t2["height"]
          and isinstance(o2["hook"], FW.WalkHook) and o2["alloc"] == "qp"
          and c2.period == t2["gait"].period and c2.duty == cfg.WALK_DUTY
          and c2.ramp == cfg.WALK_CONTACT_RAMP
          and c2.settle_s == cfg.WALK_SETTLE_S,
          "%.2f s, duty %.2f, settle %.1f s" % (c2.period, c2.duty,
                                                c2.settle_s))
    check("...walking() leaves the trot entry point's own options alone (a "
          "fresh gait; the trot's clock untouched), and hw.fold2_walk is it",
          t2["gait"].duty == cfg.DUTY and t2["gait"].settle_s == cfg.SETTLE_S
          and "hook" not in t2 and "alloc" not in t2
          and c2 is not t2["gait"]
          and F2W.walk_options()["crouch"] is POSE.FOLD2)
    f2 = POSE.FOLD2
    at_f2 = state_at(F2T.HEIGHT, foot_xy=f2.foot_xy, q_seed=f2.q,
                     srb=f2.srb_at(F2T.HEIGHT))
    lw2 = LAW.BalanceLaw(gains=_xy_gains(), foot_xy=f2.foot_xy.copy(),
                         dynamic_setpoint=False, srb=f2.srb_at(F2T.HEIGHT),
                         h_lift=F2T.HEIGHT, track_stop_deg=0.0, est_xy=True,
                         kp_joint=cfg.KP_JOINT_HOLD,
                         kd_joint=cfg.KD_JOINT_HOLD,
                         walk=WWALK.WalkPlan())
    lw2.arm(0.0, at_f2)
    lw2.ramp = REF.Quintic.ramp(F2T.HEIGHT, F2T.HEIGHT, 1.0)
    lw2.hold_joints(at_f2.q)
    tg2b = GAIT.TrotGait()
    tg2b.reset(1.0)
    lw2.feed_estimate(_est(0.996))
    o_f2 = lw2.update(1.0, at_f2, gait=tg2b, xy_hold=True)
    q_f2e, _ = lw2.walk.hold_targets(lw2, at_f2, o_f2.ref)
    close("...FOLD2 engages bumplessly too: the stance targets ARE its q_hold "
          "on the first trot sweep", q_f2e, C.unflat(lw2.q_hold), 1e-9,
          " rad")
    close("...and its footholds are placed about FOLD2's OWN sites, trunk x "
          "+-219.5 mm", np.abs(lw2.walk.sites_b),
          np.tile([0.2195, 0.065], (4, 1)), 5e-4, " m")

    # =====================================================================
    print("\n15. the fall hold: past 45 deg the joints hold the stand, "
          "nothing e-stops")
    # =====================================================================
    import contextlib
    import io

    from .. import stand as STAND

    def _tipped(deg, dq=None, qd=None, rot=C.rot_x):
        """`at_fold`'s legs (plus `dq`) under a trunk tipped `deg`."""
        R = rot(np.deg2rad(deg))
        roll, pitch, yaw = C.zyx_from_rot(R)
        orientation = IMU.TrunkOrientation(R=R, omega_b=np.zeros(3),
                                           roll=roll, pitch=pitch, yaw=yaw,
                                           age_s=0.0)
        return STATE.read(at_fold.q + (0.0 if dq is None else dq),
                          np.zeros(C.N_JOINTS) if qd is None else qd,
                          orientation, srb=fold.srb)

    def _fall_law(**kw):
        law = LAW.BalanceLaw(**dict(dict(
            foot_xy=fold.foot_xy, dynamic_setpoint=False, srb=fold.srb,
            track_stop_deg=0.0,
            fall_hold_deg=cfg.FALL_HOLD_DEG, kp_joint=cfg.KP_JOINT_HOLD,
            kd_joint=cfg.KD_JOINT_HOLD), **kw))
        law.arm(0.0, at_fold)
        return law

    fh = _fall_law()
    fh.hold_joints(at_fold.q)                    # the first HOLD
    below = fh.update(0.1, _tipped(cfg.FALL_HOLD_DEG - 5.0))
    check("5 deg short of the fall hold: the balance law, no trip",
          below.trip is None and not below.fallen and not fh.fallen)
    dq = np.deg2rad(np.linspace(-6.0, 6.0, C.N_JOINTS))
    qd = np.linspace(-1.0, 1.0, C.N_JOINTS)
    over = _tipped(cfg.FALL_HOLD_DEG + 5.0, dq=dq, qd=qd)
    fell = fh.update(0.2, over, gait=GAIT.TrotGait())
    check("5 deg past it: NO trip -- the fall hold, on that very sweep",
          fell.trip is None and fell.fallen and fh.fallen, fell.trip)
    close("...its torque is the joint PD to the stand plus the legs' weight, "
          "nothing else", fell.tau,
          cfg.KP_FALL_HOLD * (at_fold.q - over.q) - cfg.KD_FALL_HOLD * qd
          + C.flat(TRQ.all_leg_gravity_torque(over.q, over.R)), 1e-12,
          " N*m")
    check("...no wrench, no forces, no swing, whatever the gait says",
          not fell.wrench.b_d.any() and not fell.allocation.f_w.any()
          and fell.swing_s is None and fell.contact is None)
    level = fh.update(0.3, at_fold)
    check("...and it stays: a level trunk on the next sweep is still the "
          "hold", level.fallen and np.array_equal(level.q_ref, at_fold.q)
          and abs(np.abs(level.tau
                         - C.flat(TRQ.all_leg_gravity_torque(at_fold.q,
                                                             at_fold.R))
                         ).max()) < 1e-12)
    check("...and the exit report says at what tilt FROM THE SETPOINT",
          abs(fh.fall_tilt_deg - fh.tilt_from_setpoint_deg(over)) < 1e-9
          and "ENGAGED at %.1f deg" % fh.fall_tilt_deg in fh.report(),
          "%.2f deg" % fh.fall_tilt_deg)

    later = _fall_law()
    later.hold_joints(at_fold.q)
    later.hold_joints(at_fold.q + dq)            # a W step's re-latch
    later.update(0.1, _tipped(cfg.FALL_HOLD_DEG + 5.0))
    check("the stand is the FIRST HOLD's pose: a W re-latch does not move it",
          np.array_equal(later.q_fall, at_fold.q)
          and np.array_equal(later.q_hold, at_fold.q + dq))
    early = _fall_law()                          # nothing latched: the rise
    early.update(0.1, _tipped(cfg.FALL_HOLD_DEG + 5.0))
    close("fallen in the rise: the IK at h_lift on the stand's sites",
          early.q_fall, C.flat(LAW.ik_reference(early.h_lift,
                                                C.unflat(at_fold.q),
                                                early.home_xy)), 1e-12, " rad")
    pitched = _fall_law()
    pitched.hold_joints(at_fold.q)
    check("pitch past it falls too, not only roll",
          pitched.update(0.1, _tipped(cfg.FALL_HOLD_DEG + 5.0,
                                      rot=C.rot_y)).fallen)

    off = _fall_law(fall_hold_deg=None)
    gone = off.update(0.1, _tipped(cfg.FALL_HOLD_DEG + 5.0))
    check("no tilt e-stop (deleted): with the fall hold OFF, nothing acts",
          gone.trip is None and not gone.fallen, gone.trip)

    # -- the sequence -------------------------------------------------------
    fs = SEQ.StandSequence(SAFE.SafetyGate(3.0), crouch=fold,
                           balance=_fall_law(), gait=GAIT.TrotGait(),
                           step_to=TR.STEP_FOOT_XY,
                           step_gait=GAIT.TrotGait(period=TR.STEP_PERIOD_S))
    fs.phase = SEQ.PHASES.index("hold")
    fs.body = at_fold
    fs.gate.start(0.0, q=at_fold.q)
    fs.balance.hold_joints(at_fold.q)
    fs.toggle_trot(1.0)
    fs.update(1.1, at_fold)
    mode, _, trip = fs.update(1.2, _tipped(cfg.FALL_HOLD_DEG + 5.0))
    said = fs.take_notice() or ""
    check("a trot tipped past it: phase 'fall', the trot dropped, still "
          "torque, no trip", fs.phase_name == "fall" and not fs.trotting
          and mode == "torque" and trip is None, fs.phase_name)
    check("...the operator is told, in the notice", "FALL HOLD in trot" in said,
          said[:60])
    mode, _, trip = fs.update(1.3, at_fold)
    check("...a level sweep after: still the fall hold, through the gate",
          fs.phase_name == "fall" and mode == "torque" and trip is None
          and fs.out.fallen)
    check("...T and W are refused in it",
          "ignored" in fs.toggle_trot(1.4) and "ignored" in fs.toggle_step(1.4))
    refused = fs.advance(1.5, at_fold.q + dq)
    check("...and ENTER PARKS, from where the legs are -- no step home first",
          refused is None and fs.phase_name == "park" and not fs.fallen
          and np.array_equal(C.flat(fs.q_ref0), at_fold.q + dq)
          and fs.update(1.6, at_fold)[0] == "position", fs.phase_name)

    # -- the entry points -----------------------------------------------------
    check("every trot entry point flies it at 45 deg (trot.trot_options)",
          TR.trot_options()["fall_hold"] == cfg.FALL_HOLD_DEG
          and FT.stand_options()["fall_hold"] == cfg.FALL_HOLD_DEG
          and F2T.stand_options()["fall_hold"] == cfg.FALL_HOLD_DEG
          and FW.walk_options()["fall_hold"] == cfg.FALL_HOLD_DEG)
    check("...the trot's slew can carry it; the stand's cannot",
          cfg.TAU_SLEW_TROT_NM_S >= cfg.FALL_HOLD_MIN_SLEW_NM_S
          > SAFE.DEFAULT_TAU_SLEW_NM_S)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            STAND.main(["--fake", "--fall-hold", "45"])
        refused_code = None
    except SystemExit as stop:
        refused_code = stop.code
    check("hw.stand REFUSES --fall-hold on the stand's 5 N*m/s slew, before "
          "the bus", refused_code == 2 and "--fall-hold on a 5 N*m/s slew"
          in err.getvalue(), "exit %r" % refused_code)

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
