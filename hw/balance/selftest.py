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
from . import allocation as ALLOC    # noqa: E402
from . import config as cfg          # noqa: E402
from . import controller as CTRL     # noqa: E402
from . import law as LAW             # noqa: E402
from . import reference as REF       # noqa: E402
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
def pose_at(h: float) -> np.ndarray:
    """(4, 3) joints with the feet at FOOT_XY and the trunk bottom at `h`.

    `hw.kinematics`' CLOSED-FORM inverse, not `sim.stand.pose_for_height`'s
    damped least squares.  The difference is 9 nm of residual, which is
    nothing to the robot and everything to a fixture: a pose that is 9 nm off
    the commanded height makes the PD ask for 8.6 uN, and a check on "zero
    error gives exactly mg" then fails on the SOLVER rather than on the law.
    """
    targets = ST.foot_targets(STATE.height_to_origin(h))
    return HK.all_leg_ik(targets, q_seed=ST.Q_CROUCH)


def state_at(h: float, R=None, omega_b=None, qd=None, age_s: float = 0.0):
    """A `BodyState` at height `h` and orientation `R`, feet planted."""
    R = np.eye(3) if R is None else R
    roll, pitch, yaw = C.zyx_from_rot(R)
    orientation = IMU.TrunkOrientation(
        R=R, omega_b=np.zeros(3) if omega_b is None else np.asarray(omega_b),
        roll=roll, pitch=pitch, yaw=yaw, age_s=age_s)
    q = C.flat(pose_at(h))
    return STATE.read(q, np.zeros(C.N_JOINTS) if qd is None else qd,
                      orientation)


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
    check("the yaw SPRING is off and the yaw DAMPER is not",
          gains.kp_att[2] == 0.0 and gains.kd_att[2] > 0.0,
          "magnetometer untrusted; gyro omega_z is fine")

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
