"""Gate the hardware MPC path with no robot, no IMU and no simulator.

    python -m hw.cmpc.selftest

WHAT THIS CAN AND CANNOT TELL YOU
    CAN     that the controller on the hardware path -- closed-form kinematics
            handed in, the IMU hold not emulated, a stand schedule -- computes
            the SAME torques the simulator's controller computes from the same
            state; that the stand schedule is four feet down over the whole
            horizon and the trot schedule is untouched; that the estimator
            recovers height, velocity, yaw and tilt from states built to have
            them; that the phase machine keeps every one of the stand's
            position-mode joint-hold phases around the MPC, byte for byte
            against `hw.stand.HardwareStand`; that the MPC's torque map at a
            level stance agrees with `hw.balance.torque`'s boxed equation;
            and that every trip fires on the state that should fire it.

    CANNOT   whether the QP's forces will stand the robot (that is the
            simulator's 83 gates and then the robot's), whether the kinematic
            odometry is good enough behind a real IMU, or what the solve costs
            on the Pi.

THE END-TO-END RUN ON `hw.fake_bus` IS REPORTED, NOT ASSERTED, ON ONE POINT
    `hw.stand.run` stops the run when any motor goes 25 ms without a frame.
    On a laptop that never happens; on a loaded or virtualised host it does,
    and it says nothing about this code.  So the full sequence is driven on
    the fake bus and is REQUIRED to reach `done` unless the only thing that
    stopped it was that gap trip -- which is then printed as a host-timing
    note.  Every other stop reason is a failure.
"""
from __future__ import annotations

import sys
import time

import numpy as np

if __package__ in (None, ""):        # allow `python hw/cmpc/selftest.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.cmpc"

from sim import coordinates as C             # noqa: E402
from sim import kinematics as SK             # noqa: E402
from sim import params as P                  # noqa: E402
from sim import stand as ST                  # noqa: E402
from sim.cmpc import config as SIMCFG        # noqa: E402
from sim.cmpc import controller as CTL       # noqa: E402
from sim.cmpc import gait as GAIT            # noqa: E402
from sim.cmpc import swing as SWING          # noqa: E402

from .. import hardware_map as HM            # noqa: E402
from .. import imu as IMU                    # noqa: E402
from .. import kinematics as HK              # noqa: E402
from .. import safety as SAFE                # noqa: E402
from .. import stand as HS                   # noqa: E402
from ..balance import config as BCFG         # noqa: E402
from ..balance import state as BSTATE        # noqa: E402
from ..balance import torque as TRQ          # noqa: E402
from . import config as cfg                  # noqa: E402
from . import run as RUN                     # noqa: E402
from . import state as EST                   # noqa: E402

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


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def pose_at(h: float) -> np.ndarray:
    """(4, 3) joints, feet at the stand's FOOT_XY, trunk bottom at `h`."""
    return HK.all_leg_ik(ST.foot_targets(BSTATE.height_to_origin(h)),
                         q_seed=ST.Q_CROUCH)


def body_at(h: float, R=None, omega_b=None, qd=None, age_s: float = 0.0,
            q=None):
    """A `hw.balance.state.BodyState`: what the sweep hands the MPC phase."""
    R = np.eye(3) if R is None else R
    roll, pitch, yaw = C.zyx_from_rot(R)
    orientation = IMU.TrunkOrientation(
        R=R, omega_b=np.zeros(3) if omega_b is None else np.asarray(omega_b),
        roll=roll, pitch=pitch, yaw=yaw, age_s=age_s)
    q = C.flat(pose_at(h)) if q is None else C.flat(q)
    return BSTATE.read(q, np.zeros(C.N_JOINTS) if qd is None else qd,
                       orientation)


def sim_state_from(body, position, velocity, yaw: float = 0.0):
    """The simulator's `BodyState` for the same robot, NO precomputed legs."""
    R = C.rot_zyx(body.roll, body.pitch, yaw)
    return CTL.BodyState(position=np.asarray(position, float), rotation=R,
                         rpy=np.array([body.roll, body.pitch, yaw]),
                         velocity=np.asarray(velocity, float),
                         omega=R @ body.omega_b,
                         q=C.unflat(body.q).copy(), qd=C.unflat(body.qd).copy())


def gate_for_test(tau_cap: float = 2.5) -> SAFE.SafetyGate:
    return SAFE.SafetyGate(tau_cap, unconfirmed_reason="hw.cmpc.selftest")


def drive_to_mpc(stand: RUN.HardwareCmpc, body, t0: float = 0.0,
                 dwell: float = 0.2):
    """Step a HardwareCmpc limp -> ... -> mpc on a FIXED body, no bus."""
    now = t0
    stand.t_phase = now
    while stand.phase_name != "mpc":
        stand.update(now, body)
        now += 0.004
        if now - stand.t_phase >= dwell and stand.ramp_remaining(now) <= 0.0:
            refused = stand.advance(now, body.q)
            assert refused is None, refused
            stand.update(now, body)
    return now


# ===========================================================================
def main() -> int:
    print("DOG6 hardware MPC self-test  (no robot, no IMU, no MuJoCo)\n")
    h_lift = BCFG.H_LIFT
    tau_cap = 2.5

    # -- 1. config ------------------------------------------------------------
    print("config")
    check("the solve budget plus two sweeps is inside the CAN stop line",
          cfg.SOLVE_BUDGET_S + 2.0 / HS.RATE_HZ < HS.GAP_ESTOP_S,
          "%.0f + %.0f ms < %.0f ms" % (1e3 * cfg.SOLVE_BUDGET_S,
                                        2e3 / HS.RATE_HZ, 1e3 * HS.GAP_ESTOP_S))
    check("the staged torque ceiling cannot carry a trot diagonal",
          SAFE.TAU_STAGED_MAX < cfg.TROT_DIAGONAL_NM,
          "%.1f < %.2f N*m -- which is why the default gait stands"
          % (SAFE.TAU_STAGED_MAX, cfg.TROT_DIAGONAL_NM))
    check("the commandable height band holds the lift height",
          cfg.Z_MIN < ST.LIFT_HEIGHT < cfg.Z_MAX)
    check("the default MPC rate is the simulator's", cfg.MPC_HZ == SIMCFG.MPC_HZ)

    # -- 2. the gait schedules -----------------------------------------------
    print("\nsim.cmpc.gait: the two schedules")
    ts = np.linspace(0.0, 3.0, 751)
    check("STAND has four feet down at every time",
          bool(np.all(GAIT.STAND.contact(ts))) and
          bool(np.all(GAIT.STAND.horizon_contacts(0.37, SIMCFG.HORIZON,
                                                  SIMCFG.MPC_DT))),
          "%d samples and the whole horizon" % ts.size)
    check("...and no swing progress", not np.any(GAIT.STAND.swing_phase(ts)))
    check("TROT is the module's own functions, bound by name",
          GAIT.TROT.contact is GAIT.contact
          and GAIT.TROT.horizon_contacts is GAIT.horizon_contacts
          and GAIT.TROT.swing_phase is GAIT.swing_phase)
    check("Controller() with no argument runs the trot",
          CTL.Controller().schedule is GAIT.TROT)
    check("the trot never has four feet down", not np.any(
        GAIT.TROT.contact(ts).all(axis=1)), "%d samples" % ts.size)

    # -- 3. the controller hooks, against the simulator's own path -----------
    print("\nsim.cmpc.controller: the hardware hooks change nothing")
    body = body_at(h_lift)
    z0 = body.z_origin
    position = np.array([0.0, 0.0, z0])
    truth = sim_state_from(body, position, np.zeros(3))
    hw_state = CTL.BodyState(position=position.copy(), rotation=np.eye(3),
                             rpy=np.zeros(3), velocity=np.zeros(3),
                             omega=np.zeros(3), q=truth.q.copy(),
                             qd=truth.qd.copy(), feet_body=body.x_b,
                             jacobians_body=body.jac)
    close("hw.kinematics' feet agree with the chain walk the controller uses",
          body.x_b, np.stack([SK.foot_position(i, truth.q[i])
                              for i in range(4)]), 1e-12, " m")
    close("...and its Jacobians", body.jac,
          np.stack([SK.foot_jacobian(i, truth.q[i]) for i in range(4)]), 1e-12)
    close("foot_world is the same with and without precomputed feet",
          hw_state.all_feet_world(), truth.all_feet_world(), 1e-12, " m")

    sim_ctl = CTL.Controller(schedule=GAIT.STAND)
    hw_ctl = CTL.Controller(schedule=GAIT.STAND, sensor_split=False)
    sim_ctl.command.z = hw_ctl.command.z = z0
    tau_sim = sim_ctl.update(truth, 0.0)
    tau_hw = hw_ctl.update(hw_state, 0.0, imu_fresh=True)
    check("both controllers solved on the first sweep",
          sim_ctl.telemetry.status in cfg.QP_OK_STATUSES
          and hw_ctl.telemetry.status in cfg.QP_OK_STATUSES,
          "%s / %s" % (sim_ctl.telemetry.status, hw_ctl.telemetry.status))
    close("the hardware path's torques are the simulator's, same state",
          tau_hw, tau_sim, 1e-9, " N*m")
    close("...and so are the forces", hw_ctl.telemetry.forces,
          sim_ctl.telemetry.forces, 1e-9, " N")
    close("at zero error the four normal forces carry exactly the weight",
          hw_ctl.telemetry.forces[:, 2].sum(), P.WEIGHT, 0.02 * P.WEIGHT, " N")
    check("...split evenly fore and aft on a square stance",
          abs(hw_ctl.telemetry.forces[:2, 2].sum()
              - hw_ctl.telemetry.forces[2:, 2].sum()) < 0.02 * P.WEIGHT,
          "front %.1f N, rear %.1f N" % (hw_ctl.telemetry.forces[:2, 2].sum(),
                                          hw_ctl.telemetry.forces[2:, 2].sum()))
    check("the stance legs keep the controller's joint-space floor",
          SIMCFG.KP_JOINT > 0 and SIMCFG.KD_JOINT > 0
          and np.allclose(SWING.joint_pd(truth.q[0], np.array([1.0, 0, 0]),
                                         truth.q[0]),
                          [-SIMCFG.KD_JOINT, 0.0, 0.0]),
          "Kp %.1f, Kd %.2f, pure damping about the measured q"
          % (SIMCFG.KP_JOINT, SIMCFG.KD_JOINT))

    # The MPC's torque map against the balance package's boxed equation:
    # tau_i = -J_i^T R^T f_i^w, no gravity term on either side.
    f_w = hw_ctl.telemetry.forces
    boxed = TRQ.stance_torque(body, f_w, gravity=np.zeros((4, 3)))
    close("the MPC torque map is hw.balance.torque's boxed equation",
          C.flat(tau_hw), boxed, 1e-9, " N*m")
    # ...and at a tilt, where a dropped R^T would show.
    R_tilt = C.rot_zyx(np.deg2rad(6.0), np.deg2rad(-4.0), 0.3)
    body_t = body_at(h_lift, R=R_tilt)
    st_t = CTL.BodyState(position=np.array([0, 0, body_t.z_origin]),
                         rotation=R_tilt, rpy=np.array(C.zyx_from_rot(R_tilt)),
                         velocity=np.zeros(3), omega=np.zeros(3),
                         q=C.unflat(body_t.q), qd=np.zeros((4, 3)),
                         feet_body=body_t.x_b, jacobians_body=body_t.jac)
    ctl_t = CTL.Controller(schedule=GAIT.STAND, sensor_split=False)
    ctl_t.command.z = body_t.z_origin
    tau_t = ctl_t.update(st_t, 0.0, imu_fresh=True)
    close("...at a 6 deg tilt too, so R^T is applied on the hardware path",
          C.flat(tau_t), TRQ.stance_torque(body_t, ctl_t.telemetry.forces,
                                           gravity=np.zeros((4, 3))),
          1e-9, " N*m")

    # the IMU hold is the caller's, not emulated
    ctl_h = CTL.Controller(schedule=GAIT.STAND, sensor_split=False)
    ctl_h.command.z = z0
    flags = [True, False, False, True, False, True, True, False]
    for k, fresh in enumerate(flags):
        ctl_h.update(hw_state, 0.004 * k, imu_fresh=fresh)
    check("with sensor_split off, fb refreshes exactly on the caller's flags",
          ctl_h.telemetry.imu_ticks == sum(flags),
          "%d refreshes for %d flags" % (ctl_h.telemetry.imu_ticks, sum(flags)))
    ctl_s = CTL.Controller(schedule=GAIT.STAND)
    for k in range(8):
        ctl_s.update(truth, 0.004 * k)
    # 8 sweeps at 4 ms span 28 ms; a 5 ms ticker counted from a fixed origin
    # fires at 0, 8, 12, 16, 20 and 28 ms -- six times, not eight.
    check("...and with it on, the 200 Hz ticker decides, as in the simulator",
          ctl_s.telemetry.imu_ticks == 6, "%d of 8 sweeps" % ctl_s.telemetry.imu_ticks)

    ctl_20 = CTL.Controller(schedule=GAIT.STAND, mpc_dt=0.05, sensor_split=False)
    ctl_20.command.z = z0
    solves = 0
    for k in range(250):
        ctl_20.update(hw_state, 0.004 * k, imu_fresh=True)
        solves += int(ctl_20.telemetry.solved)
        ctl_20.telemetry.solved = False
    check("mpc_dt sets the solve rate (20 Hz over 1 s)", solves == 20,
          "%d solves" % solves)

    # -- 4. the estimator ------------------------------------------------------
    print("\nhw.cmpc.state: kinematic odometry")
    est = EST.KinematicOdometry()
    out = est.update(body, 0.0, np.ones(4, bool))
    close("a level planted robot reads the FK height", out.state.position[2],
          BSTATE.height_to_origin(h_lift), 1e-9, " m")
    check("...at the origin, heading zero, not moving",
          np.allclose(out.state.position[:2], 0.0)
          and np.allclose(out.state.velocity, 0.0) and out.state.rpy[2] == 0.0
          and out.n_stance == 4)
    check("the state carries the closed-form legs for the controller",
          out.state.feet_body is body.x_b and out.state.jacobians_body is body.jac)
    check("the no-IMU orientation reads fresh every sweep",
          out.imu_fresh and est.update(body, 0.004, np.ones(4, bool)).imu_fresh)

    # a tilted trunk: height from the ROTATED feet, as hw.balance.state does
    out_t = EST.KinematicOdometry().update(body_t, 0.0, np.ones(4, bool))
    close("under tilt the height is the rotated-feet height",
          out_t.state.position[2], body_t.z_origin, 1e-12, " m")
    close("...and roll/pitch are the IMU's", out_t.state.rpy[:2],
          C.zyx_from_rot(R_tilt)[:2], 1e-12, " rad")
    check("yaw starts at zero whatever the magnetometer says",
          out_t.state.rpy[2] == 0.0 and abs(out_t.yaw_mag - 0.3) < 1e-12,
          "mag %.2f rad logged, not used" % out_t.yaw_mag)

    # a trunk translating at v over planted feet: qd_i = J_i^{-1} (-v_b)
    v_true = np.array([0.12, -0.05, 0.0])
    qd = np.concatenate([np.linalg.solve(body.jac[i], -v_true) for i in range(4)])
    body_v = body_at(h_lift, qd=qd)
    est_v = EST.KinematicOdometry()
    t = 0.0
    for _ in range(250):                        # 1 s: the 20 Hz filter settles
        out_v = est_v.update(body_v, t, np.ones(4, bool))
        t += 0.004
    close("a translating trunk's velocity is recovered from the stance feet",
          out_v.state.velocity, v_true, 1e-3, " m/s")
    close("...and integrated into xy over the second",
          out_v.state.position[:2], v_true[:2] * t, 0.01, " m")

    # a trunk turning in place: the feet see omega x r, the yaw integrates
    omega_b = np.array([0.0, 0.0, 0.5])
    body_w = body_at(h_lift, omega_b=omega_b,
                     qd=np.concatenate([np.linalg.solve(
                         body.jac[i], -np.cross(omega_b, body.x_b[i]))
                         for i in range(4)]))
    est_w = EST.KinematicOdometry()
    t = 0.0
    for k in range(250):
        t = 0.004 * k                    # the integral runs from the FIRST update
        out_w = est_w.update(body_w, t, np.ones(4, bool))
    close("a yaw rate integrates into yaw", out_w.state.rpy[2], 0.5 * t,
          1e-9, " rad")
    close("...and a pure turn leaves the trunk velocity at zero",
          out_w.state.velocity, np.zeros(3), 1e-3, " m/s")
    close("omega is handed over in WORLD axes", out_w.state.omega,
          C.rot_z(out_w.state.rpy[2]) @ omega_b, 1e-12, " rad/s")
    check("the exact Euler yaw rate reduces to omega_z when level",
          EST.euler_yaw_rate(0.0, 0.0, omega_b) == 0.5
          and abs(EST.euler_yaw_rate(0.3, 0.0, [0.0, 1.0, 0.0]) - np.sin(0.3))
          < 1e-12)

    # a stale IMU: attitude held, not tripped, and reported
    body_stale = body_at(h_lift, R=R_tilt, age_s=cfg.IMU_MAX_AGE_S + 0.01)
    est_s = EST.KinematicOdometry()
    est_s.update(body, 0.0, np.ones(4, bool))
    out_s = est_s.update(body_stale, 0.004, np.ones(4, bool))
    check("a stale IMU holds the previous attitude and says so",
          out_s.imu_held and np.allclose(out_s.state.rpy[:2], 0.0),
          "held level although the stale packet says %.0f deg"
          % np.degrees(body_stale.roll))

    # -- 5. the phase machine: the stand's joint hold, around the MPC ---------
    print("\nhw.cmpc.run: the stand's position-mode joint hold is kept around the MPC")
    check("the MPC is inserted between lift and park and nowhere else",
          RUN.HardwareCmpc.PHASES == ("limp", "settle", "crouch", "lift", "mpc",
                                      "park", "done")
          and tuple(p for p in RUN.HardwareCmpc.PHASES if p != "mpc")
          == HS.HardwareStand.PHASES)
    check("every stand phase keeps its blurb", all(
        RUN.HardwareCmpc.BLURB[p] == HS.HardwareStand.BLURB[p]
        for p in HS.HardwareStand.PHASES))

    # Drive a HardwareCmpc and a plain HardwareStand through the same
    # sequence on the same body and compare what the position phases
    # command, sweep for sweep.  The fixture is a PERFECT DRIVER: in a
    # position phase the robot is wherever the last command put it, and in
    # a torque phase it stands at the lift height -- so the park ramps from
    # the lift pose to the crouch with the robot following, as on the bench.
    body_crouch = body_at(BCFG.H_CROUCH)
    body_lift = body_at(h_lift)
    def run_both(law="per-leg"):
        a = HS.HardwareStand(gate_for_test(tau_cap), law=law)
        b = RUN.HardwareCmpc(gate_for_test(tau_cap), law=law)
        rows_a, rows_b = [], []
        now = 0.0
        a.t_phase = b.t_phase = now
        body_now = body_crouch
        for _ in range(4000):
            ma, va, ta = a.update(now, body_now)
            mb_, vb, tb = b.update(now, body_now)
            rows_a.append((a.phase_name, ma, None if va is None else va.copy(), ta))
            rows_b.append((b.phase_name, mb_, None if vb is None else vb.copy(), tb))
            if ma == "position":
                body_now = body_at(h_lift, q=va)          # the driver got there
            elif ma == "torque":
                body_now = body_lift                      # the robot stands
            now += 0.004
            if (now - a.t_phase >= 0.3 and a.ramp_remaining(now) <= 0.0
                    and not a.finished):
                assert a.advance(now, body_now.q) is None
                assert b.advance(now, body_now.q) is None
                if b.phase_name == "mpc":           # b has one more phase
                    b.update(now, body_now)
                    assert b.advance(now, body_now.q) is None
            if a.finished and b.finished:
                break
        return a, b, rows_a, rows_b
    a, b, rows_a, rows_b = run_both()
    names_a = [r[0] for r in rows_a]
    names_b = [r[0] for r in rows_b]
    check("both machines finish in done, through the same phases",
          a.finished and b.finished
          and [n for n in dict.fromkeys(names_b) if n != "mpc"]
          == list(dict.fromkeys(names_a)),
          "a: %s  b: %s" % (" ".join(dict.fromkeys(names_a)),
                            " ".join(dict.fromkeys(names_b))))
    same_mode = all(ra[1] == rb[1] for ra, rb in zip(rows_a, rows_b))
    pos_rows = [(ra, rb) for ra, rb in zip(rows_a, rows_b) if ra[1] == "position"]
    worst = max(float(np.abs(ra[2] - rb[2]).max()) for ra, rb in pos_rows)
    check("settle, crouch, park, done command the SAME joint targets as "
          "hw.stand", same_mode and worst == 0.0,
          "%d position sweeps compared, worst %.0e rad" % (len(pos_rows), worst))
    check("...and limp is still a keepalive with no torque", all(
        ra[1] == "keepalive" for ra in rows_a if ra[0] == "limp"))
    check("the stand's position-mode phases are driver 0xA4 with the same "
          "speed caps", np.array_equal(a.max_dps, b.max_dps)
          and float(b.max_dps.max()) <= float(HS.MAX_SPEED_POS))
    check("no position phase tripped on either machine",
          all(ra[3] is None and rb[3] is None for ra, rb in pos_rows))

    # -- 6. the MPC phase itself ---------------------------------------------
    print("\nhw.cmpc.run: the MPC phase")
    stand = RUN.HardwareCmpc(gate_for_test(tau_cap), law="per-leg")
    now = drive_to_mpc(stand, body_lift)
    check("the handover latches the height the estimator measures",
          stand.controller is not None
          and abs(stand.controller.command.z - body_lift.z_origin) < 1e-9
          and stand.z_ramp is None,
          "z %.1f mm (origin), no ramp" % (1e3 * stand.controller.command.z))
    check("...with no velocity command in the stand gait",
          stand.controller.command.is_zero() and stand.gait == "stand")
    check("the gate was NOT restarted at the handover: the cap is already up",
          abs(stand.gate.cap_now(now) - tau_cap) < 1e-12)
    taus = []
    trips = []
    for _ in range(300):                                  # 1.2 s standing
        mode, tau, trip = stand.update(now, body_lift)
        taus.append(tau.copy())
        trips.append(trip)
        now += 0.004
    taus = np.asarray(taus)
    check("the MPC phase is torque mode and never tripped on a level stance",
          mode == "torque" and all(t is None for t in trips))
    check("...under the gate's cap", float(np.abs(taus).max()) <= tau_cap + 1e-9,
          "peak %.2f of %.1f N*m" % (np.abs(taus).max(), tau_cap))
    check("the QP solved every time it ran", stand._qp_fail_streak == 0
          and len(stand.timing.solve) > 0,
          "%d solves" % len(stand.timing.solve))
    fz = stand.controller.telemetry.forces[:, 2]
    close("standing still, the planned normal forces carry the weight",
          fz.sum(), P.WEIGHT, 0.02 * P.WEIGHT, " N")
    close("the leg-gravity term is hw.balance.torque's closed form",
          stand.tau_grav, C.flat(TRQ.all_leg_gravity_torque(body_lift.q)),
          1e-12, " N*m")
    close("...and tau_request = controller + leg gravity",
          stand.tau_request, stand.tau_mpc + stand.tau_grav, 1e-12, " N*m")
    close("the settled torque IS the request once the slew has caught up",
          taus[-1], stand.tau_request, 1e-9, " N*m")
    close("sign: at a level stance the knees HOLD the robot up like the SRB "
          "law's", np.sign(stand.tau_request.reshape(4, 3)[:, 2]),
          np.sign(C.flat(TRQ.stance_torque(
              body_lift, np.tile([0, 0, P.WEIGHT / 4], (4, 1)))).reshape(4, 3)[:, 2]),
          0.0)
    extra = stand.log_extra()
    check("the log row carries the MPC columns the CmpcLog declares",
          set(extra) == set(RUN.CmpcLog.EXTRA_FIELDS)
          and all(np.shape(np.asarray(extra[k], float)) == shape
                  for k, shape in RUN.CmpcLog.EXTRA_FIELDS.items()
                  if extra[k] is not None))
    log = RUN.CmpcLog()
    log.add(t=0.0, phase="mpc", h=0.1, **extra)
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.npz")
        log.save(path)
        with np.load(path) as arrays:
            check("...and saves with the stand's columns beside them",
                  "fz" in arrays and "f_mpc" in arrays
                  and arrays["f_mpc"].shape == (1, 4, 3)
                  and np.all(np.isnan(arrays["fz"])),
                  "%d columns" % len(arrays.files))

    # the operator: a height nudge ramps, velocity keys do nothing standing
    z_before = stand.controller.command.z
    stand.on_key("w")
    check("velocity keys are ignored in the stand gait",
          stand.controller.command.is_zero() and stand.operator.is_zero())
    stand.on_key("+")
    for _ in range(int(cfg.Z_RAMP_S / 0.004) + 5):
        stand.update(now, body_lift)
        now += 0.004
    close("a + nudges the height command by Z_NUDGE_M through a C2 ramp",
          stand.controller.command.z - z_before, cfg.Z_NUDGE_M, 1e-9, " m")
    stand.replan_height(10.0)
    for _ in range(int(cfg.Z_RAMP_S / 0.004) + 5):
        stand.update(now, body_lift)
        now += 0.004
    check("...and the command is clamped to the configured band",
          abs(stand.controller.command.z - cfg.Z_MAX) < 1e-9,
          "asked 10 m, got %.1f mm" % (1e3 * stand.controller.command.z))

    # ENTER from the MPC parks FROM WHERE THE ROBOT IS, in position mode
    assert stand.advance(now, body_lift.q) is None
    mode, values, trip = stand.update(now, body_lift)
    check("ENTER in the MPC phase parks: position mode from the measured pose",
          stand.phase_name == "park" and mode == "position"
          and np.allclose(values, body_lift.q) and trip is None)
    now += ST.RAMP_POSITION + 0.1
    mode, values, _ = stand.update(now, body_lift)
    close("...arriving at Q_CROUCH, the stand's park target",
          values, C.flat(ST.Q_CROUCH), 1e-12, " rad")

    # a --mpc-height target ramps from the latched height
    stand2 = RUN.HardwareCmpc(gate_for_test(tau_cap), law="per-leg",
                              z_target=ST.LIFT_HEIGHT + 0.02)
    drive_to_mpc(stand2, body_lift)
    check("a --mpc-height target starts a ramp AT the latched height",
          stand2.z_ramp is not None
          and abs(stand2.z_ramp.at(0.0).h - body_lift.z_origin) < 1e-9
          and abs(stand2.z_ramp.at(stand2.z_ramp.T).h
                  - (ST.LIFT_HEIGHT + 0.02)) < 1e-9)

    # -- 7. the trips -----------------------------------------------------------
    print("\nhw.cmpc.run: the trips")
    def fresh_mpc(**kw):
        s = RUN.HardwareCmpc(gate_for_test(tau_cap), law="per-leg", **kw)
        return s, drive_to_mpc(s, body_lift)
    s, n = fresh_mpc()
    _, _, trip = s.update(n, body_at(h_lift, R=C.rot_zyx(np.deg2rad(13.0), 0, 0)))
    check("tilt past the stop trips", trip is not None and "tilt" in trip,
          trip or "")
    s, n = fresh_mpc()
    s.update(n, body_lift)
    s.controller.solve = lambda state, t: None            # no solve at all
    s.controller._forces = np.full((4, 3), np.nan)
    _, _, trip = s.update(n + 0.004, body_lift)
    check("a NaN force out of the QP trips rather than reaching a motor",
          trip is not None and "non-finite" in trip, trip or "")
    s, n = fresh_mpc()
    original_solve = s.controller.solve
    def slow_solve(state, t):
        original_solve(state, t)
        s.controller.telemetry.solve_ms = 1e3 * cfg.SOLVE_BUDGET_S + 1.0
    s.controller.solve = slow_solve
    trips = []
    for k in range(int(cfg.SOLVE_OVER_STREAK * 250 / cfg.MPC_HZ) + 10):
        trips.append(s.update(n + 0.004 * k, body_lift)[2])
    first = next((k for k, t in enumerate(trips) if t), None)
    check("a solve over budget %d times in a row trips" % cfg.SOLVE_OVER_STREAK,
          first is not None and "budget" in trips[first]
          and first >= (cfg.SOLVE_OVER_STREAK - 1) * 250 / cfg.MPC_HZ,
          "first trip on sweep %s: %s" % (first, trips[first] if first is not None
                                           else "none"))
    s, n = fresh_mpc()
    original_solve = s.controller.solve
    def failing_solve(state, t):
        original_solve(state, t)
        s.controller.telemetry.status = "maximum iterations reached"
    s.controller.solve = failing_solve
    trips = [s.update(n + 0.004 * k, body_lift)[2]
             for k in range(int(cfg.QP_FAIL_STREAK * 250 / cfg.MPC_HZ) + 10)]
    check("a QP that stops solving %d times in a row trips" % cfg.QP_FAIL_STREAK,
          any(t and "returned" in t for t in trips),
          next((t for t in trips if t), ""))
    s, n = fresh_mpc()
    s.update(n, body_lift)
    s.replan_height(cfg.Z_MAX, seconds=0.001)
    trips = [s.update(n + 0.004 * (k + 1), body_lift)[2]
             for k in range(cfg.HEIGHT_STOP_STREAK + 2)]
    check("a sustained height error trips", any(t and "height" in t for t in trips),
          next((t for t in trips if t), ""))
    check("...but not before HEIGHT_STOP_STREAK sweeps",
          all(t is None for t in trips[:cfg.HEIGHT_STOP_STREAK - 1]))

    # -- 8. timing, for the record -----------------------------------------------
    print("\ntiming on this machine (the Pi is the number that matters)")
    s, n = fresh_mpc()
    for k in range(500):
        s.update(n + 0.004 * k, body_lift)
    print("  " + s.timing.report().replace("\n", "\n  "))
    sweep_us = 1e6 * np.percentile(s.timing.sweep, 50)
    # Against the SRB law on the SAME machine, so the gate does not encode a
    # laptop.  `hw.balance.selftest` holds the absolute 333 us line.
    from ..balance import law as BLAW
    srb = BLAW.BalanceLaw()
    srb.arm(0.0, body_lift)
    srb_samples = []
    for k in range(500):
        started = time.perf_counter()
        srb.update(0.004 * k, body_lift)
        srb_samples.append(time.perf_counter() - started)
    srb_us = 1e6 * np.percentile(srb_samples, 50)
    check("the MPC's no-solve sweep costs no more than 1.5x the SRB law's",
          sweep_us <= 1.5 * srb_us,
          "p50 %.0f us against %.0f us; one CAN slot is %.0f us"
          % (sweep_us, srb_us, 1e6 / (HS.RATE_HZ * HM.N_JOINTS)))

    # -- 9. the whole path, on hw.fake_bus ----------------------------------------
    print("\nthe whole sequence on hw.fake_bus, --auto, per-leg lift (no dynamics)")
    from ..fake_bus import FakeDriverBus
    from ..motor import motorbus
    ids = HM.motor_ids()
    gate = gate_for_test(1.0)
    # No height target: the fake robot has no dynamics and stays in the
    # crouch, so a target 125 mm up would (correctly) fire the height trip.
    stand = RUN.HardwareCmpc(gate, law="per-leg")
    log = RUN.CmpcLog()
    bus = FakeDriverBus(ids=ids)
    with motorbus.MotorBus(ids, bus=bus, dirs=HM.motor_directions()) as mb:
        armed = mb.arm(rate_hz=HS.RATE_HZ, verbose=False)
        check("the fake drivers arm", armed)
        started = time.perf_counter()
        stop = HS.run(mb, stand, rate_hz=HS.RATE_HZ, auto_s=0.3, log=log)
        elapsed = time.perf_counter() - started
    phases_seen = sorted({row["phase"] for row in log.rows})
    host_timing = stop is not None and "without a frame" in stop
    if host_timing:
        print("  NOTE  the run stopped on the CAN gap trip (%s) -- host "
              "timing, not this code; %.1f s in, phases logged %s"
              % (stop, elapsed, phases_seen))
    check("the sequence reaches done through the MPC phase%s"
          % (" [host gap trip, see NOTE]" if host_timing else ""),
          host_timing or (stop is None and stand.finished),
          "stopped: %r after %.1f s" % (stop, elapsed))
    check("the fake run logged MPC sweeps with their own columns",
          host_timing or ("mpc" in phases_seen and any(
              row.get("f_mpc") is not None for row in log.rows
              if row["phase"] == "mpc")),
          "phases %s" % phases_seen)

    # ----------------------------------------------------------------------
    print("\n%d checks, %d failed" % (_PASSES + len(_FAILURES), len(_FAILURES)))
    if _FAILURES:
        for name in _FAILURES:
            print("  FAILED: %s" % name)
        return 1
    print("the hardware MPC path computes what the simulator's does, the")
    print("stand's joint hold is intact around it, and the trips fire.  None")
    print("of this has seen a motor; `python -m hw.cmpc.run --fake --auto 1 "
          "--no-imu --law per-leg` next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
