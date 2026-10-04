"""hwsim -- `hw.stand.run`'s loop with MuJoCo standing in for can0 and the IMU.

WHAT RUNS UNMODIFIED (imported from the repo, built with the same arguments
each entry point's `main()` builds them with)
    hw.balance.sequence.StandSequence    the phase machine
    hw.balance.law.BalanceLaw            stages 1-5, the swing, the trips
    hw.balance.gait.TrotGait             the trot clock
    hw.safety.SafetyGate                 ramp / cap / slew, estop_reason
    hw.safety.TorqueReadback             commanded vs "measured" torque
    hw.balance.state.read                the measurement the law acts on

WHAT IS EMULATED, AND WITH WHICH NUMBERS
    CAN round robin   250 Hz sweeps of 12 slots (1/3000 s).  The law runs at
                      slot 0 and takes T_COMPUTE; motor k's frame goes out at
                      max(k*T_SLOT, T_COMPUTE) and lands T_BUS later.  Its
                      reply is sampled at landing and read at the next slot 0.
                      Every other sweep one motor (rotating, hw.stand's exact
                      rule) gets 0x9A instead: no command update, no encoder.
    encoder           360/65535 deg per LSB.  qd = hw.stand's 40 Hz low-pass
                      of the encoder difference.  Driver speed field = true qd.
    IMU               200 Hz packets, capture-to-arrival IMU_LATENCY, age =
                      now - arrival (hw.imu's definition).
    current loop      first-order lag TAU_I on the torque; iq quantised to
                      1/206.04 N*m and saturated at 2048 LSB (9.94 N*m).
    0xA4 position     driver-internal reference slewed at the frame's speed
                      cap, PD at POS_KP/POS_KD on the true joint state at the
                      physics rate, saturated at 9.94 N*m.  THE DRIVER'S REAL
                      POSITION GAINS ARE NOT KNOWN; these are sim.stand's.
    E-stop            any stop reason -> every motor iq = 0 (MotorBus.close),
                      and the plant keeps running so the drop is recorded.

Nothing here is a copy of the law.  If `hw.balance` changes, this runs the
changed code.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mujoco  # noqa: E402

from sim import coordinates as C, params as P  # noqa: E402
from hw import safety as SAFE, imu as IMU      # noqa: E402
from hw import fold_stand as FS, fold_trot as FT, trot as TR  # noqa: E402
from hw.balance import (config as BCFG, controller as BCTRL,   # noqa: E402
                        law as BLAW, posture as POSE, state as BSTATE,
                        gait as GAIT, sequence as SEQ)
from hw import kinematics as HK                # noqa: E402

N = C.N_JOINTS
RATE_HZ = 250.0
T_SWEEP = 1.0 / RATE_HZ
T_SLOT = T_SWEEP / N
ENC_LSB = math.radians(360.0 / 65535.0)
IQ_PER_NM = 206.04
IQ_MAX = 2048
TAU_SAT = IQ_MAX / IQ_PER_NM
QD_FILTER_HZ = 40.0                     # hw.stand.QD_FILTER_HZ


# ===========================================================================
# postures
# ===========================================================================
def fold_old() -> POSE.CrouchPose:
    """The 2026-09-16 fold: the capture regularised, rear legs TUCKED as
    captured (commit a779397).  Rebuilt here only as a comparison point."""
    return POSE.CrouchPose.from_hip_sites(
        "fold_old", POSE.regularise(POSE.FOLD_CAPTURED_Q),
        q_seed=C.unflat(POSE.FOLD_CAPTURED_Q),
        note="2026-09-16 fold, rear tucked (a779397)")


def lift_q(pose, h=BCFG.H_LIFT) -> np.ndarray:
    """(4, 3) joints with the feet at the posture's xy and the trunk bottom at h."""
    p = np.zeros((4, 3))
    p[:, :2] = pose.foot_xy
    p[:, 2] = -(h + BCFG.TRUNK_BOTTOM_OFFSET - P.FOOT_RADIUS)
    return HK.all_leg_ik(p, q_seed=pose.q)


# ===========================================================================
# the entry points, as their main() builds them
# ===========================================================================
@dataclasses.dataclass
class Entry:
    name: str
    crouch: object
    tau_cap: float
    tau_ceiling: float = SAFE.TAU_STAGED_MAX
    tau_slew: float = SAFE.DEFAULT_TAU_SLEW_NM_S
    overspeed_trip: bool = True
    latch: bool = bool(BCFG.SETPOINT_DYNAMIC)
    tilt_stop: float = float(BCFG.TILT_STOP_DEG)
    track_stop: float = float(np.rad2deg(BCFG.TRACK_STOP_RAD))
    roll_gains: tuple | None = None
    swing: str = "cartesian"
    gait_period: float | None = None
    step_to: object = None
    step_period: float | None = None
    # knobs that are NOT on any entry point, for experiments only
    gait_kw: dict = dataclasses.field(default_factory=dict)
    law_kw: dict = dataclasses.field(default_factory=dict)
    limits: object = None            # SafetyGate(limits=...), None = soft_limits()

    def replace(self, **kw) -> "Entry":
        return dataclasses.replace(self, **kw)

    def build(self):
        gate = SAFE.SafetyGate(self.tau_cap, ceiling=self.tau_ceiling,
                               tau_slew=self.tau_slew,
                               overspeed_trip=self.overspeed_trip,
                               limits=self.limits)
        gains = BCTRL.BalanceGains()
        if self.roll_gains is not None:
            gains.kp_att[0], gains.kd_att[0] = self.roll_gains
        kw = dict(gains=gains, rise_s=BCFG.T_RISE, h_lift=BCFG.H_LIFT,
                  gravity_legs_per_sweep=4, mu=BCFG.MU,
                  foot_xy=self.crouch.foot_xy, dynamic_setpoint=self.latch,
                  tilt_stop_deg=self.tilt_stop, track_stop_deg=self.track_stop,
                  srb=self.crouch.srb, swing=self.swing)
        kw.update(self.law_kw)
        balance = BLAW.BalanceLaw(**kw)
        gait = step_gait = None
        if self.gait_period is not None:
            gait = GAIT.TrotGait(period=self.gait_period, **self.gait_kw)
            if self.step_to is not None:
                step_gait = GAIT.TrotGait(period=self.step_period,
                                          **self.gait_kw)
        stand = SEQ.StandSequence(gate, law="srb", balance=balance,
                                  crouch=self.crouch, gait=gait,
                                  step_to=self.step_to, step_gait=step_gait)
        return stand, gate


def entry_stand(crouch=None, tau_cap=3.0) -> Entry:
    """`python -m hw.stand --tau-cap 3.0` (nominal unless `crouch`)."""
    return Entry("hw.stand", POSE.NOMINAL if crouch is None else crouch,
                 tau_cap)


def entry_fold_stand(tau_cap=3.0, crouch=None) -> Entry:
    """`python -m hw.fold_stand --tau-cap 3.0`."""
    return Entry("hw.fold_stand", POSE.FOLD if crouch is None else crouch,
                 tau_cap, latch=False, tilt_stop=FS.TILT_STOP_DEG,
                 track_stop=FS.TRACK_STOP_DEG, roll_gains=FS.ROLL_GAINS)


def entry_fold_trot() -> Entry:
    """`python -m hw.fold_trot`."""
    o = TR.trot_options(FT.PERIOD_S)
    return Entry("hw.fold_trot", POSE.FOLD, o["tau_cap"],
                 tau_ceiling=o["tau_ceiling"], tau_slew=o["tau_slew"],
                 overspeed_trip=o["overspeed_trip"], latch=False,
                 tilt_stop=FS.TILT_STOP_DEG, track_stop=FS.TRACK_STOP_DEG,
                 roll_gains=FS.ROLL_GAINS, swing="joint",
                 gait_period=o["gait"].period)


def entry_trot() -> Entry:
    """`python -m hw.trot`."""
    o = TR.trot_options(TR.PERIOD_S)
    return Entry("hw.trot", POSE.WIDE, o["tau_cap"],
                 tau_ceiling=o["tau_ceiling"], tau_slew=o["tau_slew"],
                 overspeed_trip=o["overspeed_trip"],
                 tilt_stop=TR.TILT_STOP_DEG, track_stop=TR.TRACK_STOP_DEG,
                 swing="cartesian", gait_period=o["gait"].period,
                 step_to=TR.STEP_FOOT_XY, step_period=TR.STEP_PERIOD_S)


# ===========================================================================
# the operator
# ===========================================================================
@dataclasses.dataclass
class Operator:
    """Presses keys the way a person running the entry point would.

    None for `trot_s` means no T.  `step_first` presses W before T (hw.trot's
    --auto order).  `park` False stops the run in HOLD instead of parking.
    `push` is (t_after_hold, duration_s, force_N_world(3,)) applied to the
    trunk -- the experiment HOLD exists for.
    """
    limp_s: float = 0.004
    settle_s: float = 1.0
    crouch_dwell_s: float = 0.5
    hold_s: float = 3.0
    trot_s: float | None = None
    step_first: bool = False
    hold_after_s: float = 2.0
    park: bool = True
    park_dwell_s: float = 0.5
    done_s: float = 0.5

    def keys(self, now, stand) -> list:
        name = stand.phase_name
        el = now - stand.t_phase
        if name == "limp":
            return ["enter"] if el >= self.limp_s else []
        if name == "settle":
            return ["enter"] if el >= self.settle_s else []
        if name == "crouch":
            return (["enter"] if el >= SEQ.ST.RAMP_POSITION + self.crouch_dwell_s
                    else [])
        if name in ("rise", "step"):
            return []
        if name == "hold":
            if (self.step_first and stand.step_to is not None
                    and stand.step_runs == 0 and el >= self.hold_s):
                return ["W"]
            wants_trot = self.trot_s is not None and stand.trot_runs == 0
            if wants_trot and el >= self.hold_s and (
                    not self.step_first or stand.step_runs > 0):
                return ["T"]
            if not wants_trot and el >= (self.hold_after_s if (
                    stand.trot_runs or stand.step_runs) else self.hold_s):
                return ["enter"] if self.park else ["stop"]
            return []
        if name == "trot":
            if not stand.trot_exit and now - stand.t_trot >= self.trot_s:
                return ["T"]
            return []
        if name == "park":
            return (["enter"] if el >= SEQ.ST.RAMP_POSITION + self.park_dwell_s
                    else [])
        if name == "done":
            return ["enter"] if el >= self.done_s else []
        return []


# ===========================================================================
# the plant and the loop
# ===========================================================================
@dataclasses.dataclass
class HwParams:
    steps_per_sweep: int = 12          # 12 -> dt = 1/3000 s
    t_compute: float = 0.5e-3          # law time at slot 0 (Pi: ~0.3-0.6 ms)
    t_bus: float = 0.15e-3             # frame on the wire + driver pickup
    tau_i: float = 0.3e-3              # driver current loop, first order
    imu_rate: float = 200.0
    imu_latency: float = 3.0e-3
    pos_kp: float = 120.0              # 0xA4 emulation (sim.stand KP_JOINT)
    pos_kd: float = 3.0
    friction: float | None = None      # foot/floor sliding friction
    frictionloss: float = 0.0          # joint Coulomb, N*m
    joint_damping: float | None = None
    post_stop_s: float = 1.5           # keep simulating after an e-stop
    trunk_mass_add: float = 0.0
    com_err_mm: tuple = (0.0, 0.0)
    #: EXPERIMENT, not the repo's behaviour: at the crouch->rise handover
    #: seed SafetyGate with the torque the drivers REPORT (the 0xA4 hold)
    #: and skip the cap ramp, so the handover is bumpless.
    prime_gate: bool = False
    #: produced torque = kt_scale * commanded (a torque-constant error).
    kt_scale: float = 1.0
    #: the plant's reflected rotor = armature_scale * the model's (DOG5's
    #: `params.ARMATURE`).  The bench's swing-ff run read DOG6's as ~1/1.7
    #: of it (`hw.balance.swing.feedforward_inertia`), never measured.
    armature_scale: float = 1.0


class HwSim:
    def __init__(self, entry: Entry, hw: HwParams | None = None):
        self.entry = entry
        self.hw = hw or HwParams()
        m = mujoco.MjModel.from_xml_path(P.XML_PATH)
        m.opt.timestep = T_SWEEP / self.hw.steps_per_sweep
        self.trunk = m.body("trunk").id
        self.trunk_geom = m.geom("trunk_box").id
        self.floor = m.geom("floor").id
        self.foot_geoms = [m.geom("foot_" + leg).id for leg in C.LEGS]
        self.foot_sites = [m.site("foot_" + leg).id for leg in C.LEGS]
        if self.hw.friction is not None:
            for g in self.foot_geoms + [self.floor]:
                m.geom_friction[g, 0] = self.hw.friction
        if self.hw.frictionloss:
            m.dof_frictionloss[6:] = self.hw.frictionloss
        if self.hw.joint_damping is not None:
            m.dof_damping[6:] = self.hw.joint_damping
        if self.hw.armature_scale != 1.0:
            m.dof_armature[6:] *= self.hw.armature_scale
        if self.hw.trunk_mass_add:
            m0 = m.body_mass[self.trunk]
            m.body_ipos[self.trunk] *= m0 / (m0 + self.hw.trunk_mass_add)
            m.body_mass[self.trunk] = m0 + self.hw.trunk_mass_add
        if any(self.hw.com_err_mm):
            frac = m.body_mass[self.trunk] / m.body_subtreemass[self.trunk]
            m.body_ipos[self.trunk, 0] += 1e-3 * self.hw.com_err_mm[0] / frac
            m.body_ipos[self.trunk, 1] += 1e-3 * self.hw.com_err_mm[1] / frac
        self.model = m
        self.data = mujoco.MjData(m)
        self.q_adr = m.sensor_adr[m.sensor("imu_quat").id]
        self.g_adr = m.sensor_adr[m.sensor("imu_gyro").id]

    # -- initial state -------------------------------------------------
    def place(self, q, z_origin, preroll_s=1.0):
        """Joints `q`, level trunk at `z_origin`, then `preroll_s` of an ideal
        joint hold so the feet and the contacts are settled.  Not logged."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = C.qpos_from_q(q, root_pos=(0.0, 0.0, z_origin + 1e-4))
        mujoco.mj_forward(m, d)
        qf = C.flat(q)
        for _ in range(int(round(preroll_s / m.opt.timestep))):
            d.ctrl[:] = np.clip(self.hw.pos_kp * (qf - d.qpos[7:])
                                - self.hw.pos_kd * d.qvel[6:], -TAU_SAT, TAU_SAT)
            mujoco.mj_step(m, d)
        d.time = 0.0
        self.tau_act = d.ctrl.copy()

    def place_limp(self, q, z_origin, preroll_s=1.0):
        """Joints `q` at `z_origin`, then `preroll_s` at ZERO torque."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = C.qpos_from_q(q, root_pos=(0.0, 0.0, z_origin + 1e-4))
        mujoco.mj_forward(m, d)
        d.ctrl[:] = 0.0
        for _ in range(int(round(preroll_s / m.opt.timestep))):
            mujoco.mj_step(m, d)
        d.time = 0.0
        self.tau_act = np.zeros(N)

    # -- truth -----------------------------------------------------------
    def _R_true(self):
        return self.data.xmat[self.trunk].reshape(3, 3).copy()

    def _contacts(self):
        """(4,) foot normal force, trunk-floor normal force, (4,) foot slip speed."""
        m, d = self.model, self.data
        foot = np.zeros(4)
        belly = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            g = {con.geom1, con.geom2}
            if self.floor not in g:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(f6[0])
            other = con.geom2 if con.geom1 == self.floor else con.geom1
            if other == self.trunk_geom:
                belly += fn
            elif other in self.foot_geoms:
                foot[self.foot_geoms.index(other)] += fn
        return foot, belly

    # -- the run ---------------------------------------------------------
    def run(self, operator: Operator, t_max: float = 60.0, push=None,
            record_frames=None, quiet=True, pre_update=None):
        """Drive the sequence until done, a stop, or `t_max`.

        `push` is a list of (t0, t1, (fx, fy, fz)) world forces on the trunk,
        times in run seconds.  `record_frames(sim, t, stand)` is called every
        sweep if given (for video/screens).
        """
        hw = self.hw
        m, d = self.model, self.data
        stand, gate = self.entry.build()
        self.stand, self.gate = stand, gate
        readback = SAFE.TorqueReadback(N)
        dt = m.opt.timestep
        S = hw.steps_per_sweep
        alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * N * T_SLOT)

        # the reply records, seeded with the settled state
        rec_q = np.round(d.qpos[7:] / ENC_LSB) * ENC_LSB
        rec_qd = d.qvel[6:].copy()
        rec_iq = np.round(self.tau_act * IQ_PER_NM)
        # driver state
        drv_mode = np.array(["pos"] * N, dtype=object)   # preroll held position
        drv_target = d.qpos[7:].copy()
        drv_ref = d.qpos[7:].copy()
        drv_vmax = np.full(N, np.inf)
        drv_iq = np.round(self.tau_act * IQ_PER_NM)
        tau_act = self.tau_act.copy()
        a_i = 1.0 - math.exp(-dt / hw.tau_i) if hw.tau_i > 0 else 1.0

        # IMU packets: (arrival, R, omega_b)
        imu_period = 1.0 / hw.imu_rate
        next_capture = 0.0
        packets = []
        R0 = self._R_true()
        w0 = d.sensordata[self.g_adr:self.g_adr + 3].copy()
        packets.append((-1e-3, R0, w0))

        q_prev = t_prev = None
        qd_ctrl = np.zeros(N)
        sweep = 0
        stop = None
        t_stop = None
        log = []
        notices = []
        stand.t_phase = 0.0
        n_sweeps = int(round(t_max / T_SWEEP))
        push = push or []

        for n in range(n_sweeps):
            now = d.time
            # ---- slot 0: sensors, operator, law ------------------------
            q = rec_q.copy()
            qd_driver = rec_qd.copy()
            if q_prev is not None and now > t_prev:
                qd_ctrl += alpha_qd * ((q - q_prev) / (now - t_prev) - qd_ctrl)
            q_prev, t_prev = q, now
            last = max(i for i, pk in enumerate(packets)
                       if pk[0] <= now + 1e-12 or i == 0)
            ta, Rm, wm = packets[last]
            roll, pitch, yaw = C.zyx_from_rot(Rm)
            orient = IMU.TrunkOrientation(R=Rm, omega_b=wm, roll=roll,
                                          pitch=pitch, yaw=yaw,
                                          age_s=max(0.0, now - ta))
            packets = packets[last:]          # keep the newest arrived and later
            body = BSTATE.read(q, qd_ctrl, orient, srb=stand.crouch.srb)
            stand.body = body

            mode, values = "keepalive", None
            if stop is None:
                for key in operator.keys(now, stand):
                    if key == "T":
                        notices.append((now, stand.toggle_trot(now)))
                    elif key == "W":
                        notices.append((now, stand.toggle_step(now)))
                    elif key == "stop":
                        stop, t_stop = "operator stop in hold", now
                    elif key == "enter":
                        if stand.finished:
                            stop, t_stop = "clean", now
                        else:
                            before = stand.phase_name
                            refused = stand.advance(now, q)
                            if refused:
                                notices.append((now, "ENTER refused: " + refused))
                            if (hw.prime_gate and before != "rise"
                                    and stand.phase_name == "rise"):
                                gate.started_at = now - gate.ramp_s
                                gate.previous_tau = rec_iq / IQ_PER_NM
                                gate.last_time = now
                if stop is None:
                    if pre_update is not None:
                        pre_update(self, now, stand, body)
                    mode, values, trip = stand.update(now, body)
                    note = stand.take_notice()
                    if note:
                        notices.append((now, note))
                    if trip:
                        stop, t_stop = trip, now
                if stop is None:
                    reason = gate.estop_reason(q, qd_driver, now)
                    if reason:
                        stop, t_stop = reason, now
                if stop is None:
                    tau_meas = rec_iq / IQ_PER_NM
                    reason = readback.reason(stand.tau, tau_meas,
                                             live=(mode == "torque"))
                    if reason:
                        stop, t_stop = reason, now
                if stop is not None:
                    mode, values = "keepalive", None
                    if not quiet:
                        print("[%.3f] STOP: %s" % (now, stop))
            if stop is not None and now - t_stop >= hw.post_stop_s:
                break

            # ---- truth + log -------------------------------------------
            R_true = self._R_true()
            rr, pp, yy = C.zyx_from_rot(R_true)
            foot_fn, belly_fn = self._contacts()
            out = stand.out if mode == "torque" else None
            row = dict(
                t=now, phase=stand.phase_name if stop is None else "STOP",
                mode=mode, q=q, qd=qd_ctrl.copy(), q_true=d.qpos[7:].copy(),
                qd_true=d.qvel[6:].copy(), tau_act=tau_act.copy(),
                tau_meas=rec_iq / IQ_PER_NM,
                tau_cmd=(stand.tau.copy() if mode == "torque" else
                         np.full(N, np.nan)),
                tau_req=(stand.tau_request.copy() if mode == "torque" else
                         np.full(N, np.nan)),
                q_des=(stand.q_des.copy() if mode in ("position", "torque")
                       else np.full(N, np.nan)),
                h_meas=body.h, h_cmd=BSTATE.origin_to_height(stand.h_cmd)
                if np.isfinite(stand.h_cmd) else np.nan,
                roll=rr, pitch=pp, yaw=yy,
                roll_m=body.roll, pitch_m=body.pitch,
                z_trunk=d.xpos[self.trunk][2], xy_trunk=d.xpos[self.trunk][:2].copy(),
                foot_w=np.array([d.site_xpos[s] for s in self.foot_sites]),
                foot_fn=foot_fn, belly_fn=belly_fn,
                fz=(out.allocation.fz.copy() if out is not None else
                    np.full(4, np.nan)),
                b_d=(out.wrench.b_d.copy() if out is not None else
                     np.full(6, np.nan)),
                residual=(out.allocation.residual.copy() if out is not None
                          else np.full(6, np.nan)),
                contact_w=(out.contact.copy() if out is not None and
                           out.contact is not None else np.full(4, np.nan)),
                swing_s=(out.swing_s.copy() if out is not None and
                         out.swing_s is not None else np.full(4, np.nan)),
                cap=(gate.cap_now(now) if gate.started_at is not None else 0.0),
                x_b=body.x_b.copy(),
                com_w=d.subtree_com[self.trunk].copy(),
                qpos=d.qpos.copy(),
            )
            log.append(row)
            if record_frames is not None:
                record_frames(self, now, stand)

            # ---- frames ------------------------------------------------
            sweep += 1
            land = np.empty(N, dtype=int)
            status = np.zeros(N, dtype=bool)
            for k in range(N):
                send = max(k * T_SLOT, hw.t_compute + 2e-5 * k if k < 2 else
                           k * T_SLOT)
                land[k] = min(S - 1, int(math.ceil((send + hw.t_bus) / dt - 1e-9)))
                status[k] = (sweep % 2 == 0 and k == (sweep // 2) % N
                             and stop is None)
            new_mode = np.array(["tq"] * N, dtype=object)
            new_val = np.zeros(N)
            new_vmax = np.full(N, np.inf)
            if mode == "position":
                new_mode[:] = "pos"
                new_val[:] = values
                new_vmax[:] = np.deg2rad(stand.max_dps / P.GEAR_RATIO)
            elif mode == "torque":
                new_val[:] = np.clip(np.round(values * IQ_PER_NM), -IQ_MAX, IQ_MAX)
            applied = np.zeros(N, dtype=bool)
            replied = np.zeros(N, dtype=bool)

            # ---- physics through the sweep ---------------------------------
            for s in range(S):
                t = d.time
                due = (~applied) & (land <= s)
                if due.any():
                    for k in np.flatnonzero(due):
                        applied[k] = True
                        if status[k]:
                            continue          # 0x9A: nothing changes
                        if new_mode[k] == "pos":
                            if drv_mode[k] != "pos":
                                drv_ref[k] = d.qpos[7 + k]
                            drv_mode[k] = "pos"
                            drv_target[k] = new_val[k]
                            drv_vmax[k] = new_vmax[k]
                        else:
                            drv_mode[k] = "tq"
                            drv_iq[k] = new_val[k]
                qj = d.qpos[7:]
                qdj = d.qvel[6:]
                pos = drv_mode == "pos"
                tau_des = drv_iq / IQ_PER_NM
                if pos.any():
                    step = np.clip(drv_target - drv_ref, -drv_vmax * dt,
                                   drv_vmax * dt)
                    drv_ref = np.where(pos, drv_ref + step, drv_ref)
                    pd = hw.pos_kp * (drv_ref - qj) - hw.pos_kd * qdj
                    pd = np.clip(np.round(pd * IQ_PER_NM), -IQ_MAX, IQ_MAX) / IQ_PER_NM
                    tau_des = np.where(pos, pd, tau_des)
                tau_act += a_i * (tau_des - tau_act)
                # replies: sampled when the frame lands (not for 0x9A)
                rep = due & (~status)
                if rep.any():
                    rec_q[rep] = np.round(qj[rep] / ENC_LSB) * ENC_LSB
                    rec_qd[rep] = qdj[rep]
                    rec_iq[rep] = np.round(tau_act[rep] * IQ_PER_NM)
                d.ctrl[:] = hw.kt_scale * tau_act
                # IMU
                if t >= next_capture - 1e-12:
                    R = np.zeros(9)
                    mujoco.mju_quat2Mat(R, d.sensordata[self.q_adr:self.q_adr + 4])
                    packets.append((t + hw.imu_latency, R.reshape(3, 3),
                                    d.sensordata[self.g_adr:self.g_adr + 3].copy()))
                    next_capture += imu_period
                # external push
                d.xfrc_applied[self.trunk, :] = 0.0
                for (p0, p1, f) in push:
                    if p0 <= t < p1:
                        d.xfrc_applied[self.trunk, :3] += np.asarray(f, float)
                mujoco.mj_step(m, d)

        self.log = log
        self.notices = notices
        self.stop = stop
        self.t_stop = t_stop
        return self.result()

    # -- packing ------------------------------------------------------------
    def result(self) -> dict:
        log = self.log
        out = {"phase": np.array([r["phase"] for r in log]),
               "mode": np.array([r["mode"] for r in log])}
        for key in log[0]:
            if key in ("phase", "mode"):
                continue
            out[key] = np.array([r[key] for r in log], dtype=float)
        out["stop"] = self.stop
        out["t_stop"] = self.t_stop
        out["notices"] = self.notices
        out["entry"] = self.entry.name
        out["posture"] = self.entry.crouch.name
        return out


def save(res: dict, path: str) -> None:
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, meta=np.array(repr(meta)), **arrays)


def summarize(res: dict) -> str:
    lines = ["%s from %s: stop=%r at %s" % (res["entry"], res["posture"],
                                             res["stop"], res["t_stop"])]
    ph = res["phase"]
    for name in ("settle", "crouch", "rise", "hold", "trot", "step", "park",
                 "done", "STOP"):
        sel = ph == name
        if not sel.any():
            continue
        tau = np.abs(res["tau_act"][sel])
        lines.append("  %-6s %6.2f s  |tau| peak %5.2f (%s)  mean-max %5.2f  "
                     "roll %+5.1f..%+5.1f pitch %+5.1f..%+5.1f  h %5.1f..%5.1f mm  belly %.0f N"
                     % (name, sel.sum() * T_SWEEP, tau.max(),
                        C.JOINT_NAMES[int(np.argmax(tau.max(axis=0)))],
                        tau.mean(axis=0).max(),
                        np.degrees(res["roll"][sel].min()),
                        np.degrees(res["roll"][sel].max()),
                        np.degrees(res["pitch"][sel].min()),
                        np.degrees(res["pitch"][sel].max()),
                        1e3 * (res["z_trunk"][sel].min() - BCFG.TRUNK_BOTTOM_OFFSET),
                        1e3 * (res["z_trunk"][sel].max() - BCFG.TRUNK_BOTTOM_OFFSET),
                        res["belly_fn"][sel].max()))
    return "\n".join(lines)


def trot_metrics(res) -> dict:
    """Did the trot survive?  Only the time BEFORE any stop counts: the drop
    after an e-stop is the e-stop's, not the gait's."""
    ph = res["phase"]
    trot = ph == "trot"
    out = {"stop": res["stop"], "t_stop": res["t_stop"]}
    tripped = res["stop"] not in (None, "operator stop in hold", "clean")
    if not trot.any():
        out.update(trot_s=0.0, max_tilt_trot=float("nan"), fell=True,
                   t_tilt15_after_T=None)
        return out
    t = res["t"]
    idx = np.flatnonzero(trot)
    i0 = int(idx[0])
    live = np.flatnonzero(ph != "STOP")
    i1 = int(live[-1]) + 1
    tilt = np.degrees(np.maximum(np.abs(res["roll"]), np.abs(res["pitch"])))
    lost = np.flatnonzero(tilt[i0:i1] > 15.0)
    out["trot_s"] = round(float(trot.sum() * T_SWEEP), 2)
    out["max_tilt_trot"] = round(float(tilt[trot].max()), 2)
    out["t_tilt15_after_T"] = (None if not lost.size else
                               round(float(t[i0 + lost[0]] - t[i0]), 2))
    out["fell"] = bool(lost.size) or tripped
    out["peak_tau_trot"] = round(float(np.abs(res["tau_act"][trot]).max()), 2)
    return out



if __name__ == "__main__":
    t0 = time.time()
    sim = HwSim(entry_fold_stand())
    pose = sim.entry.crouch
    sim.place(pose.q, pose.z_origin)
    res = sim.run(Operator(), t_max=25.0, quiet=False)
    print(summarize(res))
    for t, note in res["notices"]:
        print("  [%.2f] %s" % (t, note.splitlines()[0][:120]))
    print("wall %.1f s" % (time.time() - t0))
