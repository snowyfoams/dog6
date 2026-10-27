"""hwsim_leg -- `hw.swing_bench.run`'s loop with MuJoCo's hung-up RR leg in for can0.

    from doc.swing_height import hwsim_leg as HL
    res = HL.run_bench(HL.Setup(swing_ff=True), swings=30)

The robot hung up, one leg swinging: what `python -m hw.swing_bench --legs RR`
does on the bench, with the study's swing settings (doc/gait_study, the
2026-10-21 line: `Setup` below).  The plant is `model/leg_RR.xml`.

WHAT RUNS UNMODIFIED (imported from the repo)
    hw.swing_bench.SwingBench    limp -> pose -> hold -> swing; the arc, the
                                 Cartesian PD, `--swing-ff`, the swing report
    hw.balance.gait.TrotGait     the clock
    hw.safety.SafetyGate         ramp / cap / clip / slew, estop_reason
    hw.balance.state.read        FK and Jacobian, the measurement the law reads
For an in-place arc the bench's swing torque IS the walk's
(`swing_control.swing_control_torque`, the law the study flew): the PD is the
same expression, and `swing.swing_feedforward` matches
`swing_control.inertial_feedforward` to 1e-13 N*m along the 20 mm arc at the
study's pose (study.py prints the check).

ONE THING IS SET FROM OUTSIDE: the held posture.  The bench knows `lift` and
`fold`; the study flew FOLD2 at 160 mm, so `bench.q_pose` and `bench.rest` are
replaced with that stance (`fold2_pose`) right after construction -- the same
arithmetic `swing_bench.bench_pose` does for `fold`, on FOLD2's feet.

WHAT IS EMULATED -- doc/crouch_trot/hwsim.py's chain and its numbers
    CAN round robin   one sweep = 12 slots; the law runs at slot 0 and takes
                      `t_compute`; motor k's frame goes out at max(k slot,
                      t_compute) and lands `t_bus` later; its reply is sampled
                      at landing and read at the next slot 0.  RR is motors
                      9-11: its command lands 4-5 ms after the law ran.  Every
                      other sweep one motor (rotating) gets 0x9A instead: no
                      command update, no encoder.
    sweep period      `sweep_s`, default 5.51 ms -- what the study's logs
                      measured on the robot (p1_inplace_1; 5.54 in p1_lat_1),
                      not the nominal 4 ms.  The qd filter's alpha is the
                      code's, from the nominal 250 Hz.
    encoder           360/65535 deg per LSB; qd = hw.stand's 40 Hz low-pass of
                      the encoder difference.
    current loop      first-order lag `tau_i`; iq quantised to 1/206.04 N*m,
                      saturated at 2048 LSB (9.94 N*m).
    0xA4 position     the pose phase: driver reference slewed at the frame's
                      speed cap, PD at pos_kp / pos_kd on the true joint
                      state.  The driver's real gains are not known; these
                      are sim.stand's.
`Chain(ideal=True)` turns all of it off -- the law reads the true q, qd at
the sweep and its torque reaches the joint at once -- as the reference that
says what the chain itself costs.

The other nine joints are held exactly at the posture: with the trunk
pinned the legs do not couple, and the bench holds them still.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mujoco                                   # noqa: E402

from sim import coordinates as C, params as P   # noqa: E402
from hw import imu as IMU, kinematics as HK     # noqa: E402
from hw import safety as SAFE                   # noqa: E402
from hw import swing_bench as SB                # noqa: E402
from hw.balance import config as BCFG           # noqa: E402
from hw.balance import gait as GAIT             # noqa: E402
from hw.balance import posture as POSE          # noqa: E402
from hw.balance import state as BSTATE          # noqa: E402
from hw.balance import swing as SWING           # noqa: E402

LEG = 3                                         # RR
SL = slice(3 * LEG, 3 * LEG + 3)
N = C.N_JOINTS
MODEL = os.path.join(HERE, "model", "leg_RR.xml")
ENC_LSB = math.radians(360.0 / 65535.0)
IQ_PER_NM = 206.04
IQ_MAX = 2048
QD_FILTER_HZ = 40.0                             # hw.stand.QD_FILTER_HZ
RATE_HZ = 250.0                                 # hw.stand.RATE_HZ, nominal


def fold2_pose(height: float = 0.160) -> np.ndarray:
    """(4, 3) joints: FOLD2's feet at their hip-frame xy, the trunk bottom
    `height` above the floor -- `swing_bench.bench_pose("fold")`'s arithmetic
    on `posture.FOLD2`, the stance the study's walk held."""
    p = np.zeros((C.N_LEGS, 3))
    p[:, :2] = POSE.FOLD2.foot_xy
    p[:, 2] = -(height + BCFG.TRUNK_BOTTOM_OFFSET - P.FOOT_RADIUS)
    q = HK.all_leg_ik(p, q_seed=POSE.FOLD2.q)
    miss = np.abs(np.array([HK.foot_position_hip(i, q[i]) for i in range(C.N_LEGS)])
                  - p).max()
    if not miss < 1e-6:
        raise ValueError("FOLD2 at %.0f mm: IK misses by %.2f mm" % (1e3 * height, 1e3 * miss))
    return q


@dataclasses.dataclass
class Setup:
    """The bench's flags.  Defaults: the study's swing (doc/gait_study, the
    2026-10-21 line `hw.fold2_walk --duty 0.8 --contact-ramp 0.15
    --tau-slew 300`), clock as the logs measured it."""
    swing_ff: bool = False
    swing_height: float = 0.020                 # m, fold2_trot.SWING_HEIGHT
    period: float = 0.80                        # s, the logs' (code default 0.6)
    duty: float = 0.80
    ramp: float = 0.15
    tau_slew: float = 300.0                     # N*m/s
    tau_cap: float = SAFE.TAU_HARD_NM           # 9 N*m, the study's clip
    kp: tuple = tuple(BCFG.KP_SWING_WALK)       # 150 150 400 N/m
    kd: tuple = tuple(BCFG.KD_SWING_WALK)       # 6 6 40 N s/m
    ff_inertia: object = None                   # None: SWING.JOINT_INERTIA_M0
    height: float = 0.160                       # m, FOLD2's hold

    def label(self) -> str:
        return "FF %s" % ("on" if self.swing_ff else "off")


@dataclasses.dataclass
class Chain:
    """The actuation chain between the law and the joint."""
    ideal: bool = False
    sweep_s: float = 0.00551
    steps_per_sweep: int = 12
    t_compute: float = 0.5e-3
    t_bus: float = 0.15e-3
    tau_i: float = 0.3e-3
    pos_kp: float = 120.0
    pos_kd: float = 3.0
    #: hw.swing_bench.run's 0x9A every other sweep; False only to test what it does
    status_frames: bool = True
    #: plant changes, for sensitivities only
    damping: float = 0.0                        # N*m*s/rad on the three hinges
    armature_scale: float = 1.0


def build(setup: Setup, legs=(LEG,), swings: int = 0):
    """``(bench, gate)`` as `swing_bench.main` builds them from `setup`."""
    gait = GAIT.TrotGait(period=setup.period, duty=setup.duty, ramp=setup.ramp,
                         settle_s=0.0, settle_every=BCFG.SETTLE_EVERY)
    gate = SAFE.SafetyGate(setup.tau_cap, unconfirmed_reason="MuJoCo hwsim_leg",
                           ceiling=SAFE.TAU_HARD_NM, tau_slew=setup.tau_slew,
                           overspeed_trip=False)
    bench = SB.SwingBench(gate, gait, legs=list(legs), swing="cartesian",
                          swing_height=setup.swing_height, swing_ff=setup.swing_ff,
                          kp_swing=np.asarray(setup.kp, float),
                          kd_swing=np.asarray(setup.kd, float),
                          ff_inertia=setup.ff_inertia, swings=swings, posture="lift")
    # the study's stance, not the bench's lift -- the module docstring
    bench.q_pose = fold2_pose(setup.height)
    bench.rest = np.array([HK.foot_position(i, bench.q_pose[i]) for i in range(C.N_LEGS)])
    return bench, gate


def load_plant(chain: Chain, sweep_s: float):
    m = mujoco.MjModel.from_xml_path(MODEL)
    m.opt.timestep = sweep_s / chain.steps_per_sweep
    m.dof_damping[:] = chain.damping
    m.dof_armature[:] *= chain.armature_scale
    return m, mujoco.MjData(m)


def run_bench(setup: Setup, chain: Chain | None = None, swings: int = 30,
              hold_s: float = 2.0, tail_s: float = 1.0, quiet: bool = True) -> dict:
    """One bench run: ENTER at once (limp -> pose), the pose ramp and the
    hold's ramp, `hold_s` of hold, ENTER (swing), `swings` swings, the
    bench's own stop at the next four-foot window, `tail_s` of hold."""
    chain = chain or Chain()
    T = chain.sweep_s
    S = chain.steps_per_sweep
    slot = T / N
    bench, gate = build(setup, swings=swings)
    m, d = load_plant(chain, T)
    dt = m.opt.timestep
    q_full = C.flat(bench.q_pose).astype(float)
    d.qpos[:] = q_full[SL]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    site = m.site("foot_RR").id
    level = IMU.TrunkOrientation.level()
    readback = SAFE.TorqueReadback(N)
    # hw.swing_bench.run's filter constant: from the NOMINAL slot
    alpha_qd = 1.0 - np.exp(-2.0 * np.pi * QD_FILTER_HZ * N * (1.0 / (RATE_HZ * N)))
    a_i = 1.0 - math.exp(-dt / chain.tau_i) if chain.tau_i > 0 else 1.0

    # the reply records and the drivers (12 joints; only RR's three are real)
    rec_q = np.round(q_full / ENC_LSB) * ENC_LSB
    rec_iq = np.zeros(N)
    drv_mode = np.array(["tq"] * N, dtype=object)
    drv_target = q_full.copy()
    drv_ref = q_full.copy()
    drv_vmax = np.full(N, np.inf)
    drv_iq = np.zeros(N)
    tau_act = np.zeros(3)

    q_prev = t_prev = None
    qd_ctrl = np.zeros(N)
    lines, stop = [], None
    log = []
    t_swing0 = None
    t_end = None
    sweep = 0
    bench.t_phase = 0.0
    while True:
        now = d.time
        # ---- slot 0: sensors, operator, law ----------------------------------
        if chain.ideal:
            q = q_full.copy()
            q[SL] = d.qpos
            qd_ctrl = np.zeros(N)
            qd_ctrl[SL] = d.qvel
        else:
            q = rec_q.copy()
            if q_prev is not None and now > t_prev:
                qd_ctrl += alpha_qd * ((q - q_prev) / (now - t_prev) - qd_ctrl)
            q_prev, t_prev = q, now
        body = BSTATE.read(q, qd_ctrl, level)
        # the operator: ENTER at once, ENTER after `hold_s` of hold
        if bench.phase == "limp":
            lines.append("[%.3f] %s" % (now, bench.enter(now, q)))
        elif bench.phase == "hold" and not bench.done and now - bench.t_phase >= hold_s:
            lines.append("[%.3f] %s" % (now, bench.enter(now, q)))
            t_swing0 = now
        elif bench.phase == "hold" and bench.done and t_end is None:
            t_end = now + tail_s
        if t_end is not None and now >= t_end:
            break
        mode, values = bench.update(now, body)
        if bench.trip:
            stop = bench.trip
            break
        note = bench.take_notice()
        if note:
            lines.append("[%.3f] %s" % (now, note))
        tau_sent = np.zeros(N)
        if mode == "torque":
            reason = gate.estop_reason(q, qd_ctrl, now)
            if reason:
                stop = reason
                break
            tau_sent = gate.apply(values, q, now)
            for line in bench.observe(now, body, tau_sent):
                lines.append("[%.3f] %s" % (now, line))
            reason = readback.reason(tau_sent, rec_iq / IQ_PER_NM, live=not chain.ideal)
            if reason and not chain.ideal:
                stop = reason
                break
        if not quiet and lines:
            print("\n".join(lines[-1:]))

        # ---- the log: what the robot's --log keeps, and the truth ------------
        log.append(dict(
            t=now, phase=bench.phase, mode=mode, q=q[SL].copy(), qd=qd_ctrl[SL].copy(),
            q_true=d.qpos.copy(), qd_true=d.qvel.copy(),
            x_b=body.x_b[LEG].copy(), x_true=d.site_xpos[site].copy(),
            p_swing=bench.p_swing[LEG].copy(), swing_s=float(bench.swing_s[LEG]),
            contact=bool(bench.contact[LEG]),
            tau_req=(bench.tau[SL].copy() if mode == "torque" else np.full(3, np.nan)),
            tau_cmd=tau_sent[SL].copy(), tau_ff=bench.tau_ff[SL].copy(),
            tau_pd=bench.tau_pd[SL].copy(), tau_act=tau_act.copy(),
            tau_meas=rec_iq[SL] / IQ_PER_NM,
            cap=(gate.cap_now(now) if gate.started_at is not None else 0.0)))

        # ---- frames: when each motor's command lands -------------------------
        sweep += 1
        land = np.empty(N, dtype=int)
        status = np.zeros(N, dtype=bool)
        for k in range(N):
            send = max(k * slot, chain.t_compute + 2e-5 * k if k < 2 else k * slot)
            land[k] = min(S - 1, int(math.ceil((send + chain.t_bus) / dt - 1e-9)))
            status[k] = (not chain.ideal and chain.status_frames
                         and sweep % 2 == 0 and k == (sweep // 2) % N)
        if chain.ideal:
            land[:] = 0
        new_mode = np.array(["tq"] * N, dtype=object)
        new_val = np.zeros(N)
        new_vmax = np.full(N, np.inf)
        if mode == "position":
            new_mode[:] = "pos"
            new_val[:] = values
            new_vmax[:] = np.deg2rad(bench.max_dps / P.GEAR_RATIO)
        elif mode == "torque":
            new_val[:] = (tau_sent * IQ_PER_NM if chain.ideal else
                          np.clip(np.round(tau_sent * IQ_PER_NM), -IQ_MAX, IQ_MAX))
        applied = np.zeros(N, dtype=bool)

        # ---- physics through the sweep ---------------------------------------
        for s in range(S):
            due = (~applied) & (land <= s)
            for k in np.flatnonzero(due):
                applied[k] = True
                if status[k]:
                    continue                    # 0x9A: nothing changes
                if new_mode[k] == "pos":
                    if drv_mode[k] != "pos":
                        drv_ref[k] = d.qpos[k - 3 * LEG] if SL.start <= k < SL.stop else q_full[k]
                    drv_mode[k] = "pos"
                    drv_target[k] = new_val[k]
                    drv_vmax[k] = new_vmax[k]
                else:
                    drv_mode[k] = "tq"
                    drv_iq[k] = new_val[k]
            qj, qdj = d.qpos, d.qvel
            pos = drv_mode[SL] == "pos"
            tau_des = drv_iq[SL] / IQ_PER_NM
            if pos.any():
                step = np.clip(drv_target[SL] - drv_ref[SL], -drv_vmax[SL] * dt,
                               drv_vmax[SL] * dt)
                drv_ref[SL] = np.where(pos, drv_ref[SL] + step, drv_ref[SL])
                pd = chain.pos_kp * (drv_ref[SL] - qj) - chain.pos_kd * qdj
                pd = np.clip(np.round(pd * IQ_PER_NM), -IQ_MAX, IQ_MAX) / IQ_PER_NM
                tau_des = np.where(pos, pd, tau_des)
            tau_act = tau_des.copy() if chain.ideal else tau_act + a_i * (tau_des - tau_act)
            rep = due[SL] & (~status[SL])
            if rep.any() and not chain.ideal:
                rq = rec_q[SL]
                rq[rep] = np.round(qj[rep] / ENC_LSB) * ENC_LSB
                rec_q[SL] = rq
                ri = rec_iq[SL]
                ri[rep] = np.round(tau_act[rep] * IQ_PER_NM)
                rec_iq[SL] = ri
            d.ctrl[:] = tau_act
            mujoco.mj_step(m, d)
        if now > 120.0:
            stop = "time limit"
            break

    out = {"phase": np.array([r["phase"] for r in log]),
           "mode": np.array([r["mode"] for r in log])}
    for key in log[0]:
        if key in ("phase", "mode"):
            continue
        out[key] = np.array([r[key] for r in log], dtype=float)
    out.update(lines=lines, stop=stop, t_swing0=t_swing0, rest=bench.rest[LEG].copy(),
               q_pose=bench.q_pose[LEG].copy(), report=bench.report(),
               setup=dataclasses.asdict(setup), chain=dataclasses.asdict(chain),
               swing_duration=bench.gait.swing_duration, period=bench.gait.period)
    return out


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def save(res: dict, path: str) -> None:
    """Arrays as arrays; everything else (lines, setup, chain, ...) as one
    JSON string under `meta` -- `load` reads it back."""
    arrays = {k: v for k, v in res.items()
              if isinstance(v, np.ndarray) and v.dtype != object}
    meta = {k: v for k, v in res.items() if k not in arrays}
    np.savez_compressed(path, meta=np.array(json.dumps(_jsonable(meta))), **arrays)


def load(path: str) -> dict:
    raw = np.load(path, allow_pickle=False)
    res = {k: raw[k] for k in raw.files if k != "meta"}
    res.update(json.loads(str(raw["meta"])))
    for k in ("rest", "q_pose"):
        res[k] = np.asarray(res[k], float)
    return res


if __name__ == "__main__":
    for ff in (False, True):
        r = run_bench(Setup(swing_ff=ff), swings=6)
        print("\n".join(r["lines"][-3:]))
        print(r["report"], "stop:", r["stop"])
