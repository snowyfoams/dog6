"""Candidate fixes for the trot, tried in simulation WITHOUT touching the repo.

Each variant is the committed entry point plus one or more of:

    land_tc     tilt-compensated landing: the swing arc's landing point is
                moved along trunk z so it lies on the plane of the stance
                feet (R from the IMU, feet from the encoders) instead of at a
                fixed trunk-frame height.  Monkeypatch of
                hw.balance.swing.swing_reference, driven from a pre-update
                hook -- nothing the robot does not already measure.
    kp_joint    the fold's joint-swing PD gains (config.KP/KD_SWING_JOINT)
    period      the gait period
    roll        roll gains (kp, kd)
    center      the stance re-centred on the TRUE CoM (feet shifted by the
                plant's CoM error, SRB c^b corrected by the same amount): the
                upper bound of "calibrate the CoM, then step the feet under it"

    python fixes.py [group ...]      groups: fold, wide_W, wide_com
Writes data/fixes_<group>.json.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hwsim as H  # noqa: E402
from hwsim import trot_metrics as metrics  # noqa: E402
from sim import coordinates as C  # noqa: E402
from hw.balance import config as BCFG, posture as POSE, swing as SW  # noqa: E402

# ---------------------------------------------------------------------------
# tilt-compensated landing
# ---------------------------------------------------------------------------
_ORIG_SWING_REFERENCE = SW.swing_reference
_DZ = np.zeros(4)          # landing correction per leg, trunk z, m
_DZ0 = np.zeros(4)         # start correction (Cartesian swing), latched
_SWINGING = np.zeros(4, dtype=bool)
_STATE = {"on": False, "start": False, "place": 0.0, "latch_start": False}
_DXY = np.zeros((4, 2))    # placement offset per leg, trunk xy, m
_START = np.full((4, 3), np.nan)   # measured foot at liftoff, trunk frame
_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _leg_of(p) -> int:
    key = (1 if p[0] >= 0 else -1, 1 if p[1] >= 0 else -1)
    return _SIGNS.index(key)


def _patched_swing_reference(rest_b, progress, duration,
                             height=BCFG.SWING_HEIGHT, land_b=None):
    if not (_STATE["on"] or _STATE["place"]):
        return _ORIG_SWING_REFERENCE(rest_b, progress, duration, height,
                                     land_b=land_b)
    i = _leg_of(rest_b)
    start = np.array(rest_b, dtype=float)
    land = np.array(rest_b if land_b is None else land_b, dtype=float)
    land[2] += _DZ[i]
    land[:2] += _DXY[i]
    if _STATE["latch_start"] and np.isfinite(_START[i, 0]):
        start = _START[i].copy()
    elif _STATE["start"]:
        start[2] += _DZ0[i]
    return _ORIG_SWING_REFERENCE(start, progress, duration, height,
                                 land_b=land)


SW.swing_reference = _patched_swing_reference

_ORIG_SWING_TORQUE = SW.swing_torque
_HYBRID = {"on": False, "kp": 30.0, "kd": 0.8}


def _patched_swing_torque(state, leg, p_ref, v_ref, kp=None, kd=None):
    """Cartesian impedance, plus a JOINT PD on abduction toward the IK of the
    reference -- the fold's joint swing kept only on the axis it was for."""
    tau = _ORIG_SWING_TORQUE(state, leg, p_ref, v_ref, kp=kp, kd=kd)
    if _HYBRID["on"]:
        from hw import kinematics as HK
        from sim import params as P
        q = C.unflat(state.q)[leg]
        qd = C.unflat(state.qd)[leg]
        q_ref = HK.leg_ik(leg, np.asarray(p_ref, float) - P.HIP_OFFSET[leg], q_seed=q)
        tau = tau.copy()
        tau[0] += _HYBRID["kp"] * (q_ref[0] - q[0]) - _HYBRID["kd"] * qd[0]
    return tau


SW.swing_torque = _patched_swing_torque


def place_hook(sim, now, stand, body):
    """Raibert-style placement, in-place trot: land each swing foot
    k * v ahead, v = trunk horizontal velocity RELATIVE TO THE STANCE FEET
    (leg odometry: encoders + gyro, nothing else), plus land_tc if on."""
    name = stand.phase_name
    gait = (stand.gait if name == "trot" else
            stand.step_gait if name == "step" else None)
    if gait is None:
        _DXY[:] = 0.0
        _START[:] = np.nan
        _SWINGING[:] = False
        _DZ[:] = 0.0
        return
    smp = gait.sample(now)
    stance = smp.contact
    if not stance.any():
        return
    R = body.R
    qd4 = C.unflat(body.qd)
    v_feet = np.array([R @ (body.jac[i] @ qd4[i]) + np.cross(body.omega_w, R @ body.x_b[i])
                       for i in range(4)])
    v_trunk_w = -v_feet[stance].mean(axis=0)
    step_w = np.zeros(3)
    step_w[:2] = np.clip(_STATE["place"] * v_trunk_w[:2], -0.03, 0.03)
    step_b = R.T @ step_w
    zw = body.x_b @ R[2]
    ground = float(zw[stance].mean())
    law = stand.balance
    h = law.ramp.at(now - law.t0).h
    rest = SW.rest_feet_b(h, law.foot_xy)
    for i in range(4):
        if stance[i]:
            _SWINGING[i] = False
            _START[i] = np.nan
            continue
        if not _SWINGING[i]:
            _SWINGING[i] = True
            _START[i] = body.x_b[i]
            _DXY[i] = 0.0
        _DXY[i] = step_b[:2]
        if _STATE["on"]:
            p = rest[i].copy()
            p[:2] += _DXY[i]
            _DZ[i] = (ground - float(R[2] @ p)) / float(R[2, 2])


def land_tc_hook(sim, now, stand, body):
    name = stand.phase_name
    gait = (stand.gait if name == "trot" else
            stand.step_gait if name == "step" else None)
    if gait is None:
        _DZ[:] = 0.0
        _SWINGING[:] = False
        return
    smp = gait.sample(now)
    stance = smp.contact
    if not stance.any():
        return
    R = body.R
    zw = body.x_b @ R[2]
    ground = float(zw[stance].mean())
    law = stand.balance
    rest = None
    if law.swing != "joint":
        h = law.ramp.at(now - law.t0).h
        rest = SW.rest_feet_b(h, law.foot_xy)
    for i in range(4):
        if stance[i]:
            _DZ[i] = 0.0
            _SWINGING[i] = False
            continue
        if law.swing == "joint":
            p = (law._lift_x[i] if np.isfinite(law._lift_x[i, 0])
                 else body.x_b[i])
        else:
            p = rest[i]
        dz = (ground - float(R[2] @ p)) / float(R[2, 2])
        if not _SWINGING[i]:
            _DZ0[i] = dz if law.swing != "joint" else 0.0
            _SWINGING[i] = True
        _DZ[i] = dz


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def centered(pose, com_err_mm):
    """`pose` with every foot moved by the CoM error and c^b corrected."""
    d = 1e-3 * np.array([com_err_mm[0], com_err_mm[1], 0.0])
    hip = SIM_hip_sites(pose) + d
    moved = POSE.CrouchPose.from_hip_sites(pose.name + "_centered", hip,
                                           q_seed=pose.q)
    srb = pose.srb
    com = np.array(srb.com_body, dtype=float) + d
    com.flags.writeable = False
    model = BCFG.SrbModel(name="calibrated", com_body=com,
                          inertia_body=srb.inertia_body)
    return dataclasses.replace(moved, srb_pinned=model)


def SIM_hip_sites(pose):
    from sim import kinematics as SK
    return SK.hip_to_foot_stance(pose.q)


def run_variant(entry, opkw, hwkw, land_tc=False, kp_joint=None, tag="", hybrid=False,
                place=0.0):
    saved = (BCFG.KP_SWING_JOINT.copy(), BCFG.KD_SWING_JOINT.copy())
    try:
        if kp_joint is not None:
            BCFG.KP_SWING_JOINT[:] = kp_joint[0]
            BCFG.KD_SWING_JOINT[:] = kp_joint[1]
        _STATE["on"] = bool(land_tc)
        _HYBRID["on"] = bool(hybrid)
        _STATE["place"] = float(place)
        _STATE["latch_start"] = bool(place)
        _DXY[:] = 0.0
        _START[:] = np.nan
        _STATE["start"] = entry.swing != "joint"
        _DZ[:] = 0.0
        _DZ0[:] = 0.0
        _SWINGING[:] = False
        sim = H.HwSim(entry, H.HwParams(**hwkw))
        sim.place(entry.crouch.q, entry.crouch.z_origin)
        hook = (place_hook if place else land_tc_hook if land_tc else None)
        if place:
            _STATE["on"] = bool(land_tc)
        res = sim.run(H.Operator(**opkw), t_max=40.0, pre_update=hook)
    finally:
        BCFG.KP_SWING_JOINT[:] = saved[0]
        BCFG.KD_SWING_JOINT[:] = saved[1]
        _STATE["on"] = False
        _HYBRID["on"] = False
        _STATE["place"] = 0.0
        _STATE["latch_start"] = False
        _DXY[:] = 0.0
    if tag:
        H.save(res, os.path.join(HERE, "data", "fix_%s.npz" % tag))
    return res


FOLD_OP = dict(hold_s=2.0, trot_s=12.0, park=False)
WIDE_OP = dict(hold_s=2.0, trot_s=12.0, park=False)


def group_fold():
    base = H.entry_fold_trot()
    hw = {"prime_gate": True}
    return [
        ("committed", base, FOLD_OP, hw, {}),
        ("land_tc", base, FOLD_OP, hw, {"land_tc": True}),
        ("kp10", base, FOLD_OP, hw, {"kp_joint": (10.0, 0.5)}),
        ("land_tc+kp10", base, FOLD_OP, hw, {"land_tc": True, "kp_joint": (10.0, 0.5)}),
        ("P1.2", base.replace(gait_period=1.2), FOLD_OP, hw, {}),
        ("P1.2+land_tc", base.replace(gait_period=1.2), FOLD_OP, hw, {"land_tc": True}),
        ("P1.2+land_tc+kp10", base.replace(gait_period=1.2), FOLD_OP, hw,
         {"land_tc": True, "kp_joint": (10.0, 0.5)}),
        ("P1.2+land_tc+kp10 com_y5", base.replace(gait_period=1.2), FOLD_OP,
         dict(hw, com_err_mm=(0.0, 5.0)), {"land_tc": True, "kp_joint": (10.0, 0.5)}),
        ("P1.2+land_tc+kp10 combo", base.replace(gait_period=1.2), FOLD_OP,
         dict(hw, imu_latency=15e-3, com_err_mm=(3.0, 5.0), frictionloss=0.1,
              friction=0.7, kt_scale=0.9),
         {"land_tc": True, "kp_joint": (10.0, 0.5)}),
    ]


def group_wide_W():
    base = H.entry_trot()
    op = dict(WIDE_OP, step_first=True)
    return [
        ("committed", base, op, {}, {}),
        ("land_tc", base, op, {}, {"land_tc": True}),
        ("roll290", base.replace(roll_gains=(290.0, 23.0)), op, {}, {}),
        ("land_tc+roll290", base.replace(roll_gains=(290.0, 23.0)), op, {}, {"land_tc": True}),
    ]


def group_wide_com():
    base = H.entry_trot()
    out = []
    for dy in (5.0, 10.0):
        hw = {"com_err_mm": (0.0, dy)}
        out += [
            ("com_y%g committed" % dy, base, WIDE_OP, hw, {}),
            ("com_y%g land_tc" % dy, base, WIDE_OP, hw, {"land_tc": True}),
            ("com_y%g center" % dy, base.replace(crouch=centered(POSE.WIDE, (0.0, dy))),
             WIDE_OP, hw, {}),
            ("com_y%g center+land_tc" % dy,
             base.replace(crouch=centered(POSE.WIDE, (0.0, dy))), WIDE_OP, hw,
             {"land_tc": True}),
        ]
    return out


def group_fold2():
    base = H.entry_fold_trot()
    hw = {"prime_gate": True}
    out = []
    for period in (1.2, 0.8):
        out += [
            ("joint P%.1f slew200" % period,
             base.replace(gait_period=period, tau_slew=200.0), FOLD_OP, hw, {}),
            ("cart P%.1f" % period,
             base.replace(gait_period=period, swing="cartesian"), FOLD_OP, hw, {}),
            ("cart P%.1f slew200" % period,
             base.replace(gait_period=period, swing="cartesian", tau_slew=200.0),
             FOLD_OP, hw, {}),
        ]
    out += [
        ("joint P2.0 slew200", base.replace(tau_slew=200.0), FOLD_OP, hw, {}),
        ("cart P2.0", base.replace(swing="cartesian"), FOLD_OP, hw, {}),
    ]
    return out


def group_wide_center():
    """roll 290/23 with and without the re-centred stance, and naive placement."""
    base = H.entry_trot().replace(roll_gains=(290.0, 23.0))
    out = []
    for dy in (5.0, 10.0):
        hw = {"com_err_mm": (0.0, dy)}
        out += [
            ("com_y%g roll290" % dy, base, WIDE_OP, hw, {}),
            ("com_y%g roll290+center" % dy,
             base.replace(crouch=centered(POSE.WIDE, (0.0, dy))), WIDE_OP, hw, {}),
            ("com_y%g roll290+place0.15" % dy, base, WIDE_OP, hw, {"place": 0.15}),
        ]
    out.append(("ideal roll290+place0.15", base, WIDE_OP, {}, {"place": 0.15}))
    fold = H.entry_fold_trot().replace(gait_period=0.8, swing="cartesian")
    for dy in (5.0, 10.0):
        out.append(("fold cart0.8 com_y%g center" % dy,
                    fold.replace(crouch=centered(POSE.FOLD, (0.0, dy))), FOLD_OP,
                    {"com_err_mm": (0.0, dy), "prime_gate": True}, {}))
    return out


GROUPS = {"fold": group_fold, "fold2": group_fold2, "wide_W": group_wide_W,
          "wide_com": group_wide_com, "wide_center": group_wide_center}


def main(argv):
    names = [a for a in argv if a in GROUPS] or list(GROUPS)
    for g in names:
        table = {}
        for label, entry, opkw, hwkw, kw in GROUPS[g]():
            t0 = time.time()
            tag = "%s__%s" % (g, label.replace(" ", "_").replace("+", "-"))
            res = run_variant(entry, opkw, hwkw, tag=tag, **kw)
            m = metrics(res)
            table[label] = m
            print("%-8s %-28s fell=%-5s trot %5.2f s  max tilt %6.2f  tilt>15 after %s  peak tau %s  stop=%s (%.0f s)"
                  % (g, label, m["fell"], m.get("trot_s", 0), m.get("max_tilt_trot", np.nan),
                     m.get("t_tilt15_after_T"), m.get("peak_tau_trot"),
                     str(m["stop"])[:50], time.time() - t0))
            sys.stdout.flush()
        json.dump(table, open(os.path.join(HERE, "data", "fixes_%s.json" % g), "w"),
                  indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:])
