"""evalpose -- a candidate STAND posture for the trot: built, checked, scored.

    python evalpose.py describe NAME FAMILY XF XR Y H [--crouch-h MM]
    python evalpose.py run      NAME FAMILY XF XR Y H [options]   -> data/NAME.json

    FAMILY   parallel  every knee behind its hip -- posture.FOLD's fold
                       (Spot / Go1 / Cheetah-3 style)
             x         front knees forward, rear knees back -- Q_STAND,
                       posture.NOMINAL, the CAD stand
             xin       knees toward the CoM: front back, rear forward
                       (the 2026-09-17..25 fold)
    XF, XR   mm, how far AHEAD (+) of its own hip each foot sits, trunk x,
             front / rear.  posture.FOLD is +79 / +23, NOMINAL +81 / -81.
    Y        mm, |y| of the foot from the trunk centreline (track / 2).
             NOMINAL 65, FOLD 71, WIDE 85.
    H        mm, floor to trunk BOTTOM at the stand.  config.H_LIFT is 145;
             the CAD stand is 157.5.

    options  --crouch-h MM         the crouch height (floor to trunk bottom);
                                   default: the lowest feasible one (see
                                   `lowest_crouch`)
             --clocks 0.5:20 0.8:40   gait period s : swing apex mm
             --perturb all|ideal|name,name   hwsim perturbations (robustness.PERTURB)
             --stance-xy KP KD     the stance xy spring, N/m and N s/m per leg,
                                   0 0 to switch it off (default 400 25)
             --no-prime            the committed crouch->rise handover (torque
                                   ramps from zero) instead of a bumpless one
             --no-joint-hold       kp_joint 0 (hw.trot flies 5 / 0.2)
             --stand-only          skip the trots
             --keep-npz            save every run's log next to the json

Everything the robot would run is the repo's own code through
doc/crouch_trot/hwsim.py; this file only builds the posture and scores the
logs.  The one controller setting that is not on any entry point today is
the stance xy spring during the RISE (the flag exists on hw.stand since
2026-09-25, off by default): see doc/trot_posture/README.md for why a
parallel stance needs it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CROUCH_TROT = os.path.join(REPO, "doc", "crouch_trot")
for p in (REPO, CROUCH_TROT):
    if p not in sys.path:
        sys.path.insert(0, p)

import hwsim as H                                   # noqa: E402
from robustness import PERTURB                      # noqa: E402
from sim import coordinates as C, params as P, kinematics as SK   # noqa: E402
from hw import kinematics as HK, imu as IMU, calibration as CAL   # noqa: E402
from hw.balance import (posture as POSE, config as BCFG, state as BS,   # noqa: E402
                        allocation as AL, torque as TQ)

DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

#: The knee MOTOR (MG5010 mesh in sim/model) is a disc of radius 32 mm centred
#: on the knee hinge, in the leg plane.  Below this the motor is in the floor.
#: The committed FOLD crouch puts the hinge at 11 mm: the real robot kneels on
#: its knee motors there, which the simulation (visual meshes only) never saw.
KNEE_MOTOR_R = 0.032
KNEE_CLEAR = 0.003                   # margin above the motor radius
ABD_MOTOR_R = 0.032                  # the abduction motor, same housing
FAMILIES = ("parallel", "x", "xin")
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = ("abd", "pitch", "knee")


# ===========================================================================
# the posture
# ===========================================================================
def seed_for(family: str) -> np.ndarray:
    """(4, 3) joint seed that selects the knee branch of each family."""
    if family == "parallel":
        return POSE.FOLD.q.copy()
    if family == "x":
        return POSE.NOMINAL.q.copy()
    if family == "xin":
        q = POSE.FOLD.q.copy()
        # the rear mirrored fore-aft from the front: pitch and knee negated
        q[2] = POSE.FOLD.q[0] * np.array([1.0, -1.0, -1.0])
        q[3] = POSE.FOLD.q[1] * np.array([1.0, -1.0, -1.0])
        return q
    raise ValueError("family %r: one of %s" % (family, FAMILIES))


def sites_for(xf_mm, xr_mm, y_mm, h_mm) -> np.ndarray:
    """(4, 3) HIP-frame foot sites for a level trunk at `h_mm` (trunk bottom)."""
    s = np.zeros((4, 3))
    s[:2, 0] = 1e-3 * xf_mm
    s[2:, 0] = 1e-3 * xr_mm
    side = np.sign(P.HIP_OFFSET[:, 1])
    s[:, 1] = side * (1e-3 * y_mm - np.abs(P.HIP_OFFSET[:, 1]))
    s[:, 2] = -(1e-3 * h_mm + BCFG.TRUNK_BOTTOM_OFFSET - P.FOOT_RADIUS)
    return s


def solve(sites, seed):
    """IK of (4, 3) hip sites.  Returns (q, reachable(4,), knee_dir(4,)) with
    knee_dir = sign(knee x - pitch-hinge x), trunk frame: -1 is knee back."""
    q = np.empty((4, 3))
    ok = np.zeros(4, dtype=bool)
    kd = np.zeros(4)
    seed = C.unflat(seed)
    for i in range(4):
        sol = HK.leg(i).ik_full(sites[i], seed[i])
        q[i] = sol.q
        ok[i] = bool(sol.reachable)
        _, anchors, _, _, _ = SK.leg_frames(i, q[i])
        kd[i] = np.sign(anchors[2][0] - anchors[1][0])
    return q, ok, kd


def soft_limit_margin(q) -> float:
    """rad, the smallest distance of any joint from `calibration.soft_limits`."""
    low, high = CAL.soft_limits()
    qf = C.flat(q)
    return float(min((qf - low).min(), (high - qf).min()))


def knee_geometry(q, z_origin):
    """Per leg: knee hinge (trunk x, z), its height above the floor, the
    knee MOTOR's clearance to the floor, the abd motor and the trunk box."""
    rows = []
    for i in range(4):
        _, anchors, _, _, _ = SK.leg_frames(i, q[i])
        hip, hinge, knee = anchors
        knee_floor = knee[2] + z_origin
        # the abduction motor: a disc of radius ABD_MOTOR_R about the x axis
        # through the hip; the knee motor a disc of radius KNEE_MOTOR_R
        d_abd = float(np.hypot(knee[1] - hip[1], knee[2] - hip[2])) - ABD_MOTOR_R - KNEE_MOTOR_R
        box = P.TRUNK_BOX_HALF
        gap = np.maximum(np.abs(knee) - box, 0.0)
        d_box = float(np.linalg.norm(gap)) - KNEE_MOTOR_R
        rows.append(dict(leg=LEGS[i],
                         knee_x_mm=round(1e3 * knee[0], 1), knee_z_mm=round(1e3 * knee[2], 1),
                         knee_rel_hinge_x_mm=round(1e3 * (knee[0] - hinge[0]), 1),
                         knee_hinge_above_floor_mm=round(1e3 * knee_floor, 1),
                         knee_motor_floor_clear_mm=round(1e3 * (knee_floor - KNEE_MOTOR_R), 1),
                         knee_motor_abd_clear_mm=round(1e3 * d_abd, 1),
                         knee_motor_box_clear_mm=round(1e3 * d_box, 1)))
    return rows


#: Mesh-level collision check.  sim/model/dog6.xml makes the link meshes
#: visual only; here a COPY is compiled with every mesh collidable (MuJoCo
#: collides their convex hulls) and a 30 mm margin, so `mj_forward` at a pose
#: reports every pair within 30 mm with its signed distance.  Parent-child
#: pairs (trunk-hip, hip-thigh, thigh-shin) are filtered by MuJoCo itself.
#: Hulls are fatter than the parts, so a penetration under `PEN_TOL` is noise;
#: the committed FOLD crouch shows the knee motors 20.8 mm into the floor,
#: the committed stands show nothing closer than 11 mm.
PEN_TOL = 0.002                      # m, deeper than this is a real collision
MARGIN = 0.030                       # m, pairs farther apart are not reported
_COLLIDE = {}


def _collide_model():
    if "m" not in _COLLIDE:
        import mujoco
        xml = open(P.XML_PATH, encoding="utf-8").read()
        xml = xml.replace('<geom contype="0" conaffinity="0" />',
                          '<geom contype="1" conaffinity="1" margin="%.3f" />' % MARGIN)
        assets = {f: open(os.path.join(P.MESH_DIR, f), "rb").read()
                  for f in os.listdir(P.MESH_DIR)}
        m = mujoco.MjModel.from_xml_string(xml, assets)
        _COLLIDE["m"], _COLLIDE["d"], _COLLIDE["mj"] = m, mujoco.MjData(m), mujoco
    return _COLLIDE["m"], _COLLIDE["d"], _COLLIDE["mj"]


def _geom_label(m, g):
    n = m.geom(g).name
    if n:
        return n
    body = m.body(m.geom_bodyid[g]).name
    kind = (m.mesh(m.geom_dataid[g]).name if m.geom_type[g] == 7 else "prim")
    return "%s:%s" % (body, kind)


def collisions(q, z_origin) -> dict:
    """Pairs within `MARGIN` at the pose, the foot balls on the floor excluded.

    Returns min_mm (the closest pair, +30 if nothing is within the margin),
    worst (that pair), penetrating (pairs deeper than PEN_TOL, with depth),
    and the closest pair in each of three classes: a leg link against the
    floor, a leg link against the trunk (box, body, cover, abd motors), and
    leg against leg."""
    m, d, mj = _collide_model()
    d.qpos[:] = C.qpos_from_q(q, root_pos=(0.0, 0.0, float(z_origin)))
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)
    pairs = {}
    for i in range(d.ncon):
        con = d.contact[i]
        a, b = _geom_label(m, con.geom1), _geom_label(m, con.geom2)
        # the foot ball (sphere geom foot_XX) and the foot bracket mesh
        # (shin_XX:quadruped_foot*) are MEANT to touch the floor
        if "floor" in (a, b) and ("foot" in a or "foot" in b):
            continue
        key = tuple(sorted((a, b)))
        pairs[key] = min(pairs.get(key, 1.0), float(con.dist))

    def cls(key):
        a, b = key
        if "floor" in key:
            if "trunk" in a or "trunk" in b:
                return "trunk_floor"
            return "leg_floor"
        if a.startswith("trunk") or b.startswith("trunk"):
            return "leg_trunk"
        return "leg_leg"

    out = dict(min_mm=1e3 * MARGIN, worst=None, penetrating=[],
               leg_floor_mm=1e3 * MARGIN, leg_trunk_mm=1e3 * MARGIN, leg_leg_mm=1e3 * MARGIN)
    for key, dist in sorted(pairs.items(), key=lambda kv: kv[1]):
        c = cls(key)
        if c == "trunk_floor":
            continue                    # the crouch may rest its belly down
        mm = round(1e3 * dist, 1)
        if dist < out["min_mm"] * 1e-3:
            out["min_mm"], out["worst"] = mm, "%s <-> %s" % key
        k = c + "_mm"
        if mm < out[k]:
            out[k] = mm
        if dist < -PEN_TOL:
            out["penetrating"].append(["%s <-> %s" % key, mm])
    out["min_mm"] = round(out["min_mm"], 1)
    return out


#: mm, the clearance the crouch search demands between a leg link and the
#: trunk or another leg (the committed FOLD stand has 11 mm between the front
#: thigh and the abd motor; its committed crouch has 0.1), and between a leg
#: link and the floor (the feet and the belly excepted; NOMINAL's crouch rests
#: the belly down with the hip motors 3.5 mm up).  A pose that fails these is
#: a pose resting on structure the law does not model.
CROUCH_CLEAR_MM = 5.0
FLOOR_CLEAR_MM = 3.0


def crouch_ok(col: dict, clear_mm=CROUCH_CLEAR_MM, floor_mm=FLOOR_CLEAR_MM) -> bool:
    return (not col["penetrating"] and col["leg_trunk_mm"] >= clear_mm
            and col["leg_leg_mm"] >= clear_mm and col["leg_floor_mm"] >= floor_mm)


def lowest_crouch(xf, xr, y, h_stand, q_stand, step_mm=1.0, clear_mm=CROUCH_CLEAR_MM,
                  floor_mm=FLOOR_CLEAR_MM):
    """The lowest crouch (floor to trunk bottom, mm) whose IK is reachable on
    the stand's knee branch, inside the soft limits, with nothing but the
    feet (and the belly) touching the floor and no link into the trunk or
    another leg -- `crouch_ok`.  None if nothing below the stand qualifies."""
    _, _, kd_stand = solve(sites_for(xf, xr, y, h_stand), q_stand)
    for hc in np.arange(0.0, h_stand, step_mm):
        q, ok, kd = solve(sites_for(xf, xr, y, hc), q_stand)
        if not ok.all():
            continue
        if not np.array_equal(kd, kd_stand):
            continue
        if soft_limit_margin(q) <= 0.0:
            continue
        z_origin = 1e-3 * hc + BCFG.TRUNK_BOTTOM_OFFSET
        if not crouch_ok(collisions(q, z_origin), clear_mm, floor_mm):
            continue
        return float(hc)
    return None


@dataclasses.dataclass
class Spec:
    name: str
    family: str
    xf: float            # mm
    xr: float            # mm
    y: float             # mm
    h: float             # mm, the stand
    crouch_h: float | None = None   # mm; None = lowest feasible

    def as_dict(self):
        return dataclasses.asdict(self)


def build(spec: Spec):
    """-> (pose: CrouchPose with the SRB pinned at spec.h, info: dict)."""
    seed = seed_for(spec.family)
    stand_sites = sites_for(spec.xf, spec.xr, spec.y, spec.h)
    q_stand, ok, kd = solve(stand_sites, seed)
    info = dict(spec=spec.as_dict(), stand_reachable=bool(ok.all()),
                knee_dir=kd.tolist(),
                stand_soft_limit_margin_deg=round(np.degrees(soft_limit_margin(q_stand)), 1))
    hc = spec.crouch_h
    if hc is None:
        hc = lowest_crouch(spec.xf, spec.xr, spec.y, spec.h, q_stand)
        info["crouch_auto"] = True
    if hc is None:
        info["crouch_h_mm"] = None
        return None, info
    crouch_sites = sites_for(spec.xf, spec.xr, spec.y, hc)
    q_c, ok_c, kd_c = solve(crouch_sites, q_stand)
    pose = POSE.CrouchPose.from_hip_sites(spec.name, crouch_sites, q_seed=q_stand,
                                          note="evalpose %s" % spec.as_dict())
    # the SRB model pinned at THIS stand height, not config.H_LIFT
    srb = BCFG.SrbModel.from_pose(spec.name, q_stand)
    pose = dataclasses.replace(pose, srb_pinned=srb)
    info.update(crouch_h_mm=float(hc), crouch_reachable=bool(ok_c.all()),
                crouch_knee_dir=kd_c.tolist(),
                crouch_soft_limit_margin_deg=round(np.degrees(soft_limit_margin(q_c)), 1),
                q_stand_deg=np.round(np.degrees(q_stand), 2).tolist(),
                q_crouch_deg=np.round(np.degrees(pose.q), 2).tolist(),
                foot_xy_hip_mm=np.round(1e3 * pose.foot_xy, 1).tolist(),
                z_origin_crouch_mm=round(1e3 * pose.z_origin, 1),
                knees_stand=knee_geometry(q_stand, 1e-3 * spec.h + BCFG.TRUNK_BOTTOM_OFFSET),
                knees_crouch=knee_geometry(pose.q, pose.z_origin),
                reach_used_stand=np.round(np.linalg.norm(stand_sites, axis=1) / P.LEG_REACH, 3).tolist(),
                srb_com_mm=np.round(1e3 * np.asarray(srb.com_body), 1).tolist(),
                srb_inertia_diag=np.round(np.diag(np.asarray(srb.inertia_body)), 4).tolist())
    info.update(statics(q_stand, spec.h, srb, "stand"))
    info.update(statics(pose.q, hc, srb, "crouch"))
    info.update(support(q_stand))
    info["collide_stand"] = collisions(q_stand, 1e-3 * spec.h + BCFG.TRUNK_BOTTOM_OFFSET)
    info["collide_crouch"] = collisions(pose.q, pose.z_origin)
    return pose, info


# ===========================================================================
# statics: level trunk, the allocator's split, J^T torques
# ===========================================================================
def _body(q, srb):
    return BS.read(C.flat(q), np.zeros(12), IMU.TrunkOrientation.level(), srb=srb)


def statics(q, h_mm, srb, tag):
    """Static joint torques at a level trunk: four feet, and each diagonal
    pair alone (the trot's two-foot phase).  A crouch at h = 0 rests on the
    floor and carries nothing."""
    out = {}
    body = _body(q, srb)
    grav = TQ.all_leg_gravity_torque(body.q, body.R)
    weight = np.array([0.0, 0.0, BCFG.WEIGHT, 0.0, 0.0, 0.0])
    if h_mm <= 0.5 and tag == "crouch":
        tau = C.unflat(grav)
        out["%s_tau_4foot_max_nm" % tag] = np.round(np.abs(tau).max(axis=0), 3).tolist()
        out["%s_note" % tag] = "trunk on the floor: the legs carry only themselves"
        return out
    a = AL.allocate(body.r_w, weight)
    tau = C.unflat(TQ.stance_torque(body, a.f_w, grav))
    out["%s_tau_4foot_max_nm" % tag] = np.round(np.abs(tau).max(axis=0), 3).tolist()
    out["%s_tau_4foot_legs_nm" % tag] = np.round(tau, 3).tolist()
    out["%s_fz_4foot_n" % tag] = np.round(a.fz, 2).tolist()
    out["%s_front_share" % tag] = round(float(a.fz[:2].sum() / a.fz.sum()), 3)
    if tag == "stand":
        for pair, contact in (("FL_RR", [1, 0, 0, 1]), ("FR_RL", [0, 1, 1, 0])):
            ad = AL.allocate(body.r_w, weight, contact=np.array(contact, dtype=float))
            taud = C.unflat(TQ.stance_torque(body, ad.f_w, grav))
            out["stand_tau_diag_%s_max_nm" % pair] = np.round(np.abs(taud).max(axis=0), 3).tolist()
            out["stand_diag_%s_residual_nm" % pair] = np.round(ad.residual[3:], 3).tolist()
            out["stand_diag_%s_peak_nm" % pair] = round(float(np.abs(taud).max()), 3)
    return out


def support(q_stand):
    """Where the CoM sits against the feet: the support centre and each
    diagonal line, trunk frame, level."""
    com, inertia = SK.body_inertia(q_stand)
    feet = SK.all_foot_positions(q_stand)
    centre = feet[:, :2].mean(axis=0)
    out = dict(com_mm=np.round(1e3 * com, 1).tolist(),
               inertia_diag=np.round(np.diag(inertia), 4).tolist(),
               feet_trunk_mm=np.round(1e3 * feet, 1).tolist(),
               support_centre_mm=np.round(1e3 * centre, 1).tolist(),
               com_minus_centre_mm=np.round(1e3 * (com[:2] - centre), 1).tolist())
    for pair, (i, j) in (("FL_RR", (0, 3)), ("FR_RL", (1, 2))):
        a, b = feet[i, :2], feet[j, :2]
        d = b - a
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        out["com_to_diag_%s_mm" % pair] = round(float(1e3 * np.dot(com[:2] - a, n)), 2)
    out["diag_angle_deg"] = round(float(np.degrees(np.arctan2(
        abs(feet[0, 1] - feet[3, 1]), abs(feet[0, 0] - feet[3, 0])))), 1)
    out["wheelbase_mm"] = round(float(1e3 * (feet[:2, 0].mean() - feet[2:, 0].mean())), 1)
    out["track_mm"] = round(float(1e3 * (np.abs(feet[:, 1]).mean() * 2)), 1)
    return out


# ===========================================================================
# the runs
# ===========================================================================
NO_LIMITS = (-10.0 * np.ones(12), 10.0 * np.ones(12))


def entry(pose, h_mm, period=None, apex_mm=20.0, stance_xy=(400.0, 25.0),
          roll=(290.0, 23.0), joint_hold=(5.0, 0.2), latch=True):
    """hw.fold_trot's run (cap 9, slew 60, roll 290/23, no limits, Cartesian
    swing, the joint layer) from `pose`, at `h_mm`, plus the stance xy spring."""
    law_kw = dict(swing_height=1e-3 * apex_mm, h_lift=1e-3 * h_mm,
                  kp_stance_xy=float(stance_xy[0]), kd_stance_xy=float(stance_xy[1]),
                  kp_joint=float(joint_hold[0]), kd_joint=float(joint_hold[1]),
                  fall_hold_deg=BCFG.FALL_HOLD_DEG)
    return H.Entry("cand:" + pose.name, pose, 9.0, tau_ceiling=9.0, tau_slew=60.0,
                   overspeed_trip=False, latch=latch, track_stop=0.0,
                   roll_gains=roll, swing="cartesian", gait_period=period,
                   law_kw=law_kw, limits=NO_LIMITS)


def _idx(ph, name, which):
    sel = np.flatnonzero(ph == name)
    if not sel.size:
        return None
    return int(sel[0] if which == "first" else sel[-1])


def stand_metrics(res) -> dict:
    ph = res["phase"]
    ic = _idx(ph, "crouch", "last")
    ih = _idx(ph, "hold", "last")
    out = {"stop": res["stop"], "t_stop": res["t_stop"]}
    if ic is None or ih is None:
        out["stood"] = False
        return out
    xy0, xy1 = res["xy_trunk"][ic], res["xy_trunk"][ih]
    feet0, feet1 = res["foot_w"][ic], res["foot_w"][ih]
    rel0 = feet0[:, 0] - xy0[0] - P.HIP_OFFSET[:, 0]
    rel1 = feet1[:, 0] - xy1[0] - P.HIP_OFFSET[:, 0]
    rise = (ph == "rise")
    hold = (ph == "hold")
    crouch = (ph == "crouch")
    tau = np.abs(res["tau_act"])
    out.update(stood=True,
               drift_x_mm=round(float(1e3 * (xy1[0] - xy0[0])), 1),
               drift_y_mm=round(float(1e3 * (xy1[1] - xy0[1])), 1),
               foot_x_rel_hip_crouch_mm=np.round(1e3 * rel0, 1).tolist(),
               foot_x_rel_hip_hold_mm=np.round(1e3 * rel1, 1).tolist(),
               foot_slip_max_mm=round(float(1e3 * np.linalg.norm(
                   (res["foot_w"][rise | hold][:, :, :2] - feet0[None, :, :2]), axis=2).max()), 1),
               h_hold_mm=round(float(1e3 * (res["z_trunk"][ih] - BCFG.TRUNK_BOTTOM_OFFSET)), 1),
               h_min_rise_mm=round(float(1e3 * (res["z_trunk"][rise].min() - BCFG.TRUNK_BOTTOM_OFFSET)), 1),
               pitch_hold_deg=round(float(np.degrees(res["pitch"][ih])), 2),
               roll_hold_deg=round(float(np.degrees(res["roll"][ih])), 2),
               tilt_max_rise_deg=round(float(np.degrees(np.abs(
                   np.c_[res["roll"][rise], res["pitch"][rise]]).max())), 2),
               tau_crouch_hold_peak_nm=round(float(tau[crouch].max()), 3),
               tau_crouch_hold_by_joint_nm=np.round(
                   tau[crouch].max(axis=0).reshape(4, 3).max(axis=0), 3).tolist(),
               tau_rise_peak_nm=round(float(tau[rise].max()), 3),
               tau_rise_peak_joint=C.JOINT_NAMES[int(np.argmax(tau[rise].max(axis=0)))],
               tau_hold_mean_max_nm=round(float(tau[hold].mean(axis=0).max()), 3),
               tau_hold_by_joint_nm=np.round(
                   tau[hold].mean(axis=0).reshape(4, 3).max(axis=0), 3).tolist(),
               belly_max_n=round(float(res["belly_fn"][rise | hold].max()), 1))
    return out


def trot_metrics(res) -> dict:
    out = H.trot_metrics(res)
    ph = res["phase"]
    i0, i1 = _idx(ph, "trot", "first"), _idx(ph, "trot", "last")
    if i0 is not None and i1 is not None:
        trot = ph == "trot"
        out["trot_xy_drift_mm"] = np.round(1e3 * (res["xy_trunk"][i1] - res["xy_trunk"][i0]), 1).tolist()
        out["trot_yaw_drift_deg"] = round(float(np.degrees(res["yaw"][i1] - res["yaw"][i0])), 2)
        out["trot_foot_fn_peak_n"] = round(float(res["foot_fn"][trot].max()), 1)
        out["trot_h_min_mm"] = round(float(1e3 * (res["z_trunk"][trot].min() - BCFG.TRUNK_BOTTOM_OFFSET)), 1)
        out["trot_roll_rms_deg"] = round(float(np.degrees(np.sqrt(np.mean(res["roll"][trot] ** 2)))), 2)
        out["trot_pitch_rms_deg"] = round(float(np.degrees(np.sqrt(np.mean(res["pitch"][trot] ** 2)))), 2)
        ih = _idx(ph, "hold", "last")
        if ih is not None and ih > i1:
            out["hold_after_tilt_deg"] = round(float(np.degrees(
                max(abs(res["roll"][ih]), abs(res["pitch"][ih])))), 2)
    return out


def run_stand(pose, h_mm, hw=None, prime=True, stance_xy=(400.0, 25.0),
              joint_hold=(5.0, 0.2), t_max=20.0, keep=None):
    e = entry(pose, h_mm, stance_xy=stance_xy, joint_hold=joint_hold)
    sim = H.HwSim(e, H.HwParams(**dict(hw or {}, prime_gate=prime)))
    sim.place(pose.q, pose.z_origin)
    res = sim.run(H.Operator(hold_s=3.0, park=False), t_max=t_max)
    if keep:
        H.save(res, keep)
    return stand_metrics(res)


def run_trot(pose, h_mm, period, apex_mm, hw=None, prime=True, stance_xy=(400.0, 25.0),
             joint_hold=(5.0, 0.2), trot_s=12.0, t_max=32.0, keep=None):
    e = entry(pose, h_mm, period=period, apex_mm=apex_mm, stance_xy=stance_xy,
              joint_hold=joint_hold)
    sim = H.HwSim(e, H.HwParams(**dict(hw or {}, prime_gate=prime)))
    sim.place(pose.q, pose.z_origin)
    res = sim.run(H.Operator(hold_s=2.0, trot_s=trot_s, hold_after_s=2.0, park=False),
                  t_max=t_max)
    if keep:
        H.save(res, keep)
    return trot_metrics(res)


def evaluate(spec: Spec, clocks=((0.5, 20.0), (0.8, 40.0)), perturb=("ideal",),
             stance_xy=(400.0, 25.0), prime=True, joint_hold=(5.0, 0.2),
             stand_only=False, keep_npz=False, log=print) -> dict:
    t0 = time.time()
    pose, info = build(spec)
    out = {"info": info}
    if pose is None:
        out["error"] = "no feasible crouch below the stand"
        return out
    if keep_npz:
        def keep(tag):
            return os.path.join(DATA, "%s__%s.npz" % (spec.name, tag))
    else:
        def keep(tag):
            return None
    out["stand"] = run_stand(pose, spec.h, prime=prime, stance_xy=stance_xy,
                             joint_hold=joint_hold, keep=keep("stand"))
    st = out["stand"]
    log("%-18s stand  drift x %+6.1f mm  pitch %+5.2f  rise peak %.2f N*m (%s)  hold %.2f  stop=%s  (%.0f s)"
        % (spec.name, st.get("drift_x_mm", np.nan), st.get("pitch_hold_deg", np.nan),
           st.get("tau_rise_peak_nm", np.nan), st.get("tau_rise_peak_joint", "-"),
           st.get("tau_hold_mean_max_nm", np.nan), st["stop"], time.time() - t0))
    if stand_only or not st.get("stood"):
        return out
    names = list(PERTURB) if (perturb == "all" or "all" in perturb) else list(perturb)
    out["trots"] = {}
    for period, apex in clocks:
        key = "P%.1f_A%.0f" % (period, apex)
        out["trots"][key] = {}
        for pname in names:
            t1 = time.time()
            m = run_trot(pose, spec.h, period, apex, hw=PERTURB[pname], prime=prime,
                         stance_xy=stance_xy, joint_hold=joint_hold,
                         keep=keep("%s_%s" % (key, pname)))
            out["trots"][key][pname] = m
            log("%-18s %s %-15s fell=%-5s trot %5.2f s  tilt %5.2f  xy drift %s  peak %s  stop=%s  (%.0f s)"
                % (spec.name, key, pname, m["fell"], m.get("trot_s", 0),
                   m.get("max_tilt_trot", np.nan), m.get("trot_xy_drift_mm"),
                   m.get("peak_tau_trot"), str(m["stop"])[:40], time.time() - t1))
        survived = sum(1 for m in out["trots"][key].values() if not m["fell"])
        out["trots"][key]["_survived"] = "%d/%d" % (survived, len(names))
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def summarize_info(info: dict) -> str:
    s = info["spec"]
    rows = ["%s  [%s]  feet x %+.0f/%+.0f  y %.0f  stand h %.0f  crouch h %s mm" % (
        s["name"], s["family"], s["xf"], s["xr"], s["y"], s["h"], info.get("crouch_h_mm"))]
    if info.get("crouch_h_mm") is None:
        rows.append("  NO FEASIBLE CROUCH (stand reachable: %s, knee dir %s)"
                    % (info.get("stand_reachable"), info.get("knee_dir")))
        return "\n".join(rows)
    rows.append("  stand reachable %s  knee dir %s  soft-limit margin stand %.1f / crouch %.1f deg" % (
        info["stand_reachable"], info["knee_dir"], info["stand_soft_limit_margin_deg"],
        info["crouch_soft_limit_margin_deg"]))
    rows.append("  CoM (trunk) %s mm   CoM - support centre %s mm   CoM to diagonals %s / %s mm   I diag %s" % (
        info["com_mm"], info["com_minus_centre_mm"], info["com_to_diag_FL_RR_mm"],
        info["com_to_diag_FR_RL_mm"], info["inertia_diag"]))
    rows.append("  wheelbase %.0f  track %.0f  diag angle %.1f deg  reach used %s" % (
        info["wheelbase_mm"], info["track_mm"], info["diag_angle_deg"], info["reach_used_stand"]))
    rows.append("  static |tau| max (abd, pitch, knee): 4-foot %s  diag FL/RR %s  diag FR/RL %s  (front share %.2f)" % (
        info["stand_tau_4foot_max_nm"], info["stand_tau_diag_FL_RR_max_nm"],
        info["stand_tau_diag_FR_RL_max_nm"], info["stand_front_share"]))
    rows.append("  diag residual moment (Mx, My, Mz) FL/RR %s  FR/RL %s N*m" % (
        info["stand_diag_FL_RR_residual_nm"], info["stand_diag_FR_RL_residual_nm"]))
    rows.append("  crouch |tau| max %s  (%s)" % (
        info["crouch_tau_4foot_max_nm"], info.get("crouch_note", "the 0xA4 position loops hold this")))
    for tag in ("knees_stand", "knees_crouch"):
        k = info[tag]
        rows.append("  %-12s knee rel hinge x F %+.0f R %+.0f mm; hinge above floor F %.0f R %.0f; "
                    "motor clear: floor F %.0f R %.0f, abd F %.0f R %.0f, box F %.0f R %.0f mm" % (
                        tag, k[0]["knee_rel_hinge_x_mm"], k[2]["knee_rel_hinge_x_mm"],
                        k[0]["knee_hinge_above_floor_mm"], k[2]["knee_hinge_above_floor_mm"],
                        k[0]["knee_motor_floor_clear_mm"], k[2]["knee_motor_floor_clear_mm"],
                        k[0]["knee_motor_abd_clear_mm"], k[2]["knee_motor_abd_clear_mm"],
                        k[0]["knee_motor_box_clear_mm"], k[2]["knee_motor_box_clear_mm"]))
    for tag in ("collide_stand", "collide_crouch"):
        c = info[tag]
        rows.append("  %-14s closest %s mm (%s); leg-floor %s, leg-trunk %s, leg-leg %s mm; penetrating %s" % (
            tag, c["min_mm"], c["worst"], c["leg_floor_mm"], c["leg_trunk_mm"], c["leg_leg_mm"],
            c["penetrating"] or "none"))
    rows.append("  q stand (deg)  %s" % info["q_stand_deg"])
    rows.append("  q crouch (deg) %s" % info["q_crouch_deg"])
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("describe", "run"))
    ap.add_argument("name")
    ap.add_argument("family", choices=FAMILIES)
    ap.add_argument("xf", type=float)
    ap.add_argument("xr", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("h", type=float)
    ap.add_argument("--crouch-h", type=float, default=None)
    ap.add_argument("--clocks", nargs="+", default=["0.5:20", "0.8:40"])
    ap.add_argument("--perturb", nargs="+", default=["ideal"])
    ap.add_argument("--stance-xy", nargs=2, type=float, default=[400.0, 25.0])
    ap.add_argument("--no-prime", action="store_true")
    ap.add_argument("--no-joint-hold", action="store_true")
    ap.add_argument("--stand-only", action="store_true")
    ap.add_argument("--keep-npz", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    spec = Spec(a.name, a.family, a.xf, a.xr, a.y, a.h, a.crouch_h)
    if a.mode == "describe":
        pose, info = build(spec)
        print(summarize_info(info))
        return 0
    clocks = [tuple(float(v) for v in c.split(":")) for c in a.clocks]
    perturb = "all" if a.perturb == ["all"] else [p for s in a.perturb for p in s.split(",")]
    res = evaluate(spec, clocks=clocks, perturb=perturb, stance_xy=tuple(a.stance_xy),
                   prime=not a.no_prime,
                   joint_hold=(0.0, 0.0) if a.no_joint_hold else (5.0, 0.2),
                   stand_only=a.stand_only, keep_npz=a.keep_npz)
    print(summarize_info(res["info"]))
    path = a.out or os.path.join(DATA, spec.name + ".json")
    json.dump(res, open(path, "w"), indent=1, default=str)
    print("-> %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
