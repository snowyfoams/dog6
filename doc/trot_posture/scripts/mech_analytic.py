"""mech_analytic -- what the rise path implies, with no simulation.

Along the IK path from the FOLD crouch (h = 60) to the stand (h = 145), feet
fixed under the hips (the drift itself ignored), with the law's 3 s quintic:
  * the horizontal foot force a joint VISCOUS damping c = 0.1 N m s/rad maps to
    through J^-T, per leg and summed (the law does not compensate it);
  * the same for the rotor ARMATURE 0.0085 kg m^2 (I q_ddot), the inertial
    term of the hypothesis;
  * the foot-ball rolling: the ball centre (the law's foot point) moves by
    FOOT_RADIUS x (shin angle change) with NO slip.
Also the same for the mirrored stances, to show the cancellation.
    python doc/trot_posture/scripts/mech_analytic.py  -> data/mech_analytic.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(TP, "..", ".."))
for p in (TP, REPO, os.path.join(REPO, "doc", "crouch_trot")):
    if p not in sys.path:
        sys.path.insert(0, p)
import evalpose as EP                              # noqa: E402
from sim import params as P, kinematics as SK      # noqa: E402
from hw import kinematics as HK                    # noqa: E402
from hw.balance import reference as REF            # noqa: E402

C_DAMP = 0.1
I_ARM = 0.0085
R_FOOT = P.FOOT_RADIUS


def path(spec, h0, n=301):
    seed = EP.seed_for(spec.family)
    q_stand, _, _ = EP.solve(EP.sites_for(spec.xf, spec.xr, spec.y, spec.h), seed)
    hs = np.linspace(h0, spec.h, n)
    qs = []
    for h in hs:
        q, ok, _ = EP.solve(EP.sites_for(spec.xf, spec.xr, spec.y, h), q_stand)
        qs.append(q)
    return hs, np.array(qs)


def shin_angle(leg, q):
    _, anchors, _, _, _ = SK.leg_frames(leg, q)
    foot = SK.all_foot_positions(np.array([q] * 4))[leg]
    knee = anchors[2]
    v = foot - knee
    return float(np.arctan2(v[0], -v[2]))     # angle of the shin from the vertical, + foot ahead


def analyse(spec, h0, T=3.0):
    hs, qs = path(spec, h0)
    hm = 1e-3 * hs
    ramp = REF.Quintic.ramp(hm[0], hm[-1], T)
    ts = np.linspace(0.0, T, 601)
    out = dict(spec=spec.as_dict(), crouch_h=h0, T=T)
    # dq/dh by finite difference along the path
    dq_dh = np.gradient(qs, hm, axis=0)             # (n, 4, 3)
    d2q_dh2 = np.gradient(dq_dh, hm, axis=0)
    rows = []
    for t in ts:
        h = float(ramp.at(t)[0]) if hasattr(ramp, "at") else None
        rows.append(h)
    # Quintic.at returns (h, hd, hdd) hopefully; be defensive
    H_, Hd, Hdd = [], [], []
    for t in ts:
        v = ramp.at(t)
        v = np.atleast_1d(np.asarray(v, float))
        if v.size >= 3:
            H_.append(v[0]); Hd.append(v[1]); Hdd.append(v[2])
        else:
            H_.append(v[0]); Hd.append(np.nan); Hdd.append(np.nan)
    H_, Hd, Hdd = np.array(H_), np.array(Hd), np.array(Hdd)
    if np.isnan(Hd).any():
        Hd = np.gradient(H_, ts); Hdd = np.gradient(Hd, ts)
    fx_damp = np.zeros((len(ts), 4)); fx_arm = np.zeros((len(ts), 4))
    fz_damp = np.zeros((len(ts), 4))
    for k, t in enumerate(ts):
        j = int(np.clip(np.searchsorted(hm, H_[k]), 0, len(hm) - 1))
        for i in range(4):
            q = qs[j, i]
            qd = dq_dh[j, i] * Hd[k]
            qdd = dq_dh[j, i] * Hdd[k] + d2q_dh2[j, i] * Hd[k] ** 2
            J = HK.leg(i).jacobian(q)                 # 3x3, trunk frame
            JinvT = np.linalg.inv(J).T
            fd = JinvT @ (C_DAMP * qd)
            fa = JinvT @ (I_ARM * qdd)
            fx_damp[k, i], fz_damp[k, i] = fd[0], fd[2]
            fx_arm[k, i] = fa[0]
    half = ts < 0.5 * T
    out["fx_damp_sum_1st_half_mean_N"] = round(float(fx_damp[half].sum(axis=1).mean()), 4)
    out["fx_damp_sum_2nd_half_mean_N"] = round(float(fx_damp[~half].sum(axis=1).mean()), 4)
    out["fx_damp_front_rear_at_mid_N"] = [round(float(fx_damp[len(ts) // 2, :2].sum()), 4),
                                          round(float(fx_damp[len(ts) // 2, 2:].sum()), 4)]
    out["fx_damp_impulse_Ns"] = round(float(np.trapezoid(fx_damp.sum(axis=1), ts)), 4)
    out["fx_damp_peak_sum_N"] = round(float(np.abs(fx_damp.sum(axis=1)).max()), 4)
    out["fx_arm_sum_1st_half_mean_N"] = round(float(fx_arm[half].sum(axis=1).mean()), 4)
    out["fx_arm_sum_2nd_half_mean_N"] = round(float(fx_arm[~half].sum(axis=1).mean()), 4)
    out["fx_arm_impulse_Ns"] = round(float(np.trapezoid(fx_arm.sum(axis=1), ts)), 4)
    out["fx_arm_peak_sum_N"] = round(float(np.abs(fx_arm.sum(axis=1)).max()), 4)
    # the viscous impulse is T-independent; the armature one scales 1/T: say so numerically
    # the least-dissipation path: x overdamped, feet planted, the trunk velocity (xd, hd) in the
    # trunk frame gives foot velocities -(xd, 0, hd); the net horizontal viscous reaction is
    # sum_i [J_i^-T J_i^-1 (xd, 0, hd)]_x = 0  ->  xd = -(M_xz / M_xx) hd, M = sum J^-T J^-1.
    # c cancels, T cancels: the drift is geometry.  Same with the mass matrix instead of c for the
    # momentum (inertial) path, which an UNDERDAMPED free mode would follow: approximated here with
    # the armature only (I per joint), which is the hypothesis's inertial term: identical ratio.
    ratio = []
    for j in range(len(hm)):
        M = np.zeros((3, 3))
        for i in range(4):
            Jinv = np.linalg.inv(HK.leg(i).jacobian(qs[j, i]))
            M += Jinv.T @ Jinv
        ratio.append(-M[0, 2] / M[0, 0])
    ratio = np.array(ratio)
    out["least_dissipation_drift_mm"] = round(float(1e3 * np.trapezoid(ratio, hm)), 1)
    out["least_dissipation_ratio_crouch_stand"] = [round(float(ratio[0]), 3), round(float(ratio[-1]), 3)]
    # rolling
    a0 = [shin_angle(i, qs[0, i]) for i in range(4)]
    a1 = [shin_angle(i, qs[-1, i]) for i in range(4)]
    out["shin_angle_crouch_deg"] = np.round(np.degrees(a0), 1).tolist()
    out["shin_angle_stand_deg"] = np.round(np.degrees(a1), 1).tolist()
    # ball centre moves +x when the shin rotates so the foot goes from ahead to behind? sign: the
    # contact point is under the centre; rolling without slip moves the centre by R * dtheta in
    # the direction the top of the ball goes, i.e. +x when the shin tips forward (d theta > 0)
    out["roll_mm"] = np.round(1e3 * R_FOOT * (np.array(a1) - np.array(a0)), 1).tolist()
    out["roll_mean_mm"] = round(float(np.mean(out["roll_mm"])), 1)
    return out


def main():
    specs = [(EP.Spec("fold", "parallel", 79, 23, 71, 145, 60), 60.0),
             (EP.Spec("fold_c100", "parallel", 79, 23, 71, 145, 100), 100.0),
             (EP.Spec("par60_c60", "parallel", 60, -20, 71, 145, 60), 60.0),
             (EP.Spec("par60_c100", "parallel", 60, -20, 71, 145, 100), 100.0),
             (EP.Spec("par60", "parallel", 60, -20, 71, 145, None), None),
             (EP.Spec("xin", "xin", 79, -79, 71, 145, 82), 82.0),
             (EP.Spec("nominal", "x", 81, -81, 65, 145, None), None)]
    res = {}
    for spec, h0 in specs:
        if h0 is None:
            _, info = EP.build(spec)
            h0 = info["crouch_h_mm"]
        for T in (1.5, 3.0, 12.0):
            r = analyse(spec, h0, T)
            res["%s_T%.1f" % (spec.name, T)] = r
            print("%-8s T %4.1f  crouch %5.1f  damping fx sum 1st/2nd half %+.4f/%+.4f N (F/R at mid %s)  impulse %+.4f N s  |peak| %.4f ;"
                  " armature 1st/2nd %+.4f/%+.4f N impulse %+.5f |peak| %.4f ;  rolling %s mm (mean %+.1f) ; LEAST-DISSIPATION drift %+.1f mm (ratio dx/dh crouch..stand %s)" % (
                      spec.name, T, h0, r["fx_damp_sum_1st_half_mean_N"], r["fx_damp_sum_2nd_half_mean_N"],
                      r["fx_damp_front_rear_at_mid_N"], r["fx_damp_impulse_Ns"], r["fx_damp_peak_sum_N"],
                      r["fx_arm_sum_1st_half_mean_N"], r["fx_arm_sum_2nd_half_mean_N"], r["fx_arm_impulse_Ns"],
                      r["fx_arm_peak_sum_N"], r["roll_mm"], r["roll_mean_mm"],
                      r["least_dissipation_drift_mm"], r["least_dissipation_ratio_crouch_stand"]))
    json.dump(res, open(os.path.join(TP, "data", "mech_analytic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
