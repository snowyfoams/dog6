"""Skeptic check: mesh collisions along the straight-line joint ramp from the
flat zero (sim.coordinates.Q_ZERO) to each candidate's crouch, trunk resting
on its belly or lifted by whichever foot is lowest; plus the Jacobian
conditioning and the extension margin at each stand."""
import json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "doc/trot_posture")
import evalpose as EP
from sim import coordinates as C, params as P, kinematics as SK, stand as ST
from hw.balance import posture as POSE, config as BCFG
DATA = "doc/trot_posture/data"

def z_rest(q):
    """trunk origin height: belly on the floor unless a foot pushes it up."""
    fz = SK.all_foot_positions(q)[:, 2]
    return max(float(BCFG.TRUNK_BOTTOM_OFFSET), float(P.FOOT_RADIUS - fz.min())), fz

def sweep(q0, q1, n=21):
    rows = []
    for a in np.linspace(0, 1, n):
        q = q0 + a * (q1 - q0)
        z, fz = z_rest(q)
        col = EP.collisions(q, z)
        rows.append(dict(alpha=round(float(a), 2), z_origin_mm=round(1e3 * z, 1),
                         foot_z_mm=np.round(1e3 * fz, 0).tolist(),
                         min_mm=col["min_mm"], worst=col["worst"],
                         leg_leg=col["leg_leg_mm"], leg_trunk=col["leg_trunk_mm"], leg_floor=col["leg_floor_mm"],
                         pen=col["penetrating"]))
    return rows

def jac(i, q):
    def f(qq):
        return SK.all_foot_positions(C.unflat(qq) if False else qq)[i]
    J = np.zeros((3, 3)); e = 1e-6
    for j in range(3):
        qp = q.copy(); qp[i, j] += e; qm = q.copy(); qm[i, j] -= e
        J[:, j] = (SK.all_foot_positions(qp)[i] - SK.all_foot_positions(qm)[i]) / (2 * e)
    return J

cands = {
    "Y105_h160": EP.Spec("Y105_h160", "x", 81, -81, 105, 160, 17),
    "Y95_h160": EP.Spec("Y95_h160", "x", 81, -81, 95, 160, 18),
    "WIDE_h160": EP.Spec("WIDE_h160", "x", 81, -81, 85, 160, 14),
    "WIDE_h160_c40": EP.Spec("WIDE_h160_c40", "x", 81, -81, 85, 160, 40),
    "NOMINAL_c0": EP.Spec("NOMINAL_c0", "x", 81, -81, 65, 145, 0),
}
out = {}
q0 = C.Q_ZERO.copy()
print("Q_ZERO itself:", EP.collisions(q0, z_rest(q0)[0]))
for name, spec in cands.items():
    pose, info = EP.build(spec)
    q_c = pose.q
    q_s = np.radians(np.array(info["q_stand_deg"]))
    rows = sweep(q0, q_c)
    worst = min(rows, key=lambda r: r["min_mm"])
    # Jacobian at the stand: vertical force per knee torque, singular values, extension margin
    J = jac(0, q_s)
    sv = np.linalg.svd(J, compute_uv=False)
    JT = J.T                       # tau = J^T F
    tau_per_N_z = JT @ np.array([0, 0, -1.0])          # N m per N of vertical push (F on foot from floor is +z; leg pushes -z)
    hip = SK.hip_to_foot_stance(q_s)[0]
    Lplane = np.hypot(hip[0], np.hypot(hip[1], hip[2]))  # not exact; use ik geometry below
    knee = abs(q_s[0, 2])
    L2, L3 = P.L2, P.L3
    Lp = np.sqrt(L2**2 + L3**2 + 2 * L2 * L3 * np.cos(knee))   # pitch hinge -> foot in the leg plane (foot z offset ignored)
    margin = (L2 + L3) - Lp
    # how much the knee would have to open to extend by 10 / 17 mm
    def knee_for(Lt):
        c = (Lt**2 - L2**2 - L3**2) / (2 * L2 * L3)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    out[name] = dict(crouch_h=info["crouch_h_mm"], q_crouch_deg=info["q_crouch_deg"][0], q_stand_deg=info["q_stand_deg"][0],
                     crouch_col=info["collide_crouch"], stand_col=info["collide_stand"],
                     ramp_worst=worst, ramp_min_leg_leg=min(r["leg_leg"] for r in rows),
                     ramp_min_leg_trunk=min(r["leg_trunk"] for r in rows),
                     ramp_min_leg_floor=min(r["leg_floor"] for r in rows),
                     ramp_pen_alphas=[r["alpha"] for r in rows if r["pen"]],
                     ramp=rows,
                     J_sv_mm_per_rad=np.round(1e3 * sv, 1).tolist(), J_cond=round(float(sv[0] / sv[-1]), 2),
                     tau_per_N_vertical=np.round(tau_per_N_z, 4).tolist(),
                     knee_deg=round(float(np.degrees(knee)), 1), Lplane_mm=round(1e3 * Lp, 1),
                     ext_margin_mm=round(1e3 * margin, 1),
                     knee_after_10mm_ext=round(float(knee_for(Lp + 0.010)), 1),
                     knee_after_17mm_ext=round(float(knee_for(min(Lp + 0.017, L2 + L3))), 1),
                     reach_hip=round(float(np.linalg.norm(hip) / P.LEG_REACH), 3))
    print("=====", name, "crouch", info["crouch_h_mm"], "q_c", info["q_crouch_deg"][0])
    print("  stand J sv (mm/rad)", out[name]["J_sv_mm_per_rad"], "cond", out[name]["J_cond"], "tau/N vertical", out[name]["tau_per_N_vertical"])
    print("  knee %.1f  Lplane %.1f  ext margin %.1f mm  knee after +10 mm %.1f  after +17 mm %.1f" % (
        out[name]["knee_deg"], out[name]["Lplane_mm"], out[name]["ext_margin_mm"], out[name]["knee_after_10mm_ext"], out[name]["knee_after_17mm_ext"]))
    print("  ramp: min leg_leg %.1f leg_trunk %.1f leg_floor %.1f  pen at alphas %s" % (
        out[name]["ramp_min_leg_leg"], out[name]["ramp_min_leg_trunk"], out[name]["ramp_min_leg_floor"], out[name]["ramp_pen_alphas"]))
    for r in rows:
        print("    a=%.2f z=%5.1f fz=%s min=%6.1f ll=%6.1f lt=%6.1f lf=%6.1f %s %s" % (
            r["alpha"], r["z_origin_mm"], r["foot_z_mm"], r["min_mm"], r["leg_leg"], r["leg_trunk"], r["leg_floor"], r["worst"], r["pen"]))
json.dump(out, open(DATA + "/skeptic_ramp_path.json", "w"), indent=1, default=str)
