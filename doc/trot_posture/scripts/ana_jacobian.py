"""ana: leg-Jacobian conditioning at each candidate's stand (FL leg, from
q_stand_deg in cand_<NAME>.json): singular values, the weak Cartesian
direction, the joint hold's stiffness along it, force per torque quantum,
and how far the knee must flex for a 40 mm apex / how much extension remains.
Writes data/ana_jacobian.json."""
import json, os, sys, glob
import numpy as np

sys.path.insert(0, ".")
from hw import kinematics as K

D = "doc/trot_posture/data"
IQ_NM = 1.0 / 206.04          # torque quantum, hwsim's current loop
KP_JOINT = 5.0                # joint hold, N m/rad
leg = K.leg("FL")
out = {}
rows = []
for f in sorted(glob.glob(os.path.join(D, "cand_*.json"))):
    d = json.load(open(f)); info = d["info"]; name = info["spec"]["name"]
    q = np.radians(info["q_stand_deg"][0])
    p, J = leg.fk_jac(q)
    U, S, Vt = np.linalg.svd(J)
    weak = U[:, -1]                       # Cartesian direction of sigma_min
    # vertical force per unit knee torque: F = J^-T tau
    JT_inv = np.linalg.inv(J.T)
    Fk = JT_inv @ np.array([0, 0, 1.0])    # force from 1 N m at the knee
    Fh = JT_inv @ np.array([0, 1.0, 0])
    # torque to hold 29 N vertical (two-foot share) with force along the leg vs pure vertical
    tau_vert = J.T @ np.array([0, 0, -29.0])
    # Cartesian stiffness of the joint hold:  K_x = J^-T Kq J^-1
    Jinv = np.linalg.inv(J)
    Kx = Jinv.T @ (KP_JOINT * np.eye(3)) @ Jinv
    kz = Kx[2, 2]
    # force noise from one torque quantum on the knee, vertical component
    dF_quant = JT_inv @ np.array([0, 0, IQ_NM])
    # 40 mm apex: knee flex needed (foot raised 40 mm in the hip frame)
    p_hip = leg.fk_hip(q)
    q_up = leg.ik(p_hip + np.array([0, 0, 0.040]), q)
    # extension margin: planar pitch-to-foot distance vs L2+L3
    r_pf = np.linalg.norm(p_hip - leg.d1) if hasattr(leg, "d1") else np.nan
    row = dict(name=name, h=info["spec"]["h"], y=info["spec"]["y"], knee_deg=round(float(np.degrees(abs(q[2]))), 1),
               sig=np.round(1e3 * S, 1).tolist(), cond=round(float(S[0] / S[-1]), 1),
               weak_dir=np.round(weak, 2).tolist(),
               Fz_per_Nm_knee=round(float(Fk[2]), 1), Fx_per_Nm_knee=round(float(Fk[0]), 1),
               Fz_per_Nm_hip=round(float(Fh[2]), 1),
               tau_vert29=np.round(tau_vert, 2).tolist(),
               kz_hold_N_per_m=round(float(kz), 0),
               dFz_per_quantum_N=round(float(abs(dF_quant[2])), 2),
               knee_at_40mm_apex_deg=round(float(np.degrees(abs(q_up[2]))), 1),
               hip_foot_mm=round(float(1e3 * np.linalg.norm(p_hip)), 1),
               pitch_foot_mm=round(float(1e3 * r_pf), 1) if r_pf == r_pf else None)
    rows.append(row); out[name] = row
rows.sort(key=lambda r: (-r["h"], r["knee_deg"]))
print("%-14s   h   y knee |  sig(mm/rad) max/mid/min  cond | weak dir (x,y,z)   | Fz/Nm knee Fz/Nm hip | tau(29N vert) abd/hip/knee | kz_hold N/m | dFz/quantum N | knee@+40mm")
for r in rows:
    print("%-14s %3d %3d %5.1f | %6.1f %6.1f %6.1f %6.1f | %s | %7.1f %7.1f | %s | %8.0f | %5.2f | %5.1f" % (
        r["name"], r["h"], r["y"], r["knee_deg"], r["sig"][0], r["sig"][1], r["sig"][2], r["cond"], r["weak_dir"],
        r["Fz_per_Nm_knee"], r["Fz_per_Nm_hip"], r["tau_vert29"], r["kz_hold_N_per_m"], r["dFz_per_quantum_N"], r["knee_at_40mm_apex_deg"]))
json.dump(out, open(os.path.join(D, "ana_jacobian.json"), "w"), indent=1)
