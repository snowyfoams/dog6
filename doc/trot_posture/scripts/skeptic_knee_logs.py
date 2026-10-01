"""Skeptic check: knee angle from straight, leg length vs reach, abd excursion
and touchdown impact in the worst-case trot logs of the top candidates."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "doc/trot_posture")
from sim import params as P, coordinates as C, kinematics as SK
DATA = "doc/trot_posture/data"
out = {}
for name in ["Y105_h160", "Y95_h160", "WIDE_h160", "WIDE", "NOMINAL"]:
    out[name] = {}
    for clk in ["P0.5_A20", "P0.8_A40"]:
        for pert in ["ideal", "com_y_+10mm", "combo_mild", "com_x_+10mm", "mass_+0.4kg"]:
            f = os.path.join(DATA, "%s__%s_%s.npz" % (name, clk, pert))
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True)
            ph = d["phase"]; q = d["q_true"]; sw = d["swing_s"]; cw = d["contact_w"]
            m = (ph == "trot")
            if not m.any():
                continue
            q4 = q[m].reshape(-1, 4, 3)
            knee = np.degrees(np.abs(q4[:, :, 2]))
            abd = np.degrees(np.abs(q4[:, :, 0]))
            pitch = np.degrees(q4[:, :, 1])
            # leg length hip(abd hinge)->foot, per sweep per leg
            L = []
            for row in q4[::5]:
                hip = SK.hip_to_foot_stance(row)
                L.append(np.linalg.norm(hip, axis=1))
            L = np.array(L)
            hm = (ph == "hold")
            qh = q[hm].reshape(-1, 4, 3)
            out[name]["%s/%s" % (clk, pert)] = dict(
                knee_min_deg=round(float(knee.min()), 1),
                knee_hold_deg=round(float(np.degrees(np.abs(qh[:, :, 2])).mean()), 1),
                knee_sign_flip=bool((np.sign(q4[:, :, 2]) != np.sign(qh[-1, :, 2])).any()),
                knee_p01_deg=round(float(np.percentile(knee, 0.1)), 1),
                abd_max_from_down_deg=round(float((90.0 - abd).max()), 1),
                abd_min_from_down_deg=round(float((90.0 - abd).min()), 1),
                abd_hold_deg=round(float(np.degrees(np.abs(qh[:, :, 0])).mean()), 1),
                reach_max=round(float(L.max() / P.LEG_REACH), 3),
                reach_min=round(float(L.min() / P.LEG_REACH), 3),
                reach_hold=round(float(np.linalg.norm(SK.hip_to_foot_stance(qh[-1]), axis=1).mean() / P.LEG_REACH), 3),
                fn_peak_n=round(float(d["foot_fn"][m].max()), 1),
                tilt_max_deg=round(float(np.degrees(np.hypot(d["roll"][m], d["pitch"][m])).max()), 2),
            )
json.dump(out, open(os.path.join(DATA, "skeptic_knee_logs.json"), "w"), indent=1)
for n, v in out.items():
    print("=====", n)
    for k, r in v.items():
        print("  %-24s" % k, r)
