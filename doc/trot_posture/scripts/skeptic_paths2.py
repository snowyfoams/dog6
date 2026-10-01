"""Skeptic check: collisions along (a) the joint ramp from NOMINAL's crouch to
the winner's crouch (a robot parked from a previous run), (b) the winner's rise
(level trunk, feet pinned, h 17 -> 160), (c) the FOLD crouch's ramp from the
flat zero as a calibration of how pessimistic the hull check is (FOLD was flown)."""
import json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "doc/trot_posture")
import evalpose as EP
from sim import coordinates as C, params as P, kinematics as SK, stand as ST
from hw.balance import posture as POSE, config as BCFG
DATA = "doc/trot_posture/data"

def z_rest(q):
    fz = SK.all_foot_positions(q)[:, 2]
    return max(float(BCFG.TRUNK_BOTTOM_OFFSET), float(P.FOOT_RADIUS - fz.min())), fz

def sweep(q0, q1, n=21):
    rows = []
    for a in np.linspace(0, 1, n):
        q = q0 + a * (q1 - q0)
        z, fz = z_rest(q)
        col = EP.collisions(q, z)
        rows.append(dict(alpha=round(float(a), 2), z=round(1e3 * z, 1), min=col["min_mm"], worst=col["worst"],
                         ll=col["leg_leg_mm"], lt=col["leg_trunk_mm"], lf=col["leg_floor_mm"], pen=col["penetrating"]))
    return rows

def show(tag, rows):
    print("=====", tag, "min leg_leg %.1f leg_trunk %.1f leg_floor %.1f" % (
        min(r["ll"] for r in rows), min(r["lt"] for r in rows), min(r["lf"] for r in rows)))
    for r in rows:
        print("   a=%.2f z=%5.1f min=%6.1f ll=%6.1f lt=%6.1f lf=%6.1f %s %s" % (
            r["alpha"], r["z"], r["min"], r["ll"], r["lt"], r["lf"], r["worst"], [p[0][:40] + " %.1f" % p[1] for p in r["pen"]][:2]))

out = {}
win = EP.Spec("Y105_h160", "x", 81, -81, 105, 160, 17)
pose, info = EP.build(win)
q_c = pose.q
# (a) NOMINAL crouch -> winner crouch, and WIDE crouch -> winner crouch
for tag, q0 in (("NOMINAL_crouch->Y105_crouch", POSE.NOMINAL.q), ("WIDE_crouch->Y105_crouch", POSE.WIDE.q),
                ("FOLD_crouch->Y105_crouch", POSE.FOLD.q)):
    rows = sweep(q0, q_c); out[tag] = rows; show(tag, rows)
# (b) the rise: level trunk, feet pinned at the winner's sites, h from 17 to 160
rows = []
seed = q_c
for h in np.linspace(17, 160, 16):
    q, ok, kd = EP.solve(EP.sites_for(81, -81, 105, h), seed)
    seed = q
    z = 1e-3 * h + BCFG.TRUNK_BOTTOM_OFFSET
    col = EP.collisions(q, z)
    rows.append(dict(alpha=round(float(h), 1), z=round(1e3 * z, 1), min=col["min_mm"], worst=col["worst"],
                     ll=col["leg_leg_mm"], lt=col["leg_trunk_mm"], lf=col["leg_floor_mm"], pen=col["penetrating"],
                     q_deg=np.round(np.degrees(q[0]), 1).tolist(), reach=round(float(np.linalg.norm(EP.sites_for(81, -81, 105, h)[0]) / P.LEG_REACH), 3)))
out["Y105_rise"] = rows; show("Y105 rise h17->160 (alpha = h mm)", rows)
for r in rows: print("      h=%5.1f q=%s reach %.3f" % (r["alpha"], r["q_deg"], r["reach"]))
# (c) calibration: FOLD's ramp from the flat zero (flown on the robot)
rows = sweep(C.Q_ZERO.copy(), POSE.FOLD.q); out["Q_ZERO->FOLD"] = rows; show("Q_ZERO->FOLD crouch (flown)", rows)
rows = sweep(C.Q_ZERO.copy(), POSE.WIDE.q); out["Q_ZERO->WIDE"] = rows; show("Q_ZERO->WIDE crouch 40 (flown 09-17)", rows)
json.dump(out, open(DATA + "/skeptic_paths2.json", "w"), indent=1, default=str)
