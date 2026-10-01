"""Skeptic check: the PARK ramp (straight joint-space line HOLD q -> crouch q,
feet planted) -- foot slip on the floor and mesh collisions, winner vs others."""
import json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "doc/trot_posture")
import evalpose as EP
from sim import coordinates as C, params as P, kinematics as SK
from hw.balance import posture as POSE, config as BCFG
DATA = "doc/trot_posture/data"
out = {}
for name, spec in {"Y105_h160": EP.Spec("Y105_h160", "x", 81, -81, 105, 160, 17),
                   "Y95_h160": EP.Spec("Y95_h160", "x", 81, -81, 95, 160, 18),
                   "WIDE_h160_c40": EP.Spec("WIDE_h160_c40", "x", 81, -81, 85, 160, 40),
                   "WIDE_h160_c14": EP.Spec("WIDE_h160_c14", "x", 81, -81, 85, 160, 14),
                   "NOMINAL": EP.Spec("NOMINAL", "x", 81, -81, 65, 145, 0)}.items():
    pose, info = EP.build(spec)
    q_s = np.radians(np.array(info["q_stand_deg"])); q_c = pose.q
    site = SK.hip_to_foot_stance(q_c)[:, :2]
    rows = []
    for a in np.linspace(0, 1, 21):
        q = q_s + a * (q_c - q_s)
        hip = SK.hip_to_foot_stance(q)
        slip = np.linalg.norm(hip[:, :2] - site, axis=1)
        # level trunk resting on the four feet (feet at one height by symmetry)
        z = float(P.FOOT_RADIUS - SK.all_foot_positions(q)[:, 2].mean())
        z = max(z, float(BCFG.TRUNK_BOTTOM_OFFSET))
        col = EP.collisions(q, z)
        rows.append(dict(alpha=round(float(a), 2), h_mm=round(1e3 * (z - BCFG.TRUNK_BOTTOM_OFFSET), 1),
                         slip_mm=np.round(1e3 * slip, 1).tolist(), dxy_mm=np.round(1e3 * (hip[0, :2] - site[0]), 1).tolist(),
                         min=col["min_mm"], ll=col["leg_leg_mm"], lt=col["leg_trunk_mm"], lf=col["leg_floor_mm"],
                         worst=col["worst"], pen=col["penetrating"]))
    out[name] = rows
    print("=====", name, "max slip %.1f mm  min leg_leg %.1f leg_trunk %.1f leg_floor %.1f" % (
        max(max(r["slip_mm"]) for r in rows), min(r["ll"] for r in rows), min(r["lt"] for r in rows), min(r["lf"] for r in rows)))
    for r in rows[::2]:
        print("   a=%.2f h=%6.1f FL dxy=%s min=%6.1f ll=%6.1f lf=%6.1f %s %s" % (r["alpha"], r["h_mm"], r["dxy_mm"], r["min"], r["ll"], r["lf"], r["worst"], [p[0][:30] + " %.1f" % p[1] for p in r["pen"]][:2]))
json.dump(out, open(DATA + "/skeptic_park.json", "w"), indent=1, default=str)
