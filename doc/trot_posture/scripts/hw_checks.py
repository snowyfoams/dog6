"""Hardware-practicality checks on the finalists: crouch clearances vs crouch height (with the ~8 mm handover sag),
the joint path from the flat zero to the crouch, abduction angles, pendulum growth per swing."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import evalpose as EP
from sim import coordinates as C, params as P, kinematics as SK
from hw.balance import config as BCFG
out = {}
SAG = 8.0   # mm, the ramp-from-zero handover sag measured by the skeptic (7.7-7.9 mm at any crouch)
print("== crouch clearances (mm, hull-based) at the crouch and 8 mm below it (the handover sag) ==")
for name, fam, xf, xr, y, h in (("WIDE_h160", "x", 81, -81, 85, 160), ("Y95_h160", "x", 81, -81, 95, 160), ("Y105_h160", "x", 81, -81, 105, 160), ("NOMINAL", "x", 81, -81, 65, 145)):
    q_stand, _, _ = EP.solve(EP.sites_for(xf, xr, y, h), EP.seed_for(fam))
    rows = []
    for hc in (0, 8, 14, 17, 20, 25, 30, 35, 40, 50):
        q, ok, kd = EP.solve(EP.sites_for(xf, xr, y, hc), q_stand)
        if not ok.all():
            rows.append((hc, None)); continue
        c = EP.collisions(q, 1e-3 * hc + BCFG.TRUNK_BOTTOM_OFFSET)
        hs = max(hc - SAG, 0.0)
        qs, oks, _ = EP.solve(EP.sites_for(xf, xr, y, hs), q_stand)
        cs = EP.collisions(qs, 1e-3 * hs + BCFG.TRUNK_BOTTOM_OFFSET) if oks.all() else None
        tau = EP.statics(q, hc, EP.BCFG.SrbModel.from_pose(name, q_stand), "crouch").get("crouch_tau_4foot_max_nm")
        rows.append((hc, c, cs, np.degrees(q[0]).round(1).tolist(), tau))
    out[name] = rows
    print(name)
    for r in rows:
        if r[1] is None:
            print("   crouch %3d: unreachable" % r[0]); continue
        hc, c, cs, qdeg, tau = r
        print("   crouch %3d: leg-leg %5.1f  leg-trunk %5.1f  leg-floor %5.1f | after %d mm sag: leg-leg %5s leg-trunk %5s leg-floor %5s | FL q %s  0xA4 hold tau max %s" % (
            hc, c["leg_leg_mm"], c["leg_trunk_mm"], c["leg_floor_mm"], SAG,
            "%.1f" % cs["leg_leg_mm"] if cs else "-", "%.1f" % cs["leg_trunk_mm"] if cs else "-", "%.1f" % cs["leg_floor_mm"] if cs else "-", qdeg, tau))
print("\n== joint path from the flat zero (Q_ZERO, belly on the floor) to the crouch, 12 points, hull check ==")
for name, fam, xf, xr, y, h, hc in (("WIDE_h160 crouch40", "x", 81, -81, 85, 160, 40), ("WIDE_h160 crouch20", "x", 81, -81, 85, 160, 20), ("Y95 crouch30", "x", 81, -81, 95, 160, 30), ("Y105 crouch25", "x", 81, -81, 105, 160, 25), ("Y105 crouch40", "x", 81, -81, 105, 160, 40), ("NOMINAL crouch0", "x", 81, -81, 65, 145, 0)):
    q_stand, _, _ = EP.solve(EP.sites_for(xf, xr, y, h), EP.seed_for(fam))
    qc, ok, _ = EP.solve(EP.sites_for(xf, xr, y, hc), q_stand)
    worst = (99.0, None, None)
    for s in np.linspace(0, 1, 12):
        q = C.Q_ZERO + s * (qc - C.Q_ZERO)
        # the trunk: on the floor until the feet are below it; use the FK: trunk origin = foot radius - min foot z, at least on the floor
        feet = SK.all_foot_positions(q)
        z = max(BCFG.TRUNK_BOTTOM_OFFSET, P.FOOT_RADIUS - feet[:, 2].min())
        c = EP.collisions(q, z)
        m = min(c["leg_leg_mm"], c["leg_trunk_mm"], c["leg_floor_mm"])
        if m < worst[0]:
            worst = (m, round(s, 2), c["worst"])
    print("   %-20s worst clearance along the path %5.1f mm at s=%s (%s)" % (name, *worst))
print("\n== pendulum growth per swing and roll lever ==")
for name, fam, xf, xr, y, h in (("NOMINAL", "x", 81, -81, 65, 145), ("WIDE", "x", 81, -81, 85, 145), ("WIDE_h160", "x", 81, -81, 85, 160), ("Y105_h160", "x", 81, -81, 105, 160)):
    q_stand, _, _ = EP.solve(EP.sites_for(xf, xr, y, h), EP.seed_for(fam))
    com, _ = SK.body_inertia(q_stand)
    hcom = 1e-3 * h + BCFG.TRUNK_BOTTOM_OFFSET + com[2]
    w = np.sqrt(9.81 / hcom)
    feet = SK.all_foot_positions(q_stand)
    th = np.arctan2(abs(feet[0, 1] - feet[3, 1]), abs(feet[0, 0] - feet[3, 0]))
    print("   %-10s CoM height %.0f mm  omega %.2f rad/s  growth over 100 ms x%.2f, 160 ms x%.2f | diagonal angle %.1f deg: a 10 mm lateral CoM error is %.1f mm off the diagonal, %.2f N m" % (
        name, 1e3 * hcom, w, np.exp(0.1 * w), np.exp(0.16 * w), np.degrees(th), 10 * np.cos(th), BCFG.WEIGHT * 0.01 * np.cos(th)))
json.dump(out, open(os.path.join(HERE, "..", "data", "hw_checks.json"), "w"), indent=1, default=str)
