"""Post-process the skeptic runs: (1) the knee MOTOR's floor clearance at the
deepest point of today's ramp-from-zero handover (the plant collides only the
foot balls and the trunk box, so a knee motor into the floor is invisible to
it); (2) the 25 s lateral walk at 0.8 s / 40 mm, com_y +10 mm: slope per
window, is it saturating?  Reads the npz kept under $SKEPTIC_NPZ.
writes data/skeptic_analysis.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, "doc/trot_posture")
import evalpose as EP
from hw.balance import config as BCFG

NPZ = os.environ.get("SKEPTIC_NPZ", EP.DATA)
out = {"handover_sag": {}, "walk25": {}}

for name in ("Y105_h160", "Y95_h160", "WIDE_h160"):
    spec = json.load(open(f"doc/trot_posture/data/cand_{name}.json"))["info"]["spec"]
    f = os.path.join(NPZ, f"skeptic_noprime_{name}__stand_noprime_spring.npz")
    r = np.load(f, allow_pickle=True)
    keys = list(r.keys())
    ph = r["phase"]
    rise = np.flatnonzero(ph == "rise")
    i = rise[np.argmin(r["z_trunk"][rise])]
    z = float(r["z_trunk"][i])
    h = 1e3 * (z - BCFG.TRUNK_BOTTOM_OFFSET)
    t_sag = float(r["t"][i] - r["t"][rise[0]])
    row = dict(h_min_rise_mm=round(h, 1), t_after_handover_s=round(t_sag, 2),
               crouch_h_mm=json.load(open(f"doc/trot_posture/data/cand_{name}.json"))["info"]["crouch_h_mm"],
               npz_keys=keys)
    # the leg configuration at the sag: logged q if present, else the IK of
    # the posture's own foot sites at the sagged height
    if "q" in keys:
        q = np.asarray(r["q"][i]).reshape(4, 3)
        row["q_source"] = "logged"
    else:
        q, ok, _ = EP.solve(EP.sites_for(spec["xf"], spec["xr"], spec["y"], h), EP.seed_for(spec["family"]))
        row["q_source"] = "ik at the sagged height, feet at the crouch sites"
    kg = EP.knee_geometry(q, z)
    row["knee_motor_floor_clear_mm"] = [k["knee_motor_floor_clear_mm"] for k in kg]
    col = EP.collisions(q, z)
    row["collisions"] = {k: col[k] for k in ("min_mm", "worst", "leg_floor_mm", "leg_trunk_mm", "penetrating")}
    # pitch / roll at the sag and the hold
    row["tilt_at_sag_deg"] = round(float(np.degrees(max(abs(r["roll"][i]), abs(r["pitch"][i])))), 2)
    out["handover_sag"][name] = row
    print(name, {k: v for k, v in row.items() if k != "npz_keys"})

for name in ("Y105_h160", "Y95_h160", "NOMINAL"):
    f = os.path.join(NPZ, f"skeptic_long25_{name}__P0.8_A40_comy10_25s.npz")
    r = np.load(f, allow_pickle=True)
    ph = r["phase"]
    trot = np.flatnonzero(ph == "trot")
    t = r["t"][trot] - r["t"][trot[0]]
    xy = r["xy_trunk"][trot]
    yaw = np.degrees(r["yaw"][trot])
    # slope of y in 5 s windows (mm/s)
    slopes = []
    for a in np.arange(0.0, 25.0, 5.0):
        sel = (t >= a) & (t < a + 5.0)
        p = np.polyfit(t[sel], 1e3 * xy[sel, 1], 1)
        slopes.append(round(float(p[0]), 2))
    row = dict(y_slope_mm_per_s_5s_windows=slopes,
               y_total_mm=round(float(1e3 * (xy[-1, 1] - xy[0, 1])), 1),
               x_total_mm=round(float(1e3 * (xy[-1, 0] - xy[0, 0])), 1),
               yaw_total_deg=round(float(yaw[-1] - yaw[0]), 2),
               tilt_max_by_5s=[round(float(np.degrees(np.max(np.maximum(np.abs(r["roll"][trot][(t >= a) & (t < a + 5)]), np.abs(r["pitch"][trot][(t >= a) & (t < a + 5)]))))), 2) for a in np.arange(0.0, 25.0, 5.0)])
    out["walk25"][name] = row
    print(name, row)

json.dump(out, open("doc/trot_posture/data/skeptic_analysis.json", "w"), indent=1, default=str)
