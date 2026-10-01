"""Today's ramp-from-zero handover sags the trunk ~8 mm below the crouch
before the law catches it; at Y105_h160's lowest crouch (17 mm) that puts the
left and right knee motors 10 mm into each other (mesh hulls), which the plant
(foot balls + trunk box only) never sees.  How much higher must the crouch be
for the sag to stay clear?  Runs the stand with prime=False from crouch
heights 17..35 mm and checks the mesh collisions at the deepest sweep.
writes data/skeptic_crouch_margin.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, "doc/trot_posture")
import evalpose as EP
from hw.balance import config as BCFG

NPZ = os.environ.get("SKEPTIC_NPZ", EP.DATA)
name = sys.argv[1] if len(sys.argv) > 1 else "Y105_h160"
spec = dict(json.load(open(f"doc/trot_posture/data/cand_{name}.json"))["info"]["spec"])
rows = []
for hc in [float(v) for v in (sys.argv[2:] or (17, 22, 25, 30, 35))]:
    spec["crouch_h"] = hc
    pose, info = EP.build(EP.Spec(**spec))
    f = os.path.join(NPZ, f"skeptic_cm_{name}__c{hc:.0f}.npz")
    st = EP.run_stand(pose, spec["h"], prime=False, keep=f)
    r = np.load(f, allow_pickle=True)
    rise = np.flatnonzero(r["phase"] == "rise")
    i = rise[np.argmin(r["z_trunk"][rise])]
    z = float(r["z_trunk"][i]); q = np.asarray(r["q"][i]).reshape(4, 3)
    col = EP.collisions(q, z)
    zz = 1e3 * (r["z_trunk"][rise] - BCFG.TRUNK_BOTTOM_OFFSET)
    row = dict(crouch_h_mm=hc, crouch_collide=info["collide_crouch"],
               crouch_tau_hold_peak_nm=st.get("tau_crouch_hold_peak_nm"),
               crouch_soft_limit_margin_deg=info["crouch_soft_limit_margin_deg"],
               h_min_rise_mm=st.get("h_min_rise_mm"), sag_mm=round(hc - st.get("h_min_rise_mm"), 1),
               below_crouch_s=round(float(((zz < hc - 0.5).sum()) * (r["t"][1] - r["t"][0])), 2),
               sag_collide={k: col[k] for k in ("min_mm", "leg_leg_mm", "leg_floor_mm", "leg_trunk_mm", "penetrating")},
               h_hold_mm=st.get("h_hold_mm"), drift_x_mm=st.get("drift_x_mm"),
               tilt_max_rise_deg=st.get("tilt_max_rise_deg"), tau_rise_peak_nm=st.get("tau_rise_peak_nm"),
               hold_nm=st.get("tau_hold_mean_max_nm"), stop=st["stop"])
    rows.append(row)
    print(name, {k: v for k, v in row.items() if k != "crouch_collide"}, "crouch leg_leg", info["collide_crouch"]["leg_leg_mm"])
    sys.stdout.flush()
json.dump(rows, open(f"doc/trot_posture/data/skeptic_crouch_margin_{name}.json", "w"), indent=1, default=str)
