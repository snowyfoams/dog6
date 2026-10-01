"""Judge tie-break: is the 0.5-0.9 deg gap between the top acceptable stands
(Y105_h160, Y95_h160, WIDE_h160) above the run-to-run noise?  Two probes per
candidate at 0.8 s / 40 mm: the mirrored CoM offset (com_y -10 mm; any
left/right asymmetry of the sim is noise) and the same +10 mm offset flown for
18 s instead of 12 (does the max tilt grow or wander?).

usage: python judge_tiebreak.py NAME PROBE   (PROBE = mirror | long)
writes doc/trot_posture/data/judge_tiebreak_<NAME>_<PROBE>.json
"""
import json, sys
sys.path.insert(0, "doc/trot_posture")
import evalpose as EP

name, probe = sys.argv[1], sys.argv[2]
spec = json.load(open(f"doc/trot_posture/data/cand_{name}.json"))["info"]["spec"]
pose, info = EP.build(EP.Spec(**spec))
if probe == "mirror":
    m = EP.run_trot(pose, spec["h"], 0.8, 40.0, hw=dict(com_err_mm=(0.0, -10.0)), trot_s=12.0)
else:
    m = EP.run_trot(pose, spec["h"], 0.8, 40.0, hw=dict(com_err_mm=(0.0, 10.0)), trot_s=18.0, t_max=40.0)
m["name"], m["probe"] = name, probe
json.dump(m, open(f"doc/trot_posture/data/judge_tiebreak_{name}_{probe}.json", "w"), indent=1)
print(name, probe, {k: m[k] for k in ("max_tilt_trot", "trot_xy_drift_mm", "trot_yaw_drift_deg", "fell", "peak_tau_trot", "trot_foot_fn_peak_n")})
