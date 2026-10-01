"""mech_collect -- merge data/mech_<case>.json and data/mech_analytic.json into data/mech_results.json
and print the summary table (drift, trunk-vs-feet drift, pitch, h_min, floor-force impulse)."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
cases = {}
for f in sorted(glob.glob(os.path.join(DATA, "mech_*.json"))):
    name = os.path.basename(f)[5:-5]
    if name in ("results", "analytic"):
        continue
    d = json.load(open(f))
    d.pop("trace", None)
    cases[name] = d
analytic = json.load(open(os.path.join(DATA, "mech_analytic.json")))
out = {"note": ("doc/trot_posture/scripts/mech_run.py cases; drift_x_mm = trunk x at the last HOLD sweep minus the last "
                "CROUCH sweep; rel_feet_mm = -(mean foot x ahead of its hip at HOLD - at CROUCH) = trunk drift relative "
                "to the feet; impulse_* = integral of the net horizontal floor force ON the robot (mujoco.mj_contactForce, "
                "sampled every 4 ms sweep); analytic = doc/trot_posture/scripts/mech_analytic.py"),
       "cases": cases, "analytic": analytic}
json.dump(out, open(os.path.join(DATA, "mech_results.json"), "w"), indent=1, default=str)
print("%-26s %7s %7s %6s %6s %8s %8s %7s %6s  %s" % ("case", "drift", "rel", "pitch", "h_min", "J_rise", "J_hold", "Fx1st", "slip", "stop"))
for n, d in cases.items():
    if "drift_x_mm" not in d:
        print("%-26s  NO RISE: %s" % (n, str(d.get("stop"))[:70])); continue
    rel = -(sum(d["foot_x_rel_hip_hold_mm"]) - sum(d["foot_x_rel_hip_crouch_mm"])) / 4
    print("%-26s %+7.1f %+7.1f %+6.2f %6.1f %+8.4f %+8.4f %+7.3f %6.1f  %s" % (
        n, d["drift_x_mm"], rel, d["pitch_hold_deg"], d["h_min_rise_mm"], d["impulse_x_rise_Ns"],
        d["impulse_x_hold_Ns"], d.get("Fx_mean_rise_1st_half_N", float("nan")), d["foot_slip_max_mm"],
        "" if d["stop"] == "operator stop in hold" else str(d["stop"])[:50]))
print("-> %s  (%d cases)" % (os.path.join(DATA, "mech_results.json"), len(cases)))
