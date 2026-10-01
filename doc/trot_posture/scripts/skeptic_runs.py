"""Skeptic probes of the judge's pick (Y105_h160) from the simulation's blind
spots.  Each VARIANT is a small bundle of runs on one posture:

  nospring   stance xy spring OFF (hw.trot flies without it today)
  noprime    today's ramp-from-zero handover from the posture's own crouch
  srb145     SRB pinned at 145 mm (the robot's existing posture object +
             --height): Spec built with h=145, flown at h_mm = the stand
  long25     25 s at 0.8 s / 40 mm with com_y +10 mm: does the walk saturate?
  hard       joint_fric 0.3 + imu 30 ms + kt 0.85 together (+ com_y 10)
  nohold     the joint hold OFF (kp_joint 0)
  clock06    0.6 s / 30 mm, between the two tested clocks
  comy15     com_y +15 mm, beyond the tested range, 0.8 s / 40 mm

usage: python skeptic_runs.py NAME VARIANT
NAME is a cand_<NAME>.json in data/ (the spec is read from it).
writes data/skeptic_<VARIANT>_<NAME>.json; long runs keep their npz in
the directory named by $SKEPTIC_NPZ (default: data/).
"""
import json, os, sys, time
sys.path.insert(0, "doc/trot_posture")
import evalpose as EP

name, variant = sys.argv[1], sys.argv[2]
NPZ = os.environ.get("SKEPTIC_NPZ", EP.DATA)
spec = json.load(open(f"doc/trot_posture/data/cand_{name}.json"))["info"]["spec"]
spec = dict(spec)
H_RUN = spec["h"]
if variant == "srb145":
    # keep the same crouch as the study's posture, pin the SRB at 145
    cand = json.load(open(f"doc/trot_posture/data/cand_{name}.json"))["info"]
    spec["crouch_h"] = cand["crouch_h_mm"]
    spec["h"] = 145.0
pose, info = EP.build(EP.Spec(**spec))
out = {"name": name, "variant": variant, "spec_built": spec, "h_run": H_RUN,
       "crouch_h_mm": info["crouch_h_mm"], "runs": {}}
COMY10 = dict(com_err_mm=(0.0, 10.0))

def keep(tag):
    return os.path.join(NPZ, f"skeptic_{variant}_{name}__{tag}.npz")

def stand(tag, **kw):
    t0 = time.time()
    m = EP.run_stand(pose, H_RUN, **kw)
    out["runs"][tag] = m
    print("%-10s %-28s stood=%s h_hold %s h_min_rise %s drift %s/%s slip %s tilt_rise %s hold %s rise_pk %s belly %s stop=%s (%.0f s)" % (
        name, tag, m.get("stood"), m.get("h_hold_mm"), m.get("h_min_rise_mm"), m.get("drift_x_mm"),
        m.get("drift_y_mm"), m.get("foot_slip_max_mm"), m.get("tilt_max_rise_deg"),
        m.get("tau_hold_mean_max_nm"), m.get("tau_rise_peak_nm"), m.get("belly_max_n"), m["stop"], time.time() - t0))
    sys.stdout.flush()

def trot(tag, period, apex, **kw):
    t0 = time.time()
    m = EP.run_trot(pose, H_RUN, period, apex, **kw)
    out["runs"][tag] = m
    print("%-10s %-28s fell=%-5s trot %5.2f tilt %5.2f xy %s yaw %s fn %s pk %s hold_after %s stop=%s (%.0f s)" % (
        name, tag, m["fell"], m.get("trot_s", 0), m.get("max_tilt_trot", -1), m.get("trot_xy_drift_mm"),
        m.get("trot_yaw_drift_deg"), m.get("trot_foot_fn_peak_n"), m.get("peak_tau_trot"),
        m.get("hold_after_tilt_deg"), str(m["stop"])[:40], time.time() - t0))
    sys.stdout.flush()

if variant == "nospring":
    stand("stand_nospring", stance_xy=(0.0, 0.0))
    trot("P0.8_A40_comy10_nospring", 0.8, 40.0, hw=COMY10, stance_xy=(0.0, 0.0))
    trot("P0.5_A20_comy10_nospring", 0.5, 20.0, hw=COMY10, stance_xy=(0.0, 0.0))
    trot("P0.8_A40_ideal_nospring", 0.8, 40.0, stance_xy=(0.0, 0.0))
elif variant == "noprime":
    stand("stand_noprime_spring", prime=False, keep=keep("stand_noprime_spring"))
    stand("stand_noprime_nospring", prime=False, stance_xy=(0.0, 0.0), keep=keep("stand_noprime_nospring"))
    trot("P0.8_A40_comy10_noprime", 0.8, 40.0, hw=COMY10, prime=False)
elif variant == "srb145":
    stand("stand_srb145")
    trot("P0.8_A40_comy10_srb145", 0.8, 40.0, hw=COMY10)
    trot("P0.5_A20_comy10_srb145", 0.5, 20.0, hw=COMY10)
    trot("P0.8_A40_ideal_srb145", 0.8, 40.0)
elif variant == "long25":
    trot("P0.8_A40_comy10_25s", 0.8, 40.0, hw=COMY10, trot_s=25.0, t_max=45.0,
         keep=keep("P0.8_A40_comy10_25s"))
elif variant == "hard":
    HARD = dict(frictionloss=0.3, imu_latency=30e-3, kt_scale=0.85)
    trot("P0.8_A40_hard", 0.8, 40.0, hw=HARD)
    trot("P0.5_A20_hard", 0.5, 20.0, hw=HARD)
    trot("P0.8_A40_hard_comy10", 0.8, 40.0, hw=dict(HARD, **COMY10))
elif variant == "nohold":
    stand("stand_nohold", joint_hold=(0.0, 0.0))
    trot("P0.8_A40_comy10_nohold", 0.8, 40.0, hw=COMY10, joint_hold=(0.0, 0.0))
    trot("P0.8_A40_ideal_nohold", 0.8, 40.0, joint_hold=(0.0, 0.0))
    trot("P0.5_A20_comy10_nohold", 0.5, 20.0, hw=COMY10, joint_hold=(0.0, 0.0))
elif variant == "clock06":
    trot("P0.6_A30_comy10", 0.6, 30.0, hw=COMY10)
    trot("P0.6_A30_combo_mild", 0.6, 30.0, hw=EP.PERTURB["combo_mild"])
    trot("P0.6_A30_ideal", 0.6, 30.0)
elif variant == "comy15":
    trot("P0.8_A40_comy15", 0.8, 40.0, hw=dict(com_err_mm=(0.0, 15.0)))
    trot("P0.5_A20_comy15", 0.5, 20.0, hw=dict(com_err_mm=(0.0, 15.0)))
else:
    raise SystemExit("unknown variant " + variant)

path = f"doc/trot_posture/data/skeptic_{variant}_{name}.json"
json.dump(out, open(path, "w"), indent=1, default=str)
print("->", path)
