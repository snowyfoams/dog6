"""Skeptic run: hw's existing WIDE crouch (40 mm) with the SRB pinned at 145
(what hw.stand does: CrouchPose.srb derives at cfg.H_LIFT) flown at --height 160.
usage: skeptic_wide_c40.py stand | trot PERIOD APEX PERTURB"""
import json, sys
sys.path.insert(0, "doc/trot_posture"); sys.path.insert(0, "doc/crouch_trot")
import evalpose as EP
from robustness import PERTURB
spec = EP.Spec("WIDE_c40_pin145", "x", 81, -81, 85, 145, 40)
pose, info = EP.build(spec)
tag = sys.argv[1]
if tag == "stand":
    m = EP.run_stand(pose, 160.0, hw={}, prime=True, stance_xy=(400.0, 25.0))
    name = "stand"
else:
    period, apex, pert = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    m = EP.run_trot(pose, 160.0, period, apex, hw=PERTURB[pert], prime=True,
                    stance_xy=(400.0, 25.0), trot_s=12.0, t_max=32.0)
    name = "P%s_A%.0f_%s" % (period, apex, pert)
m = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()}
json.dump(dict(spec=spec.as_dict(), run=name, metrics=m, srb_com_mm=info["srb_com_mm"],
               srb_inertia_diag=info["srb_inertia_diag"]),
          open("doc/trot_posture/data/skeptic_wide_c40_%s.json" % name, "w"), indent=1, default=str)
print(name, {k: v for k, v in m.items() if k in ("fell", "max_tilt_trot", "trot_xy_drift_mm", "peak_tau_trot",
      "hold_after_tilt_deg", "trot_foot_fn_peak_n", "stood", "h_hold_mm", "tau_hold_mean_max_nm", "tau_hold_by_joint_nm",
      "tau_crouch_hold_peak_nm", "drift_x_mm", "tilt_max_rise_deg", "stop")})
