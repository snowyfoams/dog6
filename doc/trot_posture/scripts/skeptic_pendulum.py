"""Skeptic check: inverted-pendulum growth about the diagonal and the roll
lever of a lateral CoM error, from each candidate's own SRB geometry."""
import json, sys
import numpy as np
DATA = "doc/trot_posture/data"
g = 9.81; m = 5.8828
rows = {}
for name in ["Y105_h160", "Y95_h160", "WIDE_h160", "WIDE", "NOMINAL", "Y95_h145", "FOLD_asis"]:
    d = json.load(open("%s/cand_%s.json" % (DATA, name)))
    i = d["info"]; s = d["stand"]
    h_hold = s["h_hold_mm"]                       # floor to trunk bottom, sim HOLD
    com_z = i["srb_com_mm"][2]                    # rel trunk origin
    h_com = (h_hold + 35.0 + com_z) * 1e-3        # m, CoM above floor (foot ball centre ~ +15 mm but the pivot is the floor contact)
    Ixx, Iyy, Izz = i["srb_inertia_diag"]
    a = np.radians(i["diag_angle_deg"])
    I_axis = Ixx * np.cos(a)**2 + Iyy * np.sin(a)**2
    I_piv = I_axis + m * h_com**2
    lam = np.sqrt(m * g * h_com / I_piv)
    lam_pt = np.sqrt(g / h_com)                   # point-mass pendulum
    dy = 0.010
    d_perp = dy * np.cos(a)                       # lateral error -> distance to the diagonal
    M = m * g * d_perp
    r = {}
    for T in (0.100, 0.160):
        # tilt after one swing starting at rest with a constant offset d_perp:
        # theta(T) = (d/h)(cosh(lam T) - 1); growth factor of a free error e^{lam T}
        r["T%.0fms" % (1e3 * T)] = dict(growth=round(float(np.exp(lam * T)), 2),
                                        tilt_from_rest_deg=round(float(np.degrees(d_perp / h_com * (np.cosh(lam * T) - 1))), 2),
                                        growth_pointmass=round(float(np.exp(lam_pt * T)), 2))
    # static roll offset of a PD with no integrator, roll gain 290 (1/s^2) on Ixx
    static_roll = np.degrees(m * g * dy / (290.0 * Ixx))
    rows[name] = dict(h_com_mm=round(1e3 * h_com, 1), diag_deg=i["diag_angle_deg"], track=i["track_mm"],
                      Ixx=Ixx, I_axis=round(I_axis, 4), I_pivot=round(I_piv, 4), lam=round(float(lam), 2),
                      tau_ms=round(1e3 / lam, 1), d_perp_mm=round(1e3 * d_perp, 2), M_Nm=round(M, 3),
                      ang_acc=round(M / I_piv, 2), static_roll_290_deg=round(float(static_roll), 2),
                      sim_hold_after_tilt=d["trots"]["P0.8_A40"]["com_y_+10mm"].get("hold_after_tilt_deg"),
                      sim_worst_05=d["trots"]["P0.5_A20"]["com_y_+10mm"]["max_tilt_trot"],
                      sim_worst_08=d["trots"]["P0.8_A40"]["com_y_+10mm"]["max_tilt_trot"], **r)
json.dump(rows, open(DATA + "/skeptic_pendulum.json", "w"), indent=1)
for n, r in rows.items():
    print(n, json.dumps(r))
