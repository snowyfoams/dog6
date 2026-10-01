"""Judge's ranking of the 28 trot-posture candidates -> doc/trot_posture/data/ranking.md

Order of criteria: (1) survived counts, then worst tilt (0.8 s clock first,
0.5 s second; differences < 0.3 deg treated as ties) and walk; (2) crouch
height, crouch hold torque, rise drift; (3) diagonal-pair static and HOLD
torque; (4) margins (knee >= 25 deg from straight AND reach <= 0.88, link
clearance >= 5 mm, abduction, soft-limit margin, analyst flags).  Candidates
failing the knee/reach rule are placed below every acceptable survivor.
"""
import json
S = {r["name"]: r for r in json.load(open("doc/trot_posture/data/candidates_summary.json"))}
I = {n: json.load(open(f"doc/trot_posture/data/cand_{n}.json"))["info"] for n in S}

ORDER = [  # (name, note)
 ("Y105_h160",    "WINNER. Best acceptable tilt; walk 75 mm (largest); abduction carries the hold (0.62); reach 0.87 at the edge of the rule"),
 ("Y95_h160",     "RUNNER-UP. +0.6 deg vs winner, hold 0.48, knee 43, crouch margin 11 deg"),
 ("WIDE_h160",    "CONSERVATIVE. Today's WIDE raised 15 mm; lowest hold torque of all (0.38); crouch 14, soft-limit margin 7 deg"),
 ("X60_y85_h160", "tilt tie with WIDE_h160; lowest diagonal torque (0.78) but crouch 0.1 deg from the soft limit, crouch tau 0.74"),
 ("PAR60_w85_160","tilt tie with WIDE_h160; parallel family: crouch 140 (cannot fold), hold 0.84, rise drift +2 mm, walks -x"),
 ("Y95_h145",     "y95 at today's height: 6.93; crouch 18, hold 0.48"),
 ("WIDE",         "today's WIDE as-is; tie with X60_w85_145/UNDER_x_y85; crouch 40 recorded, 14 feasible (same geometry as WIDE_h160)"),
 ("X60_w85_145",  "tie with WIDE; crouch at the soft limit (0.1 deg), hold 0.61"),
 ("UNDER_x_y85",  "tie with WIDE; feet under hips: crouch 84, hold 1.04, diag 2.02, com_x sensitive"),
 ("X60_160",      "y65: 8.28; tie with XIN60/PAR80 but crouch 26 vs 112-142; walks least (22 mm)"),
 ("XIN60_160",    "tie; knees-in: crouch 142, hold 0.90, friction-sensitive"),
 ("PAR80_160",    "tie; crouch 112, rise drift +3.9, hold 0.93, reach 0.88 at the rule"),
 ("NOM_y75",      "8.87; crouch 8 (near floor), hold 0.40; cheap but 2x the winner's tilt"),
 ("FOLD_rc80",    "fold family with rear feet moved back: 8.92; crouch 112, hold 1.11, drift +2.7"),
 ("FOLD_rc70",    "tie with rc80; crouch 130, hold 1.08, most friction/mass sensitive (+1.9 deg)"),
 ("XIN_splay",    "9.26; crouch 114, hold 1.13"),
 ("X80_w85_125",  "9.67 at 0.8 but 6.60 at 0.5; low stand (knee 81), fn 39 N"),
 ("NOMINAL",      "today's NOMINAL: 9.66; floor rest (crouch 0, 0.23 N m), hold 0.42"),
 ("X60_145",      "9.85; crouch 26 at the soft limit, com_x sensitive"),
 ("UNDER_x",      "9.62; feet under hips, crouch 66, hold 1.08, com_x 8.4"),
 ("XIN80_145",    "10.16; crouch 112, hold 1.14"),
 ("X80_125",      "11.45, worst survivor; fn 53 N; floor rest"),
 ("Y95_h175",     "EXCLUDED by rule: knee 1.4 deg from straight, reach 0.92 (flown at 168 mm in sim, so its 3.85 was never at that knee)"),
 ("WIDE_h175",    "EXCLUDED by rule: knee 14, reach 0.91"),
 ("WIDE_h170",    "EXCLUDED by rule: reach 0.89 (knee 28 clears 25; flown at 164.6 mm); the natural stretch target is ~165 mm at y85"),
 ("X100_y95_h160","EXCLUDED by rule: reach 0.90; also doubles ideal tilt and hip torque"),
 ("X100_y85_h160","EXCLUDED by rule: reach 0.89"),
 ("FOLD_asis",    "today's FOLD: 0/13 at 0.8 s (CoM 20.5 mm off both diagonals), 12.5 deg at 0.5 s, hold 1.47, kneels on the knee motors"),
]
assert sorted(n for n, _ in ORDER) == sorted(S)

hdr = ("| # | name | family | xf/xr | abs y | h | crouch (tau) | surv 0.5/0.8 | worst tilt 0.5/0.8 | walk 0.5/0.8 | hold | diag | knee (reach) | note |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
rows = []
for k, (n, note) in enumerate(ORDER, 1):
    r, i = S[n], I[n]
    ct = max(i["crouch_tau_4foot_max_nm"])
    cr = f"{r['crouch']:.0f} ({ct:.2f})" + (" floor" if r["floor_rest"] else "")
    w = lambda a, b: f"{a:.2f} / {b:.2f}" if b is not None else f"{a:.2f} / fell"
    wk = lambda a, b: f"{a:.0f} / {b:.0f}" if b is not None else f"{a:.0f} / -"
    rows.append(f"| {k} | {n} | {r['family']} | {r['xf']:+.0f}/{r['xr']:+.0f} | {r['y']:.0f} | {r['h']:.0f} | {cr} | {r['s05']} / {r['s08']} | "
                f"{w(r['w05'], r['w08'])} | {wk(r['walk05'], r['walk08'])} | {r['hold_tau']:.2f} | {r['diag_knee']:.2f} | {r['knee']:.0f} ({r['reach']:.2f}) | {note} |")
table = hdr + "\n".join(rows) + "\n"

tb = {}
for n in ("Y105_h160", "Y95_h160", "WIDE_h160"):
    for p in ("mirror", "long"):
        m = json.load(open(f"doc/trot_posture/data/judge_tiebreak_{n}_{p}.json"))
        tb[(n, p)] = (m["max_tilt_trot"], m["trot_xy_drift_mm"])
tbt = ("| name | com_y +10 mm, 12 s (study) | com_y -10 mm, 12 s (mirror) | com_y +10 mm, 18 s |\n|---|---|---|---|\n" +
       "\n".join(f"| {n} | {S[n]['w08']:.2f} deg, walk {S[n]['walk08']:.0f} mm | {tb[(n,'mirror')][0]:.2f} deg, walk ({tb[(n,'mirror')][1][0]:.0f}, {tb[(n,'mirror')][1][1]:.0f}) mm | "
                 f"{tb[(n,'long')][0]:.2f} deg, walk ({tb[(n,'long')][1][0]:.0f}, {tb[(n,'long')][1][1]:.0f}) mm |"
                 for n in ("Y105_h160", "Y95_h160", "WIDE_h160")) + "\n")

md = f"""# Trot-in-place stand posture: judge's ranking

Columns: crouch = lowest feasible crouch, floor to trunk bottom, mm (crouch hold torque, max joint, N m, from info.crouch_tau_4foot_max_nm);
surv = perturbations survived of 13 at 0.5 s/20 mm and 0.8 s/40 mm; worst tilt = max tilt among survivors, deg; walk = worst trunk walk among survivors over 12 s, mm;
hold = max mean joint torque in HOLD, N m; diag = static peak joint torque on a diagonal pair, N m; knee = deg from straight at the IK stand (reach = abd-hip-to-foot / 233 mm).
Criteria in order: survival, then worst tilt (0.8 s clock first; < 0.3 deg = tie) and walk; crouch height / hold cost / rise drift; diagonal and HOLD torque; margins
(knee < 25 deg or reach > 0.88 is not acceptable as a default stand, whatever the tilt: those rows sit below every acceptable survivor).

{table}
## Tie-break probes (0.8 s / 40 mm, this judge; scripts/judge_tiebreak.py, data/judge_tiebreak_*.json)

{tbt}
The sim is deterministic and left/right symmetric to 0.06 deg and the max tilt does not grow between 12 s and 18 s (the walk is a constant 4.5-6 mm/s), so the 0.6 deg
(Y105 vs Y95) and 0.9 deg (Y95 vs WIDE_h160) gaps are real within the model; they are still a single-perturbation (com_y +10 mm) ranking.

## Verdict

- WINNER: **Y105_h160** = Spec(family="x", xf=+81, xr=-81, y=105, h=160), crouch 17 mm (auto), i.e. the CAD X-configuration (front knees forward, rear knees back, feet 81 mm ahead of / behind their hips)
  with the half-track widened from 65 to 105 mm and the stand raised from 145 to 160 mm. Worst tilt 3.75 / 4.81 deg (vs 9.66 for NOMINAL, 12.5 / fell for FOLD), 13/13 at both clocks,
  knee 40 deg from straight, reach 0.87, abduction 77.5 deg at the stand (12.5 deg out from leg-down) and 44 deg at the crouch, link clearance 5.7 mm at the crouch, crouch soft-limit margin 13.9 deg.
  Costs: the widest track loads the abduction joints in HOLD (0.62 N m, vs 0.38 for WIDE_h160), the largest lateral walk under the 10 mm CoM error (75 mm / 12 s), and reach 0.87 is at the edge of the rule.
- RUNNER-UP: **Y95_h160** = Spec("x", +81, -81, y=95, h=160), crouch 18. 4.13 / 5.40 deg (+0.6 deg on the winner), hold 0.48 N m, knee 43, reach 0.86, crouch margin 11 deg. If current draw in HOLD outweighs the last 0.6 deg, this is the pick.
- CONSERVATIVE: **WIDE_h160** = today's WIDE posture (x family, feet +81/-81, |y| 85) with the stand raised from 145 to 160 mm, crouch 14 (the WIDE geometry's true lowest crouch; the 40 recorded for WIDE was a given value, not a search).
  One number changed from a posture hw already has; 4.53 / 6.28 deg (91 % of the winner's 0.5 s gain over FOLD), the lowest HOLD torque of the whole set (0.38 N m, a quarter of FOLD's 1.47), knee 45 deg, reach 0.85. Flying WIDE as-is (h145) already gives 13/13 at both clocks at 5.54 / 7.96.
"""
open("doc/trot_posture/data/ranking.md", "w").write(md)
print(md)
