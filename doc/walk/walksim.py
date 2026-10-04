"""walksim -- `hw.fold_walk` in MuJoCo: does the walk walk?

    python doc/walk/walksim.py                  the shipped walk, 4 in parallel
    python doc/walk/walksim.py fwd10 yaw20      named scenarios only
    python doc/walk/walksim.py --list           what the names are
    python doc/walk/walksim.py --v0             the first design (impedance, 0.80)
    python doc/walk/walksim.py --grid           the osc/0.70 grid at 25/25/30
    python doc/walk/walksim.py --grid2          ... at 15/15/20
    python doc/walk/walksim.py --sens           one knob at a time at 0.10/0.15
    python doc/walk/walksim.py --repeats        3 phases each, five configs
    python doc/walk/walksim.py --qrepeats       6 phases each, the swing gains
    python doc/walk/walksim.py --xrepeats       the shipped walk, 6 phases, and
                                                the hardware unknowns at 3

BARE NAMES ARE THE ENTRY POINT AS SHIPPED: `entry()`'s defaults are
`hw.fold_walk`'s (config.WALK_SWING_LAW, WALK_DUTY, WN_SWING_OSC, ...).
Every other family spells out what it flew, so its name keeps meaning the
same runs when a default moves.  doc/walk/README.md quotes each table.

WHAT RUNS, AND WHAT IS EMULATED
    `doc/crouch_trot/hwsim.py`'s plant and loop (one knob added for this
    study, `HwParams.armature_scale`, default 1 = unchanged): the repo's own
    StandSequence, BalanceLaw (with `walk` and `alloc` exactly as
    `hw.fold_walk.walk_options()` builds them), TrotGait and SafetyGate, the
    CAN round robin, encoder quantisation, the 40 Hz qd filter, IMU latency,
    the drivers' current loop.  On top, the runner's ESTIMATOR_SLOT: the
    state estimator `hw.trot_esti.EstimatorFeed` stepped every sweep from the
    model's own accelerometer (100 Hz, 3 ms late) and fed to the law ONE SWEEP
    LATE, as `hw.stand.run` does with `--est-xy`.  The operator presses T and
    then "holds keys": the walk's command is set from a schedule.

    What it does not have: the robot's armature (the model's is DOG5's, so
    the law's M0 is exact here and is not on the robot -- the _arm06
    scenarios put the plant at the bench's 1/1.7 reading), backlash, the
    foot's real friction (MuJoCo's 1.0 unless a scenario sets it), slip the
    filter cannot see beyond what MuJoCo's contact produces.

WHAT IT REPORTS, per scenario, over the steady second half of each command
    v_cmd vs v_true   the trunk's TRUE velocity in its own heading frame
    r_cmd vs r_true   the true yaw rate
    tilt              the worst |roll|, |pitch| of the trot, true
    tau               the worst applied joint torque of the trot
    slip              loaded-foot horizontal travel per second of trot
    late              how far after the clock's touchdown the foot touched
    law / qp          the law's p50 / p95 time and the QP's worst iterations
    fell              a stop, or 15 deg of tilt, before the operator's exit
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "doc", "crouch_trot"))

import hwsim as H                    # noqa: E402
from sim import coordinates as C, params as P   # noqa: E402
from hw import fold_stand as FS, fold_trot as FT, trot as TR, imu as IMU  # noqa: E402
from hw import fold2_trot as F2       # noqa: E402
from hw.balance import config as BCFG, posture as POSE, controller as BCTRL  # noqa: E402
from hw.balance import swing as BSWING  # noqa: E402
from hw.balance.trajectory import Command, WalkReference   # noqa: E402
from hw.balance.walk import WalkPlan   # noqa: E402
from hw.trot_esti import EstimatorFeed  # noqa: E402

DATA = os.path.join(HERE, "data")
T_HOLD = 3.0                          # s of HOLD before T

#: name -> dict(schedule=[(t_after_T, vx, vy, yaw_rate_deg)], trot_s, opts)
#: Every walk starts with 1.5 s in place and ends with 2 s of stop.
def _walk(vx=0.0, vy=0.0, r=0.0, hold=5.0, t_cmd=1.5, **opts):
    return dict(schedule=[(0.0, 0.0, 0.0, 0.0), (t_cmd, vx, vy, r),
                          (t_cmd + hold, 0.0, 0.0, 0.0)],
                trot_s=t_cmd + hold + 2.0, opts=opts)


#: THE CONFIGURATIONS, EXPLICIT: a scenario's name means one thing even when
#: a default in config.py moves.  Bare names (fwd10, lat05, ...) are
#: `hw.fold_walk` AS SHIPPED -- `entry()`'s defaults, read from config.py;
#: every other family spells out what it flew.
INPLACE = [(0.0, 0.0, 0.0, 0.0)]
W253030 = np.array([25.0, 25.0, 30.0])
#: The trot's settle (gait.TrotGait's default, DOG5's re-level), which every
#: family below flew until the q_ repeats found it hurt the walk.
SETTLE_TROT = BCFG.SETTLE_S
#: v0_: the first design -- the walking impedance swing at the trot's duty.
V0 = dict(swing_law="impedance", duty=0.8, settle=SETTLE_TROT)
SCENARIOS = {
    # The flown trot in place, nothing of the walk attached: the baseline.
    "inplace_flown": dict(schedule=INPLACE, trot_s=8.0,
                          opts=dict(alloc="wls", walk=False, duty=0.8,
                                    settle=SETTLE_TROT)),
    "v0_inplace":  dict(schedule=INPLACE, trot_s=8.0, opts=dict(V0)),
    "v0_inplace_wls": dict(schedule=INPLACE, trot_s=8.0,
                           opts=dict(V0, alloc="wls")),
    "v0_fwd05":  _walk(vx=0.05, **V0),
    "v0_fwd10":  _walk(vx=0.10, **V0),
    "v0_back10": _walk(vx=-0.10, **V0),
    "v0_lat05":  _walk(vy=0.05, **V0),
    "v0_yaw20":  _walk(r=20.0, **V0),
    "v0_fwd10_ff":  _walk(vx=0.10, swing_ff=True, **V0),
    "v0_fwd10_osc": _walk(vx=0.10, swing_law="osc", duty=0.8,
                          wn_swing=W253030, settle=SETTLE_TROT),
    "v0_fwd10_d70": _walk(vx=0.10, swing_law="impedance", duty=0.7,
                          settle=SETTLE_TROT),
}
V0SET = [k for k in SCENARIOS if k.startswith("v0_")] + ["inplace_flown"]

#: THE GRID: the walk as the runs above chose it -- the task-space swing law
#: and a 0.70 duty, 180 ms of swing -- across the command envelope, then the
#: same with one thing made worse at a time.
G = dict(swing_law="osc", duty=0.7, wn_swing=W253030, settle=SETTLE_TROT)
SCENARIOS.update({
    "g_inplace": dict(schedule=[(0.0, 0, 0, 0)], trot_s=8.0, opts=dict(G)),
    "g_fwd05":  _walk(vx=0.05, **G),
    "g_fwd10":  _walk(vx=0.10, **G),
    "g_fwd15":  _walk(vx=0.15, **G),
    "g_fwd20":  _walk(vx=0.20, **G),
    "g_back05": _walk(vx=-0.05, **G),
    "g_back10": _walk(vx=-0.10, **G),
    "g_lat05":  _walk(vy=0.05, **G),
    "g_latm05": _walk(vy=-0.05, **G),
    "g_lat08":  _walk(vy=0.08, **G),
    "g_lat10":  _walk(vy=0.10, **G),
    "g_yaw20":  _walk(r=20.0, **G),
    "g_yawm20": _walk(r=-20.0, **G),
    "g_yaw40":  _walk(r=40.0, **G),
    "g_combo":  _walk(vx=0.10, r=20.0, **G),
    "g_diag":   _walk(vx=0.10, vy=0.05, **G),
    "g_fwd10_wls":  _walk(vx=0.10, alloc="wls", **G),
    "g_fwd10_fric": _walk(vx=0.10, frictionloss=0.1, imu_ms=15.0, **G),
    "g_fwd10_mu05": _walk(vx=0.10, friction=0.5, **G),
    # The law's M0 against the plant's: the robot's rotor is not DOG5's, and
    # in the task-space law M0 is the gain (swing_control.py).
    "g_fwd10_m05":  _walk(vx=0.10, ff_scale=0.5, **G),
    "g_fwd10_m20":  _walk(vx=0.10, ff_scale=2.0, **G),
    "g_fwd10_imp":  _walk(vx=0.10, swing_law="impedance", duty=0.7,
                          settle=SETTLE_TROT),
    "g_fwd10_d80":  _walk(vx=0.10, swing_law="osc", duty=0.8,
                          wn_swing=W253030, settle=SETTLE_TROT),
    "g_fwd15_p08":  _walk(vx=0.15, period=0.8, **G),
    "g_lat10_p08":  _walk(vy=0.10, period=0.8, **G),
})
GRID = [k for k in SCENARIOS if k.startswith("g_")]

#: THE GRID AGAIN with the swing's bandwidth halved where the sensitivity
#: runs found it (s_fwd10_wn15): 15/15/20 rad/s keeps the swing's torque
#: rate inside the trot's 120 N*m/s slew, which 25/25/30 did not.
G2 = dict(G, wn_swing=np.array([15.0, 15.0, 20.0]))
for _name in list(GRID):
    _sc = SCENARIOS[_name]
    _o = dict(_sc["opts"])
    if _o.get("swing_law") == "osc" and _o.get("duty") == 0.7:
        _o["wn_swing"] = G2["wn_swing"]
    SCENARIOS["h" + _name[1:]] = dict(_sc, opts=_o)
SCENARIOS["h_fwd10_wn10"] = _walk(vx=0.10, **dict(G, wn_swing=np.array([10.0, 10.0, 15.0])))
SCENARIOS["h_fwd15_wn10"] = _walk(vx=0.15, **dict(G, wn_swing=np.array([10.0, 10.0, 15.0])))
SCENARIOS["h_fwd15_slew240"] = _walk(vx=0.15, tau_slew=240.0, **G2)
SCENARIOS["h_lat08_slew240"] = _walk(vy=0.08, tau_slew=240.0, **G2)
SCENARIOS["h_yaw40_slew240"] = _walk(r=40.0, tau_slew=240.0, **G2)
SCENARIOS["h_fwd15_z10"] = _walk(vx=0.15, zeta_swing=1.0, **G2)
GRID2 = [k for k in SCENARIOS if k.startswith("h_")]

#: REPEATS: one run is one gait phase at the command.  Each case again with
#: the command 0, 71 and 142 ms later -- a different phase of the 0.6 s
#: clock at the step, everything else equal -- so "it walks" is 3 of 3.
_W15 = np.array([15.0, 15.0, 20.0])
_W1525 = np.array([15.0, 25.0, 20.0])
_CASES = {"fwd05": dict(vx=0.05), "fwd10": dict(vx=0.10),
          "back10": dict(vx=-0.10), "lat05": dict(vy=0.05),
          "latm05": dict(vy=-0.05), "yaw20": dict(r=20.0),
          "yaw40": dict(r=40.0), "fwd15": dict(vx=0.15), "inplace": dict()}
_CONFIGS = {
    "w25": (dict(G), ["inplace", "fwd05", "fwd10", "back10", "lat05",
                      "latm05", "yaw20"]),
    "w15": (dict(G, wn_swing=_W15), ["inplace", "fwd05", "fwd10", "back10",
                                     "lat05", "yaw20"]),
    "w1525": (dict(G, wn_swing=_W1525), ["fwd10", "lat05", "latm05"]),
    "w15s240": (dict(G, wn_swing=_W15, tau_slew=240.0),
                ["fwd10", "lat05", "yaw40", "fwd15"]),
    "w25s240": (dict(G, tau_slew=240.0), ["fwd10", "lat05", "yaw40"]),
}
for _cname, (_copts, _cases) in _CONFIGS.items():
    for _case in _cases:
        for _seed in range(3):
            SCENARIOS["r_%s_%s_%d" % (_cname, _case, _seed)] = _walk(
                t_cmd=1.5 + 0.071 * _seed, **_CASES[_case], **_copts)
REPEATS = [k for k in SCENARIOS if k.startswith("r_")]

#: THE SWING'S GAINS BY AXIS.  The foot's apparent mass is 0.31 / 0.36 kg in
#: x / y and 5.8 kg in z (the reflected rotor through a short z lever), so a
#: z bandwidth costs torque RATE -- the gate's 120 N*m/s -- and an x/y one
#: costs next to nothing.  Six phases each, the command 0..500 ms into the
#: 0.6 s clock.
_QCASES = dict(_CASES, combo=dict(vx=0.10, r=20.0), lat08=dict(vy=0.08))
_QCONFIGS = {
    "w25": dict(G),
    "w2520": dict(G, wn_swing=np.array([25.0, 25.0, 20.0])),
    "w3520": dict(G, wn_swing=np.array([35.0, 35.0, 20.0])),
    "w2520n": dict(G, wn_swing=np.array([25.0, 25.0, 20.0]), settle=0.0),
}
for _cname, _copts in _QCONFIGS.items():
    for _case in ("fwd10", "lat05", "latm05", "fwd15", "lat08", "yaw40",
                  "combo"):
        for _seed in range(6):
            SCENARIOS["q_%s_%s_%d" % (_cname, _case, _seed)] = _walk(
                t_cmd=1.5 + 0.1 * _seed, **_QCASES[_case], **_copts)
QREPEATS = [k for k in SCENARIOS if k.startswith("q_")]
#: The batch as it was run: w3520 stopped after 31 of its 42 (it was already
#: the worst), then the settle comparison on its own.
QFINAL = [k for k in QREPEATS if k.startswith("q_w2520n_")]


def repeat_table(results) -> str:
    """Repeats folded: per config and case, falls of n, worst tilt and
    torque, and the mean true velocity of the runs that stood."""
    groups = {}
    for r in results:
        _, cname, case, _seed = r["name"].split("_")
        groups.setdefault((cname, case), []).append(r)
    lines = ["%-9s %-8s %-6s %-21s %-14s %-12s %s" % (
        "config", "case", "fell", "cmd vx vy r", "true (stood)", "tilt max",
        "tau max")]
    for (cname, case), rs in groups.items():
        stood = [r for r in rs if not r["fell"]]
        segs = [next((s for s in r.get("segments", []) if any(s["cmd"])),
                     None) for r in stood]
        segs = [s for s in segs if s is not None]
        cmd = segs[0]["cmd"] if segs else [0, 0, 0]
        if segs:
            v = np.mean([[s["v_true"][0], s["v_true"][1], s["r_true"]]
                         for s in segs], axis=0)
            true = "%+.3f %+.3f %+5.1f" % tuple(v)
        else:
            true = "-"
        lines.append("%-9s %-8s %d/%d    %+.2f %+.2f %+5.0f    %-21s %5.1f deg   %5.2f"
                     % (cname, case, len(rs) - len(stood), len(rs), cmd[0],
                        cmd[1], cmd[2], true,
                        max(r.get("max_tilt", float("nan")) for r in rs),
                        max(r.get("peak_tau", float("nan")) for r in rs)))
    return "\n".join(lines)

#: SENSITIVITY: the grid's walk at 0.10 / 0.15 m/s with one knob moved.
_KNOBS = {"slew240": dict(tau_slew=240.0), "slew360": dict(tau_slew=360.0),
          "wn15": dict(wn_swing=np.array([15.0, 15.0, 20.0])),
          "kv06": dict(kv=0.06), "kv10": dict(kv=0.10),
          "h30": dict(swing_height=0.030)}
for _k, _o in _KNOBS.items():
    SCENARIOS["s_fwd10_" + _k] = _walk(vx=0.10, **{**G, **_o})
    SCENARIOS["s_fwd15_" + _k] = _walk(vx=0.15, **{**G, **_o})
SENS = [k for k in SCENARIOS if k.startswith("s_")]
#: `hw.fold_walk` AS SHIPPED (entry()'s defaults), across the envelope and
#: with one thing made worse at a time.  `python doc/walk/walksim.py`.
#: _arm06: the plant's rotor 0.6 of DOG5's -- what the bench's swing-ff run
#: read DOG6's as (swing.feedforward_inertia) -- with the law's M0 left at
#: DOG5's (_arm06), or set to the plant's own (_arm06fit, --ff-armature).
_ARM = 0.6 * P.ARMATURE
SCENARIOS.update({
    "inplace": dict(schedule=INPLACE, trot_s=8.0, opts={}),
    "fwd05":   _walk(vx=0.05),
    "fwd10":   _walk(vx=0.10),
    "fwd15":   _walk(vx=0.15),
    "back05":  _walk(vx=-0.05),
    "back10":  _walk(vx=-0.10),
    "lat05":   _walk(vy=0.05),
    "latm05":  _walk(vy=-0.05),
    "lat08":   _walk(vy=0.08),
    "yaw20":   _walk(r=20.0),
    "yawm20":  _walk(r=-20.0),
    "yaw40":   _walk(r=40.0),
    "combo":   _walk(vx=0.10, r=20.0),
    "diag":    _walk(vx=0.10, vy=0.05),
    "fwd10_wls":  _walk(vx=0.10, alloc="wls"),
    "fwd10_fric": _walk(vx=0.10, frictionloss=0.1, imu_ms=15.0),
    "fwd10_mu05": _walk(vx=0.10, friction=0.5),
    "fwd10_imp":  _walk(vx=0.10, swing_law="impedance"),
    "inplace_arm06": dict(schedule=INPLACE, trot_s=8.0,
                          opts=dict(armature_scale=0.6)),
    "fwd10_arm06": _walk(vx=0.10, armature_scale=0.6),
    "fwd10_arm06fit": _walk(vx=0.10, armature_scale=0.6, ff_armature=_ARM),
    "lat05_arm06": _walk(vy=0.05, armature_scale=0.6),
    "lat05_arm06fit": _walk(vy=0.05, armature_scale=0.6, ff_armature=_ARM),
})
STANDARD = ["inplace_flown", "inplace", "fwd05", "fwd10", "fwd15", "back05",
            "back10", "lat05", "latm05", "lat08", "yaw20", "yawm20", "yaw40",
            "combo", "diag", "fwd10_wls", "fwd10_fric", "fwd10_mu05",
            "fwd10_imp", "inplace_arm06", "fwd10_arm06", "fwd10_arm06fit",
            "lat05_arm06", "lat05_arm06fit"]

#: THE SHIPPED WALK AGAIN, across the gait phase: six phases a case, and the
#: hardware unknowns at three -- the joint friction and IMU latency, a
#: slippery floor, DOG6's lighter rotor with M0 inherited and fitted.
#: (fwd10, fwd15, lat05, latm05, lat08, yaw40 and combo at six phases are
#: q_w2520n_*: the same options, the same phases, the same runs.)
_XVARIANTS = {"ship": ({}, ("inplace", "fwd05", "back10", "yaw20", "yawm20"),
                       6),
              "fric": (dict(frictionloss=0.1, imu_ms=15.0),
                       ("fwd10", "lat05", "yaw20"), 3),
              "mu05": (dict(friction=0.5), ("fwd10", "lat05", "yaw20"), 3),
              "arm06": (dict(armature_scale=0.6),
                        ("inplace", "fwd10", "lat05", "yaw20"), 3),
              "arm06fit": (dict(armature_scale=0.6, ff_armature=_ARM),
                           ("inplace", "fwd10", "lat05", "yaw20"), 3)}
_XCASES = dict(fwd05=dict(vx=0.05), fwd10=dict(vx=0.10), back10=dict(vx=-0.10),
               lat05=dict(vy=0.05), latm05=dict(vy=-0.05), lat08=dict(vy=0.08),
               yaw20=dict(r=20.0), yawm20=dict(r=-20.0), yaw40=dict(r=40.0),
               combo=dict(vx=0.10, r=20.0), inplace=dict())
for _vname, (_vopts, _vcases, _n) in _XVARIANTS.items():
    for _case in _vcases:
        for _seed in range(_n):
            SCENARIOS["x_%s_%s_%d" % (_vname, _case, _seed)] = _walk(
                t_cmd=1.5 + 0.6 * _seed / _n, **_XCASES[_case], **_vopts)
XREPEATS = [k for k in SCENARIOS if k.startswith("x_")]

# ===========================================================================
# THE QP AS EVERY ENTRY POINT'S ALLOCATOR (2026-10-04), and FOLD2's walk
# ===========================================================================
F2O = dict(posture="fold2")
#: The trot entry points in place, as flown (the least squares) and as they
#: now ship (the QP): hw.fold_trot (no prefix) and hw.fold2_trot (f2_),
#: then pushed sideways 0.2 s, 3 s into the trot, and with the hardware
#: unknowns.  `--allocs`.
FLOWN = dict(walk=False, duty=0.8, settle=SETTLE_TROT)
for _pre, _po in (("", {}), ("f2_", F2O)):
    for _alloc in ("wls", "qp"):
        _o = dict(FLOWN, alloc=_alloc, **_po)
        SCENARIOS["%sflown_%s" % (_pre, _alloc)] = dict(
            schedule=INPLACE, trot_s=8.0, opts=_o)
        for _f in (5, 10):
            SCENARIOS["%sflown_%s_push%d" % (_pre, _alloc, _f)] = dict(
                schedule=INPLACE, trot_s=8.0, opts=_o,
                push=("trot", 3.0, 0.2, (0.0, float(_f), 0.0)))
        SCENARIOS["%sflown_%s_fric" % (_pre, _alloc)] = dict(
            schedule=INPLACE, trot_s=8.0,
            opts=dict(_o, frictionloss=0.1, imu_ms=15.0))
        SCENARIOS["%sflown_%s_mu05" % (_pre, _alloc)] = dict(
            schedule=INPLACE, trot_s=8.0, opts=dict(_o, friction=0.5))
ALLOCS = [k for k in SCENARIOS if k.startswith(("flown_", "f2_flown_"))]

#: FOLD2 WALKING, `hw.fold2_walk` as shipped: the fold's walk on the knees-
#: out stance (same swing law, clock, keys).  Single runs at phase 0
#: (`--fold2`), then six phases a case and the hardware unknowns at three
#: (`--f2repeats`), with 0.15 / 0.20 m/s and 0.10 sideways to find where it
#: stops.
SCENARIOS.update({
    "f2_inplace": dict(schedule=INPLACE, trot_s=8.0, opts=dict(F2O)),
    "f2_fwd05":  _walk(vx=0.05, **F2O),
    "f2_fwd10":  _walk(vx=0.10, **F2O),
    "f2_fwd15":  _walk(vx=0.15, **F2O),
    "f2_fwd20":  _walk(vx=0.20, **F2O),
    "f2_back10": _walk(vx=-0.10, **F2O),
    "f2_lat05":  _walk(vy=0.05, **F2O),
    "f2_latm05": _walk(vy=-0.05, **F2O),
    "f2_lat08":  _walk(vy=0.08, **F2O),
    "f2_lat10":  _walk(vy=0.10, **F2O),
    "f2_yaw20":  _walk(r=20.0, **F2O),
    "f2_yawm20": _walk(r=-20.0, **F2O),
    "f2_yaw40":  _walk(r=40.0, **F2O),
    "f2_combo":  _walk(vx=0.10, r=20.0, **F2O),
    "f2_diag":   _walk(vx=0.10, vy=0.05, **F2O),
    "f2_fwd10_wls":  _walk(vx=0.10, alloc="wls", **F2O),
    "f2_fwd10_fric": _walk(vx=0.10, frictionloss=0.1, imu_ms=15.0, **F2O),
    "f2_fwd10_mu05": _walk(vx=0.10, friction=0.5, **F2O),
    "f2_fwd10_arm06": _walk(vx=0.10, armature_scale=0.6, **F2O),
    "f2_fwd10_arm06fit": _walk(vx=0.10, armature_scale=0.6,
                               ff_armature=_ARM, **F2O),
    "f2_lat05_arm06": _walk(vy=0.05, armature_scale=0.6, **F2O),
    "f2_lat05_arm06fit": _walk(vy=0.05, armature_scale=0.6,
                               ff_armature=_ARM, **F2O),
})
F2STANDARD = [k for k in SCENARIOS if k.startswith("f2_")
              and not k.startswith("f2_flown")]
_YCASES = dict(_XCASES, fwd15=dict(vx=0.15), fwd20=dict(vx=0.20),
               lat10=dict(vy=0.10))
_YVARIANTS = {
    "f2ship": ({}, ("inplace", "fwd05", "fwd10", "fwd15", "fwd20", "back10",
                    "lat05", "latm05", "lat08", "lat10", "yaw20", "yawm20",
                    "yaw40", "combo"), 6),
    "f2fric": (dict(frictionloss=0.1, imu_ms=15.0),
               ("fwd10", "lat05", "yaw20"), 3),
    "f2mu05": (dict(friction=0.5), ("fwd10", "lat05", "yaw20"), 3),
    "f2arm06": (dict(armature_scale=0.6),
                ("inplace", "fwd10", "lat05", "yaw20"), 3),
    "f2arm06fit": (dict(armature_scale=0.6, ff_armature=_ARM),
                   ("inplace", "fwd10", "lat05", "yaw20"), 3)}
for _vname, (_vopts, _vcases, _n) in _YVARIANTS.items():
    for _case in _vcases:
        for _seed in range(_n):
            SCENARIOS["y_%s_%s_%d" % (_vname, _case, _seed)] = _walk(
                t_cmd=1.5 + 0.6 * _seed / _n, **_YCASES[_case], **_vopts,
                **F2O)
F2REPEATS = [k for k in SCENARIOS if k.startswith("y_")]


#: The two fold stances as their trot entry points hand them to `hw.stand`:
#: FOLD with the slanted rise (hw.fold_trot), FOLD2 straight up, its stand's
#: feet the crouch's (hw.fold2_trot).  Same hold, clock and apex.
POSTURES = {
    "fold": dict(walk="hw.fold_walk", trot="hw.fold_trot", crouch=POSE.FOLD,
                 height=FT.HEIGHT, stand_xy=FT.STAND_XY,
                 slide_from=FT.SLIDE_FROM_H, period=FT.PERIOD_S,
                 apex=FT.SWING_HEIGHT),
    "fold2": dict(walk="hw.fold2_walk", trot="hw.fold2_trot",
                  crouch=POSE.FOLD2, height=F2.HEIGHT, stand_xy=None,
                  slide_from=None, period=F2.PERIOD_S, apex=F2.SWING_HEIGHT),
}


def entry(alloc: str = "qp", walk: bool = True, swing_ff: bool = False,
          settle: float | None = BCFG.WALK_SETTLE_S,
          kv: float = BCFG.STEP_KV, kp_swing=None, kd_swing=None,
          swing_law: str = BCFG.WALK_SWING_LAW, wn_swing=None,
          period: float | None = None, duty: float | None = BCFG.WALK_DUTY,
          swing_height: float | None = None,
          tau_slew: float | None = None, zeta_swing: float | None = None,
          ff_scale: float | None = None, ff_armature: float | None = None,
          posture: str = "fold"):
    """(`hwsim.Entry`, WalkPlan or None) -- `hw.fold_walk` (or, `posture`
    "fold2", `hw.fold2_walk`) as `hw.stand.main` builds it at its defaults:
    the trot entry point's stand and trot, the walk plan, the allocator, the
    estimator's x/y loop on.  `walk` False is the trot entry point itself."""
    pz = POSTURES[posture]
    o = TR.trot_options(pz["period"])
    srb = pz["crouch"].srb_at(pz["height"], pz["stand_xy"])
    wide = (np.full(C.N_JOINTS, -1e9), np.full(C.N_JOINTS, 1e9))   # --no-limits
    gains = BCTRL.BalanceGains()
    gains.kp_att[0], gains.kd_att[0] = FS.ROLL_GAINS
    gains.kp_pos[:2] = BCFG.KP_XY_EST
    gains.kd_pos[:2] = BCFG.KD_XY_EST
    plan = None
    if walk:
        # The harness's box is WIDE: it measures what the law can do, and
        # the keys' box (config.WALK_*_MAX) is what it found.
        plan = WalkPlan(reference=WalkReference(vx_max=0.30, vy_max=0.20,
                                                yaw_rate_max=math.radians(60.0)),
                        kv=kv,
                        kp_swing=kp_swing, kd_swing=kd_swing,
                        swing_ff=swing_ff, swing_law=swing_law,
                        wn_swing=wn_swing, zeta_swing=zeta_swing,
                        ff_inertia=(
                            BSWING.feedforward_inertia(ff_armature)
                            if ff_armature is not None else
                            None if ff_scale is None else
                            ff_scale * BSWING.JOINT_INERTIA_M0))
    gait_kw = {} if settle is None else dict(settle_s=settle)
    if duty is not None:
        # The contact ramp has to fit the four-foot window, (duty-0.5)/(2 duty)
        # of stance: 0.19 at duty 0.80, 0.14 at 0.70 (gait.TrotGait refuses).
        gait_kw["duty"] = duty
        gait_kw["ramp"] = min(BCFG.CONTACT_RAMP,
                              0.9 * (duty - 0.5) / (2.0 * duty))
    slanted = ({} if pz["stand_xy"] is None else
               dict(stand_xy=pz["stand_xy"], slide_from_h=pz["slide_from"]))
    e = H.Entry(pz["walk"] if walk else pz["trot"], pz["crouch"],
                o["tau_cap"], tau_ceiling=o["tau_ceiling"],
                tau_slew=o["tau_slew"] if tau_slew is None else tau_slew,
                overspeed_trip=o["overspeed_trip"],
                latch=False, tilt_stop=FS.TILT_STOP_DEG,
                track_stop=FS.TRACK_STOP_DEG, roll_gains=FS.ROLL_GAINS,
                swing="cartesian",
                gait_period=o["gait"].period if period is None else period,
                limits=wide,
                gait_kw=gait_kw,
                law_kw=dict(gains=gains, h_lift=pz["height"], srb=srb,
                            residual_trip=False, rise_track=True,
                            **slanted,
                            kp_joint=BCFG.KP_JOINT_HOLD,
                            kd_joint=BCFG.KD_JOINT_HOLD,
                            swing_height=(pz["apex"] if swing_height is None
                                          else swing_height),
                            kp_swing=BCFG.KP_SWING, kd_swing=BCFG.KD_SWING,
                            est_xy=True, alloc=alloc, walk=plan))
    return e, plan


class Driver:
    """`hw.stand.run`'s ESTIMATOR_SLOT and the operator's keys, per sweep."""

    def __init__(self, sim, schedule, push=None, pushes=None):
        #: (phase, t_after, duration_s, (fx, fy, fz)) or None; `pushes` is the
        #: list `HwSim.run` reads every step -- appended once the phase starts.
        self.push, self.pushes = push, pushes
        self.feed = EstimatorFeed(None)
        m = sim.model
        self.acc_adr = m.sensor_adr[m.sensor("imu_acc").id]
        self.pending = None
        self.next_acc = 0.0
        self.acc = np.full(3, np.nan)
        self.t_acc = -1.0
        self.schedule = sorted(schedule)
        self.rows = []
        self.trace = []

    def command(self, t_trot: float) -> Command:
        cmd = (0.0, 0.0, 0.0)
        for t0, vx, vy, r in self.schedule:
            if t_trot >= t0:
                cmd = (vx, vy, r)
        return Command(vx=cmd[0], vy=cmd[1], yaw_rate=math.radians(cmd[2]))

    def __call__(self, sim, now, stand, body):
        if self.push is not None and stand.phase_name == self.push[0]:
            _, t_after, dur, force = self.push
            self.pushes.append((now + t_after, now + t_after + dur, force))
            self.push = None
        if self.pending is not None:
            stand.feed_estimate(self.pending)
        if now >= self.next_acc - 1e-12:              # the 0x40 stream, 100 Hz
            self.acc = sim.data.sensordata[self.acc_adr:self.acc_adr + 3].copy()
            self.t_acc = now
            self.next_acc += 0.010
        orient = IMU.TrunkOrientation(
            R=body.R, omega_b=body.omega_b, roll=body.roll, pitch=body.pitch,
            yaw=body.yaw, age_s=body.imu_age_s, acc_b=self.acc,
            acc_age_s=now - self.t_acc + 0.003)
        self.feed.update(now, stand, body, orient)
        self.pending = self.feed.estimate()
        plan = stand.balance.walk
        if plan is not None:
            plan.set_command(self.command(now - stand.t_trot)
                             if stand.trotting else None)
        out = stand.out
        ref = None if out is None else out.ref
        self.rows.append((now, stand.phase_name,
                          np.nan if ref is None else ref.v[0],
                          np.nan if ref is None else ref.v[1],
                          np.nan if ref is None else ref.yaw_rate,
                          float("nan") if ref is None else float(ref.leashed)))
        nan3, nan2 = np.full((4, 3), np.nan), np.full((4, 2), np.nan)
        est = None if out is None else out.estimate
        self.trace.append(dict(
            t=now,
            p_swing=nan3 if out is None or out.p_swing is None else out.p_swing,
            land_w=nan2 if out is None or out.land_w is None else out.land_w,
            ref_p=np.full(2, np.nan) if ref is None else ref.p,
            ref_v=np.full(2, np.nan) if ref is None else ref.v,
            ref_yaw=np.nan if ref is None else ref.yaw,
            est_p=np.full(3, np.nan) if est is None else est.p_w,
            est_v=np.full(3, np.nan) if est is None else est.v_w,
            acc_lin=np.full(3, np.nan) if out is None else out.wrench.acc_lin,
            tau_req=np.full(12, np.nan) if out is None else out.tau,
            f_w=nan3 if out is None else out.allocation.f_w,
            xy_on=float("nan") if out is None else float(out.xy_on)))


def run(name: str) -> dict:
    sc = SCENARIOS[name]
    opts = dict(sc["opts"])
    hwkw = dict(frictionloss=opts.pop("frictionloss", 0.0),
                imu_latency=1e-3 * opts.pop("imu_ms", 3.0),
                friction=opts.pop("friction", None),
                armature_scale=opts.pop("armature_scale", 1.0))
    t0 = time.time()
    e, plan = entry(**opts)
    sim = H.HwSim(e, H.HwParams(**hwkw))
    sim.place(e.crouch.q, e.crouch.z_origin)
    # A never-active first entry: `HwSim.run` keeps THIS list only if it is
    # not empty, and the Driver appends the scenario's push to it.
    pushes = [(-1.0, -1.0, (0.0, 0.0, 0.0))]
    drv = Driver(sim, sc["schedule"], push=sc.get("push"), pushes=pushes)
    op = H.Operator(hold_s=T_HOLD, trot_s=sc["trot_s"], park=False,
                    hold_after_s=1.5)
    res = sim.run(op, t_max=3.0 + 1.0 + 3.0 + T_HOLD + sc["trot_s"] + 3.0,
                  push=pushes, quiet=True, pre_update=drv)
    out = metrics(name, res, drv, sim)
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "walk_%s.json" % name), "w") as fh:
        json.dump(out, fh, indent=1)
    trace = {}
    if drv.trace:
        # The driver runs before the law each sweep, so its row k holds the
        # law's output of sweep k-1; align it with the log by time.
        tt = np.array([r["t"] for r in drv.trace])
        for key in drv.trace[0]:
            trace["w_" + key] = np.array([r[key] for r in drv.trace], dtype=float)
        trace["w_t"] = tt
    np.savez_compressed(os.path.join(DATA, "walk_%s.npz" % name),
                        **{k: v for k, v in res.items()
                           if isinstance(v, np.ndarray)}, **trace)
    return out


def metrics(name, res, drv, sim) -> dict:
    sc = SCENARIOS[name]
    ph, t = res["phase"], res["t"]
    trot = np.flatnonzero(ph == "trot")
    m = H.trot_metrics(res)
    out = dict(name=name, stop=m["stop"], fell=bool(m["fell"]),
               trot_s=m.get("trot_s", 0.0),
               max_tilt=m.get("max_tilt_trot", float("nan")),
               peak_tau=m.get("peak_tau_trot", float("nan")))
    law = sim.stand.balance
    a = np.asarray(law.timing.samples) if law.timing.samples else np.zeros(1)
    out["law_p50_us"] = round(1e6 * float(np.percentile(a, 50)), 0)
    out["law_p95_us"] = round(1e6 * float(np.percentile(a, 95)), 0)
    if law.qp is not None:
        out["qp_iter_max"] = law.qp.iter_max_seen
        out["qp_capped"] = law.qp.capped
    if law.walk is not None:
        out["leash_sweeps"] = law.walk.reference.leash_sweeps
        out["step_clamped"] = (0 if law.walk.planner is None
                               else law.walk.planner.clamped)
    if not trot.size:
        return out
    i0 = int(trot[0])
    t_trot0 = float(t[i0])
    xy, yaw = res["xy_trunk"], np.unwrap(res["yaw"])
    live = np.flatnonzero(ph != "STOP")
    i_end = int(min(trot[-1], live[-1]))
    segs = []
    sched = sorted(sc["schedule"])
    for k, (ts, vx, vy, r) in enumerate(sched):
        te = sched[k + 1][0] if k + 1 < len(sched) else sc["trot_s"]
        # the steady second half of the segment
        a_t, b_t = t_trot0 + ts + 0.5 * (te - ts), t_trot0 + te
        idx = np.flatnonzero((t >= a_t) & (t <= b_t) & (np.arange(t.size) <= i_end))
        if idx.size < 50:
            continue
        j0, j1 = int(idx[0]), int(idx[-1])
        dt = float(t[j1] - t[j0])
        dxy = xy[j1] - xy[j0]
        psi = float(np.mean(yaw[j0:j1 + 1]))
        c, s = math.cos(psi), math.sin(psi)
        v_b = np.array([c * dxy[0] + s * dxy[1], -s * dxy[0] + c * dxy[1]]) / dt
        segs.append(dict(cmd=[vx, vy, r],
                         v_true=[round(float(v_b[0]), 3), round(float(v_b[1]), 3)],
                         r_true=round(math.degrees((yaw[j1] - yaw[j0]) / dt), 1)))
    out["segments"] = segs
    # slip and touchdown lateness over the trot
    fn, fw, cw = res["foot_fn"], res["foot_w"], res["contact_w"]
    i1 = i_end
    slip = 0.0
    for leg in range(4):
        loaded = fn[i0:i1, leg] >= 1.0
        d = np.linalg.norm(np.diff(fw[i0:i1, leg, :2], axis=0), axis=1)
        slip += float(d[loaded[1:] & loaded[:-1]].sum())
    out["slip_mm_per_s"] = round(1e3 * slip / max(1e-9, float(t[i1] - t[i0])), 1)
    late = []
    for leg in range(4):
        c_ = cw[:, leg] > 0
        for k in range(i0 + 1, i1):
            if c_[k] and not c_[k - 1]:
                j = k
                while j < i1 and fn[j, leg] < 1.0:
                    j += 1
                late.append((j - k) * H.T_SWEEP)
    if late:
        out["late_ms_mean"] = round(1e3 * float(np.mean(late)), 0)
        out["late_ms_max"] = round(1e3 * float(np.max(late)), 0)
    leashed = np.array([r[5] for r in drv.rows if r[1] == "trot"], float)
    if np.isfinite(leashed).any():
        out["leashed_frac"] = round(float(np.nanmean(leashed)), 3)
    return out


def table(results) -> str:
    lines = ["%-14s %-5s %-24s %-22s %6s %6s %6s %6s %9s %8s"
             % ("scenario", "fell", "cmd vx vy r", "true vx vy r", "tilt",
                "tau", "slip", "late", "law p50/95", "qp it")]
    for r in results:
        segs = [s for s in r.get("segments", []) if any(s["cmd"])] or \
            r.get("segments", [])[:1]
        s = segs[0] if segs else dict(cmd=[0, 0, 0], v_true=[0, 0], r_true=0)
        lines.append("%-14s %-5s %+.2f %+.2f %+5.0f       %+.3f %+.3f %+6.1f   %5.1f  %5.2f %6.1f %6s %4.0f/%-4.0f %6s"
                     % (r["name"], "YES" if r["fell"] else "no",
                        s["cmd"][0], s["cmd"][1], s["cmd"][2],
                        s["v_true"][0], s["v_true"][1], s["r_true"],
                        r.get("max_tilt", float("nan")),
                        r.get("peak_tau", float("nan")),
                        r.get("slip_mm_per_s", float("nan")),
                        str(r.get("late_ms_mean", "-")),
                        r.get("law_p50_us", float("nan")),
                        r.get("law_p95_us", float("nan")),
                        str(r.get("qp_iter_max", "-"))))
    return "\n".join(lines)


def main(argv) -> int:
    if "--list" in argv:
        for k, v in SCENARIOS.items():
            print("%-14s %s %s" % (k, v["schedule"], v["opts"]))
        return 0
    names = ([a for a in argv if a in SCENARIOS]
             or (GRID if "--grid" in argv else
                 GRID2 if "--grid2" in argv else
                 REPEATS if "--repeats" in argv else
                 QREPEATS if "--qrepeats" in argv else
                 XREPEATS if "--xrepeats" in argv else
                 ALLOCS if "--allocs" in argv else
                 F2STANDARD if "--fold2" in argv else
                 F2REPEATS if "--f2repeats" in argv else
                 (ALLOCS + F2STANDARD + F2REPEATS) if "--fold2all" in argv
                 else
                 (XREPEATS + V0SET + STANDARD) if "--final" in argv
                 else
                 V0SET if "--v0" in argv else
                 SENS if "--sens" in argv else STANDARD))
    t0 = time.time()
    with Pool(min(4, len(names))) as pool:
        results = pool.map(run, names)
    print(table(results))
    if all(r["name"][:2] in ("r_", "q_", "x_", "y_") for r in results):
        print()
        print(repeat_table(results))
    print("wall %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
