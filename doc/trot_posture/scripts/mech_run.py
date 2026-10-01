"""mech_run -- controlled rise experiments on the trunk-x drift hypothesis.

    python doc/trot_posture/scripts/mech_run.py CASE [CASE ...]   -> data/mech_<case>.json
    python doc/trot_posture/scripts/mech_run.py --list

Each case rises from a crouch through doc/trot_posture/evalpose + doc/crouch_trot/hwsim
(repo controller code unmodified) and records, besides evalpose.stand_metrics:
  * the net horizontal floor force ON the robot every sweep (pre_update hook,
    mujoco.mj_contactForce on every floor contact, the sign fixed so the vertical
    component is upward, which a compressive floor contact always is), and its
    integral over the rise, the hold, and both;
  * the mean Fx over the first and second half of the rise (the "sign follows
    the rise acceleration" claim);
  * the trunk and whole-robot CoM x at crouch end / rise end / hold end.
The rise length is set in BOTH places that know it: sim.stand.RAMP_LIFT (which
hw.balance.sequence.ramp_remaining and the Operator read) and law_kw rise_s
(the law's quintic).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(TP, "..", ".."))
for p in (TP, REPO, os.path.join(REPO, "doc", "crouch_trot")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco                                   # noqa: E402
import evalpose as EP                           # noqa: E402
import hwsim as H                               # noqa: E402
from sim import stand as ST, params as P        # noqa: E402
from hw.balance import config as BCFG           # noqa: E402

DATA = os.path.join(TP, "data")
os.makedirs(DATA, exist_ok=True)

FOLD = dict(spec=("fold", "parallel", 79, 23, 71, 145, 60))
LEG_BODIES = ["%s_%s" % (k, leg) for k in ("hip", "thigh", "shin") for leg in ("FL", "FR", "RL", "RR")]


def case_def(**kw):
    d = dict(spec=FOLD["spec"], rise_s=3.0, prime=True, stance_xy=(0.0, 0.0),
             joint_hold=(5.0, 0.2), hw={}, armature=None, leg_mass_scale=None,
             damping=None, push=None, hold_s=3.0, loop_hz=None, noslip=None, law_consistent=False)
    d.update(kw)
    return d


CASES = {
    # the baseline and its handover
    "base":        case_def(),
    "asflown":     case_def(prime=False),
    # (1) rise time
    "rise1.5":     case_def(rise_s=1.5),
    "rise6":       case_def(rise_s=6.0),
    "rise12":      case_def(rise_s=12.0),
    # (2) armature
    "arm0":        case_def(armature=0.0),
    # (3) armature 0 + 1 % leg links
    "arm0_light":  case_def(armature=0.0, leg_mass_scale=0.01),
    "light":       case_def(leg_mass_scale=0.01),
    # viscous joint damping (an alternative, non-inertial, velocity-proportional source)
    "damp0":       case_def(damping=0.0),
    "arm0_light_damp0": case_def(armature=0.0, leg_mass_scale=0.01, damping=0.0),
    "arm0_light_damp0_rise6": case_def(armature=0.0, leg_mass_scale=0.01, damping=0.0, rise_s=6.0),
    "damp0_rise1.5": case_def(damping=0.0, rise_s=1.5),
    "damp0_rise6":   case_def(damping=0.0, rise_s=6.0),
    # (4) joint Coulomb friction
    "fric0.1":     case_def(hw=dict(frictionloss=0.1)),
    "fric0.3":     case_def(hw=dict(frictionloss=0.3)),
    "fric1.0":     case_def(hw=dict(frictionloss=1.0)),
    # (5) the stance xy spring, prime True
    "sp100_10":    case_def(stance_xy=(100.0, 10.0)),
    "sp200_15":    case_def(stance_xy=(200.0, 15.0)),
    "sp400_25":    case_def(stance_xy=(400.0, 25.0)),
    "sp800_50":    case_def(stance_xy=(800.0, 50.0)),
    # minimum-spring search, kd = kp / 16
    "sp25_2":      case_def(stance_xy=(25.0, 1.6)),
    "sp50_3":      case_def(stance_xy=(50.0, 3.1)),
    "sp150_9":     case_def(stance_xy=(150.0, 9.4)),
    "sp300_19":    case_def(stance_xy=(300.0, 18.8)),
    "sp600_38":    case_def(stance_xy=(600.0, 37.5)),
    # (5) prime False
    "np_sp100_10": case_def(prime=False, stance_xy=(100.0, 10.0)),
    "np_sp200_15": case_def(prime=False, stance_xy=(200.0, 15.0)),
    "np_sp300_19": case_def(prime=False, stance_xy=(300.0, 18.8)),
    "np_sp400_25": case_def(prime=False, stance_xy=(400.0, 25.0)),
    "np_sp600_38": case_def(prime=False, stance_xy=(600.0, 37.5)),
    "np_sp800_50": case_def(prime=False, stance_xy=(800.0, 50.0)),
    "np_sp1600_100": case_def(prime=False, stance_xy=(1600.0, 100.0)),
    # (8) damping only
    "kd25":        case_def(stance_xy=(0.0, 25.0)),
    "kd100":       case_def(stance_xy=(0.0, 100.0)),
    # (6) mirrored families
    "xin":         case_def(spec=("xin", "xin", 79, -79, 71, 145, 82)),
    "nominal":     case_def(spec=("nominal", "x", 81, -81, 65, 145, None)),
    # (7) another parallel stance
    "par60":       case_def(spec=("par60", "parallel", 60, -20, 71, 145, None)),
    "par60_rise6": case_def(spec=("par60", "parallel", 60, -20, 71, 145, None), rise_s=6.0),
    # the loop: 1 kHz with ~zero compute / bus latency (stale-Jacobian, zero-order-hold family)
    "fastloop":          case_def(loop_hz=1000.0, hw=dict(t_compute=0.05e-3, t_bus=0.02e-3)),
    "fastloop_damp0":    case_def(loop_hz=1000.0, hw=dict(t_compute=0.05e-3, t_bus=0.02e-3), damping=0.0),
    "fastloop_arm0_light_damp0": case_def(loop_hz=1000.0, hw=dict(t_compute=0.05e-3, t_bus=0.02e-3),
                                          damping=0.0, armature=0.0, leg_mass_scale=0.01),
    # the static (gravity-comp mismatch) part of the light-leg cases: T^2 scaling
    "light_rise12":      case_def(leg_mass_scale=0.01, rise_s=12.0),
    "arm0_light_rise12": case_def(armature=0.0, leg_mass_scale=0.01, rise_s=12.0),
    "arm0_light_damp0_rise12": case_def(armature=0.0, leg_mass_scale=0.01, damping=0.0, rise_s=12.0),
    "arm0_light_damp0_rise1.5": case_def(armature=0.0, leg_mass_scale=0.01, damping=0.0, rise_s=1.5),
    # friction, no joint damping: Coulomb alone
    "damp0_fric0.3":     case_def(damping=0.0, hw=dict(frictionloss=0.3)),
    # the spring at a 1.5 s and a 12 s rise: does the ~12 mm floor move?
    "sp400_25_rise1.5":  case_def(stance_xy=(400.0, 25.0), rise_s=1.5),
    "sp400_25_rise12":   case_def(stance_xy=(400.0, 25.0), rise_s=12.0),
    # armature and damping both off; 1 % armature with 1 % legs (0 is unstable in the position hold)
    "arm0_damp0":        case_def(armature=0.0, damping=0.0),
    "arm1pct_light":     case_def(armature=0.0085 * 0.01, leg_mass_scale=0.01),
    "arm1pct_light_damp0": case_def(armature=0.0085 * 0.01, leg_mass_scale=0.01, damping=0.0),
    "arm1pct_light_damp0_rise12": case_def(armature=0.0085 * 0.01, leg_mass_scale=0.01, damping=0.0, rise_s=12.0),
    "arm10pct_light":    case_def(armature=0.0085 * 0.1, leg_mass_scale=0.01),
    # contact creep: MuJoCo's noslip solver on
    "noslip":            case_def(noslip=20),
    "damp0_noslip":      case_def(damping=0.0, noslip=20),
    "sp400_25_noslip":   case_def(stance_xy=(400.0, 25.0), noslip=20),
    # the damping coefficient: does the drift depend on c (viscous-geometric picture says no)
    "damp0.02":          case_def(damping=0.02),
    "damp0.05":          case_def(damping=0.05),
    "damp0.5":           case_def(damping=0.5),
    "damp0.02_rise12":   case_def(damping=0.02, rise_s=12.0),
    "light_rise1.5":     case_def(leg_mass_scale=0.01, rise_s=1.5),
    # light legs with the LAW told the truth (cfg.MASS / WEIGHT and the leg gravity term scaled, in-process)
    "lightC_arm10pct":   case_def(armature=0.0085 * 0.1, leg_mass_scale=0.01, law_consistent=True),
    "lightC_arm10pct_rise12": case_def(armature=0.0085 * 0.1, leg_mass_scale=0.01, law_consistent=True, rise_s=12.0),
    "lightC_arm10pct_damp0": case_def(armature=0.0085 * 0.1, leg_mass_scale=0.01, law_consistent=True, damping=0.0),
    "lightC":            case_def(leg_mass_scale=0.01, law_consistent=True),
    # a constant 0.3 N forward force for 4 s in HOLD: the x mode's stiffness / damping
    "cpush_nojoint":        case_def(push=(0.5, 4.0, 0.3), hold_s=8.0, joint_hold=(0.0, 0.0)),
    "cpush_nojoint_damp0":  case_def(push=(0.5, 4.0, 0.3), hold_s=8.0, joint_hold=(0.0, 0.0), damping=0.0),
    "cpush_nojoint_sp400":  case_def(push=(0.5, 4.0, 0.3), hold_s=8.0, joint_hold=(0.0, 0.0), stance_xy=(400.0, 25.0)),
    "cpush_nojoint_kd25":   case_def(push=(0.5, 4.0, 0.3), hold_s=8.0, joint_hold=(0.0, 0.0), stance_xy=(0.0, 25.0)),
    "cpush_joint":          case_def(push=(0.5, 4.0, 0.3), hold_s=8.0),
    # parallel stances from a forced deep crouch (the plant ignores the knee-floor penetration)
    "par60_c60":         case_def(spec=("par60", "parallel", 60, -20, 71, 145, 60)),
    "par60_c100":        case_def(spec=("par60", "parallel", 60, -20, 71, 145, 100)),
    "fold_c100":         case_def(spec=("fold", "parallel", 79, 23, 71, 145, 100)),
    "fold_c100_sp400":   case_def(spec=("fold", "parallel", 79, 23, 71, 145, 100), stance_xy=(400.0, 25.0)),
    "fold_c100_asflown": case_def(spec=("fold", "parallel", 79, 23, 71, 145, 100), prime=False),
    "fold_c100_np_sp400": case_def(spec=("fold", "parallel", 79, 23, 71, 145, 100), prime=False, stance_xy=(400.0, 25.0)),
    # push in HOLD: 1 N forward for 1 s, 1 s into the hold -- stiffness of the held trunk
    "push_hold":        case_def(push=(1.0, 1.0, 1.0), hold_s=6.0),
    "push_hold_nojoint": case_def(push=(1.0, 1.0, 1.0), hold_s=6.0, joint_hold=(0.0, 0.0)),
    "push_hold_sp400":  case_def(push=(1.0, 1.0, 1.0), hold_s=6.0, stance_xy=(400.0, 25.0)),
}


class FloorForce:
    """pre_update hook: net floor force on the robot each sweep."""
    def __init__(self):
        self.t, self.F, self.phase, self.ncon = [], [], [], []
        self.f6 = np.zeros(6)

    def __call__(self, sim, now, stand, body):
        m, d = sim.model, sim.data
        F = np.zeros(3)
        n = 0
        for i in range(d.ncon):
            con = d.contact[i]
            if sim.floor not in (con.geom1, con.geom2):
                continue
            mujoco.mj_contactForce(m, d, i, self.f6)
            fw = con.frame.reshape(3, 3).T @ self.f6[:3]
            if fw[2] < 0.0:          # the floor can only push the robot UP
                fw = -fw
            F += fw
            n += 1
        self.t.append(now); self.F.append(F); self.phase.append(stand.phase_name); self.ncon.append(n)


def run_case(name, c, keep_npz=False):
    t0 = time.time()
    s = c["spec"]
    spec = EP.Spec(s[0], s[1], s[2], s[3], s[4], s[5], s[6])
    pose, info = EP.build(spec)
    out = dict(case=name, cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
               crouch_h_mm=info.get("crouch_h_mm"), knee_dir=info.get("knee_dir"))
    if pose is None:
        out["error"] = "no feasible crouch"
        return out
    # the rise length, in both places
    ST.RAMP_LIFT = float(c["rise_s"])
    e = EP.entry(pose, spec.h, stance_xy=c["stance_xy"], joint_hold=c["joint_hold"])
    e = e.replace(law_kw=dict(e.law_kw, rise_s=float(c["rise_s"])))
    H.RATE_HZ, H.T_SWEEP = 250.0, 1.0 / 250.0
    hwp = dict(c["hw"], prime_gate=c["prime"])
    if c.get("loop_hz"):
        H.RATE_HZ = float(c["loop_hz"]); H.T_SWEEP = 1.0 / H.RATE_HZ
        hwp["steps_per_sweep"] = int(round(H.T_SWEEP * 3000.0))     # keep dt = 1/3000
    H.T_SLOT = H.T_SWEEP / H.N
    sim = H.HwSim(e, H.HwParams(**hwp))
    m = sim.model
    if c["armature"] is not None:
        m.dof_armature[6:] = c["armature"]
    if c["damping"] is not None:
        m.dof_damping[6:] = c["damping"]
    if c.get("noslip"):
        m.opt.noslip_iterations = int(c["noslip"])
    if c["leg_mass_scale"] is not None:
        for b in LEG_BODIES:
            bid = m.body(b).id
            m.body_mass[bid] *= c["leg_mass_scale"]
            m.body_inertia[bid] *= c["leg_mass_scale"]
        mujoco.mj_setConst(m, sim.data)
    out["total_mass_kg"] = float(m.body_subtreemass[0])
    if c.get("law_consistent"):
        from hw.balance import torque as TRQ
        k = float(c["leg_mass_scale"])
        BCFG.MASS = float(m.body_subtreemass[0]); BCFG.WEIGHT = BCFG.MASS * 9.81
        if not hasattr(TRQ, "_orig_leg_gravity_torque"):
            TRQ._orig_leg_gravity_torque = TRQ.leg_gravity_torque
            TRQ._orig_all_leg_gravity_torque = TRQ.all_leg_gravity_torque
        TRQ.leg_gravity_torque = lambda *a, **kw: k * TRQ._orig_leg_gravity_torque(*a, **kw)
        TRQ.all_leg_gravity_torque = lambda *a, **kw: k * TRQ._orig_all_leg_gravity_torque(*a, **kw)
        out["law_mass_kg"] = BCFG.MASS
    sim.place(pose.q, pose.z_origin)
    hook = FloorForce()
    push = None
    t_hold0 = 0.004 + 1.0 + ST.RAMP_POSITION + 0.5 + c["rise_s"]
    if c["push"] is not None:
        dt_after, dur, fx = c["push"]
        push = [(t_hold0 + dt_after, t_hold0 + dt_after + dur, (fx, 0.0, 0.0))]
        out["push_window_s"] = [push[0][0], push[0][1]]
    t_max = t_hold0 + c["hold_s"] + 3.0
    res = sim.run(H.Operator(hold_s=c["hold_s"], park=False), t_max=t_max, pre_update=hook, push=push)
    if keep_npz:
        H.save(res, os.path.join(DATA, "mech_%s.npz" % name))
    out.update(EP.stand_metrics(res))
    out["notices"] = [str(n[1]).splitlines()[0][:100] for n in res["notices"]][:8]
    # --- the floor force ---
    t = np.array(hook.t); F = np.array(hook.F); ph = np.array(hook.phase)
    rise = ph == "rise"; hold = ph == "hold"
    dt = H.T_SWEEP
    out["phases_seen"] = sorted(set(ph.tolist()))
    if not (rise | hold).any():
        out["wall_s"] = round(time.time() - t0, 1)
        return out
    out["impulse_x_rise_Ns"] = round(float(F[rise, 0].sum() * dt), 4)
    out["impulse_x_hold_Ns"] = round(float(F[hold, 0].sum() * dt), 4)
    out["impulse_x_Ns"] = round(float(F[rise | hold, 0].sum() * dt), 4)
    out["impulse_y_Ns"] = round(float(F[rise | hold, 1].sum() * dt), 4)
    out["Fx_peak_N"] = round(float(np.abs(F[rise | hold, 0]).max()), 3)
    out["Fz_mean_hold_N"] = round(float(F[hold, 2].mean()), 2) if hold.any() else None
    if rise.any():
        tr = t[rise]; fr = F[rise, 0]
        half = tr < tr[0] + 0.5 * c["rise_s"]
        out["Fx_mean_rise_1st_half_N"] = round(float(fr[half].mean()), 4)
        out["Fx_mean_rise_2nd_half_N"] = round(float(fr[~half].mean()), 4)
        out["Fx_mean_hold_N"] = round(float(F[hold, 0].mean()), 4) if hold.any() else None
    # --- trunk and CoM x through the phases ---
    phr = res["phase"]
    ic = EP._idx(phr, "crouch", "last"); ir = EP._idx(phr, "rise", "last"); ih = EP._idx(phr, "hold", "last")
    i_r0 = EP._idx(phr, "rise", "first")
    if ic is not None and ir is not None:
        out["drift_x_rise_end_mm"] = round(float(1e3 * (res["xy_trunk"][ir, 0] - res["xy_trunk"][ic, 0])), 1)
        out["com_x_shift_mm"] = round(float(1e3 * (res["com_w"][ih, 0] - res["com_w"][ic, 0])), 1)
        out["com_x_shift_rise_end_mm"] = round(float(1e3 * (res["com_w"][ir, 0] - res["com_w"][ic, 0])), 1)
        # trunk x velocity at rise end and hold end
        tt = res["t"]; x = res["xy_trunk"][:, 0]
        k = 25
        out["vx_rise_end_mm_s"] = round(float(1e3 * (x[ir] - x[ir - k]) / (tt[ir] - tt[ir - k])), 1)
        out["vx_hold_end_mm_s"] = round(float(1e3 * (x[ih] - x[ih - k]) / (tt[ih] - tt[ih - k])), 1)
        # the push response: trunk x before / peak / after
        if c["push"] is not None:
            p0, p1 = push[0][0], push[0][1]
            w0 = np.flatnonzero(tt >= p0 - 0.02)[0]
            w_after = np.flatnonzero(tt >= p1 + 2.0)
            out["push_x_before_mm"] = round(float(1e3 * (x[w0] - x[ic])), 1)
            out["push_x_max_mm"] = round(float(1e3 * (x[w0:].max() - x[ic])), 1)
            if w_after.size:
                out["push_x_after2s_mm"] = round(float(1e3 * (x[w_after[0]] - x[ic])), 1)
            # x relative to the push start, sampled through the push and after it
            samp = {}
            for dtq in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0):
                w = np.flatnonzero(tt >= p0 + dtq)
                if w.size:
                    samp["%.2f" % dtq] = round(float(1e3 * (x[w[0]] - x[w0])), 1)
            out["push_x_vs_t_mm"] = samp
            out["push_end_pitch_deg"] = round(float(np.degrees(res["pitch"][np.flatnonzero(tt >= p1)[0]])), 2)
        # trace, decimated, for the record
        sel = np.arange(max(0, i_r0 - 50), min(len(tt), ih + 1), 10)
        out["trace"] = dict(t=np.round(tt[sel] - tt[i_r0], 3).tolist(),
                            x_mm=np.round(1e3 * (x[sel] - x[ic]), 2).tolist(),
                            h_mm=np.round(1e3 * (res["z_trunk"][sel] - BCFG.TRUNK_BOTTOM_OFFSET), 1).tolist(),
                            pitch_deg=np.round(np.degrees(res["pitch"][sel]), 3).tolist(),
                            Fx_N=np.round(np.interp(tt[sel], t, F[:, 0]), 4).tolist())
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-npz", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(CASES)); return 0
    for name in a.cases:
        out = run_case(name, CASES[name], keep_npz=a.keep_npz)
        path = os.path.join(DATA, "mech_%s.json" % name)
        json.dump(out, open(path, "w"), indent=1, default=str)
        print("%-22s drift %+7.1f mm  pitch %+6.2f  h_min %6.1f  J_x %+8.4f N s (rise %+8.4f hold %+8.4f)  Fx 1st/2nd half %+.3f/%+.3f  Fz %s  stop=%s  (%.0f s)" % (
            name, out.get("drift_x_mm", np.nan), out.get("pitch_hold_deg", np.nan), out.get("h_min_rise_mm", np.nan),
            out.get("impulse_x_Ns", np.nan), out.get("impulse_x_rise_Ns", np.nan), out.get("impulse_x_hold_Ns", np.nan),
            out.get("Fx_mean_rise_1st_half_N", np.nan), out.get("Fx_mean_rise_2nd_half_N", np.nan),
            out.get("Fz_mean_hold_N"), out.get("stop"), out.get("wall_s", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
