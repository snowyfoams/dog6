"""DOG6 three-leg stance: lift one foot off the floor and WAVE it.

    python -m sim.wave                  # viewer; ENTER steps the phases
    python -m sim.wave --headless       # no window, auto-steps, GATES the run
    python -m sim.wave --check          # the offline gates alone, no MuJoCo
    python -m sim.wave --leg FR         # any of FL FR RL RR

`sim.stand`'s sequence with five phases inserted between `lift` and `park`:

    0 settle   |
    1 crouch   |  sim.stand, unchanged
    2 lift     |
    3 shift    TORQUE.  Four feet planted.  The trunk slides sideways over the
               feet until the CoM sits SUPPORT_MARGIN inside the triangle the
               other three feet will make
    4 raise    TORQUE.  The chosen foot is UNLOADED first -- its share of the
               weight fades to zero while it is still on the floor -- and only
               then lifted, forward and up, to the raised point
    5 wave     TORQUE.  Three feet carry the robot; the raised paw swings side
               to side, WAVE_CYCLES times at WAVE_HZ.  Then holds
    6 lower    TORQUE.  The paw returns to the spot it left, and is RELOADED
    7 unshift  TORQUE.  The trunk slides back over the centre of the feet
    8 park     |  sim.stand, unchanged
    9 done     |

ONE LAW, FOUR LEGS, ONE FORMULA -- THE SWING LEG IS NOT A SPECIAL CASE
    Per leg, in that leg's own HIP frame, exactly `sim.stand`'s compliance:

        f   = KP_CART (p_des - p) + KD_CART (v_des - J qd) + (0, 0, -fz_i)
        tau = J^T f + leg_gravity_torque

    What changes from the stand is only what `p_des`, `v_des` and `fz_i` are.
    A stance foot has `p_des` pinned on the floor and `fz_i` a share of the
    weight; the swing foot has `p_des` moving along a trajectory and `fz_i`
    zero.  The handover between the two is the CONTACT WEIGHT c_i in [0, 1],
    which fades the foot's share out continuously before it leaves and back in
    after it lands.  There is no instant at which a leg switches controllers,
    so there is no step in any torque.

THE LOAD SPLIT IS SOLVED, NOT ASSUMED
    `sim.stand` hands every foot mg/4.  That is only right with the CoM in the
    middle of four square feet, and the whole point of this sequence is to
    leave that arrangement.  `static_load_split` solves the three statics
    equations -- the vertical sum and the two tilting moments about the CoM --
    for four normal forces by weighted minimum norm, with the contact weights
    as the weighting.  At c = (1, 1, 1, 1) and a centred CoM it IS mg/4; at
    c = (0, 1, 1, 1) it is the unique three-foot solution; in between it is
    the continuous path from one to the other.  Hand the springs the right
    feedforward and they have nothing to deflect against, which is what keeps
    a law with no attitude feedback LEVEL on three feet.

    It is the same 3x4 problem `hw.balance.allocation` solves as 6x12, in the
    two rows a flat stand actually loads.  Kept separate because this file is
    `sim` and must import no `hw`; `check` gates the two agree.

THE SHIFT, AND WHY 35 mm
    At the lift pose the CoM lies on the FR-RL diagonal to within 0.03 mm --
    DOG6's CAD mirrors cleanly.  Lift FL and that diagonal becomes an EDGE of
    the support triangle, so with no shift the robot balances on a line.
    `support_shift` moves the trunk perpendicular to that edge, toward the far
    corner, until the CoM is SUPPORT_MARGIN inside: (-9, -34) mm for FL, which
    is mostly to the right and a little back.  35 mm is what the raised leg
    costs back: lifting 0.88 kg of leg 0.1 m forward moves the CoM ~2 mm
    toward the edge, and the waving paw adds +-1 N of lateral reaction.

    EVERY xy IS PINNED IN THE BODY FRAME, as in the stand.  Nothing here
    measures where the trunk is in the world; the shift is a change in where
    the feet are told to be relative to the trunk, and the springs do the
    rest.  `sim.stand`'s docstring says why that is honest on flat ground.

WHAT THE PAW DOES
    The raised point is RAISED_HIP, mirrored per leg: 120 mm toward the
    robot's own end of the body (forward for a front leg), 50 mm outboard, and
    30 mm below the hip -- 120 mm off the floor at the lift height.  The wave
    is lateral, +-WAVE_AMPLITUDE about that point, which on this leg is mostly
    the ABDUCTION joint swinging.  The amplitude ramps in and out over half a
    cycle, so the reference has no velocity step at either end.

    Every reference here is closed form and `v_des` is its ANALYTIC rate --
    never a finite difference -- for the reason `hw.balance.reference` gives:
    the damping term must not be fed the derivative of the reference's own
    quantisation.  `check` gates the targets' reachability and joint limits
    over the whole envelope, because the simulator enforces neither.

THE HARDWARE RUNS THIS FILE
    `hw.stand --wave FL` inserts the same five phases and calls
    `ThreeLegStance.torque` with `kin=hw.kinematics`, exactly as `hw.stand`
    runs `sim.stand.compliance_torque`.  The trajectory, the load split and
    the force law are properties of the robot, not of the simulator.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# NO `import mujoco` HERE -- see sim.stand.  The robot imports this file.

if __package__ in (None, ""):        # allow `python sim/wave.py` too
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sim"

from . import coordinates as C       # noqa: E402
from . import kinematics as K        # noqa: E402
from . import params as P            # noqa: E402
from . import stand as ST            # noqa: E402

__all__ = ["WAVE_PHASES", "DURATIONS", "SUPPORT_MARGIN", "RAISED_HIP",
           "WAVE_AMPLITUDE", "WAVE_HZ", "WAVE_CYCLES", "COM_XY_LIFT",
           "quintic", "raised_point", "support_shift", "static_load_split",
           "ThreeLegStance", "WaveController", "check", "run_headless"]


# ===========================================================================
# the numbers
# ===========================================================================
#: The five phases this file adds, in order.  `hw.stand` and `WaveController`
#: both splice them between `lift` and `park`.
WAVE_PHASES = ("shift", "raise", "wave", "lower", "unshift")

#: How far inside the support triangle the CoM is put before a foot leaves.
#: See the module docstring for what 35 mm pays for.  [SIM TUNING]
SUPPORT_MARGIN = 0.035                  # m

#: Where the paw is held, in the HIP frame of a FRONT-LEFT leg: forward,
#: outboard, a little below the hip.  `raised_point` mirrors it per leg.
#: 50 mm outboard is 23 deg of abduction, well inside ABD_LIM.  [SIM TUNING]
RAISED_HIP = np.array([0.120, 0.050, -0.030])

#: The wave: lateral, about the raised point.  Three cycles at 1.2 Hz is a
#: wave a person recognises as one; +-40 mm is +-18 deg of abduction.
WAVE_AMPLITUDE = 0.040                  # m
WAVE_HZ = 1.2
WAVE_CYCLES = 3

#: Phase timing.  The shift is slow because it is 34 mm of trunk through
#: springs that were tuned to hold xy, not to move it; the unload is what
#: lets the foot leave with no force step.  [SIM TUNING]
T_SHIFT = 1.5
T_UNLOAD = 0.6
T_RAISE = 1.0
T_WAVE = WAVE_CYCLES / WAVE_HZ          # 2.5 s
T_ENVELOPE = 0.5 / WAVE_HZ              # the amplitude ramp, half a cycle

DURATIONS = {
    "shift": T_SHIFT,
    "raise": T_UNLOAD + T_RAISE,
    "wave": T_WAVE,
    "lower": T_RAISE + T_UNLOAD,
    "unshift": T_SHIFT,
}

#: The whole-robot CoM at the lift pose, trunk frame, xy only.  PINNED, for
#: `hw.balance.config`'s reason: both the shift and the load split use the
#: same constant, so the regulated point is consistent even where the true
#: CoM wanders by a few mm as the paw moves.  (0.00, +0.03) mm.  [DERIVED]
_POSE_LIFT = ST.pose_for_height(ST.LIFT_HEIGHT, q_seed=ST.Q_CROUCH)
COM_XY_LIFT = np.array(K.body_inertia(_POSE_LIFT)[0][:2], dtype=float)
COM_XY_LIFT.flags.writeable = False

#: The z spring on THREE feet, and why it is four times the stand's.
#:
#: THE HORIZONTAL SPRINGS CANNOT STOP A TIP.  A tip is a rotation about an
#: edge of the support polygon, and that edge lies ON THE FLOOR -- so every
#: horizontal foot force, however stiff the xy springs, has zero moment about
#: it.  Only vertical forces count, and only the far foot's changes with the
#: tilt.  The tipping stiffness about an edge is therefore
#:
#:     k_z d_far^2  -  W h
#:
#: with d_far the far foot's distance from the edge and h the CoM height.
#: Lift FL and the FR-RL diagonal becomes an edge: d_far = 125 mm, h = 135 mm,
#: and at the stand's 500 N/m the two terms are 7.8 N*m/rad EACH.  Zero net
#: stiffness, and the first MuJoCo run of this file tipped over the diagonal
#: exactly as that predicts.  2000 N/m makes it 31 against 7.8 -- 2 Hz about
#: the diagonal, with the far foot carrying the restoring force.  The side
#: edge (FR-RR, d_far 130 mm) is the next weakest and comes out the same.
#:
#: The stand's 500 is right for the stand: four feet, the CoM in the middle,
#: and a z axis meant to be compliant.  Three feet with the CoM 35 mm from an
#: edge is a different problem, so the gain is BLENDED up over the shift and
#: back down over the unshift rather than switched.  [SIM TUNING]
KP_THREE = np.array([2000.0, 2000.0, 2000.0])          # N/m
KD_THREE = np.array([30.0, 30.0, 60.0])                # N*s/m

#: Trip threshold for the swing foot's Cartesian tracking error -- `hw.stand`
#: reads it; the simulator reports it.  A paw 50 mm from its reference has
#: hit something or lost a motor.
SWING_TRACK_STOP = 0.050                # m


# ===========================================================================
# the pieces
# ===========================================================================
def quintic(s: float) -> tuple[float, float]:
    """``(sigma, dsigma/ds)`` of 10 s^3 - 15 s^4 + 6 s^5, clamped to [0, 1].

    Zero velocity AND zero acceleration at both ends.  `sim.stand.smoothstep`
    is only C1 and its implied acceleration steps at s = 0 and s = 1; with
    the damping term fed an analytic `v_des` that step would land in the
    torque at every phase edge.  Same polynomial as `hw.balance.reference`.
    """
    s = min(1.0, max(0.0, float(s)))
    return (s * s * s * (10.0 + s * (-15.0 + 6.0 * s)),
            30.0 * s * s * (1.0 - s) * (1.0 - s))


def raised_point(leg) -> np.ndarray:
    """RAISED_HIP mirrored for `leg`: toward its own end, outboard, down."""
    i = K.leg_index(leg)
    return np.array([C.END_SIGN[i] * RAISED_HIP[0],
                     C.SIDE_SIGN[i] * RAISED_HIP[1],
                     RAISED_HIP[2]])


def support_shift(leg, feet_xy, com_xy=None, margin: float = SUPPORT_MARGIN
                  ) -> np.ndarray:
    """(2,) how far the TRUNK moves, trunk frame, before `leg` can leave.

    `feet_xy` is (4, 2) in the TRUNK frame.  The two stance feet that share
    an end or a side with the swing foot span the critical edge of the
    support triangle; the third is the far corner.  The trunk is moved along
    that edge's inward normal until the CoM is `margin` inside it.  Zero if
    it already is.

    The feet move the OTHER way in the body frame: a target that was at
    `p` in a hip frame is at ``p - shift`` afterwards.
    """
    i = K.leg_index(leg)
    feet_xy = np.asarray(feet_xy, dtype=float).reshape(C.N_LEGS, 2)
    com_xy = COM_XY_LIFT if com_xy is None else np.asarray(com_xy, dtype=float)
    stance = [k for k in range(C.N_LEGS) if k != i]
    adjacent = [k for k in stance
                if C.END_SIGN[k] == C.END_SIGN[i] or C.SIDE_SIGN[k] == C.SIDE_SIGN[i]]
    far = [k for k in stance if k not in adjacent][0]
    a, b = feet_xy[adjacent[0]], feet_xy[adjacent[1]]
    edge = b - a
    normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
    if float(normal @ (feet_xy[far] - a)) < 0.0:
        normal = -normal
    inside = float(normal @ (com_xy - a))
    return max(0.0, float(margin) - inside) * normal


def static_load_split(feet_xy, com_xy, weight: float, contact=None) -> np.ndarray:
    """(4,) normal force per foot that carries `weight` with NO tilting moment.

    Three equations -- the vertical sum, the moment about x and the moment
    about y, all taken about `com_xy` -- in four unknowns, closed by weighted
    minimum norm with the contact weights as the weighting:

        f = diag(c) A^T (A diag(c) A^T)^-1 (W, 0, 0)

    A foot with c = 0 gets exactly zero and drops out of the 3x3 without
    making it singular, as long as three feet remain and are not collinear.
    `feet_xy` is (4, 2) in the trunk frame; `contact` is (4,) in [0, 1].
    """
    feet_xy = np.asarray(feet_xy, dtype=float).reshape(C.N_LEGS, 2)
    c = (np.ones(C.N_LEGS) if contact is None
         else np.asarray(contact, dtype=float).reshape(C.N_LEGS))
    arm = feet_xy - np.asarray(com_xy, dtype=float).reshape(2)
    A = np.vstack([np.ones(C.N_LEGS), arm[:, 1], -arm[:, 0]])      # (3, 4)
    b = np.array([float(weight), 0.0, 0.0])
    y = np.linalg.solve((A * c) @ A.T, b)
    return c * (A.T @ y)


# ===========================================================================
# the law
# ===========================================================================
class ThreeLegStance:
    """The five phases' targets and torques.  No clock, no simulator, no I/O.

    `enter(phase, q)` latches what a phase needs from the measured pose;
    `torque(phase, t, q, qd)` is the law, `t` seconds into that phase.  Both
    `WaveController` (MuJoCo) and `hw.stand` (the robot) drive exactly this
    object; `kin` is `sim.kinematics` or `hw.kinematics`, which `sim.selftest`
    gates equal.
    """

    PHASES = WAVE_PHASES

    def __init__(self, leg="FL", *, kin=K, margin: float = SUPPORT_MARGIN,
                 raised=None, amplitude: float = WAVE_AMPLITUDE,
                 hz: float = WAVE_HZ, cycles: int = WAVE_CYCLES,
                 kp=ST.KP_CART, kd=ST.KD_CART, kp_three=KP_THREE,
                 kd_three=KD_THREE, com_xy=None):
        self.leg = K.leg_index(leg)
        self.leg_name = C.LEGS[self.leg]
        self.kin = kin
        self.margin = float(margin)
        self.raised = raised_point(self.leg) if raised is None else np.asarray(
            raised, dtype=float).reshape(3)
        self.amplitude = float(amplitude)
        self.hz = float(hz)
        self.cycles = int(cycles)
        self.kp = np.asarray(kp, dtype=float).reshape(3)
        self.kd = np.asarray(kd, dtype=float).reshape(3)
        self.kp_three = np.asarray(kp_three, dtype=float).reshape(3)
        self.kd_three = np.asarray(kd_three, dtype=float).reshape(3)
        self.com_xy = COM_XY_LIFT if com_xy is None else np.asarray(
            com_xy, dtype=float).reshape(2)

        # latched at `enter`
        self.xy0 = np.array(ST.FOOT_XY, dtype=float)      # (4, 2) hip frame
        self.z0 = np.full(C.N_LEGS, -(ST.LIFT_HEIGHT - P.FOOT_RADIUS))
        self.shift = np.zeros(2)
        self.p_floor = np.array([self.xy0[self.leg, 0], self.xy0[self.leg, 1],
                                 self.z0[self.leg]])
        # the last call's diagnostics
        self.p_des = np.zeros((C.N_LEGS, 3))
        self.v_des = np.zeros((C.N_LEGS, 3))
        self.contact = np.ones(C.N_LEGS)
        self.fz = np.full(C.N_LEGS, P.WEIGHT / C.N_LEGS)
        self.foot_hip = np.zeros((C.N_LEGS, 3))
        self.swing_error = 0.0
        self.h_stance = float("nan")
        self.blend = 0.0            # 0: the stand's gains, 1: KP_THREE

    # -- timing ----------------------------------------------------------
    @property
    def t_wave(self) -> float:
        return self.cycles / self.hz

    def duration(self, phase: str) -> float:
        """Seconds until `phase`'s reference has arrived and holds."""
        if phase == "wave":
            return self.t_wave
        return DURATIONS[phase]

    @property
    def height_cmd(self) -> float:
        """Trunk-ORIGIN height the stance feet are told to hold, `sim.stand`'s frame."""
        return float(P.FOOT_RADIUS - self.z0.mean())

    # -- latching --------------------------------------------------------
    def enter(self, phase: str, q) -> None:
        """Latch what `phase` needs from the MEASURED pose `q` (4, 3)."""
        q = C.unflat(q)
        if phase == "shift":
            # FROM WHERE THE FEET ARE, as every handover in sim.stand: the
            # lift law settled somewhere near FOOT_XY and LIFT_HEIGHT, and
            # starting the springs from anywhere else is a step input.
            feet = np.stack([self.kin.foot_position_hip(i, q[i])
                             for i in range(C.N_LEGS)])
            self.xy0 = feet[:, :2].copy()
            self.z0 = feet[:, 2].copy()
            self.shift = support_shift(self.leg, feet[:, :2] + P.HIP_OFFSET[:, :2],
                                       self.com_xy, self.margin)
        elif phase == "raise":
            # Where the paw actually is when it leaves, so `lower` puts it
            # back on the same spot and not on the spring's reference.
            self.p_floor = np.array(self.kin.foot_position_hip(
                self.leg, q[self.leg]), dtype=float)

    # -- the references ----------------------------------------------------
    def _pinned(self, alpha: float) -> np.ndarray:
        """(4, 3) stance targets with the trunk `alpha` of the way shifted."""
        p = np.empty((C.N_LEGS, 3))
        p[:, :2] = self.xy0 - alpha * self.shift
        p[:, 2] = self.z0
        return p

    def _wave_offset(self, t: float) -> tuple[float, float]:
        """Lateral (offset, rate) of the paw, with the amplitude enveloped."""
        if t >= self.t_wave:
            return 0.0, 0.0
        up, dup = quintic(t / T_ENVELOPE)
        down, ddown = quintic((self.t_wave - t) / T_ENVELOPE)
        env = up * down
        denv = (dup * down - up * ddown) / T_ENVELOPE
        w = 2.0 * np.pi * self.hz
        s, c = np.sin(w * t), np.cos(w * t)
        return (self.amplitude * env * s,
                self.amplitude * (denv * s + env * w * c))

    def targets(self, phase: str, t: float):
        """``(p_des (4, 3), v_des (4, 3), contact (4,))`` in the HIP frames.

        Also sets `blend`, the stiffness crossfade: the stand's gains at the
        start of `shift`, KP_THREE from its end until the end of `unshift`.
        """
        t = max(0.0, float(t))
        v = np.zeros((C.N_LEGS, 3))
        contact = np.ones(C.N_LEGS)
        i = self.leg
        self.blend = 1.0

        if phase == "shift":
            alpha, dalpha = quintic(t / T_SHIFT)
            p = self._pinned(alpha)
            v[:, :2] = -(dalpha / T_SHIFT) * self.shift
            self.blend = alpha
        elif phase == "unshift":
            alpha, dalpha = quintic(t / T_SHIFT)
            p = self._pinned(1.0 - alpha)
            v[:, :2] = (dalpha / T_SHIFT) * self.shift
            self.blend = 1.0 - alpha
        elif phase == "raise":
            p = self._pinned(1.0)
            contact[i] = 1.0 - quintic(t / T_UNLOAD)[0]
            p[i] = self.p_floor
            if t > T_UNLOAD:
                sigma, dsigma = quintic((t - T_UNLOAD) / T_RAISE)
                p[i] = self.p_floor + sigma * (self.raised - self.p_floor)
                v[i] = (dsigma / T_RAISE) * (self.raised - self.p_floor)
        elif phase == "wave":
            p = self._pinned(1.0)
            contact[i] = 0.0
            offset, rate = self._wave_offset(t)
            p[i] = self.raised + np.array([0.0, offset, 0.0])
            v[i] = np.array([0.0, rate, 0.0])
        elif phase == "lower":
            p = self._pinned(1.0)
            sigma, dsigma = quintic(t / T_RAISE)
            p[i] = self.raised + sigma * (self.p_floor - self.raised)
            v[i] = (dsigma / T_RAISE) * (self.p_floor - self.raised)
            contact[i] = quintic((t - T_RAISE) / T_UNLOAD)[0] if t > T_RAISE else 0.0
        else:
            raise ValueError("not a wave phase: %r" % phase)
        return p, v, contact

    # -- the law -----------------------------------------------------------
    def torque(self, phase: str, t: float, q, qd, gravity=None) -> np.ndarray:
        """(4, 3) joint torques.  THE ONE COPY OF THE LAW -- see the docstring.

        `gravity` is the (4, 3) leg-weight torque if the caller has it (the
        robot refreshes it on its own schedule); left None it is
        `sim.kinematics.leg_gravity_torque` at this pose.
        """
        q = C.unflat(q)
        qd = C.unflat(qd)
        p_des, v_des, contact = self.targets(phase, t)
        fz = static_load_split(p_des[:, :2] + P.HIP_OFFSET[:, :2], self.com_xy,
                               P.WEIGHT, contact)
        kp = self.kp + self.blend * (self.kp_three - self.kp)
        kd = self.kd + self.blend * (self.kd_three - self.kd)

        tau = np.zeros((C.N_LEGS, 3))
        for i in range(C.N_LEGS):
            foot, jac = self.kin.leg_state(i, q[i])
            p = foot - P.HIP_OFFSET[i]
            self.foot_hip[i] = p
            # f is what the LEG APPLIES TO THE WORLD: the feedforward pushes
            # DOWN by this foot's share, which is zero for the paw in the air.
            force = (kp * (p_des[i] - p) + kd * (v_des[i] - jac @ qd[i])
                     + np.array([0.0, 0.0, -fz[i]]))
            leg_gravity = (K.leg_gravity_torque(i, q[i]) if gravity is None
                           else gravity[i])
            tau[i] = jac.T @ force + leg_gravity

        self.p_des, self.v_des, self.contact, self.fz = p_des, v_des, contact, fz
        self.swing_error = float(np.linalg.norm(p_des[self.leg] - self.foot_hip[self.leg]))
        # Height from the feet that are CARRYING something -- a paw in the air
        # measures nothing, which `sim.stand.height_from_fk` would average in.
        self.h_stance = float(P.FOOT_RADIUS - (contact @ self.foot_hip[:, 2]) / contact.sum())
        return tau

    def q_ref(self, q_seed) -> np.ndarray:
        """(4, 3) joint angles at the last targets -- a tracking trip's reference."""
        return self.kin.all_leg_ik(self.p_des, q_seed=C.unflat(q_seed))

    def status(self) -> str:
        return ("c=%s fz=%s N  paw err %4.1f mm  h3=%.4f"
                % (np.array2string(self.contact, precision=2, suppress_small=True),
                   np.array2string(self.fz, precision=1, suppress_small=True),
                   1e3 * self.swing_error, self.h_stance))


# ===========================================================================
# the controller
# ===========================================================================
class WaveController(ST.StandController):
    """`sim.stand.StandController` with the five wave phases spliced in."""

    PHASES = ("settle", "crouch", "lift") + WAVE_PHASES + ("park", "done")
    BLURB = dict(ST.StandController.BLURB, **{
        "shift":   "TORQUE, four feet -- trunk slides over the feet, CoM into the "
                   "support triangle",
        "raise":   "TORQUE -- the paw is UNLOADED, then lifted to the raised point",
        "wave":    "TORQUE, THREE feet -- the paw waves side to side, then holds",
        "lower":   "TORQUE -- the paw goes back where it left, then is RELOADED",
        "unshift": "TORQUE, four feet -- trunk slides back to centre",
    })

    def __init__(self, leg="FL", **law_kwargs) -> None:
        super().__init__()
        self.law = ThreeLegStance(leg, **law_kwargs)

    def _enter_next(self, data, q) -> None:
        super()._enter_next(data, q)
        if self.phase_name in WAVE_PHASES:
            self.law.enter(self.phase_name, q)

    def _law(self, name: str, elapsed: float, q, qd) -> np.ndarray:
        if name in WAVE_PHASES:
            self.h_cmd = self.law.height_cmd
            return self.law.torque(name, elapsed, q, qd)
        return super()._law(name, elapsed, q, qd)

    def status(self, data) -> str:
        line = super().status(data)
        if self.phase_name in WAVE_PHASES:
            line += "  " + self.law.status()
        return line


# ===========================================================================
# the gates
# ===========================================================================
_FAILURES: list[str] = []
_PASSES = 0


def _check(label: str, ok: bool, detail: str = "") -> None:
    global _PASSES
    if ok:
        _PASSES += 1
        print("  ok    %-62s %s" % (label, detail))
    else:
        _FAILURES.append(label)
        print("  FAIL  %-62s %s" % (label, detail))


def check(leg="FL") -> bool:
    """The offline gates: geometry, reachability, continuity.  No MuJoCo.

    Runnable on the robot host, which is the point -- `hw.stand --wave` runs
    the same targets through the same IK, and a target the leg cannot reach
    is found here rather than by a tracking trip with the robot in the air.
    """
    from hw import kinematics as HK      # the closed form says "unreachable"

    law = ThreeLegStance(leg)
    i = law.leg
    pose = _POSE_LIFT
    print("sim.wave offline gates, leg %s" % law.leg_name)

    # -- the shift ---------------------------------------------------------
    feet = K.all_foot_positions(pose)
    shift = support_shift(i, feet[:, :2], COM_XY_LIFT)
    stance = [k for k in range(C.N_LEGS) if k != i]
    after = feet[:, :2] - shift
    # The CoM's distance inside each edge of the triangle, the critical edge
    # (the two feet adjacent to the swing foot) first.
    adjacent = [k for k in stance
                if C.END_SIGN[k] == C.END_SIGN[i] or C.SIDE_SIGN[k] == C.SIDE_SIGN[i]]
    far = [k for k in stance if k not in adjacent][0]
    margins = []
    for a, b in ((adjacent[0], adjacent[1]), (adjacent[1], far), (far, adjacent[0])):
        edge = after[b] - after[a]
        n = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        margins.append(abs(float(n @ (COM_XY_LIFT - after[a]))))
    _check("the CoM starts ON the critical diagonal (no shift = a line)",
           np.linalg.norm(support_shift(i, feet[:, :2], COM_XY_LIFT, 0.0)) < 1e-4,
           "DOG6's CAD mirrors cleanly: %.3f mm off it"
           % (1e3 * np.linalg.norm(support_shift(i, feet[:, :2], COM_XY_LIFT, 0.0))))
    _check("after the shift the CoM is SUPPORT_MARGIN inside the critical edge",
           abs(margins[0] - SUPPORT_MARGIN) < 1e-9,
           "shift (%+.1f, %+.1f) mm, margins %.0f/%.0f/%.0f mm"
           % (*(1e3 * shift), *(1e3 * np.asarray(margins))))
    _check("...and at least 0.8 x SUPPORT_MARGIN inside the other two",
           min(margins[1:]) >= 0.8 * SUPPORT_MARGIN,
           "the side edge is the next nearest, at %.0f mm" % (1e3 * min(margins[1:])))

    # -- the load split ----------------------------------------------------
    even = static_load_split(feet[:, :2], COM_XY_LIFT, P.WEIGHT)
    _check("four square feet, centred CoM: the split is mg/4",
           np.max(np.abs(even - P.WEIGHT / 4)) < 0.02,
           "%s N" % np.array2string(even, precision=2))
    c3 = np.ones(C.N_LEGS)
    c3[i] = 0.0
    three = static_load_split(after, COM_XY_LIFT, P.WEIGHT, c3)
    arm = after - COM_XY_LIFT
    _check("three feet: the swing foot carries exactly zero",
           three[i] == 0.0, "%s N" % np.array2string(three, precision=2))
    _check("...the other three carry the weight with no tilting moment",
           abs(three.sum() - P.WEIGHT) < 1e-9
           and abs(three @ arm[:, 0]) < 1e-9 and abs(three @ arm[:, 1]) < 1e-9)
    _check("...and every one of them pushes DOWN (f_z > 0)",
           bool(np.all(three[stance] > 5.0)),
           "min %.1f N" % three[stance].min())
    try:
        from hw.balance import allocation as ALLOC
        r_w = np.zeros((C.N_LEGS, 3))
        r_w[:, :2] = arm
        r_w[:, 2] = -(ST.LIFT_HEIGHT - P.FOOT_RADIUS)
        alloc = ALLOC.allocate(r_w, [0, 0, P.WEIGHT, 0, 0, 0], contact=c3,
                               fz_min=0.0, lam=0.0)
        _check("the 3x4 split agrees with hw.balance.allocation's 6x12",
               np.max(np.abs(alloc.fz - three)) < 1e-3,
               "worst %.2e N" % np.max(np.abs(alloc.fz - three)))
    except ImportError:
        pass

    # -- the trajectory: reachable, inside the limits, continuous ----------
    law.enter("shift", pose)
    worst_step = 0.0
    unreachable = outside = 0
    worst_v = 0.0
    prev = None
    dt = 0.004
    seeds = pose.copy()
    for phase in WAVE_PHASES:
        # `seeds` is the IK of the previous target: a leg that tracked
        # perfectly, which is what the latch at `raise` reads on the robot.
        law.enter(phase, seeds)
        n = int(round((law.duration(phase) + 0.5) / dt))
        for k in range(n + 1):
            p, v, c = law.targets(phase, k * dt)
            if prev is not None:
                worst_step = max(worst_step, float(np.abs(p - prev).max()))
                fd = (p - prev) / dt
                worst_v = max(worst_v, float(np.abs(fd - v).max()))
            prev = p
            for leg_k in range(C.N_LEGS):
                sol = HK.leg(leg_k).ik_full(p[leg_k], q_seed=seeds[leg_k])
                seeds[leg_k] = sol.q
                unreachable += not sol.reachable
                outside += not P.within_limits(sol.q)
    _check("every target over the whole envelope is REACHABLE", unreachable == 0,
           "%d unreachable" % unreachable)
    _check("...and inside params.JOINT_LIMITS", outside == 0, "%d outside" % outside)
    _check("the references are continuous across every phase edge",
           worst_step < 1.5e-3, "largest 4 ms step %.2f mm" % (1e3 * worst_step))
    _check("v_des IS the derivative of p_des (analytic, not differenced)",
           worst_v < 0.02, "worst %.4f m/s against a 4 ms difference" % worst_v)
    peak_v = max(np.linalg.norm(law.targets("wave", t)[1][i])
                 for t in np.linspace(0, law.t_wave, 500))
    _check("the paw's peak speed is modest", peak_v < 0.5, "%.2f m/s" % peak_v)
    _check("the raised paw is well off the floor",
           law.raised[2] - law.z0[i] > 0.08,
           "%.0f mm above its floor spot" % (1e3 * (law.raised[2] - law.z0[i])))

    print("  %d checks, %d failed" % (_PASSES + len(_FAILURES), len(_FAILURES)))
    return not _FAILURES


# ===========================================================================
# running it
# ===========================================================================
def run_headless(leg="FL", quiet: bool = False):
    """The whole sequence in MuJoCo with no window, then the physical gates.

    What `check` cannot see: whether the robot STAYS UP.  Four things are
    recorded every step through the five wave phases and gated afterwards:
    the paw's height above the floor, the trunk's roll and pitch, the trunk
    height against the command, and how far each stance foot slid.  Returns
    the controller, the rows, and whether every gate passed.
    """
    import mujoco

    ctrl = WaveController(leg)
    law = ctrl.law
    i = law.leg
    dwell = {"settle": 2.0, "crouch": ST.RAMP_POSITION + 0.5,
             "lift": ST.RAMP_LIFT + 1.0, "park": ST.RAMP_POSITION + 0.5,
             "done": 1.0}
    for phase in WAVE_PHASES:
        dwell[phase] = law.duration(phase) + 0.5

    site = [None]
    record = {name: [] for name in ("phase", "paw_z", "roll", "pitch", "h_err",
                                    "slide", "tau", "err", "paw_y")}
    stance_xy = {}

    def on_step(ctrl, model, data):
        if ctrl.phase_name not in WAVE_PHASES:
            return
        if site[0] is None:
            site[0] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                       for n in C.FOOT_SITE_NAMES]
        feet_w = np.array([data.site_xpos[s] for s in site[0]])
        if not stance_xy:
            for k in range(C.N_LEGS):
                stance_xy[k] = feet_w[k, :2].copy()
        w, x, y, z = data.qpos[C.QPOS_ROOT_QUAT]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        roll, pitch, _ = C.zyx_from_rot(R)
        record["phase"].append(ctrl.phase_name)
        record["paw_z"].append(float(feet_w[i, 2]))
        record["paw_y"].append(float(law.foot_hip[i, 1]))
        record["roll"].append(float(roll))
        record["pitch"].append(float(pitch))
        record["h_err"].append(float(data.qpos[2] - ctrl.h_cmd))
        record["slide"].append(max(float(np.linalg.norm(feet_w[k, :2] - stance_xy[k]))
                                   for k in range(C.N_LEGS) if k != i))
        record["tau"].append(float(np.abs(ctrl.tau).max()))
        record["err"].append(float(law.swing_error))

    ctrl, data, rows = ST.run_headless(dwell, quiet=quiet, controller=ctrl,
                                       on_step=on_step)

    phase = np.array(record["phase"])
    paw_z = np.array(record["paw_z"])
    roll = np.degrees(record["roll"])
    pitch = np.degrees(record["pitch"])
    h_err = 1e3 * np.array(record["h_err"])
    slide = 1e3 * np.array(record["slide"])
    tau = np.array(record["tau"])
    err = 1e3 * np.array(record["err"])
    paw_y = np.array(record["paw_y"])
    in_wave = phase == "wave"
    three = np.isin(phase, ("raise", "wave", "lower"))

    print("\nsim.wave physical gates, leg %s (MuJoCo, %d steps through the wave "
          "phases)" % (law.leg_name, len(phase)))
    clearance = paw_z[in_wave].min() - P.FOOT_RADIUS
    _check("the paw stays off the floor through the whole wave",
           clearance > 0.05, "lowest %.0f mm clearance" % (1e3 * clearance))
    swing_pp = paw_y[in_wave].max() - paw_y[in_wave].min()
    _check("...and actually waves: peak-to-peak lateral travel >= 80 % of 2A",
           swing_pp >= 1.6 * law.amplitude,
           "%.0f mm against %.0f commanded" % (1e3 * swing_pp, 2e3 * law.amplitude))
    _check("the paw tracks its reference", err.max() < 1e3 * SWING_TRACK_STOP
           and np.sqrt(np.mean(err[in_wave] ** 2)) < 15.0,
           "worst %.1f mm, rms %.1f mm in the wave"
           % (err.max(), np.sqrt(np.mean(err[in_wave] ** 2))))
    _check("the trunk stays level on three feet: |roll|, |pitch| < 3 deg",
           max(np.abs(roll[three]).max(), np.abs(pitch[three]).max()) < 3.0,
           "worst roll %+.2f, pitch %+.2f deg"
           % (roll[three][np.argmax(np.abs(roll[three]))],
              pitch[three][np.argmax(np.abs(pitch[three]))]))
    _check("...and holds its height within 10 mm", np.abs(h_err).max() < 10.0,
           "worst %+.1f mm" % h_err[np.argmax(np.abs(h_err))])
    _check("no stance foot slides more than 5 mm", slide.max() < 5.0,
           "worst %.1f mm" % slide.max())
    _check("the peak torque through the wave phases is under TAU_STAGED_MAX",
           tau.max() < 3.0, "%.2f N*m against 3.0" % tau.max())
    final = rows[-1]
    _check("the sequence ends parked in the crouch",
           ctrl.finished and abs(final[2] - ST.CROUCH_HEIGHT) < 2e-3,
           "h_fk %.4f m" % final[2])
    return ctrl, rows, not _FAILURES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DOG6 three-leg stance: lift a paw "
                                             "and wave it")
    ap.add_argument("--leg", choices=C.LEGS, default="FL",
                    help="which foot leaves the floor")
    ap.add_argument("--headless", action="store_true",
                    help="no window; auto-step the phases and gate the run")
    ap.add_argument("--check", action="store_true",
                    help="the offline gates only (no MuJoCo)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.check or args.headless:
        ok = check(args.leg)
        if args.check:
            return 0 if ok else 1
        print("\nDOG6 wave, headless.  lift %.4f m, shift margin %.0f mm, "
              "paw +-%.0f mm at %.1f Hz x %d.\n"
              % (ST.LIFT_HEIGHT, 1e3 * SUPPORT_MARGIN, 1e3 * WAVE_AMPLITUDE,
                 WAVE_HZ, WAVE_CYCLES))
        _, rows, ok_run = run_headless(args.leg, quiet=args.quiet)
        print("\n%-8s %10s %10s %10s %10s" % ("phase", "h_cmd", "h_fk", "h_true", "fk err"))
        for name, h_cmd, h_fk, h_true in rows:
            print("%-8s %10.4f %10.4f %10.4f %+9.2f mm"
                  % (name, h_cmd, h_fk, h_true, 1000.0 * (h_fk - h_true)))
        if _FAILURES:
            print("\nFAILED: " + "; ".join(_FAILURES))
            return 1
        print("\nall %d gates passed" % _PASSES)
        return 0

    ST.run(WaveController(args.leg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
