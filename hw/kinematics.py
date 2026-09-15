"""DOG6 leg kinematics for the hardware path: closed-form FK, Jacobian and IK.

WHY A SECOND KINEMATICS EXISTS AT ALL, WHEN `sim.kinematics` SAYS IT MUST NOT
    `sim.kinematics`' docstring is right that a second forward kinematics
    agreeing BY INSPECTION is the failure mode this project is laid out to
    prevent.  This one does not agree by inspection -- `sim.selftest` drives
    both over random poses and fails if they part company at 1e-12.  That
    gate is the entire licence for this file; without it, delete it.

    It earns its place because the two are not the same computation:

        sim.kinematics    recursive body-chain walk (the MJCF's own form).
                          Produces hinge points, body rotations and link CoMs
                          as a side effect, which the gravity torque and the
                          composite inertia need.
        hw.kinematics     four trig evaluations and a 3x3 product, hand
                          derived.  Produces the foot, its Jacobian and a
                          CLOSED-FORM inverse, and imports no simulator.  A
                          control sweep on the robot runs this.

    Measured: 8.2 us against 69 us for `sim.kinematics.leg_state` on the same
    pose, so twelve joints cost 0.10 ms of a 4 ms sweep instead of 0.83 ms.
    The inverse is the bigger gap -- 20 us against 463 us for
    `sim.kinematics.leg_ik`, and see `Leg.ik` for why the difference is not
    really about speed.  None of this is a reason to trust the file; the gate
    is.

WHAT THE ROBOT ACTUALLY HAS TO CARRY
    This module, `hw/__init__.py`, `sim/__init__.py`, `sim/coordinates.py`,
    `sim/params.py`, and numpy.  Five files.  No MuJoCo, no model/, no meshes,
    no `sim.kinematics`, no solver -- checked by importing it with `mujoco`
    blocked out of `sys.meta_path`, from a directory holding nothing else.

    `params` is a table of literals plus numpy; it reads no file at import
    (`load_export` is the only I/O and it is lazy).  So the geometry is shared
    with `sim` rather than copied, which is the point: there is one set of
    link offsets in this project, `sim.selftest` gates it against
    model/dog6.xml, and a second copy living here would be free to drift from
    the CAD while still passing every check this file could run on its own.

TOPOLOGY
    q1  about  x_hat        abduction / roll, through the hip point
    q2  about  R1 @ z_hat   hip pitch
    q3  about  R1 @ z_hat   knee -- PARALLEL to the pitch axis

    Three offsets, all expressed in the base frame at q = 0, all straight out
    of `params.LEG_GEOMETRY`:

        d1 = hip_to_pitch       hip hinge   -> pitch hinge
        d2 = pitch_to_knee      pitch hinge -> knee hinge
        d3 = knee_to_foot       knee hinge  -> foot site

    There are no intermediate link frames and no DH parameters.  This is the
    axis-offset (PoE-equivalent) parameterisation.

CLOSED FORM
        v2    = Rz(q2)  @ d2
        v3    = Rz(q23) @ d3                    q23 = q2 + q3
        Sigma = v2 + v3
        w     = d1 + Sigma = (P, Q, Z)          Z is CONSTANT (= z1 + z2 + z3)
        p     = hip + R1 @ w

                     [  0      -Sigma_y   -v3_y ]
        J    = R1 @  [ -Z       Sigma_x    v3_x ]
                     [  Q        0          0   ]

        det J = Q * (v2_x * v3_y - v2_y * v3_x)

    Both bracketed matrices are free of q1: the roll factors out entirely.
    The third row (Q, 0, 0) is the statement that q2 and q3 turn about z_hat
    and so cannot change the third component of w.

    THAT CONSTANT Z IS ALSO WHAT MAKES THE INVERSE CLOSED FORM -- see `Leg.ik`.

    `hip` is OUTSIDE R1, and that is not a detail: the abduction axis passes
    THROUGH the hip point, so the trunk-to-hip offset is not carried by the
    roll.  Being a constant it drops out of the Jacobian entirely, which is
    why J is the same matrix in the trunk frame and in the hip frame.

THE REAR LEGS ARE HANDLED BY THEIR NUMBERS, NOT BY A SIGN IN THIS FILE
    DOG6's rear legs are the front legs turned 180 deg about z, so d1, d2 and
    d3 have their x components NEGATED -- and their z components do NOT flip,
    because a turn about z does not touch z.  The 5 mm foot offset in d3 is
    -0.005 on all four legs.

    Nothing above assumes a sign on any component of d, so this needs no
    branch: the rear legs are the same closed form fed the rear numbers.  The
    joint AXES stay x_hat and z_hat on all four legs, exactly as
    `coordinates` fixes them and exactly as model/dog6.xml declares them.

    THE TEMPTING ALTERNATIVE IS WRONG.  Giving the rear legs a base frame
    rotated 180 deg about z would make all four sets of offsets identical, and
    it would cost: the abduction axis becomes -x_hat in that frame, so q1
    changes sign, the first Jacobian column changes sign, the result needs
    rotating back, and hw's q1 no longer means what the MJCF's and the
    firmware's q1 mean.  That moves a sign out of data that `sim.selftest`
    checks against the XML and into code that nothing checks.

    One consequence worth knowing: x2 * x3 is positive on the rear legs too
    (two negatives), so with DOG6's y2 = y3 = 0 the determinant collapses to

        det J = Q * x2 * x3 * sin(q3)

    identically on all four legs.  The knee singularity (sin q3 = 0) and the
    abduction singularity (Q = 0, foot on the roll axis) are therefore the
    same surfaces front and rear, and `manipulability` takes one threshold for
    the whole robot.  Only the SIGN of det differs, which is the mirror.

NOT GATED BY `hw.CONFIRMED_ON_DOG6`
    That flag guards tables read off the assembled robot -- CAN ids, direction
    signs, encoder offsets.  This file contains none: it is CAD geometry, it
    is checked against the model, and it is true before the robot is powered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/kinematics.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

X_HAT = np.array([1.0, 0.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])

__all__ = ["Leg", "LegCore", "IKSolution", "LEGS", "leg", "hat", "rot_x", "rot_z",
           "foot_position", "foot_position_hip", "foot_jacobian", "leg_state",
           "all_foot_positions", "leg_ik", "all_leg_ik", "describe"]


def hat(a) -> np.ndarray:
    """Skew-symmetric matrix [a]_x, so that ``hat(a) @ b == np.cross(a, b)``."""
    ax, ay, az = a
    return np.array([[0.0, -az, ay],
                     [az, 0.0, -ax],
                     [-ay, ax, 0.0]])


def rot_x(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def rot_z(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


TWO_PI = 2.0 * np.pi


class LegCore(NamedTuple):
    """Shared intermediates.  Every downstream quantity is built from these."""
    R1: np.ndarray      # (3,3)  Rx(q1)
    v2: np.ndarray      # (3,)   Rz(q2)  @ d2
    v3: np.ndarray      # (3,)   Rz(q23) @ d3
    Sigma: np.ndarray   # (3,)   v2 + v3
    w: np.ndarray       # (3,)   d1 + Sigma == (P, Q, Z), the roll-frame foot


class IKSolution(NamedTuple):
    """What `Leg.ik_full` knows that `Leg.ik` throws away."""
    q: np.ndarray       # (3,)  the joint angles
    reachable: bool     # False if the target is outside the leg's workspace,
                        # in which case `q` is the nearest pose, not a solution
    branch: tuple       # (roll_sign, knee_sign), the two elbow choices taken


class _IKConstants(NamedTuple):
    """Per-leg quantities the inverse needs, folded once at construction."""
    Z: float            # z1 + z2 + z3, the invariant the roll is solved from
    a: float            # |d2 in the xy plane|, and the angle it sits at
    alpha: float
    b: float            # |d3 in the xy plane|, and the angle it sits at
    beta: float
    a2b2: float         # a*a + b*b
    ab2: float          # 2*a*b


def _near(x: float, ref: float) -> float:
    """`x` shifted by a whole number of turns to land closest to `ref`.

    A joint angle is only defined modulo 2 pi, and the branch the solver hands
    back has to be the branch the encoder is ON -- a knee reported as +5.5 rad
    when it sits at -0.78 is a 2 pi step commanded into the leg.
    """
    return ref + (x - ref + np.pi) % TWO_PI - np.pi


@dataclass(frozen=True)
class Leg:
    """Geometry of one leg.  Immutable: the offsets are calibration constants.

    `hip` is that leg's abduction hinge in the TRUNK frame.  Left at zero it
    is absent and every result is in the leg's own hip frame.
    """

    d1: np.ndarray
    d2: np.ndarray
    d3: np.ndarray
    hip: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: This leg's stance pose -- the default IK seed, and so the elbow branch
    #: the inverse returns when nobody says otherwise.  Zeros would be the flat
    #: calibration pose, which is very nearly singular: see `coordinates`.
    stand: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _ik: _IKConstants = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        for name in ("d1", "d2", "d3", "hip", "stand"):
            v = np.array(getattr(self, name), dtype=float).reshape(3)
            v.flags.writeable = False
            object.__setattr__(self, name, v)
        a = float(np.hypot(self.d2[0], self.d2[1]))
        b = float(np.hypot(self.d3[0], self.d3[1]))
        object.__setattr__(self, "_ik", _IKConstants(
            Z=float(self.d1[2] + self.d2[2] + self.d3[2]),
            a=a, alpha=float(np.arctan2(self.d2[1], self.d2[0])),
            b=b, beta=float(np.arctan2(self.d3[1], self.d3[0])),
            a2b2=a * a + b * b, ab2=2.0 * a * b))

    # ---------------------------------------------------------------- core

    def _core(self, q) -> LegCore:
        """Four trig evaluations; everything else is algebra."""
        q1, q2, q3 = q
        c2, s2 = np.cos(q2), np.sin(q2)
        q23 = q2 + q3
        c23, s23 = np.cos(q23), np.sin(q23)

        x2, y2, z2 = self.d2
        x3, y3, z3 = self.d3

        v2 = np.array([x2 * c2 - y2 * s2,
                       x2 * s2 + y2 * c2,
                       z2])
        v3 = np.array([x3 * c23 - y3 * s23,
                       x3 * s23 + y3 * c23,
                       z3])

        Sigma = v2 + v3
        return LegCore(R1=rot_x(q1), v2=v2, v3=v3, Sigma=Sigma, w=self.d1 + Sigma)

    # ------------------------------------------------------------ kinematics

    def fk(self, q) -> np.ndarray:
        """Foot position, trunk frame if `hip` was given.  Shape (3,)."""
        k = self._core(np.asarray(q, dtype=float))
        return self.hip + k.R1 @ k.w

    def fk_hip(self, q) -> np.ndarray:
        """Foot position in this leg's own hip frame, `hip` left off."""
        k = self._core(np.asarray(q, dtype=float))
        return k.R1 @ k.w

    def fk_orientation(self, q) -> np.ndarray:
        """Shin orientation ``Rx(q1) @ Rz(q2 + q3)``.  Shape (3,3).

        Meaningless for a point foot, but needed for contact-moment models.
        """
        q1, q2, q3 = np.asarray(q, dtype=float)
        return rot_x(q1) @ rot_z(q2 + q3)

    def jacobian(self, q) -> np.ndarray:
        """Position Jacobian dp/dq.  Shape (3,3)."""
        return self.fk_jac(q)[1]

    def fk_jac(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Foot position and position Jacobian from ONE pass.

        The Jacobian costs no extra trig: Sigma, v3, Q and Z are already
        available from the FK.  Only the final 3x3 product is added.
        """
        k = self._core(np.asarray(q, dtype=float))
        _, Q, Z = k.w

        inner = np.array([[0.0, -k.Sigma[1], -k.v3[1]],
                          [-Z, k.Sigma[0], k.v3[0]],
                          [Q, 0.0, 0.0]])
        return self.hip + k.R1 @ k.w, k.R1 @ inner

    def jacobian_angular(self, q) -> np.ndarray:
        """Angular Jacobian ``[a1 | a2 | a3]``.  Shape (3,3), rank 2.

        Columns 2 and 3 coincide because the pitch axes are parallel: the foot
        orientation has only two controllable degrees of freedom.
        """
        a23 = rot_x(np.asarray(q, dtype=float)[0]) @ Z_HAT
        return np.column_stack([X_HAT, a23, a23])

    # -------------------------------------------------------- the inverse

    def ik(self, p_hip, q_seed=None) -> np.ndarray:
        """Joint angles putting this leg's foot at `p_hip`, in the HIP frame.

        CLOSED FORM, NOT A SOLVER, AND THE DIFFERENCE IS NOT MAINLY SPEED.
        `sim.kinematics.leg_ik` is damped least squares: it iterates, it stops
        when a residual falls under a tolerance, and its cost depends on where
        the target is.  All three are wrong things to put in a 250 Hz loop on
        a robot.  This runs a fixed number of operations, returns the EXACT
        pose (3e-16 m over the whole joint torus, not 1e-6), and says outright
        when the target cannot be reached instead of quietly handing back
        however close it got.

        THE TOPOLOGY MAKES IT SEPARABLE, WHICH IS THE WHOLE TRICK.  q2 and q3
        both turn about z_hat, so neither can change the third component of
        w = Rx(-q1) @ p_hip.  That component is the constant Z = z1 + z2 + z3.
        So the roll falls out of ONE scalar equation that knows nothing about
        the other two joints:

            uz cos q1 - uy sin q1 = Z        (u = p_hip)

        an A cos + B sin = C, solved as q1 = atan2(-uy, uz) +- acos(Z / r)
        with r = hypot(uy, uz).  What is left is a planar 2R in the rolled
        frame, solved by the law of cosines.

        ON DOG6 Z IS -5 mm, AND THAT IS THE FOOT BALL'S OUT-OF-PLANE OFFSET,
        NOT A LINK LENGTH -- d3 is (0.105, 0, -0.005), 105.1 mm long.  It sets
        the condition r >= |Z|, and with it the chain's global minimum
        |p_hip| = |Z|: fold the 2R until its end lands on the roll axis and
        only the 5 mm off-plane term survives.  That minimum is a fact about
        the CHAIN.  The leg cannot hold it -- it wants the knee at -164 deg,
        past `params.KNEE_LIM`, with the shin doubled back over the thigh.
        Inside JOINT_LIMITS the foot gets no closer than 27 mm.

        Nothing here enforces that, and the simulator does not either: the
        MJCF leaves every hinge unlimited and the legs cannot self-collide.

        BRANCHES.  Two roll solutions times two knee solutions is four exact
        answers, and they are all correct.  `q_seed` chooses between them:
        the one closest to it wins, joint by joint and modulo 2 pi.  Pass the
        previous sweep's measured pose and the leg cannot flip its elbow or
        take a 2 pi step under you.  With no seed, `stand` is used.

        UNREACHABLE TARGETS return the nearest pose the leg can hold, with
        both acos arguments clamped -- finite, continuous, and safe to command.
        Use `ik_full` if you need to know that it happened; a foot placement
        that quietly saturates is a gait that limps for a reason nothing logs.

        Joint limits are NOT applied.  `params.clamp_q` afterwards if needed.
        """
        return self.ik_full(p_hip, q_seed).q

    def ik_full(self, p_hip, q_seed=None) -> IKSolution:
        """`ik`, plus whether the target was reachable and which branch won."""
        k = self._ik
        u = np.asarray(p_hip, dtype=float).reshape(3)
        ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
        seed = self.stand if q_seed is None else np.asarray(q_seed, dtype=float)
        s1, s2, s3 = float(seed[0]), float(seed[1]), float(seed[2])

        # -- the roll, from the invariant third component ------------------
        r = float(np.hypot(uy, uz))
        arg = k.Z / r if r > 0.0 else 2.0            # r == 0 cannot hold Z != 0
        roll_ok = abs(arg) <= 1.0
        half = float(np.arccos(arg if roll_ok else (1.0 if arg > 0.0 else -1.0)))
        phi = float(np.arctan2(-uy, uz))

        best, best_d, best_branch, reachable = None, np.inf, (0, 0), roll_ok
        for roll_sign in ((+1, -1) if half != 0.0 else (+1,)):
            q1 = _near(phi + roll_sign * half, s1)
            # -- what is left is a planar 2R in the rolled frame ------------
            X = ux - self.d1[0]
            Y = np.cos(q1) * uy + np.sin(q1) * uz - self.d1[1]
            cos_psi = (X * X + Y * Y - k.a2b2) / k.ab2
            knee_ok = abs(cos_psi) <= 1.0
            psi0 = float(np.arccos(
                cos_psi if knee_ok else (1.0 if cos_psi > 0.0 else -1.0)))
            base = float(np.arctan2(Y, X)) - k.alpha
            for knee_sign in ((+1, -1) if psi0 != 0.0 else (+1,)):
                psi = knee_sign * psi0
                q3 = _near(psi - k.beta + k.alpha, s3)
                q2 = _near(base - float(np.arctan2(
                    k.b * np.sin(psi), k.a + k.b * np.cos(psi))), s2)
                d = max(abs(q1 - s1), abs(q2 - s2), abs(q3 - s3))
                if d < best_d:
                    best, best_d = (q1, q2, q3), d
                    best_branch = (roll_sign, knee_sign)
                    reachable = roll_ok and knee_ok
        return IKSolution(np.array(best), reachable, best_branch)

    # -------------------------------------------------------- singularities

    def det_jacobian(self, q) -> float:
        """det(J), in the factored form ``Q * (v2 x v3)_z``.

        Two singular families:
            (v2 x v3)_z == 0 -> planar 2R boundary (knee straight or folded)
            Q == 0           -> abduction singularity (foot on the roll axis)
        Independent of q1, as det(R1) == 1.
        """
        k = self._core(np.asarray(q, dtype=float))
        cross_z = k.v2[0] * k.v3[1] - k.v2[1] * k.v3[0]
        return float(k.w[1] * cross_z)

    def manipulability(self, q) -> float:
        """|det J|.  Cheap workspace-boundary monitor for the control loop."""
        return abs(self.det_jacobian(q))

    # -------------------------------------------------------------- statics

    def joint_torque(self, q, f_foot) -> np.ndarray:
        """``tau = J^T f``, with `f_foot` the force the LEG applies to the
        world -- so a stance leg holding the robot up passes a positive z."""
        return self.fk_jac(q)[1].T @ np.asarray(f_foot, dtype=float).reshape(3)


# ===========================================================================
# the four DOG6 legs
# ===========================================================================
#: Built from `params.LEG_GEOMETRY`, which is gated against model/dog6.xml.
#: Not a copy of those numbers -- a view of them.
LEGS: dict[str, Leg] = {
    name: Leg(d1=P.LEG_GEOMETRY[name].hip_to_pitch,
              d2=P.LEG_GEOMETRY[name].pitch_to_knee,
              d3=P.LEG_GEOMETRY[name].knee_to_foot,
              hip=P.LEG_GEOMETRY[name].hip,
              stand=C.Q_STAND[C.LEG_INDEX[name]])
    for name in C.LEGS
}


def leg(which) -> Leg:
    """The `Leg` for "FL".."RR" or 0..3 -- `coordinates` fixes which is which."""
    if isinstance(which, str):
        if which not in LEGS:
            raise ValueError(f"unknown leg {which!r}; expected one of {C.LEGS}")
        return LEGS[which]
    index = int(which)
    if not 0 <= index < C.N_LEGS:
        raise ValueError(f"leg index must be 0..{C.N_LEGS - 1}, got {which}")
    return LEGS[C.LEGS[index]]


# ---------------------------------------------------------------------------
# `sim.kinematics`' names, so a controller can be pointed at either module.
# ---------------------------------------------------------------------------
def foot_position(which, q) -> np.ndarray:
    """The foot site in the TRUNK frame, metres."""
    return leg(which).fk(q)


def foot_position_hip(which, q) -> np.ndarray:
    """The foot site relative to that leg's own hip hinge."""
    return leg(which).fk_hip(q)


def foot_jacobian(which, q) -> np.ndarray:
    """``d(foot_position)/dq``, 3x3.  The same matrix in either frame."""
    return leg(which).fk_jac(q)[1]


def leg_state(which, q) -> tuple[np.ndarray, np.ndarray]:
    """``(foot_in_trunk_frame, jacobian)`` from ONE pass."""
    return leg(which).fk_jac(q)


def all_foot_positions(q) -> np.ndarray:
    """(4, 3) foot sites in the trunk frame, for a (4, 3) whole-robot pose."""
    q = C.unflat(q)
    return np.stack([LEGS[n].fk(q[i]) for i, n in enumerate(C.LEGS)], axis=0)


def leg_ik(which, p_hip, q_seed=None) -> np.ndarray:
    """Joint angles putting a foot at `p_hip`, in that leg's HIP frame.

    `sim.kinematics.leg_ik`'s name and arguments, minus its `iters` and `tol`:
    this one does not iterate, so there is nothing for them to mean.
    """
    return leg(which).ik(p_hip, q_seed)


def all_leg_ik(p_hip, q_seed=None) -> np.ndarray:
    """(4, 3) IK for four feet given as (4, 3) hip-frame positions."""
    p_hip = np.asarray(p_hip, dtype=float).reshape(C.N_LEGS, 3)
    seeds = (None,) * C.N_LEGS if q_seed is None else C.unflat(q_seed)
    return np.stack([LEGS[n].ik(p_hip[i], seeds[i])
                     for i, n in enumerate(C.LEGS)], axis=0)


def describe() -> str:
    lines = ["DOG6 closed-form leg kinematics (hw path, no simulator)",
             "  at Q_STAND:"]
    for i, name in enumerate(C.LEGS):
        item, q = LEGS[name], C.Q_STAND[i]
        core = item._core(q)
        lines.append("    %s  d1x %+.6f  d2x %+.4f  d3x %+.4f   "
                     "w = (P %+.6f, Q %+.6f, Z %+.4f)   det J %+.4e"
                     % (name, item.d1[0], item.d2[0], item.d3[0],
                        *core.w, item.det_jacobian(q)))
    lines.append("  Z = z1 + z2 + z3 = %+.4f m on every leg at every pose"
                 % (LEGS["FL"].d1[2] + LEGS["FL"].d2[2] + LEGS["FL"].d3[2]))
    lines.append("  gated against sim.kinematics and MuJoCo by `python -m sim.selftest`")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
