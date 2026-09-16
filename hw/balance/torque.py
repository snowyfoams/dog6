"""Stage 5: foot force -> joint torque, and the leg gravity the SRB leaves out.

    tau_i = -J_i^T R^T f_i^w  +  tau_grav,i(q_i, R)

THE THIRD AND LAST WORLD/BODY CROSSING
    The allocator returns f_i^w, the ground reaction ON THE ROBOT, in the
    WORLD frame.  The Jacobian maps joint rates to foot velocity in the BODY
    frame.  Virtual work only applies once both are in the same frame, so the
    force comes back first: f_i^b = R^T f_i^w.

    A MISSING R^T IS EXACTLY ZERO ERROR AT R = I, so it passes every level
    bench test and then grows linearly with tilt -- in the one regime the
    attitude loop exists to handle.  `selftest` checks it at a tilt, not at
    the identity, for that reason.

    The minus sign is the other half.  `f_w` is what the GROUND applies to the
    ROBOT; `J^T f` wants what the LEG applies to the WORLD, which is its
    negative.  Getting it backwards does not merely sag -- it commands the
    robot to pull itself into the floor at whatever the cap allows, which
    `sim.stand` measured as the trunk dropping 0.1925 m to 0.0349 m in 1.5 s.
    The offline check is that at a level stand with f_w = (0, 0, +mg/4) this
    equals `sim.stand.compliance_torque`'s own -mg/4 feedforward at the same
    pose, and `selftest` runs it.

WHAT SURVIVES OF THE CHEETAH FEEDFORWARD
    The complete joint-space feedforward for a leg tracking a Cartesian
    reference is

        tau_ff = J^T Lambda (a_ref - Jdot qdot) + C qdot + G

    with Lambda the operational-space inertia at the foot.  Cheetah 3 needs
    all of it because its feet SWING.  On a stand they do not: a_ref = 0, and
    qdot is small enough that the inertial terms and C qdot are a few per cent
    of the per-foot static load.  So the whole expression collapses to G --
    the leg gravity torque, and nothing else.

    IT IS NOT OPTIONAL.  The SRB model lumps the legs into the trunk and so
    assumes they are massless; DOG6's are 60 % of its mass.  Two balances are
    being kept and only one of them is in the wrench:

        EXTERNAL   sum f = m g about the whole-body CoM.  Sets how BIG f is.
                   That is `controller` and `allocation`.
        INTERNAL   with the trunk as base, each joint carries the foot GRF AND
                   the weight of every link distal to it.  Sets how f maps to
                   tau.  That is this file.

    On DOG5 the same term was measured against MuJoCo's floating-base inverse
    dynamics: -J^T f alone was off by 0.482 N*m and with it matched to machine
    precision.  As a fraction of |J^T f| it was 63-65 % on abduction, 29-33 %
    on hip pitch and 2 % on the knee -- and OPPOSITE in sign.  Dropping it is
    what made DOG5's RR abduction run away.

TWO IMPLEMENTATIONS, AND THE SECOND ONE IS WHY THE LAW FITS IN A SLOT
    `_gravity_walk` is the reference: `sim.kinematics.leg_frames` plus the
    tilt, which is the same computation `sim.kinematics.leg_gravity_torque`
    does and gated equal to it at R = I.  MEASURED at 465 us for four legs,
    which was 67 % of the entire control law and the single reason it did not
    fit in a 333 us slot.

    `leg_gravity_torque` is the closed form.  Same derivation, no chain walk,
    no (3, 3, 3) intermediates -- the roll factors out exactly as it does in
    `hw.kinematics`, leaving two planar rotations acting on constant vectors
    and nine scalar products.  MEASURED at 36 us for four legs, 13x faster,
    and `selftest` gates the two against each other over 300 random poses AND
    random orientations, where they agree to 2.2e-16.  That gate is the entire
    licence for the second copy; without it, delete it.

    With the closed form the whole law is 210 us of the 333 us slot, all four
    legs refreshed every sweep, and the 12 ms cross-leg skew the per-leg law
    carries is simply gone rather than traded against something.

WHY THE GRAVITY TERM IS TILTED HERE AND IS NOT IN `sim.kinematics`
    `sim.kinematics.leg_gravity_torque` takes gravity along trunk -z, which is
    right only for a level trunk.  It says so in its own docstring and offers
    no way to pass an orientation, because it has no way to know the base's.
    This file has R.  At the lift pose a 10 deg pitch moves the term by
    0.042 N*m -- 11 % of it, and invisible to every level bench test.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/torque.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import kinematics as SK     # noqa: E402
from sim import params as P          # noqa: E402

__all__ = ["leg_gravity_torque", "all_leg_gravity_torque", "stance_torque",
           "gravity_direction_body"]

_DOWN_W = np.array([0.0, 0.0, -1.0])


def gravity_direction_body(R=None) -> np.ndarray:
    """Which way is down, in the TRUNK frame.  (0, 0, -1) at a level trunk."""
    if R is None:
        return _DOWN_W
    return np.asarray(R, dtype=float).reshape(3, 3).T @ _DOWN_W


# ===========================================================================
# the reference implementation -- correctness, not speed
# ===========================================================================
def _gravity_walk(leg, q, R=None) -> np.ndarray:
    """`sim.kinematics.leg_gravity_torque` with the gravity direction tilted.

    NOT CALLED BY THE CONTROL LOOP.  It is the thing `leg_gravity_torque` is
    gated against, and the thing to read if the closed form ever disagrees
    with the robot -- this one is a plain statement of the statics and the
    other is an optimisation of it.
    """
    _, anchors, axes, _, link_coms = SK.leg_frames(leg, q)
    inertials = P.LINK_INERTIALS[SK.leg_name(leg)]
    down_b = gravity_direction_body(R)

    tau = np.zeros(3)
    for link, inertial in enumerate(inertials):
        weight = inertial.mass * P.GRAVITY
        # A joint only carries the links at or below it in the chain.
        for joint in range(link + 1):
            dcom = np.cross(axes[joint], link_coms[link] - anchors[joint])
            tau[joint] -= weight * float(dcom @ down_b)
    return tau


# ===========================================================================
# the closed form the loop runs
# ===========================================================================
def _constants(name: str):
    """Per-leg geometry and inertial data, folded once at import.

    Everything below is expressed in the ROLL FRAME -- the leg's frame after
    Rx(q1) has been taken out.  That is legitimate because the abduction axis
    IS trunk +x and passes through the hip point, so Rx factors out of every
    position in the chain exactly as it does out of `hw.kinematics`' Jacobian:

        point_trunk = hip + Rx(q1) @ point_roll

    A cross product is carried by a rotation with det = 1 unchanged, and the
    dot product against gravity then only needs gravity rolled the other way
    -- one 3-vector, once per leg.  So q1 never appears below.
    """
    geom = P.LEG_GEOMETRY[name]
    inertials = P.LINK_INERTIALS[name]
    return (np.asarray(geom.hip_to_pitch, dtype=float),
            np.asarray(geom.pitch_to_knee, dtype=float),
            np.stack([np.asarray(item.com, dtype=float) for item in inertials]),
            np.array([item.mass for item in inertials]) * P.GRAVITY)


_LEG_CONSTANTS = {name: _constants(name) for name in C.LEGS}


def _gravity_closed(name: str, q, down_b) -> np.ndarray:
    """The closed form for one leg, given gravity already in the TRUNK frame.

    Split out from `leg_gravity_torque` so the four-leg call can compute the
    trunk-frame gravity direction ONCE -- it is the same 3-vector on all four
    legs and the 3x3 product that makes it is a fifth of the whole cost.

    THE DERIVATION, because nine scalar products are unreadable otherwise.
    With w_l the link CoMs and a_j the hinge points, both in the ROLL frame,
    and e_j the hinge axes there -- x_hat, z_hat, z_hat, the last two being
    the same vector because the pitch and knee axes are PARALLEL:

        tau_j = - sum_{l >= j} m_l g  ( e_j x (w_l - a_j) ) . g_roll

    and both cross products are trivial by hand,

        x_hat x v = (0, -v_z, +v_y)          z_hat x v = (-v_y, +v_x, 0)

    so each joint touches exactly two components of g_roll.  The hip link
    carries no planar rotation at all, which is why w_0 is a constant.
    """
    d1, d2, coms, weights = _LEG_CONSTANTS[name]
    q1, q2, q3 = (float(v) for v in np.asarray(q, dtype=float).reshape(3))

    # Gravity in the roll frame, Rx(-q1) @ down_b.  x is untouched by a roll.
    c1, s1 = np.cos(q1), np.sin(q1)
    grx = down_b[0]
    gry = c1 * down_b[1] + s1 * down_b[2]
    grz = -s1 * down_b[1] + c1 * down_b[2]

    c2, s2 = np.cos(q2), np.sin(q2)
    c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)

    # The chain in the roll frame: hinge points, then link CoMs.
    a1x, a1y, a1z = d1                                          # pitch hinge
    a2x = a1x + d2[0] * c2 - d2[1] * s2                         # knee hinge
    a2y = a1y + d2[0] * s2 + d2[1] * c2
    a2z = a1z + d2[2]

    w0x, w0y, w0z = coms[0]                                     # hip link
    w1x = a1x + coms[1][0] * c2 - coms[1][1] * s2               # thigh
    w1y = a1y + coms[1][0] * s2 + coms[1][1] * c2
    w1z = a1z + coms[1][2]
    w2x = a2x + coms[2][0] * c23 - coms[2][1] * s23             # shin
    w2y = a2y + coms[2][0] * s23 + coms[2][1] * c23
    w2z = a2z + coms[2][2]

    m0, m1, m2 = weights
    # abduction: x_hat x (w - hip), all three links
    tau0 = -(m0 * (w0y * grz - w0z * gry)
             + m1 * (w1y * grz - w1z * gry)
             + m2 * (w2y * grz - w2z * gry))
    # hip pitch: z_hat x (w - a1), thigh and shin
    tau1 = -(m1 * ((w1x - a1x) * gry - (w1y - a1y) * grx)
             + m2 * ((w2x - a1x) * gry - (w2y - a1y) * grx))
    # knee: z_hat x (w2 - a2), shin only
    tau2 = -m2 * ((w2x - a2x) * gry - (w2y - a2y) * grx)
    return np.array([tau0, tau1, tau2])


def leg_gravity_torque(leg, q, R=None) -> np.ndarray:
    """(3,) torque this leg's motors hold against its OWN link weights.

    `R` is the trunk orientation, world <- body.  Left None it is the
    identity and this reduces exactly to `sim.kinematics.leg_gravity_torque`.

    Sign: what the MOTOR must produce to HOLD the links up, so it is ADDED to
    the contact term rather than subtracted.  No foot contact is assumed, so
    subtracting this from torque feedback isolates the external load -- which
    is what the readback check in `hw.stand` uses it for.
    """
    return _gravity_closed(SK.leg_name(leg), q, gravity_direction_body(R))


def all_leg_gravity_torque(q, R=None) -> np.ndarray:
    """(4, 3) of the above.  `q` is (4, 3) or flat (12,).

    MEASURED: 36 us for four legs against 465 us for the chain walk.  That
    difference is what puts the whole law inside a 333 us slot with all four
    legs refreshed every sweep -- so the 12 ms cross-leg skew the per-leg law
    carries is gone rather than traded against something else.
    """
    q = C.unflat(q)
    down_b = gravity_direction_body(R)
    return np.stack([_gravity_closed(name, q[i], down_b)
                     for i, name in enumerate(C.LEGS)])


# ===========================================================================
# the boxed equation
# ===========================================================================
def stance_torque(state, f_w, gravity=None) -> np.ndarray:
    """(12,) joint torques for the allocated foot forces.  The boxed equation.

    `state` is a `state.BodyState` -- it already carries R and the four
    Jacobians, computed once in stage 1 rather than a second time here.
    `f_w` is (4, 3), the ground reaction ON THE ROBOT in the WORLD frame, as
    `allocation.allocate` returns it.

    `gravity` is the (4, 3) leg-weight term if the caller already has it (the
    runner can sub-rate it); left None it is computed here at this sweep's R.
    """
    f_w = np.asarray(f_w, dtype=float).reshape(C.N_LEGS, 3)
    if gravity is None:
        gravity = all_leg_gravity_torque(state.q, state.R)
    gravity = np.asarray(gravity, dtype=float).reshape(C.N_LEGS, 3)

    # f^b = R^T f^w for every foot at once: (R^T f)^T = f^T R, so `f_w @ R`.
    f_b = f_w @ state.R
    tau = np.empty((C.N_LEGS, 3))
    for i in range(C.N_LEGS):
        tau[i] = -(state.jac[i].T @ f_b[i]) + gravity[i]
    return C.flat(tau)


def describe() -> str:
    from sim import stand as ST
    q = ST.pose_for_height(ST.LIFT_HEIGHT, q_seed=ST.Q_CROUCH)
    level = all_leg_gravity_torque(q)
    tilted = all_leg_gravity_torque(q, C.rot_y(np.deg2rad(10.0)))
    walk = np.stack([_gravity_walk(i, q[i], C.rot_y(np.deg2rad(10.0)))
                     for i in range(C.N_LEGS)])
    return "\n".join([
        "DOG6 stage 5: tau = -J^T R^T f^w + tau_grav",
        "  at the lift pose, leg gravity alone (N*m):",
        "    level      %s" % np.array2string(level, precision=3),
        "    10 deg nose-down pitch",
        "               %s" % np.array2string(tilted, precision=3),
        "    the tilt moves it by %.3f N*m -- what a level bench test cannot see"
        % float(np.abs(tilted - level).max()),
        "  closed form vs the chain walk at that tilt: %.2e N*m"
        % float(np.abs(tilted - walk).max()),
        "  gated over random poses and orientations by",
        "  `python -m hw.balance.selftest`",
    ])


if __name__ == "__main__":
    print(describe())
