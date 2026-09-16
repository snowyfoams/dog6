"""The stand's balance controller: an SRB wrench law with a force allocator.

    python -m hw.balance.selftest        gate all of it, no robot, no IMU
    python -m hw.stand --law srb         run it
    python -m hw.stand --law per-leg     run what it replaces

WHAT THIS REPLACES, AND WHAT IT DOES NOT TOUCH
    Only the LIFT phase of `hw.stand`.  The limp, settle, crouch, park and
    done phases are the drivers' own 0xA4 position loops and nothing here
    reaches them.

    The law it replaces is `sim.stand.compliance_torque` -- a Cartesian
    spring-damper per leg with a fixed mg/4 feedforward, four independent
    controllers with no shared model of the body.  THAT LAW STAYS IN PLACE,
    unused, as the A/B baseline: without it the first question after a bad
    run, "is this worse than what we had?", has no answer.

WHAT THE OLD LAW COULD NOT SEE
    Its four legs each track a foot position in their own hip frame, and
    nothing in it names the trunk's ORIENTATION.  No sensor read it and no
    variable represented it, so roll and pitch were emergent: whatever the
    four springs happened to produce.  Load sharing was fixed at mg/4
    regardless of where the feet actually were, and the six-dimensional
    internal-force null space was picked by accident.

    This law names all of it.  Attitude is measured, the error is an SO(3)
    log map, and the gains that act on it are roll and pitch gains DIRECTLY,
    independent of the height gains and of the stance geometry.

THE FIVE STAGES, AND THE TWO FRAMES
    Once per sweep, at slot 0.  The two frames meet in exactly three places
    and every one of them is a single named line:

      1 state       R, omega^b from the IMU; q, qd from the encoders
      2   ...       omega^w = R omega^b,  r_i^w = R (x_i^b - c^b),
                    h from the feet                        BODY -> WORLD
      3 controller  PD on (p_c,z, R) -> desired accelerations -> b_d
      4 allocation  A f^w = b_d by weighted least squares, then the cone
      5 torque      tau_i = -J_i^T R^T f_i^w + tau_grav,i   WORLD -> JOINT

    Everything between 2 and 5 is WORLD frame; everything outside is body or
    joint.  Keeping that boundary sharp is what makes the implementation
    checkable by inspection, and it is why every array that has a frame
    carries the frame in its name -- `x_b` and `r_w` never share an identifier.

    WORLD IS CHOSEN OVER BODY because the friction cone, gravity and the
    height reference are all world-frame objects; in the body frame the cone
    is a rotated pyramid whose face normals have to be rebuilt every sweep.
    The price is one similarity transform on the inertia per sweep, and on
    DOG6 not even that -- see `config.INERTIA_BODY`.

THE MODULES
    config       every number.  Gains, the cone, the weights, the FIXED CoM
                 and inertia, the trip thresholds.  Nothing else holds one.
    state        stages 1-2.  Measurement only: no gain appears in it.
    reference    the C2 quintic S-curve in h, and the conversion to a CoM
                 height command.  Owns the trajectory, not the law.
    controller   stage 3.  THE CONTROLLER: the only place a gain touches the
                 trunk.
    allocation   stage 4.  Pure transmission -- how the wrench is split across
                 four feet.  No trunk feedback in it at all.
    torque       stage 5, plus the tilt-aware leg gravity term the SRB model's
                 massless-leg assumption leaves out.
    selftest     gates the lot offline, including against the old law.

    That split is DOG5's (`feedback_estimator` / `dynamic_model` /
    `force_totorque` / the runner) with DOG6's names, and it is where it is
    for DOG5's reason: the boundary between "measured", "decided" and
    "transmitted" is the boundary a bad run has to be bisected along.

WHAT IS STILL OPEN
    R_BODY_IMU is an identity PLACEHOLDER, so nothing downstream of the
    mounting rotation has been checked on a real board.  Every gain in
    `config` is marked [UNTUNED].  And the rear-foot slide that ended the
    2026-09-15 runs is invisible to a law that reads joint encoders only:
    the cone bounds what the allocator ASKS for, and nothing here measures
    whether the foot then stayed put.
"""
from __future__ import annotations

__all__ = ["config", "state", "reference", "controller", "allocation", "torque"]
