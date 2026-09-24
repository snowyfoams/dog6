"""Stages 1 and 2: what is measured, and the one place body becomes world.

    sensors -> BodyState

NO GAIN APPEARS IN THIS FILE.  It is measurement and frame arithmetic, and
the split matters: after a bad run the first question is whether the numbers
the law acted on were right, and that question has to be answerable without
reading the law.  `controller` decides; this only reports.

THE THREE CROSSINGS, AND TWO OF THEM ARE HERE
    Everything the IMU and the encoders give is in the BODY or JOINT frame.
    Everything the SRB law, the grasp map and the friction cone want is in the
    WORLD frame.  The boundary is crossed in exactly three places in the whole
    controller and two of them are the two lines below (the third is the
    torque map, in `torque`):

        omega^w = R omega^b
        r_i^w   = R (x_i^b - c^b)

    Every array that has a frame carries the frame in its NAME.  `x_b` is a
    foot in the trunk frame; `r_w` is the same foot, about the CoM, in the
    world.  They are never spelled the same way, because the failure this
    convention exists to prevent is a body-frame vector reaching a world-frame
    equation and being absorbed as a small calibration error.

WHICH WORLD, THOUGH -- THE HEADING IS PINNED, NOT TRACKED
    `read` builds R from the DETA10's full ZYX triple, so the world it hands
    back is the MAGNETOMETER'S: x points wherever the magnetometer calls
    north.  That frame is not a frame to control in.  The magnetometer sits
    next to twelve motors and `hw.imu` labels its yaw untrusted, so a world
    pinned to it turns whenever the motors load up.

    `rezero_yaw` pins it instead.  `sequence` calls it with the heading
    measured at the handover -- the sweep after the crouch, the first sweep
    torque is live -- and from there on world x is where the trunk was
    pointing and the reported yaw is the heading error, zero on the sweep it
    was taken.  `BodyState.yaw_offset` says which world a given measurement
    is in.  DOG5's `C_from_rp` convention; `rezero_yaw` has the arithmetic
    and the reason.

THE HEIGHT, AND WHY IT IS THREE NUMBERS
    The kinematics give the TRUNK ORIGIN; the operator reads the TRUNK BOTTOM;
    the PD needs the CoM.  One chain, and only the last step is not a constant:

        z_origin = -mean_i (R x_i^b)_z + FOOT_RADIUS
        h        = z_origin - TRUNK_BOTTOM_OFFSET          (a constant)
        p_c,z    = z_origin + (R c^b)_z

    ROTATING THE FOOT VECTOR IS NOT OPTIONAL and is the difference from
    `sim.stand.height_from_fk`, which averages the BODY-frame z and so reports
    the height of a trunk it assumes is level.  Under tilt those differ by
    (1 - cos) times the lever, which for a 237 mm foot at 10 deg is 3.6 mm --
    small, but a systematic error in exactly the state the attitude loop is
    fighting.

    BOTH EXPRESSIONS ASSUME FOUR PLANTED FEET.  Leg kinematics cannot see
    whether a foot is loaded; lift one and this averages in a leg that is
    measuring nothing.  That assumption is honest for a four-foot stand on
    flat ground and it is the one the 2026-09-15 runs showed the robot can
    violate.

    A TROT VIOLATES IT ON PURPOSE, TWICE A CYCLE, so `on_stance` re-takes both
    means over the feet the SCHEDULE has down.  Averaged over all four, a
    diagonal pair at its 40 mm apex reads as the trunk 20 mm LOWER, and the
    height loop answers with ~19 N it should not -- measured offline,
    2026-09-16.  DOG5's estimator averaged its planted set for the same
    reason.  The schedule is not a contact sensor, and this does not pretend
    to be one.

THE RATE TERM NOTHING ELSE HAS
        zdot_origin = -mean_i ( R J_i qdot_i  +  omega^w x R x_i^b )_z

    The second term is the contribution of TRUNK ROTATION to foot velocity.
    Without it the rate estimate is wrong whenever the trunk is turning --
    which is precisely when k_d,z is doing work, so the error arrives exactly
    where it is least welcome.  It is one cross product per leg.

STALENESS IS REPORTED, NEVER WAITED FOR
    The IMU streams on its own clock, not on the sweep.  This takes the most
    recent packet whatever its age and says what that age is.  A sweep that
    blocks on the IMU is a sweep that misses its CAN deadline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/state.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import config as cfg          # noqa: E402

__all__ = ["BodyState", "read", "on_stance", "rezero_yaw", "height_to_origin",
           "origin_to_height"]


def _wrap_pi(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(np.remainder(a + np.pi, 2.0 * np.pi) - np.pi)


def origin_to_height(z_origin: float) -> float:
    """Trunk-ORIGIN height -> the floor-to-trunk-BOTTOM height an operator reads."""
    return float(z_origin) - cfg.TRUNK_BOTTOM_OFFSET


def height_to_origin(h: float) -> float:
    """The inverse.  `sim.stand` is written in the origin frame; this package
    is written in `h`, and these two functions are the only conversion."""
    return float(h) + cfg.TRUNK_BOTTOM_OFFSET


@dataclass(frozen=True)
class BodyState:
    """One sweep's measurement, both frames, with the frame in every name."""

    # -- stage 1: as measured ------------------------------------------
    q: np.ndarray            # (12,) rad, joint frame
    qd: np.ndarray           # (12,) rad/s, joint frame -- the FILTERED
                             #       encoder difference, not the driver field
    R: np.ndarray            # (3, 3) world <- trunk
    omega_b: np.ndarray      # (3,) rad/s, TRUNK frame -- what the gyro gives
    roll: float              # rad, for the tilt trip and the log ONLY
    pitch: float             # rad
    yaw: float               # rad, magnetometer, UNTRUSTED
    imu_age_s: float
    imu_stale: bool          # past cfg.IMU_MAX_AGE_S: freeze, do not trip

    # -- stage 1: the legs, body frame ---------------------------------
    x_b: np.ndarray          # (4, 3) foot sites, TRUNK frame, about the origin
    jac: np.ndarray          # (4, 3, 3) foot Jacobians, TRUNK frame

    # -- stage 2: world ------------------------------------------------
    omega_w: np.ndarray      # (3,) rad/s  = R omega_b
    r_w: np.ndarray          # (4, 3) foot moment arms about the CoM, WORLD
    z_origin: float          # m, trunk origin above the floor
    h: float                 # m, FLOOR TO TRUNK BOTTOM -- the ruler's number
    p_cz: float              # m, CoM height, WORLD.  What the PD regulates
    zdot_origin: float       # m/s
    p_cz_dot: float          # m/s

    #: rad, the magnetometer heading the world frame's x axis is pinned to --
    #: `rezero_yaw`.  0.0 means "the world is the magnetometer's own", which
    #: is what every sweep before the handover reports.  It is carried on the
    #: state rather than held beside it so that a second rezero to the same
    #: heading is a no-op: which world frame this measurement is in is a
    #: property of the measurement.
    yaw_offset: float = 0.0

    @property
    def tilt_deg(self) -> float:
        """max(|roll|, |pitch|) in degrees -- the quantity the tilt trip reads."""
        return float(np.degrees(max(abs(self.roll), abs(self.pitch))))

    def status(self) -> str:
        """One line for the 2 Hz operator stream.  Only things a human can act on."""
        return ("h %6.1f mm  rp %+5.1f/%+5.1f deg  w %+5.1f/%+5.1f/%+5.1f dps"
                "  imu %3.0f ms%s"
                % (1e3 * self.h, np.degrees(self.roll), np.degrees(self.pitch),
                   *np.degrees(self.omega_b), 1e3 * self.imu_age_s,
                   "  STALE" if self.imu_stale else ""))


def read(q, qd, orientation, srb=None) -> BodyState:
    """Stages 1 and 2 from one sweep's sensors.

    `q`, `qd` are flat (12,) in JOINT coordinates -- `calibration.joint_state`
    with the encoder-differenced velocity, not the driver's speed field, which
    `calibration.joint_state` documents as a field that has been seen to lie.

    `orientation` is an `imu.TrunkOrientation`.  `TrunkOrientation.level()`
    stands in for it on the no-IMU path, and everything below then degenerates
    to the level-trunk arithmetic `sim.stand` already does -- which is what
    makes the no-IMU run a genuine ablation of the attitude loop rather than a
    different program.

    `srb` is the pinned `config.SrbModel` -- c^b and I^b for the stance being
    held.  None is `config.SRB`, the nominal one.  IT MUST BE THE SAME OBJECT
    `reference.com_command` IS GIVEN: the CoM offset only cancels out of the
    height error because both sides convert with the same constant.

    MEASURED: 0.12 ms for all four legs including the IK-free Jacobians.
    """
    srb = cfg.SRB if srb is None else srb
    q = np.asarray(q, dtype=float).reshape(C.N_JOINTS)
    qd = np.asarray(qd, dtype=float).reshape(C.N_JOINTS)
    q4 = C.unflat(q)
    qd4 = C.unflat(qd)
    R = np.asarray(orientation.R, dtype=float).reshape(3, 3)

    x_b = np.empty((C.N_LEGS, 3))
    jac = np.empty((C.N_LEGS, 3, 3))
    for i in range(C.N_LEGS):
        x_b[i], jac[i] = HK.leg_state(i, q4[i])

    omega_b = np.asarray(orientation.omega_b, dtype=float).reshape(3)
    omega_w = R @ omega_b

    # -- the two crossings ---------------------------------------------
    x_w = x_b @ R.T                       # (4, 3): row i is R @ x_b[i]
    r_w = (x_b - srb.com_body) @ R.T      # about the PINNED CoM; see config

    return BodyState(
        q=q, qd=qd, R=R, omega_b=omega_b,
        roll=float(orientation.roll), pitch=float(orientation.pitch),
        yaw=float(orientation.yaw),
        imu_age_s=float(orientation.age_s),
        imu_stale=bool(orientation.age_s > cfg.IMU_MAX_AGE_S),
        x_b=x_b, jac=jac,
        omega_w=omega_w, r_w=r_w,
        **_height(x_w, jac, qd4, R, omega_w, srb, None),
    )


def _height(x_w, jac, qd4, R, omega_w, srb, stance) -> dict:
    """The height chain of the module docstring, averaged over `stance`
    ((4,) bool; None is all four).  Returns the five BodyState fields."""
    feet = slice(None) if stance is None else np.asarray(stance, dtype=bool)
    z_origin = float(P.FOOT_RADIUS - x_w[feet, 2].mean())
    com_w = R @ srb.com_body

    # Foot velocity in the world has two parts: the leg folding under a fixed
    # trunk, and the trunk turning under fixed joints.  Dropping the second
    # is only correct at omega = 0.
    xdot_b = np.einsum("kij,kj->ki", jac[feet], qd4[feet])   # J_i qdot_i
    xdot_w = xdot_b @ R.T + np.cross(omega_w, x_w[feet])
    zdot_origin = float(-xdot_w[:, 2].mean())
    return dict(
        z_origin=z_origin,
        h=origin_to_height(z_origin),
        p_cz=z_origin + float(com_w[2]),
        zdot_origin=zdot_origin,
        # c^b is a constant in the BODY frame, so its world-frame rate is the
        # trunk's rotation alone.  Zero at a level trunk, which is why it is
        # easy to leave out and easy to leave out wrongly.
        p_cz_dot=zdot_origin + float(np.cross(omega_w, com_w)[2]),
    )


def on_stance(state: BodyState, stance, srb=None) -> BodyState:
    """`state` with the height and its rate re-taken over the `stance` feet.

    Everything else -- attitude, the feet, the moment arms -- is untouched;
    only the two means change.  No kinematics is redone.  With every foot in
    `stance` it returns fields equal to `read`'s.
    """
    srb = cfg.SRB if srb is None else srb
    stance = np.asarray(stance, dtype=bool).reshape(C.N_LEGS)
    if not stance.any():
        raise ValueError("no foot in stance: there is no height to measure")
    # Only the world z rows are needed, so only the third row of R is used:
    # (R x)_z = R[2] . x.  The same chain as `_height`, at a third the cost.
    R = state.R
    x_b = state.x_b[stance]
    z_origin = float(P.FOOT_RADIUS - (x_b @ R[2]).mean())
    xdot_b = np.einsum("kij,kj->ki", state.jac[stance],
                       C.unflat(state.qd)[stance])
    w = state.omega_w
    x_w = x_b @ R.T
    spin_z = w[0] * x_w[:, 1] - w[1] * x_w[:, 0]        # (omega x x_w)_z
    zdot_origin = float(-((xdot_b @ R[2]) + spin_z).mean())
    com_w = R @ srb.com_body
    return replace(state, z_origin=z_origin, h=origin_to_height(z_origin),
                   p_cz=z_origin + float(com_w[2]), zdot_origin=zdot_origin,
                   p_cz_dot=zdot_origin + float(w[0] * com_w[1]
                                                - w[1] * com_w[0]))


def rezero_yaw(state: BodyState, yaw_offset: float) -> BodyState:
    """`state` in the world frame whose x axis is the heading `yaw_offset`.

    DOG5'S "WORLD := BODY HERE", PORTED.  DOG5's estimator built its
    world-from-body rotation from roll and pitch alone (`C_from_rp`) and let
    the heading enter the wrench as a SCALAR beside it, so that -- its words
    -- "the frame does not start rotating with the heading".  DOG6 gets the
    full ZYX triple from the DETA10 in one matrix, so the same convention is
    reached from the other side: turn the world frame by the heading that was
    measured at the handover, and from that sweep on the reported yaw IS the
    heading error and the world's x axis IS where the trunk was pointing.

    THIS IS WHY IT MATTERS, AND IT IS NOT COSMETIC.  `controller` forms the
    attitude error as ``log(R_des R^T)``, and the log map does not separate.
    With a heading error of psi still inside R, a pure roll error r does not
    come out as ``(r, 0, -psi)``: the roll correction is scaled by
    ``(psi/2)/tan(psi/2)`` and a term ``r*psi/2`` is delivered about the world
    Y AXIS instead -- a phantom PITCH moment made out of a magnetometer
    reading.  At psi = 90 deg that is 21 % of the roll correction lost and
    39 % of it misdirected.  Zeroing `config.KP_YAW` does not help: it zeroes
    the yaw ROW of the gain, not the yaw inside the log.

    IT IS EXACT, AND IT TOUCHES FOUR FIELDS.  Rz commutes with everything the
    height chain reads -- ``(Rz v)_z = v_z``, and ``Rz(a x b) = Rz a x Rz b``
    -- so z_origin, h, p_cz and both rates come through untouched and are not
    recomputed.  The attitude pair is untouched for the same reason: a
    rotation about world z changes no roll and no pitch.  What moves is R,
    the two world-frame vectors built from it, and the yaw scalar.

    A SECOND CALL AT THE SAME HEADING IS A NO-OP, because the state carries
    the frame it is already in.  That is what lets `sequence` rezero on both
    the `advance` and the `update` path without subtracting the offset twice.
    """
    psi = float(yaw_offset) - float(state.yaw_offset)
    if psi == 0.0:
        return state
    Rz = C.rot_z(-psi)
    return replace(state,
                   R=Rz @ state.R,
                   yaw=_wrap_pi(state.yaw - psi),
                   omega_w=Rz @ state.omega_w,
                   r_w=state.r_w @ Rz.T,
                   yaw_offset=float(yaw_offset))
