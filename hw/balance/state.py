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

from dataclasses import dataclass

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

__all__ = ["BodyState", "read", "height_to_origin", "origin_to_height"]


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


def read(q, qd, orientation) -> BodyState:
    """Stages 1 and 2 from one sweep's sensors.

    `q`, `qd` are flat (12,) in JOINT coordinates -- `calibration.joint_state`
    with the encoder-differenced velocity, not the driver's speed field, which
    `calibration.joint_state` documents as a field that has been seen to lie.

    `orientation` is an `imu.TrunkOrientation`.  `TrunkOrientation.level()`
    stands in for it on the no-IMU path, and everything below then degenerates
    to the level-trunk arithmetic `sim.stand` already does -- which is what
    makes the no-IMU run a genuine ablation of the attitude loop rather than a
    different program.

    MEASURED: 0.12 ms for all four legs including the IK-free Jacobians.
    """
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
    r_w = (x_b - cfg.COM_BODY) @ R.T      # about the PINNED CoM; see config

    z_origin = float(P.FOOT_RADIUS - x_w[:, 2].mean())
    com_w = R @ cfg.COM_BODY

    # Foot velocity in the world has two parts: the leg folding under a fixed
    # trunk, and the trunk turning under fixed joints.  Dropping the second
    # is only correct at omega = 0.
    xdot_b = np.einsum("kij,kj->ki", jac, qd4)          # (4, 3) = J_i qdot_i
    xdot_w = xdot_b @ R.T + np.cross(omega_w, x_w)
    zdot_origin = float(-xdot_w[:, 2].mean())

    return BodyState(
        q=q, qd=qd, R=R, omega_b=omega_b,
        roll=float(orientation.roll), pitch=float(orientation.pitch),
        yaw=float(orientation.yaw),
        imu_age_s=float(orientation.age_s),
        imu_stale=bool(orientation.age_s > cfg.IMU_MAX_AGE_S),
        x_b=x_b, jac=jac,
        omega_w=omega_w, r_w=r_w,
        z_origin=z_origin,
        h=origin_to_height(z_origin),
        p_cz=z_origin + float(com_w[2]),
        zdot_origin=zdot_origin,
        # c^b is a constant in the BODY frame, so its world-frame rate is the
        # trunk's rotation alone.  Zero at a level trunk, which is why it is
        # easy to leave out and easy to leave out wrongly.
        p_cz_dot=zdot_origin + float(np.cross(omega_w, com_w)[2]),
    )
