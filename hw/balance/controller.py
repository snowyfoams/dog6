"""Stage 3: the balance controller.  THE ONLY PLACE A GAIN TOUCHES THE TRUNK.

    (BodyState, ComCommand, R_des) -> b_d, the 6-vector the allocator owes

Everything upstream (`state`) is measurement; everything downstream
(`allocation`, `torque`) is bookkeeping -- how to split a wrench across four
feet and how to turn a foot force into three motor torques.  Neither has a
gain in it.  So this file is the controller, and it is about twenty lines.

THE LAW -- MIT CHEETAH 3'S BALANCE CONTROLLER, eqs (2)-(3)
    The PD does NOT produce a wrench.  It produces a pair of desired
    ACCELERATIONS, and the robot's own mass and inertia turn those into the
    wrench:

        [ pddot_c,d ]   [ Kp,p (p_c,d - p_c)   + Kd,p (pdot_c,d - pdot_c) ]
        [ wdot_b,d  ] = [ Kp,w log(R_d R^T)^v + Kd,w (w_b,d - w)          ]

        b_d = [ m (pddot_c,d + g) ;  I_G wdot_b,d ]
        I_G = R I^b R^T

    WHY IT IS WORTH WRITING IT THIS WAY.  Gravity enters ONCE, as m*g in the
    force row, and the allocator distributes it by the geometry the robot
    actually has -- not as a fixed mg/4 per foot, which is what the per-leg law
    assumed and which is wrong the moment the trunk is not level or the feet
    are not square.  And dividing the moment by I_G makes Kp,w and Kd,w the
    ROLL AND PITCH GAINS DIRECTLY, in 1/s^2 and 1/s, independent of the height
    gains and of the stance geometry: one number is one closed-loop bandwidth
    on every axis.  Under the per-leg law there was no such number to turn.

THE ONE THING NOT TO GET WRONG: THE ERROR AND THE RATE MUST SHARE A FRAME
    The form above is the WORLD-frame pair -- log(R_d R^T)^v with
    omega^w = R omega^b.  The BODY-frame pair is log(R_d^T R)^v with the
    measured omega^b.  They are related by e_R^w = R e_R^b and they COINCIDE
    AT R_d = I, so a level bench test cannot tell them apart.  This file is
    world throughout, because that is the frame the grasp map and the cone are
    built in and a moment handed to the allocator in any other frame is
    applied about the wrong axes.

    Concretely: were the moment computed in the body frame and consumed in the
    world, a robot pitched 10 deg would have part of its roll correction
    delivered as a yaw moment.

    Note the sign.  log(R_d R^T)^v is the rotation that carries the MEASURED
    attitude onto the DESIRED one, so it is the correction, and it enters with
    a PLUS.  log(R R_d^T)^v is its negative and would drive the robot away.

NO INTEGRATOR
    Against a constant disturbance moment the loop parks at Kp,w^-1 M_d,
    which is not zero.  Add an integrator on wdot_b,d only if a residual tilt
    shows up in a log -- and if one does, check `config.COM_BODY` first: a
    pinned CoM puts exactly this kind of constant bias in the moment row.

WHAT DOG6 COMMANDS
    wdot_b,d = 0, and R_d = Rz(psi_0) with psi_0 LATCHED when torque arms.
    Roll and pitch desired are zero after a flat calibration.  Of the three
    CoM axes only z is commanded; x and y are switched off at the gain, for
    the reason `config.KP_XY` gives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/controller.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402

from . import config as cfg          # noqa: E402

__all__ = ["BalanceGains", "Wrench", "balance_wrench", "level_attitude"]


@dataclass
class BalanceGains:
    """The six acceleration gains, as three diagonal pairs.

    Mutable on purpose: `hw.stand` overrides individual entries from the
    command line, and an ablation (attitude gains zeroed, the other half of
    the A/B) is one assignment rather than a second code path.
    """

    kp_pos: np.ndarray = field(
        default_factory=lambda: np.array([cfg.KP_XY, cfg.KP_XY, cfg.KP_Z]))
    kd_pos: np.ndarray = field(
        default_factory=lambda: np.array([cfg.KD_XY, cfg.KD_XY, cfg.KD_Z]))
    kp_att: np.ndarray = field(
        default_factory=lambda: np.array([cfg.KP_ATT, cfg.KP_ATT, cfg.KP_YAW]))
    kd_att: np.ndarray = field(
        default_factory=lambda: np.array([cfg.KD_ATT, cfg.KD_ATT, cfg.KD_YAW]))

    def ablate_attitude(self) -> "BalanceGains":
        """Zero both attitude gains: the ablation half of the A/B.

        The trunk will NOT push back.  What is left is the height loop plus
        gravity, split across four feet by the actual geometry -- which is the
        per-leg law's mg/4 done properly, and is the honest intermediate
        between the old law and this one.
        """
        self.kp_att = np.zeros(3)
        self.kd_att = np.zeros(3)
        return self

    @property
    def attitude_off(self) -> bool:
        return not (self.kp_att.any() or self.kd_att.any())

    def __str__(self) -> str:
        return ("gains kp_z %.0f kd_z %.0f  kp_att %.0f kd_att %.0f  "
                "kp_yaw %.0f kd_yaw %.0f"
                % (self.kp_pos[2], self.kd_pos[2], self.kp_att[0],
                   self.kd_att[0], self.kp_att[2], self.kd_att[2]))


class Wrench(NamedTuple):
    """`b_d` and the four intermediates a log needs to explain it."""

    b_d: np.ndarray          # (6,) [force; moment], WORLD, about the CoM
    acc_lin: np.ndarray      # (3,) m/s^2   the PD's own output, before m
    acc_ang: np.ndarray      # (3,) rad/s^2 the PD's own output, before I_G
    e_R: np.ndarray          # (3,) rad, the SO(3) log-map attitude error
    inertia_w: np.ndarray    # (3, 3) I_G = R I^b R^T

    @property
    def force(self) -> np.ndarray:
        return self.b_d[:3]

    @property
    def moment(self) -> np.ndarray:
        return self.b_d[3:]


def level_attitude(yaw: float) -> np.ndarray:
    """R_des for a stand: level, facing `yaw`.  Rz(psi_0) and nothing else.

    LATCH psi_0 WHEN TORQUE ARMS, do not track it.  Absolute yaw is the
    magnetometer's and `hw.imu` labels it untrusted -- but with
    `config.KP_YAW` at zero the yaw column of the error is multiplied by zero,
    so what psi_0 actually does here is keep R_d a proper rotation near R and
    keep the log map away from its large-angle branch.  Set KP_YAW non-zero
    and this number starts to matter; that is the moment to check the
    magnetometer under power first.
    """
    return C.rot_z(float(yaw))


def balance_wrench(state, com_cmd, R_des, gains: BalanceGains, *,
                   omega_des=None, hold_attitude: bool = False) -> Wrench:
    """The PD of the module docstring.  A pure function -- no state, no clock.

    `state` is a `state.BodyState`, `com_cmd` a `reference.ComCommand`.

    `hold_attitude` is the IMU-staleness response: it zeroes the attitude
    accelerations and leaves the height loop and gravity running.  FREEZE, NOT
    TRIP -- a trip during the lift is a drop, and holding is the behaviour a
    robot with no attitude sensor has anyway (it is what the per-leg law did
    for its whole life).  Freezing the ERROR instead, and keeping the moment
    the last good packet asked for, would push on a world model that has
    stopped updating.
    """
    R = state.R

    # -- translation ---------------------------------------------------
    # Only z is commanded; the x and y entries of the gains are zero, so the
    # zeros paired with them here are never read.  Written out in full anyway,
    # because a 3-vector whose first two entries are structurally absent is
    # how a later feedforward gets quietly dropped into the wrong slot.
    p_error = np.array([0.0, 0.0, com_cmd.p_cz - state.p_cz])
    v_error = np.array([0.0, 0.0, com_cmd.p_cz_dot - state.p_cz_dot])
    acc_lin = gains.kp_pos * p_error + gains.kd_pos * v_error

    # -- attitude ------------------------------------------------------
    # The exponential-map error on SO(3), never a difference of Euler angles.
    e_R = C.log_so3(np.asarray(R_des, dtype=float).reshape(3, 3) @ R.T)
    omega_des = np.zeros(3) if omega_des is None else np.asarray(
        omega_des, dtype=float).reshape(3)
    if hold_attitude:
        acc_ang = np.zeros(3)
    else:
        acc_ang = (gains.kp_att * e_R
                   + gains.kd_att * (omega_des - state.omega_w))

    # -- the desired dynamics, eq (3) ----------------------------------
    # I_G is the composite inertia carried into the world by a similarity
    # transform.  I^b is PINNED (see config), so this is the only pose
    # dependence left in it -- and at a level trunk it is the identity.
    inertia_w = R @ cfg.INERTIA_BODY @ R.T
    force = cfg.MASS * (acc_lin - cfg.GRAVITY_W)
    moment = inertia_w @ acc_ang

    return Wrench(b_d=np.concatenate([force, moment]),
                  acc_lin=acc_lin, acc_ang=acc_ang, e_R=e_R,
                  inertia_w=inertia_w)


def describe() -> str:
    gains = BalanceGains()
    return "\n".join([
        "DOG6 balance controller (Cheetah 3 eqs 2-3), WORLD frame throughout",
        "  %s" % gains,
        "  the PD returns ACCELERATIONS; m and I_G make the wrench",
        "  attitude error  log(R_d R^T)^vee, paired with omega^w = R omega^b",
        "  gravity enters ONCE, as m*g in the force row -- not mg/4 per foot",
        "  no integrator: against a constant moment it parks at Kp^-1 M",
    ])


if __name__ == "__main__":
    print(describe())
