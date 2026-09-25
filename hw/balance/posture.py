"""The crouch a lift STARTS FROM, as an object rather than a constant.

    posture.NOMINAL     `sim.stand.Q_CROUCH` -- trunk on the floor, h = 0
    posture.FOLD        the 2026-09-16 hand-captured fold, rear parallel, h = 60 mm
    posture.WIDE        NOMINAL's feet 20 mm further out, h = 40 mm

WHY THIS EXISTS.  `hw.stand` used to name `sim.stand.Q_CROUCH` in four places
and `sim.stand.FOOT_XY` in a fifth, and between them those five references
assert things the SRB law does not actually require: that the crouch puts the
trunk on the floor, and that the feet are under the nominal stance.  Neither is
true of a posture somebody folds the robot into by hand, and the point of this
file is to find out whether the law cares.

WHAT A POSE OWES THE SEQUENCE, AND IT IS EXACTLY THREE THINGS
    q            what the CROUCH and PARK phases drive the drivers to
    foot_xy      where `law.ik_reference` PINS the feet while the trunk rises.
                 In each leg's OWN HIP frame, like `sim.stand.FOOT_XY`.
    z_origin     the trunk origin's height at this pose with a LEVEL trunk,
                 which is what the status line and `h_cmd` report against.

    `BalanceLaw.arm` already latches h0 from the MEASUREMENT, so a crouch that
    does not start at zero needs nothing special there -- that was written for
    a different reason (a ramp started where the robot is not is a step into
    k_d,z) and it pays for itself here.

    THE ALLOCATOR NEVER NEEDED ANY OF THIS.  It builds the grasp map from the
    MEASURED foot positions, so an odd stance is already ordinary to it; what
    a non-nominal crouch breaks is only the TRACKING TRIP's reference, which
    is why `foot_xy` is a property of the pose and not a module constant.

A HAND-CAPTURED POSE IS NOT A POSE UNTIL IT HAS BEEN REGULARISED
    `FOLD` was posed by hand on the robot and read off the encoders.  Used
    raw it carries two defects that are the operator's, not the posture's:

      left/right skew   the two front feet sat 2.4 mm apart in x and the two
                        rear 6.6 mm apart.  A quadruped stance IS mirror
                        symmetric; anything else puts a roll moment into the
                        allocation that the attitude loop then spends its
                        authority cancelling, for no reason.
      unequal height    the four feet spanned 21.5 mm of trunk-frame z.  On a
                        flat floor that is not a foot-height difference at
                        all, it is the TRUNK PITCHED 3.43 deg -- and baking it
                        into the reference would mean commanding a level trunk
                        that the pose cannot hold, at every height.

    So the raw capture is recorded below as PROVENANCE and the flown pose is
    DERIVED from it here: x and |y| averaged within each axle, mirrored, and
    one z for all four.  Then the rear axle is REPLACED by the front one
    carried PARALLEL (2026-09-25): every leg folds the same way, knee behind
    its hip -- the front knees tucked under the abd motors, the rear ones out
    behind the rear hips -- and the rear thigh and shin are exactly the
    front's.  NOT symmetric front to back: the rear feet sit 61 mm nearer the
    trunk origin than the front ones.  The captured rear fold is kept only in
    `FOLD_CAPTURED_Q`.  `selftest` gates the symmetry, the equal height and
    the parallel legs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__ in (None, ""):        # allow `python hw/balance/posture.py`
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "hw.balance"

from sim import coordinates as C     # noqa: E402
from sim import kinematics as SK     # noqa: E402
from sim import params as P          # noqa: E402
from sim import stand as ST          # noqa: E402

from .. import kinematics as HK      # noqa: E402
from . import config as cfg          # noqa: E402

__all__ = ["CrouchPose", "NOMINAL", "FOLD", "WIDE", "POSTURES",
           "FOLD_CAPTURED_Q", "regularise", "WIDE_SPLAY", "WIDE_H"]

#: The raw capture, 2026-09-16, read off the encoders with the robot folded
#: by hand.  FLAT, in the joint frame, FL/FR/RL/RR.  KEPT SO THE DERIVATION
#: BELOW CAN BE CHECKED AND RE-RUN -- it is not what flies; `FOLD` is.
FOLD_CAPTURED_Q = np.array([
    -1.4828, +2.3931, -2.3796,      # FL   abd, pitch, knee
    +1.4884, -2.4275, +2.3814,      # FR
    -1.5530, -0.8561, -2.1453,      # RL
    +1.5053, +0.9356, +2.0983,      # RR
])


def regularise(q) -> np.ndarray:
    """(4, 3) hip-frame foot sites: mirror symmetric, one height, from `q`.

    The two defects a hand-posed capture has, removed in the one frame where
    removing them is meaningful.  Doing it in FOOT space and solving the IK
    afterwards is what makes the symmetry EXACT -- averaging joint angles
    instead would leave the feet slightly off, because the map from one to the
    other is not linear.
    """
    hip = SK.hip_to_foot_stance(C.unflat(np.asarray(q, dtype=float).ravel()))
    out = np.empty_like(hip)
    for lo, hi in ((0, 2), (2, 4)):                  # the front axle, the rear
        out[lo:hi, 0] = hip[lo:hi, 0].mean()
        out[lo:hi, 1] = np.abs(hip[lo:hi, 1]).mean() * np.array([+1.0, -1.0])
    out[:, 2] = hip[:, 2].mean()                     # ONE height, all four
    return out


@dataclass(frozen=True)
class CrouchPose:
    """A crouch the lift can start from.  Everything else is derived."""

    name: str
    q: np.ndarray                    # (4, 3) rad, joint frame
    foot_xy: np.ndarray              # (4, 2) m, each leg's OWN HIP frame
    z_origin: float                  # m, trunk ORIGIN above the floor, level
    note: str = ""
    #: An SRB model to use INSTEAD of deriving one.  `NOMINAL` passes
    #: `config.SRB` so that the path `hw.stand` has always flown stays
    #: bit-identical: `config.NOMINAL_POSE` is solved with `sim.stand`'s
    #: damped least squares and `CrouchPose` uses the closed-form inverse, and
    #: the two differ by about a nanometre.  Physically nothing; a needless
    #: diff in a number the robot flies.
    srb_pinned: object = None

    @property
    def h(self) -> float:
        """Floor to trunk BOTTOM -- the number a ruler reads."""
        return self.z_origin - cfg.TRUNK_BOTTOM_OFFSET

    @property
    def srb(self) -> cfg.SrbModel:
        """The pinned c^b and I^b for THIS posture, at the LIFT height.

        THE SINGLE-RIGID-BODY MODEL IS PINNED AT THE STANCE IT WILL HOLD, and
        `config`'s own numbers are pinned at the NOMINAL one -- feet at
        `sim.stand.FOOT_XY`, trunk at `LIFT_HEIGHT`.  Hold a different stance
        and both numbers move: folding the legs forward carries the whole
        robot's CoM 13.8 mm BACK of the trunk origin, against the nominal
        0.0, and 13.8 mm on a 57.7 N robot is 0.80 N*m of pitch moment the
        law never asked for.  With no integrator the attitude loop parks
        against exactly that -- which is the -2.1 deg of nose-up the first
        fold run held, to within the measurement.

        DERIVED AT `H_LIFT`, NOT AT THE CROUCH.  The stance that matters is
        the one the HOLD phase sits in; the crouch is passed through in three
        seconds.  Same convention as `config.NOMINAL_POSE`.

        Computed once and cached -- it is a constant, and the whole point is
        that it stays one.
        """
        if self.srb_pinned is not None:
            return self.srb_pinned
        cached = self.__dict__.get("_srb")
        if cached is None:
            p = np.zeros((C.N_LEGS, 3))
            p[:, :2] = self.foot_xy
            p[:, 2] = -(cfg.H_LIFT + cfg.TRUNK_BOTTOM_OFFSET - P.FOOT_RADIUS)
            cached = cfg.SrbModel.from_pose(self.name,
                                            HK.all_leg_ik(p, q_seed=self.q))
            object.__setattr__(self, "_srb", cached)
        return cached

    @property
    def reach_used(self) -> np.ndarray:
        """Fraction of LEG_REACH each leg is extended to."""
        p = np.zeros((C.N_LEGS, 3))
        p[:, :2] = self.foot_xy
        p[:, 2] = -(self.z_origin - P.FOOT_RADIUS)
        return np.linalg.norm(p, axis=1) / P.LEG_REACH

    @classmethod
    def from_hip_sites(cls, name: str, hip_xyz, q_seed=None,
                       note: str = "") -> "CrouchPose":
        """Build from (4, 3) hip-frame foot sites; the joints come from the IK."""
        hip_xyz = np.asarray(hip_xyz, dtype=float).reshape(C.N_LEGS, 3)
        q = HK.all_leg_ik(hip_xyz, q_seed=q_seed)
        return cls(name=name, q=q, foot_xy=hip_xyz[:, :2].copy(),
                   z_origin=float(P.FOOT_RADIUS
                                  - SK.all_foot_positions(q)[:, 2].mean()),
                   note=note)

    def describe(self) -> str:
        x = SK.all_foot_positions(self.q)
        rows = ["%s crouch -- h %.1f mm floor to trunk bottom (origin %.1f)"
                % (self.name, 1e3 * self.h, 1e3 * self.z_origin)]
        if self.note:
            rows.append("  %s" % self.note)
        rows.append("  leg      abd    pitch     knee  |  foot x,y,z (mm, trunk)")
        for i, label in enumerate(("FL", "FR", "RL", "RR")):
            rows.append("  %-3s %7.2f %8.2f %8.2f  | %8.1f %7.1f %7.1f"
                        % (label, *np.degrees(self.q[i]), *(1e3 * x[i])))
        rows.append("  foot z spread %.3f mm   reach used %s"
                    % (1e3 * (x[:, 2].max() - x[:, 2].min()),
                       np.array2string(self.reach_used, precision=3)))
        rows.append("  SRB at the lift height: %s" % self.srb.describe())
        return "\n".join(rows)


#: What `hw.stand` has always flown: `sim.stand`'s crouch, trunk on the floor.
NOMINAL = CrouchPose(
    name="nominal", q=ST.Q_CROUCH.copy(), foot_xy=ST.FOOT_XY.copy(),
    z_origin=float(ST.CROUCH_HEIGHT), srb_pinned=cfg.SRB,
    note="sim.stand.Q_CROUCH -- shins vertical, trunk resting on the floor")

def _fold_sites() -> np.ndarray:
    """The capture's FRONT axle, regularised, carried PARALLEL onto the rear.

    2026-09-25, the operator's decision: front and rear legs PARALLEL, every
    knee behind its hip -- the front fold, the knee tucked under the abd
    motor, and the rear leg the same shape: the same thigh, the same shin,
    hung from the rear pitch hinge.  The rear knee then sits out behind the
    rear hip, and the rear foot almost under the rear abd hinge, 2.3 mm
    ahead of it.

    THE REAR FOOT IS NOT THE FRONT'S HIP-FRAME SITE.  The pitch hinge is L1
    along the abduction axis from the abd hinge -- AHEAD of it on a front
    leg, BEHIND it on a rear one -- so the same thigh and shin land 2 L1
    (56.4 mm) further back in the hip frame.  y and z are the front's.

    (2026-09-17 to 09-25 the rear was the front MIRRORED fore-aft instead,
    knee toward the CoM.  The capture's own rear was parallel by hand only:
    its thigh 6 deg steeper than the front's, and regularised, its foot
    11 mm further forward.)
    """
    hip = regularise(FOLD_CAPTURED_Q)
    hip[2:] = hip[:2] - P.HIP_TO_PITCH[:2] + P.HIP_TO_PITCH[2:]
    return hip


def _fold_seed() -> np.ndarray:
    """The front legs' captured joints, carried parallel onto the rear.

    abd and knee the front's; pitch half a turn round, because the rear
    chain lies along -x at the zero, so the same thigh direction is the
    front's pitch -+ pi.  Picks the IK branch whose knee points the front's
    way, back.  The same foot has a second elbow, pitch ~ -179 deg, knee
    forward at the trunk's own height -- inside the trunk.
    """
    q = C.unflat(FOLD_CAPTURED_Q).copy()
    q[2:] = q[:2]
    q[2:, 1] -= np.pi * np.sign(q[:2, 1])
    return q


#: The hand-folded crouch, regularised, rear parallel to the front.  DERIVED
#: from `FOLD_CAPTURED_Q` at import, so the capture and the pose cannot drift
#: apart.
FOLD = CrouchPose.from_hip_sites(
    "fold", _fold_sites(), q_seed=_fold_seed(),
    note="hand-posed 2026-09-16, regularised: mirror symmetric, one foot "
         "height; rear parallel to the front, 2026-09-25.  The trunk does NOT "
         "start on the floor.")

#: THE WIDE CROUCH, 2026-09-17: the feet splayed out, for the trot.
#:
#: WHY IT IS NOT ON THE FLOOR.  NOMINAL's foot is 20 mm below the abduction
#: axis and its knee 85 mm above it, so splaying the foot out tips the leg
#: plane and swings the knee INBOARD, over the trunk.  Measured on the MJCF
#: meshes with the trunk down, left-to-right thigh/shin clearance:
#:
#:     splay    abd       L-R clearance
#:     0 mm     -90.0      74.5 mm      NOMINAL
#:     5 mm     -76.4      28.8
#:     10 mm      --       -3.7         the two knees collide
#:     20 mm    -47.7     -31.8
#:
#: Raising the crouch puts the foot further below the axis, so the same splay
#: costs less abduction.  At WIDE_H = 40 mm, +20 mm is abd -71.8 deg, L-R
#: 46.9 mm, leg-to-trunk 19.0 mm.  The price: the trunk is in the air, so this
#: is not a zero-torque park any more.  FOLD is not either.
WIDE_SPLAY = 0.020                   # m, |y| added to NOMINAL's feet
WIDE_H = 0.040                       # m, floor to trunk bottom at the crouch


def _wide_sites() -> np.ndarray:
    hip = SK.hip_to_foot_stance(NOMINAL.q)
    hip[:, 1] += np.sign(hip[:, 1]) * WIDE_SPLAY
    hip[:, 2] -= WIDE_H - NOMINAL.h
    return hip


WIDE = CrouchPose.from_hip_sites(
    "wide", _wide_sites(), q_seed=NOMINAL.q,
    note="NOMINAL's feet %.0f mm further out, trunk %.0f mm off the floor"
         % (1e3 * WIDE_SPLAY, 1e3 * WIDE_H))

POSTURES = {p.name: p for p in (NOMINAL, FOLD, WIDE)}


if __name__ == "__main__":
    for pose in (NOMINAL, FOLD, WIDE):
        print(pose.describe())
        print()
    raw = C.unflat(FOLD_CAPTURED_Q)
    print("what regularising the capture moved (rear: also refolded parallel "
          "to the front):")
    print("  joints, per leg (deg):  %s"
          % np.array2string(np.degrees(np.abs(FOLD.q - raw)).max(axis=1),
                            precision=2))
    print("  feet (mm):              %s"
          % np.array2string(1e3 * np.linalg.norm(
              SK.hip_to_foot_stance(FOLD.q) - SK.hip_to_foot_stance(raw),
              axis=1), precision=1))
