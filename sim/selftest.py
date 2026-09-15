"""Gate `coordinates` + `params` + `kinematics` + `hw.kinematics` against
model/dog6.xml.

    python -m sim.selftest

WHY THIS FILE EXISTS
    The three modules beside it are a HAND-MAINTAINED COPY of numbers that
    really live in model/dog6.xml, which is itself generated from the Fusion
    360 CAD by a pipeline that stays in D:\\mujoco\\dog6_description.  A copy
    that nothing compares is a copy that will drift, and the drift is silent:
    a wrong link length still produces a plausible-looking foot position, and
    a wrong sign still produces a robot that stands up.

    So every number in `params` is read back out of the MJCF here, and every
    map in `kinematics` is checked against MuJoCo's own answer -- not once at
    a nominal pose, but over random poses across the whole joint circle with
    the trunk tilted, because one pose cannot distinguish a correct chain from
    several wrong ones.

    This is the file to run after re-copying a rebuilt model into model/.

WHAT A FAILURE MEANS
    Almost always: the CAD moved, model/dog6.xml was rebuilt and re-copied,
    and `params` was not updated to match.  The MJCF is the authority -- it is
    the generated artifact -- so fix the Python, not the XML.

    The exception is section 5.  `hw.kinematics` is a SECOND, independently
    derived forward kinematics -- the closed form the robot itself will run,
    which cannot call MuJoCo.  A failure there is the two derivations
    disagreeing, and the MJCF says which one is wrong.

    THIS IS THE ONE PLACE `sim` REACHES INTO `hw`, AND IT IS DELIBERATE.  The
    dependency inside `sim` still runs one way; `selftest` sits above both
    stacks because being the gate is its whole job, and a cross-check that
    lived in `hw` would be a module marking its own homework.
"""
from __future__ import annotations

import sys

import numpy as np

if __package__ in (None, ""):        # allow `python sim/selftest.py` too
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sim"

from . import coordinates as C       # noqa: E402
from . import kinematics as K        # noqa: E402
from . import params as P            # noqa: E402

from hw import kinematics as HW      # noqa: E402  see "WHAT A FAILURE MEANS"

try:
    import mujoco
except ImportError:                  # pragma: no cover
    sys.exit("mujoco is not installed in this interpreter.\n"
             "  D:\\mujoco\\.venv\\Scripts\\python.exe -m sim.selftest\n"
             "or  pip install -r requirements.txt")


N_RANDOM = 500
_FAILURES: list[str] = []
_PASSES = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _PASSES
    if ok:
        _PASSES += 1
        print("  ok    %-58s %s" % (label, detail))
    else:
        _FAILURES.append(label)
        print("  FAIL  %-58s %s" % (label, detail))


def close(label: str, a, b, tol: float, unit: str = "") -> None:
    """Check a max-abs difference and report it, so a passing run still shows
    the margin -- a gate that passes at 0.9 of its tolerance is worth seeing."""
    worst = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    check(label, worst <= tol, "worst %.3g%s (tol %.3g)" % (worst, unit, tol))


def _named(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


# ===========================================================================
def main() -> int:
    model = mujoco.MjModel.from_xml_path(P.XML_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(6)

    print("DOG6 sim self-test")
    print("  model  %s" % P.XML_PATH)
    print("  mujoco %s\n" % mujoco.__version__)

    # -- 1. the names and the state layout ---------------------------------
    print("names and layout")
    jnt = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
           for i in range(model.njnt)]
    check("joint names match coordinates.JOINT_NAMES",
          tuple(jnt[1:]) == C.JOINT_NAMES, "%d hinges after the freejoint" % (len(jnt) - 1))
    check("the base is a freejoint named root",
          jnt[0] == "root" and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)
    check("qpos / qvel are %d / %d" % (C.NQ, C.NV),
          (model.nq, model.nv) == (C.NQ, C.NV), "got %d / %d" % (model.nq, model.nv))
    check("12 actuators, one per hinge", model.nu == C.N_JOINTS, "nu = %d" % model.nu)
    check("foot sites present", all(
        _named(model, mujoco.mjtObj.mjOBJ_SITE, n) >= 0 for n in C.FOOT_SITE_NAMES))
    check("imu site and its three sensors present",
          _named(model, mujoco.mjtObj.mjOBJ_SITE, C.IMU_SITE) >= 0 and all(
              _named(model, mujoco.mjtObj.mjOBJ_SENSOR, n) >= 0 for n in C.IMU_SENSORS))

    # -- 2. params vs the XML ----------------------------------------------
    print("\nparams against the MJCF")
    close("total mass", model.body_subtreemass[1], P.MASS, 1e-9, " kg")

    xml_geom, xml_mass, xml_com, xml_inertia = [], [], [], []
    for leg in C.LEGS:
        names = (f"hip_{leg}", f"thigh_{leg}", f"shin_{leg}")
        ids = [_named(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        site = _named(model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}")
        xml_geom.append([model.body_pos[ids[0]], model.body_pos[ids[1]],
                         model.body_pos[ids[2]], model.site_pos[site]])
        xml_mass.append([model.body_mass[i] for i in ids])
        xml_com.append([model.body_ipos[i] for i in ids])
        # body_inertia is the diagonal in the body's PRINCIPAL frame, with
        # body_iquat the rotation to it.  Rebuild the full tensor to compare.
        tensors = []
        for i in ids:
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.body_iquat[i])
            rot = rot.reshape(3, 3)
            tensors.append(rot @ np.diag(model.body_inertia[i]) @ rot.T)
        xml_inertia.append(tensors)

    ours_geom = np.array([[P.LEG_GEOMETRY[leg].hip, P.LEG_GEOMETRY[leg].hip_to_pitch,
                           P.LEG_GEOMETRY[leg].pitch_to_knee,
                           P.LEG_GEOMETRY[leg].knee_to_foot] for leg in C.LEGS])
    # 5e-8 IS THE MJCF'S OWN TEXT PRECISION, NOT SLOP.  The XML writes body
    # positions to seven decimal places -- `pos="0.1563145 0.06 0"` -- so half
    # of the last written digit is 5e-8, and `params` legitimately carries more
    # digits than the file it was generated alongside (2.9e-9 worst, on the
    # hip).  The next check is the one that makes this harmless.
    close("LEG_GEOMETRY == body pos + foot site pos", ours_geom, xml_geom, 5e-8, " m")

    # THE ROUNDING CANCELS, AND THAT IS WHY IT COSTS NOTHING.  `hip` rounds up
    # by 2.9 nm and `hip_to_pitch` rounds down by the same 2.9 nm, so the point
    # the chain actually propagates -- the pitch hinge, at hip + hip_to_pitch --
    # is bit-identical to the MJCF's.  It is also why the FK gate below agrees
    # to 4e-16 rather than to the 2.9e-9 the line above reports.
    close("...and the rounding cancels: the pitch hinge is exact",
          ours_geom[:, 0] + ours_geom[:, 1],
          np.asarray(xml_geom)[:, 0] + np.asarray(xml_geom)[:, 1], 1e-15, " m")

    close("LINK_MASS == body mass", P.LINK_MASS, xml_mass, 1e-12, " kg")
    close("LINK_COM == body ipos", P.LINK_COM, xml_com, 5e-10, " m")
    # 1e-10, because this comparison round-trips through MuJoCo's storage
    # rather than reading the file: the MJCF's `fullinertia` is diagonalised at
    # compile time into principal moments plus `body_iquat`, and rebuilding the
    # tensor from those is ill-conditioned here -- the thigh's iyy and izz
    # differ by 5 %, which leaves the eigenvector directions poorly determined.
    # The 1.2e-11 seen is that decomposition, not a wrong number.
    close("LINK_INERTIA == body inertia tensors", P.LINK_INERTIA, xml_inertia,
          1e-10, " kg m^2")

    trunk = _named(model, mujoco.mjtObj.mjOBJ_BODY, C.TRUNK_BODY)
    close("TRUNK_MASS", model.body_mass[trunk], P.TRUNK_MASS, 1e-12, " kg")
    close("TRUNK_COM", model.body_ipos[trunk], P.TRUNK_COM, 5e-10, " m")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[trunk])
    rot = rot.reshape(3, 3)
    close("TRUNK_INERTIA", rot @ np.diag(model.body_inertia[trunk]) @ rot.T,
          P.TRUNK_INERTIA, 1e-12, " kg m^2")

    close("ARMATURE == the XML joint default", model.dof_armature[6:], P.ARMATURE, 0.0)
    close("JOINT_DAMPING == the XML joint default", model.dof_damping[6:],
          P.JOINT_DAMPING, 0.0)
    close("TIMESTEP", model.opt.timestep, P.TIMESTEP, 0.0, " s")

    # The software limits are NOT the model's, and that is the point.
    check("the MJCF leaves every hinge UNLIMITED (so JOINT_LIMITS is policy)",
          not bool(np.any(model.jnt_limited)),
          "params.JOINT_LIMITS is enforced by callers, not by MuJoCo")

    foot_geoms = [_named(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}")
                  for leg in C.LEGS]
    close("FOOT_RADIUS == the foot sphere geoms",
          [model.geom_size[g][0] for g in foot_geoms], P.FOOT_RADIUS, 1e-12, " m")
    box = _named(model, mujoco.mjtObj.mjOBJ_GEOM, "trunk_box")
    close("TRUNK_BOX_HALF", model.geom_size[box], P.TRUNK_BOX_HALF, 1e-12, " m")
    check("robot collision geoms are contype/conaffinity %d/%d (never self-collide)"
          % (P.ROBOT_CONTYPE, P.ROBOT_CONAFFINITY),
          all(model.geom_contype[g] == P.ROBOT_CONTYPE
              and model.geom_conaffinity[g] == P.ROBOT_CONAFFINITY
              for g in foot_geoms + [box]))

    imu = _named(model, mujoco.mjtObj.mjOBJ_SITE, C.IMU_SITE)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.site_quat[imu])
    close("R_BODY_IMU == the imu site orientation in the MJCF",
          rot.reshape(3, 3), C.R_BODY_IMU, 1e-12)

    # -- 3. the keyframes --------------------------------------------------
    print("\nposes")
    key_stand = model.key_qpos[C.KEYFRAMES["stand"]]
    close("Q_STAND == the `stand` keyframe (exact pi vs the XML's 8 dp)",
          C.flat(C.Q_STAND), key_stand[C.QPOS_JOINTS], 5e-9, " rad")
    close("STAND_HEIGHT == the `stand` keyframe height",
          key_stand[2], P.STAND_HEIGHT, 5e-7, " m")
    key_home = model.key_qpos[C.KEYFRAMES["home"]]
    close("Q_ZERO == the `home` keyframe", C.flat(C.Q_ZERO),
          key_home[C.QPOS_JOINTS], 0.0, " rad")
    close("HOME_HEIGHT == the `home` keyframe height", key_home[2], P.HOME_HEIGHT,
          1e-12, " m")

    # The claim Q_STAND is worth making: driven to exact pi/2 and pi/3, every
    # foot ball sits exactly on the floor.
    data.qpos[:] = C.qpos_from_q(C.Q_STAND, root_pos=(0, 0, P.STAND_HEIGHT))
    mujoco.mj_forward(model, data)
    feet = np.array([data.site_xpos[_named(model, mujoco.mjtObj.mjOBJ_SITE, n)]
                     for n in C.FOOT_SITE_NAMES])
    # 1e-9: STAND_HEIGHT is written to nine decimals, so it carries ~5e-10 of
    # its own rounding.  The claim being gated is "exactly on the floor" to
    # within the precision the number is stated at -- a sub-nanometre residual,
    # against a 15 mm ball.
    close("at Q_STAND every foot ball rests exactly on the floor",
          feet[:, 2], P.FOOT_RADIUS, 1e-9, " m")
    check("the stance is exactly square",
          np.allclose(np.abs(feet[:, 0]), np.abs(feet[0, 0]), atol=1e-12)
          and np.allclose(np.abs(feet[:, 1]), np.abs(feet[0, 1]), atol=1e-12),
          "%.3f x %.3f m" % (2 * abs(feet[0, 0]), 2 * abs(feet[0, 1])))

    # -- 4. FK, over the whole joint circle, with the trunk tilted ---------
    print("\nkinematics against MuJoCo")
    worst_fk = worst_jac = 0.0
    for _ in range(N_RANDOM):
        q = rng.uniform(-np.pi, np.pi, size=(C.N_LEGS, 3))
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        data.qpos[:] = C.qpos_from_q(q, root_pos=rng.uniform(-1, 1, 3), root_quat=quat)
        mujoco.mj_forward(model, data)

        rot_wb = data.xmat[trunk].reshape(3, 3)      # R_{world <- trunk}
        for i, name in enumerate(C.FOOT_SITE_NAMES):
            site = _named(model, mujoco.mjtObj.mjOBJ_SITE, name)
            mine = K.foot_position(i, q[i])
            theirs = rot_wb.T @ (data.site_xpos[site] - data.xpos[trunk])
            worst_fk = max(worst_fk, float(np.max(np.abs(mine - theirs))))

    close("foot_position == mj site position, %d random poses, tilted trunk"
          % N_RANDOM, worst_fk, 0.0, 1e-12, " m")

    # The Jacobian comparison keeps the trunk at identity so world axes and
    # trunk axes coincide and mj_jacSite's columns can be read directly.
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(N_RANDOM):
        q = rng.uniform(-np.pi, np.pi, size=(C.N_LEGS, 3))
        data.qpos[:] = C.qpos_from_q(q)
        mujoco.mj_forward(model, data)
        for i, name in enumerate(C.FOOT_SITE_NAMES):
            site = _named(model, mujoco.mjtObj.mjOBJ_SITE, name)
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            cols = slice(6 + 3 * i, 9 + 3 * i)
            worst_jac = max(worst_jac, float(
                np.max(np.abs(K.foot_jacobian(i, q[i]) - jacp[:, cols]))))
    close("foot_jacobian == mj_jacSite, %d random poses" % N_RANDOM,
          worst_jac, 0.0, 1e-12)

    # -- 5. hw.kinematics: the closed form the robot will run --------------
    # `sim.kinematics` walks the chain the way the MJCF is built; `hw` is a
    # hand-derived closed form that touches none of the same intermediates.
    # Both are compared to MuJoCo here rather than only to each other, so a
    # shared misreading of the geometry table cannot pass by cancelling.
    print("\nhw.kinematics (closed form) against MuJoCo and sim.kinematics")
    worst_p = worst_j = worst_ph = worst_rot = worst_ang = worst_det = 0.0
    worst_fac = 0.0
    for _ in range(N_RANDOM):
        q = rng.uniform(-np.pi, np.pi, size=(C.N_LEGS, 3))
        data.qpos[:] = C.qpos_from_q(q)          # trunk at identity, as above
        mujoco.mj_forward(model, data)
        for i, leg in enumerate(C.LEGS):
            site = _named(model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{leg}")
            shin = _named(model, mujoco.mjtObj.mjOBJ_BODY, f"shin_{leg}")
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            cols = slice(6 + 3 * i, 9 + 3 * i)

            p, jac = HW.leg_state(i, q[i])
            worst_p = max(worst_p, float(np.max(np.abs(
                p - (data.site_xpos[site] - data.xpos[trunk])))))
            worst_j = max(worst_j, float(np.max(np.abs(jac - jacp[:, cols]))))
            worst_ph = max(worst_ph, float(np.max(np.abs(
                HW.foot_position_hip(i, q[i]) - K.foot_position_hip(i, q[i])))))
            # The shin's orientation is Rx(q1) Rz(q2+q3) -- a claim about the
            # chain that a foot POINT cannot distinguish, so it is worth its
            # own line.
            worst_rot = max(worst_rot, float(np.max(np.abs(
                HW.leg(i).fk_orientation(q[i]) - data.xmat[shin].reshape(3, 3)))))
            worst_ang = max(worst_ang, float(np.max(np.abs(
                HW.leg(i).jacobian_angular(q[i]) - jacr[:, cols]))))
            # det in the factored form, against a plain LU determinant.
            worst_det = max(worst_det, abs(
                HW.leg(i).det_jacobian(q[i]) - float(np.linalg.det(jac))))
            # ...and against the collapsed form the rear legs are the test of:
            # y2 = y3 = 0 on DOG6, so (v2 x v3)_z is exactly x2 x3 sin(q3),
            # and x2 x3 > 0 on the rear legs too because BOTH are negated.
            core = HW.leg(i)._core(q[i])
            x2, x3 = HW.leg(i).d2[0], HW.leg(i).d3[0]
            worst_fac = max(worst_fac, abs(
                HW.leg(i).det_jacobian(q[i]) - core.w[1] * x2 * x3 * np.sin(q[i][2])))

    close("hw foot_position == mj site position, %d poses x 4 legs" % N_RANDOM,
          worst_p, 0.0, 1e-12, " m")
    close("hw foot_jacobian == mj_jacSite", worst_j, 0.0, 1e-12)
    close("hw foot_position_hip == sim.kinematics'", worst_ph, 0.0, 1e-12, " m")
    close("hw fk_orientation == mj shin body orientation", worst_rot, 0.0, 1e-12)
    close("hw jacobian_angular == mj_jacSite's rotational columns",
          worst_ang, 0.0, 1e-12)
    check("...and it is rank 2, because the pitch and knee axes are parallel",
          all(np.linalg.matrix_rank(HW.leg(i).jacobian_angular(
              rng.uniform(-np.pi, np.pi, 3))) == 2 for i in range(C.N_LEGS)))
    close("hw det_jacobian == det(J), in its factored form",
          worst_det, 0.0, 1e-15)
    close("...and collapses to Q * x2 * x3 * sin(q3) on ALL FOUR legs",
          worst_fac, 0.0, 1e-15)

    # The rear legs are the whole reason the closed form needs no per-leg
    # branch: they are the front legs with the x components of d negated, z
    # untouched, and no sign anywhere on an axis.
    front, rear = HW.LEGS["FL"], HW.LEGS["RL"]
    check("the rear legs are the front legs with d's x negated, z untouched",
          all(np.allclose(getattr(rear, n), np.array([-1.0, 1.0, 1.0])
                          * getattr(front, n), atol=1e-15)
              for n in ("d1", "d2", "d3"))
          and rear.d3[2] == front.d3[2] == P.LEG_GEOMETRY["FL"].knee_to_foot[2],
          "d3_z = %+.4f m on both" % front.d3[2])
    # The inverse.  `sim.leg_ik` iterates to a tolerance; this one is exact,
    # so it is held to machine precision rather than to a solver's contract --
    # and over the WHOLE joint torus, not just the working envelope, because a
    # closed form has no envelope to stay inside.
    worst_ik = worst_branch = worst_clear = worst_stand = 0.0
    unreachable = 0
    for _ in range(N_RANDOM):
        for i in range(C.N_LEGS):
            q = rng.uniform(-np.pi, np.pi, size=3)
            target = K.foot_position_hip(i, q)
            sol = HW.leg(i).ik_full(target, q_seed=q)
            worst_ik = max(worst_ik, float(np.max(np.abs(
                K.foot_position_hip(i, sol.q) - target))))
            err_q = float(np.max(np.abs(sol.q - q)))
            worst_branch = max(worst_branch, err_q)
            if abs(np.sin(q[2])) > 0.2:          # clear of the knee singularity
                worst_clear = max(worst_clear, err_q)
            unreachable += not sol.reachable
            # ...and seeded from the stance, which is how a controller calls
            # it: the branch must still be the one the leg is standing on.
            near = C.Q_STAND[i] + rng.uniform(-0.9, 0.9, size=3)
            stand_sol = HW.leg_ik(i, K.foot_position_hip(i, near))  # no seed
            worst_stand = max(worst_stand, float(np.max(np.abs(stand_sol - near))))

    close("hw leg_ik is EXACT over the whole joint torus, %d poses x 4 legs"
          % N_RANDOM, worst_ik, 0.0, 1e-14, " m")
    # THE JOINT-SPACE FIGURE IS LOOSER THAN THE POSITION ONE, AND acos IS WHY.
    # The knee comes out of an arc cosine, whose derivative blows up as its
    # argument approaches +-1 -- which is the leg straight or folded.  The
    # worst cases below all sit at |sin q3| < 0.005: the ANGLE there is
    # ill-determined by the position, and the position itself stays exact,
    # which is the next line.  Any closed-form IK has this; a solver hides it
    # by reporting the residual it stopped at instead.
    close("...and recovers the very joint angles FK was given",
          worst_branch, 0.0, 1e-9, " rad")
    close("...to machine precision once clear of the knee singularity",
          worst_clear, 0.0, 1e-12, " rad")
    close("...and unseeded, picks the Q_STAND branch across the envelope",
          worst_stand, 0.0, 1e-12, " rad")
    check("every FK-generated target is reported reachable",
          unreachable == 0, "%d of %d flagged" % (unreachable, 4 * N_RANDOM))

    # A target the leg cannot meet has to degrade, and say that it did.  Ask
    # for the hip hinge itself: |p_hip|^2 = P^2 + Q^2 + Z^2 and Z is fixed, so
    # the chain's global minimum is |Z| -- 5 mm, which is the foot BALL's
    # out-of-plane offset and nothing to do with any link length (|d3| is
    # 105.1 mm).  A correct closed form finds that minimum exactly.
    #
    # IT IS A STATEMENT ABOUT THE CHAIN, NOT ABOUT THE ROBOT, and the next two
    # lines are why it is written down rather than left implied.  Reaching it
    # needs the knee at -164.4 deg, past KNEE_LIM, with the shin doubled back
    # over the thigh -- a pose the real leg is inside itself in.  The MJCF
    # leaves every hinge unlimited and cannot self-collide, so nothing in the
    # simulator objects; `params.JOINT_LIMITS` is the only thing that does,
    # and only if a caller applies it.
    at_hip = HW.leg(0).ik_full((0.0, 0.0, 0.0))
    close("hw leg_ik at the hip hinge finds the chain's true minimum, |Z|",
          np.linalg.norm(K.foot_position_hip(0, at_hip.q)),
          abs(HW.LEGS["FL"].d3[2]), 1e-15, " m")
    check("...at a pose OUTSIDE params.JOINT_LIMITS, which nothing enforces",
          not P.within_limits(at_hip.q),
          "knee %.1f deg vs KNEE_LIM %.1f deg -- clamp_q before commanding"
          % (np.degrees(at_hip.q[2]), np.degrees(P.KNEE_LIM)))
    far = HW.leg(0).ik_full((0.0, 0.0, -0.9))
    check("hw leg_ik degrades instead of exploding on an unreachable target",
          np.all(np.isfinite(far.q)) and not far.reachable,
          "asked for a foot 0.9 m below a %.3f m leg, reachable=%s"
          % (P.LEG_REACH, far.reachable))
    # Against `sim.leg_ik`.  COMPARED AT THE FOOT, NOT AT THE JOINTS, and the
    # reason is that sim's solver stops at a 1e-6 m residual by design: the
    # two answers can differ by that much of foot travel, which divided
    # through a Jacobian is tens of microradians of joint -- 4e-5 measured,
    # and not a disagreement.  At the foot the triangle inequality bounds it
    # by sim's own tolerance, since hw's own residual is zero.
    pairs = [(i, t) for i in range(C.N_LEGS)
             for t in [K.foot_position_hip(i, C.Q_STAND[i] + d)
                       for d in rng.uniform(-0.6, 0.6, size=(25, 3))]]
    both = [(HW.leg_ik(i, t), K.leg_ik(i, t)) for i, t in pairs]
    close("hw leg_ik lands where sim's damped least squares lands, to sim's tol",
          max(float(np.max(np.abs(K.foot_position_hip(i, a)
                                  - K.foot_position_hip(i, b))))
              for (i, _), (a, b) in zip(pairs, both)), 0.0, 1e-6, " m")
    close("...on the same elbow branch, not merely the same point",
          max(float(np.max(np.abs(a - b))) for a, b in both), 0.0, 1e-3, " rad")

    check("hw takes its geometry from params, not a second copy of it",
          all(np.array_equal(HW.LEGS[n].d1, P.LEG_GEOMETRY[n].hip_to_pitch)
              and np.array_equal(HW.LEGS[n].d2, P.LEG_GEOMETRY[n].pitch_to_knee)
              and np.array_equal(HW.LEGS[n].d3, P.LEG_GEOMETRY[n].knee_to_foot)
              and np.array_equal(HW.LEGS[n].hip, P.LEG_GEOMETRY[n].hip)
              for n in C.LEGS),
          "exact equality, not a tolerance -- these must be the same numbers")

    # -- 6. leg statics against MuJoCo's own bias force --------------------
    # qfrc_bias at zero velocity IS the generalised gravity force, so the
    # joint entries are exactly the torque the motors hold.  The trunk is
    # lifted clear of the floor; contacts do not enter qfrc_bias anyway, but
    # a pose that interpenetrates the floor is not worth reasoning about.
    worst_tau = 0.0
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, size=(C.N_LEGS, 3))
        data.qpos[:] = C.qpos_from_q(q, root_pos=(0.0, 0.0, 1.0))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for i in range(C.N_LEGS):
            mine = K.leg_gravity_torque(i, q[i])
            theirs = data.qfrc_bias[6 + 3 * i: 9 + 3 * i]
            worst_tau = max(worst_tau, float(np.max(np.abs(mine - theirs))))
    close("leg_gravity_torque == mj qfrc_bias, 200 random poses",
          worst_tau, 0.0, 1e-12, " N m")

    # -- 7. the whole-robot composite --------------------------------------
    com, inertia = K.body_inertia(C.Q_STAND)
    data.qpos[:] = C.qpos_from_q(C.Q_STAND)
    mujoco.mj_forward(model, data)
    mujoco.mj_comPos(model, data)
    theirs_com = data.subtree_com[trunk] - data.xpos[trunk]
    close("body_inertia CoM == mj subtree_com", com, theirs_com, 1e-12, " m")

    # MuJoCo's own composite inertia about the subtree CoM, in world axes --
    # the trunk is at identity here so world axes are trunk axes.
    points, masses, tensors = [], [], []
    for b in range(1, model.nbody):
        irot = np.zeros(9)
        mujoco.mju_quat2Mat(irot, model.body_iquat[b])
        rot = data.xmat[b].reshape(3, 3) @ irot.reshape(3, 3)
        points.append(data.xipos[b] - data.xpos[trunk])
        masses.append(model.body_mass[b])
        tensors.append(rot @ np.diag(model.body_inertia[b]) @ rot.T)
    points, masses = np.asarray(points), np.asarray(masses)
    ref = (masses[:, None] * points).sum(0) / masses.sum()
    theirs_I = np.zeros((3, 3))
    for p, m, own in zip(points, masses, tensors):
        d = p - ref
        theirs_I += own + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    # 1e-8: thirteen bodies' worth of the same principal-axis round-trip noted
    # above, each then carried out to a parallel-axis lever, so the per-link
    # 1e-11 accumulates.  Against an Ixx of 0.036 this is a relative 5e-8.
    close("body_inertia tensor == MuJoCo's composite", inertia, theirs_I,
          1e-8, " kg m^2")
    check("the legs dominate it: Ixx is %.4f against the trunk's own %.4f"
          % (inertia[0, 0], P.TRUNK_INERTIA[0, 0]),
          inertia[0, 0] > 3.0 * P.TRUNK_INERTIA[0, 0],
          "%.1fx" % (inertia[0, 0] / P.TRUNK_INERTIA[0, 0]))

    # -- 8. IK round-trips FK ----------------------------------------------
    print("\ninverse kinematics")
    worst_ik = 0.0
    reachable = 0
    for _ in range(N_RANDOM):
        i = int(rng.integers(C.N_LEGS))
        # Sample poses AROUND the stance rather than over the whole circle:
        # the IK's job is the working envelope, and a random point on the
        # joint torus is often on the far elbow branch where "round-trips"
        # is not even well posed.
        q = C.Q_STAND[i] + rng.uniform(-0.6, 0.6, size=3)
        target = K.foot_position_hip(i, q)
        solved = K.leg_ik(i, target)                 # no seed: Q_STAND branch
        err = float(np.max(np.abs(K.foot_position_hip(i, solved) - target)))
        worst_ik = max(worst_ik, err)
        reachable += 1
    # The bound IS `leg_ik`'s own default `tol`, so this gates the contract the
    # solver states rather than an independent number -- expect it to land just
    # under, because the solver stops as soon as it is under.  A result at 98 %
    # of this tolerance is convergence, not a near miss.  What would be a real
    # failure is a pose that does not converge at all.
    close("leg_ik round-trips FK to its own 1 um tol, %d poses" % reachable,
          worst_ik, 0.0, 1e-6, " m")
    close("...and to 10 nm when asked for it",
          max(float(np.max(np.abs(
              K.foot_position_hip(0, K.leg_ik(0, t, tol=1e-8)) - t)))
              for t in [K.foot_position_hip(0, C.Q_STAND[0] + d)
                        for d in rng.uniform(-0.6, 0.6, size=(20, 3))]),
          0.0, 1e-8, " m")

    check("leg_ik degrades instead of exploding on an unreachable target",
          np.all(np.isfinite(K.leg_ik(0, (0.0, 0.0, -0.9)))),
          "asked for a foot 0.9 m below a %.3f m leg" % P.LEG_REACH)

    # -- 9. the derived stance numbers -------------------------------------
    print("\nderived stance geometry")
    check("reach_used is DOG6's 77%, not DOG5's 92%",
          0.76 < K.reach_used() < 0.78, "%.1f%%" % (100 * K.reach_used()))
    check("reach_room solves the quadratic, not the difference of norms",
          K.reach_room() > 0.10,
          "%.1f mm along +x" % (1000 * K.reach_room()))
    check("the four legs weigh the same",
          np.allclose(P.LINK_MASS.sum(1), P.LINK_MASS[0].sum(), atol=1e-9),
          "%.6f kg each" % P.LEG_MASS)
    close("the four legs are geometric mirrors",
          np.abs(K.hip_to_foot_stance()), np.abs(K.hip_to_foot_stance()[0]),
          1e-12, " m")

    # ----------------------------------------------------------------------
    print("\n%d checks, %d failed" % (_PASSES + len(_FAILURES), len(_FAILURES)))
    if _FAILURES:
        for name in _FAILURES:
            print("  FAILED: %s" % name)
        return 1
    print("model/dog6.xml, sim/{coordinates,params,kinematics}.py and "
          "hw/kinematics.py all agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
