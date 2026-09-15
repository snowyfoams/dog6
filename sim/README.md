# `sim` — DOG6 as a model

Everything that can be true about DOG6 before a motor is powered: the CAD's
geometry, its mass properties, the frames those are expressed in, and the maps
between joint angles and foot positions. Nothing here talks to hardware.

```
sim/
  coordinates.py   frames, signs, naming, pose conventions.   no numbers
  params.py        masses, inertias, lengths, limits, motors. no logic
  kinematics.py    FK, Jacobians, IK, leg statics, composite inertia
  leg_dynamics.py  one leg's M, C qd + G and Lambda, by RNEA
  selftest.py      the description, gated against model/dog6.xml
  cmpc/            convex MPC -- see sim/cmpc/README.md
  model/
    dog6.xml           GENERATED — see "the model is an artifact" below
    meshes/            20 STLs, mm, component-local frames
    robot_export.json  the build pipeline's machine-readable summary
```

Run it:

```
D:\mujoco\.venv\Scripts\python.exe -m sim.selftest        # 62 gates
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.selftest   # 83 gates
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.run        # drive it
D:\mujoco\.venv\Scripts\python.exe -m sim.params          # the numbers
D:\mujoco\.venv\Scripts\python.exe -m sim.kinematics      # the stance
```

The first four files are the robot as a **description** — they say what is
true of the design and nothing about how to control it. `cmpc/` is the first
thing built on top: a reproduction of the MIT Cheetah 3 convex MPC. It walks at
0.4 m/s and turns at 90 °/s on ground-truth state and a timetable gait.

`leg_dynamics.py` exists because that controller's swing law needs the
operational-space inertia Λ, and the Coriolis and gravity torques — none of
which forward kinematics provides. It is RNEA over a three-link chain, gated
against MuJoCo's own `mj_fullM` (7.7e-13) and `qfrc_bias` (8.3e-12).

## Why three files and not one

They could be one file; DOG5's equivalent nearly was. They are three because
the three kinds of mistake are different, and each is caught a different way.

A **frame** error is a sign that survives every scalar check — the masses are
right, the lengths are right, and the robot walks sideways. A **parameter**
error is a plausible number in the right place, caught only by reading it back
out of the artifact it was copied from. A **kinematics** error is neither: it
is caught by comparing against an integrator over many poses, because one pose
cannot distinguish a correct chain from several wrong ones.

Keeping them apart lets `selftest` check each the way that actually catches it,
and lets a controller import a convention without importing a table of
inertias. The dependency runs one way and never back:

```
coordinates  <-  params  <-  kinematics  <-  selftest
```

## The model is an artifact, not source

`model/dog6.xml` is **generated**, and its generator deliberately does not live
here. It comes out of the Fusion 360 design `quadruped_robot` in two stages,
both in `D:\mujoco\dog6_description`:

```
Fusion 360 "quadruped_robot"
     |  stage 1: read-only, through the Fusion MCP
     v
  dog6_raw.json + meshes/*.stl
     |  stage 2: build_dog6_mjcf.py
     v
  dog6.xml, robot_export.json        --> copied into sim/model/
```

When the CAD moves: rebuild there, copy `dog6.xml`, `meshes/` and
`robot_export.json` into `model/`, then run `python -m sim.selftest`. That is
what tells you which numbers in `params.py` went stale — do not edit the XML to
match the Python, it is the generated side.

## What the self-test proves

62 gates. The ones worth knowing about:

| | gate | measured |
|---|---|---|
| FK | `foot_position` vs MuJoCo's site, 500 random poses **with the trunk tilted** | 4.4e-16 m |
| Jacobian | `foot_jacobian` vs `mj_jacSite`, 500 poses | 1.7e-16 |
| statics | `leg_gravity_torque` vs `qfrc_bias`, 200 poses | 3.9e-16 N·m |
| inertia | `body_inertia` vs MuJoCo's composite | 1.7e-9 kg·m² |
| IK | `leg_ik` round-trips FK from the default seed | meets its own 1 µm `tol` |
| geometry | every `LEG_GEOMETRY` entry vs the MJCF's body `pos` | 2.9e-9 m |

The random-pose gates exist because a single nominal pose is one equation and
the chain has many ways to be wrong. Tilting the trunk matters separately: FK
returns trunk-frame coordinates, and a bug that only shows up when world and
trunk axes differ is invisible at identity.

**The 2.9 nm on the geometry is the MJCF's text precision, not drift.** The XML
writes `pos="0.1563145 0.06 0"` — seven decimals — so `hip` rounds up by 2.9 nm
and `hip_to_pitch` rounds down by the same amount. Their sum, which is the point
the chain actually propagates, is bit-identical. `selftest` gates the sum
separately at 1e-15 to make that explicit; it is why FK agrees to 4e-16 and not
to 2.9e-9.

## Two numbers that are placeholders

Marked in the source, repeated here because they are easy to forget:

- **`coordinates.R_BODY_IMU`** is the identity. The IMU has not been mounted.
  The same rotation is baked into `model/dog6.xml` as the `imu` site's quat and
  `selftest` gates that they agree, so it cannot be changed in one place only.
- **`params.TAU_MAX_SIM = 8.0`** N·m is a *simulation* number — the motor's
  9.94 N·m saturation with 20 % held back. Before DOG6 runs on hardware this
  goes back to a low staging cap.

Everything in `params.py` under `[INHERITED]` is DOG5's, adopted because DOG6
has the same motors and nobody has powered DOG6. The software joint limits are
in that category, and note that **`model/dog6.xml` carries no joint ranges at
all** — `params.JOINT_LIMITS` is a policy callers must apply, not something
MuJoCo will enforce. `selftest` asserts the hinges are unlimited so that stays
visible.

## The two things DOG6 does that DOG5 could not

Both fall out of one geometric fact: **DOG6 stands under its hips.**

```
                   DOG6              DOG5
leg extension      76.9 %            92.3 %
step room (+x)     106.9 mm          ~6 mm
Ixx about the CoM  0.0362 kg m^2     0.0658
hip spacing        0.313 x 0.120 m   0.441 x 0.120
thigh / shin       100.0 / 105.0 mm  120.0 / 112.4
```

DOG5's flat-zero leg had to abduct 90° and then reach 120 mm *forward* to touch
the floor, which used almost all of its reach and left a step nowhere to go.
DOG6's foot hangs 25.7 mm ahead of and 177.5 mm below its hip. That also makes
a forward step nearly *perpendicular* to the leg instead of radial, which is why
`kinematics.reach_room()` solves a quadratic rather than subtracting two norms —
the two disagree by a factor of 2.5 here (106.9 mm against 42.2 mm).

The abduction and pitch axes also **intersect** on DOG6: `hip_to_pitch` is a
pure 28.19 mm axial offset where DOG5's carried a −52.9 mm z term. The hip is a
plain elbow, not a dogleg.
