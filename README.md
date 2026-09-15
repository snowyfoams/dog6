# DOG6

A quadruped, sim first. The hardware is on the bench being assembled; this
repository is what can be established about the robot before a motor is
powered, arranged so that the day hardware arrives it plugs into a description
that has already been checked.

```
dog6/
  sim/     the robot as a model — geometry, mass, frames, kinematics
    cmpc/  convex MPC, reproducing MIT Cheetah 3 (IROS 2018)
  hw/      empty, on purpose. The robot does not exist yet.
```

```
D:\mujoco\.venv\Scripts\python.exe -m sim.selftest        # 62 gates, the model
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.selftest   # 83 gates, the controller
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.run        # drive it: Xbox pad, or WASD + QE
D:\mujoco\.venv\Scripts\python.exe -m sim.cmpc.teleop     # the reference alone, animated
```

145 gates, all passing. Run them after touching anything here.

The robot trots at **0.4 m/s** and turns at **90 °/s** in simulation, on
ground-truth state and a timetable gait. Above 0.4 m/s the 8 N·m torque clamp
saturates and it falls at 0.5 — the envelope is set by the actuators, not by
the controller. `sim/cmpc/README.md` has the numbers and the three bugs worth
recording.

## Why two modules, and why one of them is empty

The split is not organisational tidiness — it is a claim about what is known.

`sim` contains things that are **true of the design**: the Fusion 360 CAD's
geometry and mass properties, the frame and sign conventions, and the maps
between joint angles and foot positions. Every number in it is either measured
off the CAD or checked against `sim/model/dog6.xml`, and `sim/selftest.py` is
what keeps that honest.

`hw` would contain things that are **true of the built robot**: CAN ids,
direction signs, encoder zeros, the IMU's mounting rotation, what the motors
actually do at a commanded current. None of that has been observed, because
there is no robot to observe. So the module holds a `CONFIRMED_ON_DOG6 = False`
and nothing else.

The temptation this structure exists to resist is porting DOG5's hardware layer
across and calling it DOG6's. DOG5 is the same motors and, on the operator's
word, the same wiring — which makes it an excellent starting point and not a
measurement. A wrong CAN id sends a knee command to an abduction motor; a wrong
direction sign is a joint whose feedback and command disagree, which under a
torque law is a robot driving itself into the floor at full current. Neither
shows up in simulation. `hw/README.md` has the bring-up order.

## What is in `sim`

```
sim/
  coordinates.py   frames, signs, naming, pose conventions.   no numbers
  params.py        masses, inertias, lengths, limits, motors. no logic
  kinematics.py    FK, Jacobians, IK, leg statics, composite inertia
  selftest.py      all three, gated against model/dog6.xml
  model/           dog6.xml, 20 meshes, robot_export.json — GENERATED
```

Three files rather than one because the three kinds of mistake are different. A
frame error is a sign that survives every scalar check. A parameter error is a
plausible number in the right place. A kinematics error is caught only by
comparing against an integrator over many poses. `sim/README.md` goes into it.

`model/dog6.xml` is an **artifact**, not source. It is generated from the Fusion
360 design `quadruped_robot` by a two-stage pipeline that stays in
`D:\mujoco\dog6_description`. When the CAD moves, rebuild there, copy the XML,
meshes and `robot_export.json` into `sim/model/`, and run the self-test — it
will tell you which numbers in `params.py` went stale. Do not edit the XML to
match the Python.

## The robot

| | DOG6 | DOG5 |
|---|---|---|
| mass | 5.883 kg | 5.815 kg |
| trunk | 2.349 kg (40 %) | 2.619 kg (45 %) |
| hip / thigh / shin | 0.400 / 0.412 / 0.072 kg | 0.392 / 0.369 / 0.038 |
| hip spacing | 0.313 × 0.120 m | 0.441 × 0.120 m |
| thigh / shin | 100.0 / 105.0 mm | 120.0 / 112.4 mm |
| foot ball | r = 15 mm | r = 20 mm |
| stand height | 0.19254 m (drawn) | 0.190 m (chosen) |
| **leg extension at stance** | **76.9 %** | **92.3 %** |
| **step room, +x** | **106.9 mm** | **~6 mm** |
| I<sub>xx</sub> about the CoM | 0.0362 kg·m² | 0.0658 |
| motors | 12 × MG5010-i10, 330 g, 10:1 | same |

The two bold rows are the reason DOG6 exists. **DOG6 stands under its hips**;
DOG5 stood with its feet 120 mm in front of them, because its flat-zero leg had
to abduct 90° and then reach forward to touch the floor. That used 92 % of its
reach and left a step essentially nowhere to go. It also means a forward step on
DOG6 is nearly *perpendicular* to the leg rather than radial, so it costs almost
nothing in reach.

Smaller and worth knowing: DOG6's abduction and pitch axes **intersect**, so the
hip is a plain 28.19 mm elbow where DOG5's was a dogleg with a −52.9 mm z
offset. And its roll inertia is 45 % smaller on a stance 33 % narrower, so the
roll mode is much faster.

## Conventions carried from DOG5

Kept deliberately, so that DOG5's calibration procedure — which has actually
been performed on a real robot — still describes DOG6, and so a controller
written against one reads the other correctly.

- **Frame**: x forward, y left, z up, origin at the centre of the
  abduction-axis plane.
- **Zero**: every joint reads 0 with the robot flat on its belly, legs extended
  fore/aft. It is the only pose a human can put a robot into on a bench well
  enough to zero an encoder against. It is also nearly singular, which is why
  nothing plans through it.
- **Leg order**: FL, FR, RL, RR, matching the MuJoCo joint order, so `tau[i]`
  needs no permutation anywhere.
- **Collision**: link meshes are visual only; collision is a trunk box plus four
  foot spheres on contact classes that meet the floor and never each other.

## Status

- [x] `sim` — model, frames, parameters, kinematics, leg dynamics. 62 gates.
- [x] `sim.cmpc` — convex MPC: linearised body model, condensed QP, timetable
      gait, swing law. 83 gates. Trots, strafes and turns, from an Xbox pad.
- [ ] state estimation — the controller is handed ground truth. Everything it
      achieves is an upper bound on what it does behind a real filter.
- [ ] `hw` — waiting on the assembly. See `hw/README.md`.

An older trot stack exists against this model in `D:\mujoco\dog6_trot`; it has
not been folded in here, and `sim.cmpc` does not depend on it.
