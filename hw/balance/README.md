# `hw.balance` — the stand's balance controller

```
python -m hw.balance.selftest        78 checks, no robot, no IMU, no simulator
python -m hw.balance.config          every number, with its provenance
python -m hw.stand --law srb         run it
python -m hw.stand --law per-leg     run what it replaces
```

Implements the design note in [`doc/dog6_stand_control.tex`](../../doc): an
SRB wrench law with a weighted-least-squares force allocator, replacing the
per-leg Cartesian compliance law in the **lift phase of `hw.stand` only**.
Nothing here touches the limp, settle, crouch, park or done phases — those are
the drivers' own 0xA4 loops.

## What the old law could not see

| | per-leg compliance | this |
|---|---|---|
| trunk height | regulated | regulated |
| trunk roll / pitch | **emergent** — no sensor reads it, no variable names it | measured, SO(3) log-map error, its own gains |
| load sharing | fixed `mg/4` per foot | allocated by the actual geometry |
| internal force (6 DOF) | picked by accident | picked deliberately, min-weighted-norm |
| a foot pulling **up** on the floor | nothing forbids it | `fz_min > 0` forbids it |
| gravity | `mg/4` per foot, four times | once, as `m*g` in the wrench |

## The five stages, and the two frames

```
1 state       R, ω^b from the IMU; q, q̇ from the encoders      body / joint
2   …         ω^w = R ω^b,  r_i^w = R (x_i^b − c^b),  h        BODY → WORLD
3 controller  PD on (p_c,z, R) → accelerations → b_d            world
4 allocation  A f^w = b_d by weighted LS, then the cone         world
5 torque      τ_i = −J_i^T R^T f_i^w + τ_grav,i                WORLD → JOINT
```

Everything between 2 and 5 is world frame; everything outside is body or
joint. **Every array that has a frame carries the frame in its name** — `x_b`
and `r_w` are never spelled the same way.

| file | what it owns |
|---|---|
| `config.py` | every number. Gains, cone, weights, the pinned CoM/inertia, the trips |
| `state.py` | stages 1–2. Measurement only — no gain appears in it |
| `reference.py` | the C² quintic S-curve in `h`, and the CoM conversion |
| `controller.py` | stage 3. **The controller**: the only place a gain touches the trunk |
| `allocation.py` | stage 4. Pure transmission — no trunk feedback at all |
| `torque.py` | stage 5, plus the tilt-aware leg-gravity term |
| `law.py` | the five chained; one call per sweep. `hw.stand` owns *when*, this owns *what* |
| `selftest.py` | gates the lot offline |

## Three things that are exactly zero error at the identity

Each would pass every level bench test and then grow with tilt. All three are
checked **at a tilt**, never at `R = I`:

- the `R^T` in stage 5 — dropping it costs 0.35 N·m at 10°, 0 at 0°
- the world/body attitude-error pair — `e^w = R e^b`, identical at `R_d = I`
- the `ω × R x^b` term in `ż` — and at a *level square stance* it cancels by
  symmetry even when ω ≠ 0, which is why it is so easy to leave out

## Heights are floor to trunk **bottom**

The number a ruler reads. `sim.stand`'s `CROUCH_HEIGHT` / `LIFT_HEIGHT` /
`height_from_fk` are all trunk-**origin** heights, 35.01 mm higher, and
`state.origin_to_height` is the only conversion. DOG5 printed 191 mm where a
ruler read ~160 for want of exactly this distinction.

```
crouch  h =   0.0 mm     lift  h = 115.0 mm     CAD stand  h = 157.5 mm
```

## The CoM is pinned, and that is a decision

`config.COM_BODY` is evaluated once at the **lift pose** and used everywhere.
The true value swings 37 mm over the ramp (the legs are 60 % of the mass and
unfold downward), so this is an approximation — but it is the *consistent*
one: the same constant appears in the measured `p_c,z` and in the commanded
`p_c,z,d`, so it cancels out of the error entirely and the PD's z channel sees
precisely the trunk-height error. The chain-rule coefficient a pose-dependent
CoM would force into the velocity reference (0.677) becomes exactly 1.

What it costs is the moment arms, taken about a point up to 37 mm off early in
the ramp — a bias in the moment row that looks like a small constant pitch
offset and shrinks to zero at the top. **If a log shows a residual tilt that
shrinks as the robot rises, this is the first thing to unpin.**

## Timing, measured

Whole law, all four leg-gravity terms refreshed every sweep:

| | |
|---|---|
| `state.read` (stages 1–2) | 82 µs |
| reference | 4 µs |
| controller | 11 µs |
| allocation | 37 µs |
| leg gravity ×4, **closed form** | 36 µs |
| leg gravity ×4, chain walk | *465 µs* |
| `stance_torque` | 8 µs |
| IK for the tracking trip | 67 µs |
| **total** | **210 µs** of a 333 µs slot |

The closed-form leg gravity is what makes it fit. It agrees with
`sim.kinematics.leg_frames`' walk to 2.2e-16 over 300 random poses **and**
random orientations, which is the only licence for a second copy.

## What is still open

- **`coordinates.R_BODY_IMU` is an identity placeholder.** The board is not
  mounted. Nothing downstream of it has been checked on real hardware, and
  when it becomes a measurement it has to move on **both** sides at once —
  left-multiplied into the gyro, right-multiplied and transposed into the
  attitude (`imu.trunk_rotation`).
- **Every gain is `[UNTUNED]`.** The attitude pair is what the whole change
  exists to make tunable, so sweep it first, in sim.
- **Nothing measures whether a foot stayed put.** The cone bounds what the
  allocator *asks* the floor for. The rear-foot slide that ended the
  2026-09-15 runs is still invisible to a law reading joint encoders only.
- **The handover is still one sweep.** The torque cap ramps over 1.0 s, which
  bounds how hard the new law can push; it does not blend the two laws.
