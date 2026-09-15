# `hw` — the protocol is here, the wiring is not

```
python -m hw.selftest              gate the plumbing, no robot needed  (53 checks)
python -m hw.bringup plan          where the bring-up has got to
python -m hw.bringup scan --fake   run the whole CAN path against software drivers
```

DOG5's **protocol** has been ported across — same gearmotors, same drivers,
same frames, and a frame layout does not change with a chassis. DOG5's
**tables** have not, and that is the shape of this whole module.

| the two facts that are not portable | |
|---|---|
| a CAN id | what somebody flashed into a driver |
| a direction | which way round somebody bolted a motor |

Neither is derivable. Neither survives being copied. So
[hardware_map.py](hardware_map.py) is **empty** — twelve rows of `None` —
and everything needing joint coordinates raises `MapIncomplete` rather than
running on a guess. Everything in the **motor's own frame** still works, which
is what lets the bring-up tools discover the map in the first place.

## Building the map

**Discovery, not verification.** DOG5's tools checked a table that already
existed; these build one.

| | step | what it establishes |
|---|---|---|
| 1 | `bringup scan` | which CAN ids answer. Probes 1..32 — no assumption of 1..12 |
| 2 | `bringup spin --id 7 --go` | turns motor 7 in its **own** frame. You watch which limb moves → that is the `can_id` |
| 3 | `bringup spin --id 7 --joint FL.knee --go` | same step, now printing what a **positive joint angle** does to that foot → read off the `direction` |
| 4 | `bringup check --joint FL.knee --go` | re-drive in **joint** coordinates, through the measured direction. Passing earns `confirmed=True` |
| 5 | `bringup setzero --yes` | encoders zeroed flat on the belly. Power-cycle, re-read ≈ 0 |
| 6 | `bringup imu --go` | mounting rotation **measured**, not assumed |
| 7 | first torque | `safety.TAU_START_MAX` = 1.0 N·m, robot supported |

`spin` commands with direction `+1` because no direction is known yet — that
is what "unknown" has to mean. No joint coordinates appear anywhere in it, so
it runs on a completely empty map. Step 3 prints both candidate rows:

```
[spin] IF CAN 7 is FL.knee, a POSITIVE JOINT angle moves that foot
[spin]   dx +9.34  dy +0.00  dz -4.79 mm   (dominant +x), trunk frame
[spin] so after the move below:
[spin]   the FL foot went +x           -> direction = +1
[spin]   the FL foot went the other way -> direction = -1
[spin]   a DIFFERENT limb moved         -> CAN 7 is not FL.knee

[spin] write what you saw into hw/hardware_map.py -- ONE of:
    HardwareJoint('FL', 'knee', 'knee_FL', can_id=7, direction=+1),
    HardwareJoint('FL', 'knee', 'knee_FL', can_id=7, direction=-1),
```

That prediction is computed from [kinematics.py](kinematics.py), never typed
into a table — a stale copy would confirm a sign against the wrong
expectation, which is the exact failure the empty map exists to prevent.
`python -m hw.hardware_map` prints all twelve predictions at once.

**Steps 1–6 command no torque.** `scan` streams iq=0 keep-alives (which hold
the drivers' 50 ms input watchdog open without producing motion — every motor
stays back-drivable); `spin` and `check` use the drivers' own 0xA4 position
loop with a low speed cap.

## Three states, checked separately

| | |
|---|---|
| **empty** | no id or no direction. `MapIncomplete`, **no override anywhere** |
| **wired** | filled in from `spin`, but nobody has re-driven and watched it |
| **confirmed** | `check` passed. `CONFIRMED_ON_DOG6` is the conjunction |

The middle state is real: filling a row in is a note about what you saw;
confirming it is a separate observation that the note was right. DOG5
collapsed the two and the check lived in an operator's memory.

## What is here

| file | on an empty map | what it is |
|---|---|---|
| `kinematics.py` | **runs** | closed-form FK, Jacobian, IK. Also the yardstick step 3 reads against. Gated against MuJoCo by `sim.selftest` |
| `motor/` | **runs** | CAN transport + LK protocol, four modules **byte-for-byte from DOG5**. Addresses raw ids |
| `hardware_map.py` | — | the two facts. Twelve rows of `None` |
| `calibration.py` | **half** | motor frame runs; joint coordinates raise |
| `imu.py` | **runs** (signs unverified) | DETA10 → trunk frame |
| `safety.py` | **refuses** | ramp, cap, limit block, slew, e-stop trips |
| `fake_bus.py` | needs explicit ids | twelve drivers in software, same protocol |
| `bringup.py` | — | scan / spin / check / setzero / imu / plan |
| `stand.py` | **refuses** | `sim.stand`'s limp → settle → crouch → lift → park on the robot. Driver 0xA4 for position, `SafetyGate` for the lift, every motor re-sent inside the 50 ms input-lost window. `--fake` runs it against `fake_bus` |
| `selftest.py` | — | 53 checks, no robot |

## The staging ladder, with DOG6's own numbers

Measured through `hw.kinematics`' Jᵀ over a ±0.6 rad envelope around `Q_STAND`:

| | worst joint torque |
|---|---|
| holding the robot up on all four | **2.20 N·m** |
| one trot diagonal, half the weight | **4.46 N·m** |
| one leg's own weight, no contact | 0.19 N·m |

So `TAU_STAGED_MAX = 3.0` is enough to **stand** and deliberately not enough to
**trot**. `params.TAU_MAX_SIM` is 8.0 and is a *simulation* number —
`SafetyGate` refuses it by name.

## Two values `sim` is still holding a place for

- **`coordinates.R_BODY_IMU`** — identity today, and `hw.imu` keeps it separate
  from the DETA10's own NED→FLU convention precisely so the measured value has
  somewhere to go. The same rotation is baked into `model/dog6.xml` as the imu
  site's quaternion and `sim.selftest` gates that the two agree, so changing one
  without the other fails rather than passing silently.
- **`params.TAU_MAX_SIM`** — see the ladder above.

## What a green `hw.selftest` does not mean

It runs the protocol path against `fake_bus` under a **synthetic** map it
installs itself (ids ascending in sevens, six signs negative — nothing that
could be mistaken for a measurement). `fake_bus` obeys whatever map it is
given, so this can never tell you a real map is right. It says the plumbing is
sound. It never says it is safe to power the robot.
