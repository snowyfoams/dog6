# `hw` — the protocol came from DOG5, the wiring came from DOG6

```
python -m hw.selftest              gate the plumbing, no robot needed  (48 checks)
python -m hw.balance.selftest      gate the balance controller         (78 checks)
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

Neither is derivable. Neither survives being copied. So they were **read off
DOG6** on 2026-09-15 — all twelve rows of
[hardware_map.py](hardware_map.py) found with `spin`, then re-driven in joint
coordinates with `check` and confirmed one at a time:

```
CAN ids   FL = 1-3    FR = 4-6    RR = 7-9    RL = 10-12
          (RR before RL on the bus — NOT canonical order)
dir −1    CAN 3, 6, 7, 9, 10, 12         dir +1   the other six
zero      all twelve drivers zeroed at Q_ZERO, flat on the belly
```

This table, that zero, [kinematics.py](kinematics.py) and the leg fold in
`sim/coordinates.py` are the **reference** the rest of the project measures
joint angles against. `hw.stand` then ran its position phases on them; the
**lift phase did not hold**, which is what [balance/](balance/) addresses —
the four facts above are not what is in question there.

The guards stayed. Everything needing joint coordinates asks the table at
**call time**, so a replaced or re-flashed driver raises `MapIncomplete`
instead of moving the wrong joint. Everything in the **motor's own frame**
needs no map at all, which is what let the bring-up tools discover the table
and what they would use to re-measure one row.

## Building the map

**Discovery, not verification.** DOG5's tools checked a table that already
existed; these built one. Below is the procedure that produced the table
above — run it again for any row whose driver is replaced or re-flashed.

| | step | what it establishes |
|---|---|---|
| 1 | `bringup scan` | which CAN ids answer. Probes 1..32 — no assumption of 1..12 |
| 2 | `bringup spin --id 7 --go` | turns motor 7 in its **own** frame. You watch which limb moves → that is the `can_id` |
| 3 | `bringup spin --id 7 --joint FL.knee --go` | same step, now printing what a **positive joint angle** does to that foot → read off the `direction` |
| 4 | `bringup check --joint FL.knee --go` | re-drive in **joint** coordinates, through the measured direction. Passing earns `confirmed=True` |
| 5 | `bringup setzero --yes` | encoders zeroed flat on the belly. Power-cycle, re-read ≈ 0 |
| 6 | `bringup imu --go` | ✅ mounting rotation **measured** — identity, board aligned with the trunk as on DOG5. Signs **verified** |
| 7 | first torque | `safety.TAU_START_MAX` = 1.0 N·m, robot supported |

`spin` commands with direction `+1` whatever the table says — that is what
"unknown" has to mean, and it is why `spin` is still the right tool for a row
being re-measured: it cannot be biased by the value already in that row. No
joint coordinates appear anywhere in it. Step 3 prints both candidate rows:

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
expectation, which is the exact failure the discovery procedure exists to
prevent. `python -m hw.hardware_map` prints all twelve predictions at once,
now beside the measured id and direction for each.

**Steps 1–6 command no torque.** `scan` streams iq=0 keep-alives (which hold
the drivers' 50 ms input watchdog open without producing motion — every motor
stays back-drivable); `spin` and `check` use the drivers' own 0xA4 position
loop with a low speed cap.

## Three states, checked separately

| | | DOG6 today |
|---|---|---|
| **empty** | no id or no direction. `MapIncomplete`, **no override anywhere** | 0 rows |
| **wired** | filled in from `spin`, but nobody has re-driven and watched it | 0 rows |
| **confirmed** | `check` passed. `CONFIRMED_ON_DOG6` is the conjunction | **12 rows** |

The middle state is real: filling a row in is a note about what you saw;
confirming it is a separate observation that the note was right. DOG5
collapsed the two and the check lived in an operator's memory.

The first two states are not dead code — they are where any replaced driver
starts, and `hw.selftest` still gates the torque refusal by punching a hole in
the table for the length of a `with` block.

## What is here

| file | needs the map? | what it is |
|---|---|---|
| `kinematics.py` | no | closed-form FK, Jacobian, IK. The yardstick all twelve directions were read against on the robot. Gated against MuJoCo by `sim.selftest` |
| `motor/` | no | CAN transport + LK protocol, four modules **byte-for-byte from DOG5**. Addresses raw ids |
| `hardware_map.py` | — | the two facts. **Twelve measured rows**, 2026-09-15 |
| `calibration.py` | motor frame no, joint frame yes | the gain, `EncoderUnwrap`, `soft_limits`; plus the joint conversions, which raise if a row goes missing |
| `imu.py` | **streams** (200 Hz, 0 CRC); signs **verified** | DETA10 → trunk frame; `orientation()` gives R and ω^b in SI. `R_BODY_IMU` is identity **by measurement** — same board, same mounting as DOG5. The adapter stays dumb: "level" is latched by the law at zero torque, not trimmed here — see [balance/README](balance/README.md) |
| `balance/` | yes | the stand's SRB balance controller + force allocator, **every gain untuned**. [README](balance/README.md) |
| `fold_stand.py` | no | `hw.stand` from `posture.FOLD` — a hand-captured crouch at h = 60 mm with the feet nowhere near the nominal stance. SRB only, IMU datum fixed. Needs `--tau-cap 3.0` |
| `safety.py` | yes | ramp, cap, limit block, slew, e-stop trips. Refuses a map with a hole in it, **no override** |
| `fake_bus.py` | needs explicit ids | twelve drivers in software, same protocol |
| `bringup.py` | scan/spin no, check/setzero yes | scan / spin / check / setzero / imu / plan |
| `record.py` | yes | the twelve joint angles, read and kept. **Zero torque** — 0xA1 iq=0 keep-alives only, so the legs stay back-drivable and a pose put in by hand can be read off. Live line, ENTER for a pose table, `--out FILE.csv` streams every sweep. `--fake` too |
| `stand.py` | yes | `sim.stand`'s limp → settle → crouch → lift → park on the robot. Driver 0xA4 for position, `SafetyGate` for the lift, every motor re-sent inside the 50 ms input-lost window. Both lift laws, `--law srb` / `--law per-leg`. `--fake` runs it against `fake_bus` |
| `selftest.py` | — | 48 checks, no robot |

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
could be mistaken for a measurement). Running it against DOG6's real table
would prove *less*: ids `1..12` in ascending order are exactly the arrangement
in which an off-by-one indexes the right value by accident.

`fake_bus` obeys whatever map it is given, so this can never tell you a real
map is right — only the robot did that, on 2026-09-15. It says the plumbing is
sound. It never says it is safe to power the robot.

It needs `python-can` installed even for `--fake`, because `motorbus` builds
real `can.Message` frames before handing them to the software drivers.
