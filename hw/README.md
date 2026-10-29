# `hw` — the protocol came from DOG5, the wiring came from DOG6

```
python -m hw.selftest              gate the plumbing, no robot needed  (48 checks)
python -m hw.balance.selftest      gate the balance controller         (78 checks)
python -m hw.bringup plan          where the bring-up has got to
python -m hw.bringup scan --fake   run the whole CAN path against software drivers
python -m hw.stand                 stand up, six phases, ENTER steps them
python -m hw.stand --wave FL       ...and stand on three legs, waving that paw
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
| `can_link.py` | no | can0 as the **program's** job, every real-bus entry point: before the socket opens, `down` → 1 Mbit/s, `restart-ms 100` → `up` (the last run's queued frames and a bus-off cleared — what `ip link set can0 down/up` by hand was for), a software **replug** of the PCAN-USB's hub port when can0 is missing or the adapter will not take a frame, a status-read probe; the same reset after the motors are stopped. Refuses a link another process has open. `python -m hw.can_link [reset|replug]` by hand |
| `realtime.py` | no | the loop's 4 ms kept: the GC frozen before `arm()` and run by the loop itself in slot 3 (an automatic pass took 13.8 ms of CPU mid-run in the fake loop on 2026-10-05, and a 27 ms gap is a stop), every thread at nice −20 on the real bus, and `SendGuard` — the 25 ms stop line counted on frames the kernel **took**, saying on a stop whether the link refused, another process had the core, or this one was computing |
| `hardware_map.py` | — | the two facts. **Twelve measured rows**, 2026-09-15 |
| `calibration.py` | motor frame no, joint frame yes | the gain, `EncoderUnwrap`, `soft_limits`; plus the joint conversions, which raise if a row goes missing |
| `imu.py` | **streams** (200 Hz, 0 CRC); signs **verified** | DETA10 → trunk frame; `orientation()` gives R and ω^b in SI. `R_BODY_IMU` is identity **by measurement** — same board, same mounting as DOG5. The adapter stays dumb: "level" is latched by the law at zero torque, not trimmed here — see [balance/README](balance/README.md) |
| `velocity_estimator.py` | offline gates pass; streams on the robot | trunk velocity and position by integrating the raw 0x40 accel — DOG5's EKF prediction step with the AHRS's roll/pitch for gravity, body-frame state so yaw never enters, sensor-clock dt. Position is in DOG5's odometry frame (heading fixed at init, gyro yaw). Pure integration drifts (bias × t, × t²/2); `zero()` / `set_velocity()` / `set_position()` are for `hw.balance` to re-anchor it |
| `balance/` | yes | the stand's SRB balance controller + force allocator, **every gain untuned**. [README](balance/README.md) |
| `fold_stand.py` | no | `hw.stand` from `posture.FOLD` — a hand-captured crouch at h = 60 mm with the feet nowhere near the nominal stance. SRB only, IMU datum fixed. Needs `--tau-cap 3.0` |
| `safety.py` | yes | ramp, cap, limit block, slew, e-stop trips. Refuses a map with a hole in it, **no override** |
| `fake_bus.py` | needs explicit ids | twelve drivers in software, same protocol |
| `bringup.py` | scan/spin no, check/setzero yes | scan / spin / check / setzero / imu / plan |
| `record.py` | yes | the twelve joint angles, read and kept. **Zero torque** — 0xA1 iq=0 keep-alives only, so the legs stay back-drivable and a pose put in by hand can be read off. Live line, ENTER for a pose table, `--out FILE.csv` streams every sweep. `--fake` too |
| `swing_bench.py` | yes, **robot hung up** | the swing leg on its own: position ramp to the lift pose, torque hold of every foot at its site (swing PD + gravity, no balance law, no IMU), ENTER runs the gait clock and the legs in `--legs` follow the arc — `hw.trot`'s swing law and flags exactly. One line per swing from the FK: apex asked vs reached, x/y excursion, touchdown speed, tau_ff / PD / slew-clip peaks. `--log FILE.npz`, `--fake --auto 1 --swings 4` |
| `swing_ff.py` | yes, **robot hung up** | the swing feedforward's A/B on the robot, the question of `doc/swing_height` (branch `swing-analysis`): position ramp to `posture.FOLD2`'s hold at 160 mm, every leg on the joint layer (Kp 5 / Kd 0.2, `--kp-joint` and the rest) + gravity, no IMU; ENTER runs the clock — time alone, period 0.8 / duty 0.8 / ramp 0.15, the study's — and the legs in `--legs` follow the 20 mm arc on the walk's Cartesian PD, the feedforward OFF or ON **for the whole run** (`--swing-ff`; `--sched` schedules the swing PD on the foot's apparent mass, Kp = ωn²Λ(q), Kd = 2ζωnΛ(q), `swing.GainSchedule`): the A/B is two runs, each to its own log (`swing_ff_off_<time>.npz` / `swing_ff_on_<time>.npz`, or `--log FILE.npz`). One line per swing with the study's metrics (apex and where, lag, rms, h_td, vz_td), the per-leg table at exit, the torque split in the log. No `--fake` |
| `gravity_bench.py` | yes, **robot hung up, level** | G(q), the legs' own gravity torque, at `posture.FOLD2`'s hold (`--height`, default 160 mm, `hw.fold2_trot`'s stand; several heights in turn). Every joint held by a joint PD alone (Kp 5 / Kd 0.2, the joint layer's; no gravity feedforward), the whole leg brought slowly (2 deg/s) into the target from 8 deg below and from 8 deg above, the torque averaged 1 s once the leg has stopped: G = the mean of the two sides, tau_s = half their difference, both from the commanded and the driver-measured torque, beside `balance.torque`'s model G. Sides are angles ("from −" = the joint coming up in angle) so tau_s is positive on every joint. `--log FILE.npz`, `--fake --auto 1`. **Not flown** |
| `stand.py` | yes | `sim.stand`'s limp → settle → crouch → lift → park on the robot. Driver 0xA4 for position, `SafetyGate` for the lift, every motor re-sent inside the 50 ms input-lost window. Both lift laws, `--law srb` / `--law per-leg`. `--fake` runs it against `fake_bus`. `--wave LEG` splices `sim.wave`'s shift → raise → wave → lower → unshift after the hold (plain `hw.stand` only): torque phases on `hw.kinematics`, refused unless the lift arrived; the paw's and the joints' tracking trips, no tilt trip. **Simulated, not flown** |
| `wide_trot.py` | yes | `hw.fold_trot`'s run from `posture.WIDE` at 160 mm, the stance xy spring on, 0.5 s / 20 mm — the stand posture the 2026-10-01 simulation study (`doc/trot_posture/`) found best for the trot in place; `--track 95 / 105` for the feet further out |
| `fold_walk.py` | yes | `hw.fold_trot` steered from the keyboard while trotting: W/S x, A/D y, Q/E yaw, SPACE stop. Since 2026-10-06 the law presses SPACE itself: the **governor** (`balance/governor.py`, `--pause-lean`, default 5°) pauses the keys on a low-passed lean and resumes them calm, as the operator did by hand. One walking reference feeds the x/y rows, the heading, the joint layer's (world-anchored) stance targets and the footholds; at zero command each swing lands on its own site in the trunk frame, hw.fold_trot's in-place target (`--place-at-rest`: placed as before); the QP allocator by default; the trot at duty 0.70; since 2026-10-20 (the operator's robot test) the swing is the Cartesian impedance and no trot settles by default (`--swing-law osc --settle 0.2` is the walk before). **Simulated, not flown** — what MuJoCo says it can and cannot do is [doc/walk/README.md](../doc/walk/README.md) |
| `fold2_walk.py` | yes | the same walk over `hw.fold2_trot` (knees out, CoM on both diagonals): `fold_walk.walking` around FOLD2's stand. **Flown: the best run, 2026-10-07** (operator: very long, no fall) on `--kp-joint 4 --kp-joint-abd 5 --swing-law impedance --duty 0.8 --contact-ramp 0.2 --settle 0.2 --swing-roll --tau-slew 300` -- not the defaults (the impedance swing is, since 2026-10-20, and that day's test dropped the settle); see the module docstring — [doc/walk/README.md](../doc/walk/README.md) |
| `com_check.py` | reads a log | the built robot's CoM from a log's **measured** torques, **one row per HOLD**, against the pin the law flew there (`com_pin`), and each T's re-centring. Since 2026-10-06 the run does this itself: **every HOLD measures the CoM and T applies it** (`--com-auto`, the default on every trot that tracks its rise; `balance/com_estimate.py`) -- the pin and the stance sites move by measured - pinned and the trunk slides 0.5 s over the planted feet onto it before the trot starts. In MuJoCo a CoM 3 / 5 mm off the CAD's is the one hardware unknown that makes the trot in place rock harder every 10 s and fells every walk within seconds; re-centring the stance on the measured CoM brings both back to the ideal plant -- [doc/walk, section 11](../doc/walk/README.md) |
| `selftest.py` | — | 78 checks, no robot, no adapter touched |

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
