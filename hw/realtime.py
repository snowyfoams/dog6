"""What keeps the CAN loop on its 4 ms: the GC kept out of it, the CPU kept for
it, and -- when it stalls anyway -- which of the two it was.

`hw.stand.run` and `hw.swing_bench.run` use all three.  Nothing here touches
the bus.


"CAN 6 went 27.3 ms without a frame, past the 25 ms stop line", 2026-10-05
    The operator's stop, "after a while" of running; no log of that run was
    kept, so which of the two below it was is not known.  It is a STALL of
    the whole loop, ~23 ms on top of the 4 ms sweep, and two things on this
    Pi make stalls that long -- both measured that day on `hw.fake_bus` (the
    same loop, no robot, no CAN):

    PYTHON'S GARBAGE COLLECTOR.  `hw.fold_trot --fake` stopped in HOLD with
    "CAN 1 went 27.8 ms without a frame" -- the same stop -- right after a
    generation-1 pass of 23.6 ms (wall clock, the Pi busy with other runs).
    Unloaded, with `--log`, the one generation-1 pass in 225 s armed took
    13.8 ms of CPU.  The passes are rare (11 in those 225 s) and they get
    LONGER as the run goes on, because what they walk grows: the twelve
    per-motor reply-latency lists `RoundRobinBus` appends 3000 floats a
    second to cost 4.9 ms on their own at that length, and the log's rows
    come on top -- which would fit "after a while".

    THE OTHER PROCESSES ON THE PI.  The same afternoon another session ran
    four MuJoCo sweeps on the four cores (~87 % each) while the operator
    flew; the robot's loop -- a busy-wait at nice 0, in the one cgroup every
    session, sim and editor shares -- got 77 % of a core, and can0 carried
    2090 frames/s against ~2700.  With six busy processes beside it, the fake
    loop at nice 0 ran 8641 slots longer than 3 ms in 120 s (worst 13.2 ms,
    the thread off the CPU in each) and managed 1650 slots/s of the 3000 it
    asks for; at nice 10 it stopped in CROUCH on a 41.9 ms gap with no GC
    pass in the loop at all.


THE GC: FROZEN BEFORE ARM, OFF IN THE LOOP, COLLECTED IN A SLOT OF ITS OWN
    `QuietGC.start()` runs one full collection and then `gc.freeze()`s
    everything the program built -- modules, the law, the posture, the
    estimator, the bus and its lists -- so no collection walks any of it
    again, and switches the automatic collector off.  It is called BEFORE
    `arm()`: the full collection takes ~20-35 ms, and from `arm()` on
    nothing may stall.

    The loop then collects by itself, at `GC_SLOT`, after that slot's frame
    is out: the young generation every sweep, the middle one every
    `GC_MID_EVERY` sweeps; never the old one, so a cycle the loop drops
    after a middle pass waits for `stop()`, which counts it.  Measured on the
    fake loop: young passes at most 20 us unloaded and 77 us under six busy
    processes, middle ones 12-28 us, and 0 objects left over at the stop in
    every run -- this loop makes no cyclic garbage at all.  `stop()` turns
    the collector back on.


THE CPU: NICE -20 FOR THE WHOLE PROCESS
    `boost()` renices every thread of this process to `NICE` (`sudo -n
    renice`, as `hw.can_link` runs `ip`): in the one cgroup everything here
    shares, weight 88761 against a sim's 1024.  The same six busy processes
    as above, the loop at nice -20: 46 slots over 3 ms in 120 s, worst 6.5
    ms, 2960 slots/s.  Threads started later (the IMU reader) inherit it, so
    call it before the IMU opens.  `--fake` runs are NOT boosted: a fake
    run from another session must not outrank a real one.

    Not SCHED_FIFO, which DOG5's `pace` suggests.  The loop busy-waits
    (`motorbus.pace` never sleeps for a 333 us slot), and this kernel's fair
    server (debugfs: 50 ms of every 1000 ms, per core) exists to run normal
    tasks an RT task starves -- 50 ms is the drivers' whole window.  A
    real-time loop here would have to sleep in every slot first; not tried.


WHAT A STALL SAYS
    `SendGuard` keeps, per motor, the time of the last frame the KERNEL
    TOOK -- before 2026-10-05 the gap was counted from the last ATTEMPT, so
    a link refusing every frame (ENOBUFS, ENETDOWN) looked healthy to it
    until the missed-reply streak, 80 ms in, after the drivers had already
    gone limp.  On a stop it says which it was:

        refused   the kernel would not take the frames: the adapter or the link
        off-CPU   the whole loop stood still and this thread was NOT running:
                  another process had the core
        computing the whole loop stood still and this thread WAS running: GC,
                  the law, a print -- with the last GC pass's length
"""
from __future__ import annotations

import gc
import os
import subprocess
import time

#: The nice value `boost()` asks for: the strongest a normal task gets.
NICE = -20

#: The slot the loop's own collections run in, after its frame: not slot 0
#: (the law) and not `hw.stand.ESTIMATOR_SLOT` (6).
GC_SLOT = 3

#: The middle generation every this many sweeps -- once a second at 250 Hz.
GC_MID_EVERY = 250


def boost(say=print) -> str:
    """Every thread of this process to nice `NICE`.  Returns a line for the
    banner; never raises -- a run without it is a run with today's timing."""
    pid = os.getpid()
    try:
        tids = sorted(os.listdir("/proc/%d/task" % pid), key=int)
    except OSError:
        tids = [str(pid)]
    cmd = ["renice", "--priority", str(NICE), "-p"] + tids
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
        ok = out.returncode == 0
        why = out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as failure:
        ok, why = False, str(failure)
    if ok:
        return ("CPU: nice %d on all %d threads (a sim at nice 0 cannot take "
                "the loop's core for long)" % (NICE, len(tids)))
    return ("CPU: NOT boosted (%s) -- the loop runs at nice %d and shares its "
            "core with whatever else runs; stalls past 25 ms come from that"
            % (why or "renice failed", os.getpriority(os.PRIO_PROCESS, 0)))


class QuietGC:
    """The garbage collector, kept to `GC_SLOT` for the length of a run."""

    def __init__(self):
        self.frozen = 0
        self.setup_ms = 0.0
        self.young_us_max = 0.0
        self.mid_us_max = 0.0
        self.passes = 0
        self.last_ms = 0.0               # the last pass, for SendGuard
        self.last_at = None
        self._was_enabled = None

    def start(self) -> str:
        """Full collection, freeze, automatic collector off.  BEFORE arm()."""
        t = time.perf_counter()
        self._was_enabled = gc.isenabled()
        gc.collect()
        gc.freeze()
        gc.disable()
        self.frozen = gc.get_freeze_count()
        self.setup_ms = 1e3 * (time.perf_counter() - t)
        return ("GC: %d objects frozen (%.0f ms, before arming), automatic "
                "collection OFF; the loop collects young every sweep in slot "
                "%d, middle every %d sweeps" % (self.frozen, self.setup_ms,
                                                GC_SLOT, GC_MID_EVERY))

    def step(self, sweep: int) -> None:
        """At `GC_SLOT`, after that slot's frame went out."""
        mid = sweep % GC_MID_EVERY == 0
        t = time.perf_counter()
        gc.collect(1 if mid else 0)
        dt = time.perf_counter() - t
        self.passes += 1
        self.last_ms, self.last_at = 1e3 * dt, t
        if mid:
            self.mid_us_max = max(self.mid_us_max, 1e6 * dt)
        else:
            self.young_us_max = max(self.young_us_max, 1e6 * dt)

    def stop(self) -> str:
        """The collector back as it was; what the run left for it.  "" if
        `start()` never ran; safe to call twice."""
        if self._was_enabled is None:
            return ""
        # Still frozen, so this walks only what the run made: what it finds
        # unreachable is the cyclic garbage the loop's own passes left.
        t = time.perf_counter()
        left = gc.collect()
        left_ms = 1e3 * (time.perf_counter() - t)
        gc.unfreeze()
        if self._was_enabled:
            gc.enable()
        self._was_enabled = None
        return ("GC: %d passes in the loop, young max %.0f us, middle max %.0f "
                "us; %d unreachable objects left at the stop (%.0f ms to "
                "collect, after the motors stopped)"
                % (self.passes, self.young_us_max, self.mid_us_max, left,
                   left_ms))


class SendGuard:
    """The per-motor gap, counted on frames the kernel took, with a verdict.

    `ok_at` starts from each motor's last frame in `arm()` -- the first loop
    frame's gap is checked as it always was."""

    def __init__(self, mb, ids, limit_s: float, window_s: float, gcq=None):
        self.mb = mb
        self.ids = list(ids)
        self.limit = float(limit_s)
        self.window = float(window_s)
        self.gc = gcq
        self.ok_at = [mb.rec(mid).last_cmd_t for mid in self.ids]
        self.refused = [0] * len(self.ids)
        self.refused_total = 0
        self.any_at = None               # the last frame taken, any motor
        self.cpu_at = None               # this thread's CPU time then
        self.worst = [0.0] * len(self.ids)

    def check(self, k: int, now: float) -> str | None:
        """Before slot k's frame: the stop reason, or None."""
        previous = self.ok_at[k]
        if previous is None:
            return None
        gap = now - previous
        if gap > self.worst[k]:
            self.worst[k] = gap
        if gap <= self.limit:
            return None
        return self._reason(k, gap, now)

    def sent(self, k: int, ok: bool, now: float) -> None:
        """After slot k's send; `ok` is what the send returned."""
        if ok:
            self.ok_at[k] = now
            self.refused[k] = 0
            self.any_at = now
            self.cpu_at = time.thread_time()
        else:
            self.refused[k] += 1
            self.refused_total += 1

    def worst_ms(self) -> float:
        return 1e3 * max(self.worst)

    def _reason(self, k: int, gap: float, now: float) -> str:
        mid = self.ids[k]
        head = ("CAN %d went %.1f ms without a frame, past the %.0f ms stop "
                "line (the drivers' window is %.0f ms)"
                % (mid, 1e3 * gap, 1e3 * self.limit, 1e3 * self.window))
        if self.refused[k]:
            error = getattr(self.mb, "last_send_error", None)
            return ("%s -- REFUSED: its last %d frame%s went to the kernel and "
                    "were not taken%s.  The link or the adapter, not the loop"
                    % (head, self.refused[k], "" if self.refused[k] == 1 else "s",
                       " (%s)" % error if error else ""))
        if self.any_at is None:
            return head
        still = now - self.any_at
        cpu = time.thread_time() - self.cpu_at
        if still < 0.5 * gap:
            return head
        if cpu < 0.5 * still:
            return ("%s -- OFF-CPU: the whole loop stood still %.1f ms (no frame "
                    "to any motor) and this thread ran %.1f ms of it: another "
                    "process had the core.  `top`, and what else is running"
                    % (head, 1e3 * still, 1e3 * cpu))
        recent = (self.gc is not None and self.gc.last_at is not None
                  and self.gc.last_at >= self.any_at)
        return ("%s -- COMPUTING: the whole loop stood still %.1f ms and this "
                "thread ran %.1f ms of it%s"
                % (head, 1e3 * still, 1e3 * cpu,
                   ": a GC pass of %.1f ms in it" % self.gc.last_ms if recent
                   else ": the law, a print, or a GC pass outside the loop's own"))
