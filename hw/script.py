"""The operator's keys from a file, on the phase clock: the same presses, every run.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.fold2_walk --script hw/scripts/p1_lat.txt --log p1_lat_1.npz
    $V -m hw.fold2_walk --fake --no-imu --auto 1 --script hw/scripts/p1_lat.txt
    $V -m hw.script hw/scripts/p1_lat.txt            print the expanded schedule

WHY
    A comparison between two runs is only as good as the sameness of the
    runs.  Pressed by hand, A comes 4.8 s after T in one run and 6.1 s in the
    next, the reversal lands at a different point of the gait cycle each
    time, and one burst is 110 s long and the next 130.  Scripted, every run
    of a kind is the same run, and what differs between two runs is the one
    flag on the command line.  The thesis can quote the file.

THE FILE
    One event per line; `#` starts a comment; blank lines are ignored:

        <phase>[<n>]  +<seconds>  <key> [<key> ...]
        <phase>[<n>]  +<seconds>  shuttle <K1> <K2> leg=<s> total=<s> [pause=<s>]

    <phase>   the phase the event is timed from, as `StandSequence.phase_name`
              reads it: limp settle crouch rise hold recentre trot fall park done
    [<n>]     the n-th ENTRY into that phase this run -- hold[2] is the HOLD
              after the trot -- ; omitted is the first entry; [*] is every one
    +<s>      seconds after that entry.  For the trot it is seconds after the
              trot STARTED (`t_trot`): after the HOLD's CoM wait and the slide
              onto it, not after T was pressed
    <key>     ENTER  SPACE  T  W  S  A  D  Q  E, the runner's own keys, case
              folded; several on a line go out one per sweep, in order.
              X is REFUSED: an e-stop is never scripted.
    shuttle   K1 for `leg` seconds, then K2 for `leg` seconds, and so on until
              `total` seconds have passed since the first press, then SPACE.
              Without `pause` a reversal is the other key PRESSED TWICE on
              consecutive sweeps: the keys sum through zero (`balance.keys`),
              so the reference crosses zero without resting on it and no swing
              flies "at rest".  With `pause` a reversal is SPACE, `pause`
              seconds of the trot in place, then the other key once.

WHAT THE RUNNER DOES WITH IT (`hw.stand.run`)
    Every sweep, where it would read the keyboard, it asks the script.  The
    keyboard still works, and ANY key the operator presses CANCELS the rest
    of the script and says so: the person at the robot overrides the file,
    and X is X.  An event whose phase ends before its time is SKIPPED, and
    the report says so -- a fall mid-shuttle drops the shuttle's remaining
    presses, and a `fall +3 ENTER` line then parks.  `--auto` beside a script
    (--fake only) steps the position phases only -- limp, settle, crouch,
    park, done -- and the script owns HOLD and the trot.  The script never
    reaches the law: it is a key source and nothing else.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

__all__ = ["KeyScript", "Event", "PHASES", "KEYS", "AUTO_PHASES", "parse",
           "shuttle"]

#: `StandSequence.phase_name`'s values, the anchors a line may name.
PHASES = ("limp", "settle", "crouch", "rise", "hold", "recentre", "trot",
          "fall", "park", "done")

#: Key word -> the character `hw.stand.run` reads from the keyboard.
KEYS = {"ENTER": "\r", "SPACE": " ", "T": "t", "W": "w", "S": "s", "A": "a",
        "D": "d", "Q": "q", "E": "e"}
_NAMES = {ch: name for name, ch in KEYS.items()}

#: The walk's keys a shuttle may alternate between.
_WALK_KEYS = ("W", "S", "A", "D", "Q", "E")


def key_name(ch) -> str:
    """The word for a character the runner read: ENTER, SPACE, T, ... or
    the character itself, quoted, when it is none of ours."""
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == " ":
        return "SPACE"
    if isinstance(ch, str) and ch.lower() in _NAMES:
        return _NAMES[ch.lower()]
    return repr(ch)

#: With a script present, the phases `--auto` may still step by itself.
AUTO_PHASES = ("limp", "settle", "crouch", "park", "done")


@dataclass
class Event:
    """One line of the file (a shuttle line becomes several)."""

    phase: str
    index: int | None        # 1-based entry; None = the first; 0 = every
    t: float                 # s after the entry
    keys: list               # the characters, in order
    text: str                # as written, for the report and the log
    line: int                # the file's line number
    #: entry number -> the `now` it fired, or a "skipped: ..." string.
    outcome: dict = field(default_factory=dict)

    def wants(self, entry: int) -> bool:
        if self.index is None:
            return entry == 1
        if self.index == 0:
            return True
        return entry == self.index

    @property
    def anchor(self) -> str:
        if self.index is None:
            return self.phase
        return "%s[%s]" % (self.phase, "*" if self.index == 0 else self.index)

    @property
    def keys_text(self) -> str:
        return " ".join(_NAMES[ch] for ch in self.keys)


def _key(token: str, line: int) -> str:
    name = token.upper()
    if name == "X":
        raise ValueError("line %d: X is the e-stop and is never scripted"
                         % line)
    if name not in KEYS:
        raise ValueError("line %d: %r is not a key (%s)"
                         % (line, token, " ".join(KEYS)))
    return KEYS[name]


def _anchor(token: str, line: int) -> tuple:
    """'hold[2]' -> ('hold', 2); 'hold' -> ('hold', None); 'hold[*]' -> 0."""
    name, index = token, None
    if "[" in token:
        if not token.endswith("]"):
            raise ValueError("line %d: %r -- expected phase[n]" % (line, token))
        name, inside = token[:-1].split("[", 1)
        if inside == "*":
            index = 0
        else:
            try:
                index = int(inside)
            except ValueError:
                raise ValueError("line %d: %r -- [n] must be a whole number "
                                 "or *" % (line, token)) from None
            if index < 1:
                raise ValueError("line %d: %r -- entries count from 1"
                                 % (line, token))
    name = name.lower()
    if name not in PHASES:
        raise ValueError("line %d: %r is not a phase (%s)"
                         % (line, name, " ".join(PHASES)))
    return name, index


def _seconds(token: str, line: int) -> float:
    if not token.startswith("+"):
        raise ValueError("line %d: the time is written +SECONDS, not %r"
                         % (line, token))
    try:
        t = float(token[1:])
    except ValueError:
        raise ValueError("line %d: %r is not a number of seconds"
                         % (line, token)) from None
    if t < 0.0:
        raise ValueError("line %d: a negative time" % line)
    return t


def shuttle(k1: str, k2: str, leg: float, total: float,
            pause: float = 0.0) -> list:
    """``[(t, [keys], text), ...]`` relative to the first press: K1 for `leg`
    s, K2 for `leg` s, ... until `total` s, then SPACE.  Module docstring."""
    k1, k2 = k1.upper(), k2.upper()
    for k in (k1, k2):
        if k not in _WALK_KEYS:
            raise ValueError("shuttle keys are the walk's (%s), not %r"
                             % (" ".join(_WALK_KEYS), k))
    if k1 == k2:
        raise ValueError("a shuttle needs two different keys")
    if not leg > 0.0 or not total > 0.0 or pause < 0.0:
        raise ValueError("shuttle: leg and total must be positive, pause >= 0")
    out = []
    keys = (k1, k2)
    n = 0
    while True:
        start = n * (leg + pause)
        if start >= total:
            break
        k = keys[n % 2]
        if n == 0:
            out.append((start, [KEYS[k]], "%s (leg 1)" % k))
        elif pause > 0.0:
            out.append((start, [KEYS[k]], "%s (leg %d)" % (k, n + 1)))
        else:
            out.append((start, [KEYS[k], KEYS[k]],
                        "%s %s (leg %d: through zero)" % (k, k, n + 1)))
        end = start + leg
        if end >= total:
            out.append((total, [KEYS["SPACE"]], "SPACE (end)"))
            break
        if pause > 0.0:
            out.append((end, [KEYS["SPACE"]], "SPACE (pause %d)" % (n + 1)))
        n += 1
    if out[-1][1] != [KEYS["SPACE"]]:
        out.append((total, [KEYS["SPACE"]], "SPACE (end)"))
    return out


def parse(text: str, name: str = "<script>") -> list:
    """The file's text -> the events, in file order (a shuttle expanded)."""
    events = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        tokens = body.split()
        if len(tokens) < 3:
            raise ValueError("line %d: expected `<phase>[n] +<s> <keys>`, got "
                             "%r" % (lineno, body))
        phase, index = _anchor(tokens[0], lineno)
        t0 = _seconds(tokens[1], lineno)
        if tokens[2].lower() == "shuttle":
            opts = dict(leg=None, total=None, pause=0.0)
            keys = []
            for tok in tokens[3:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k not in opts:
                        raise ValueError("line %d: shuttle takes leg=, total=, "
                                         "pause=, not %r" % (lineno, k))
                    try:
                        opts[k] = float(v)
                    except ValueError:
                        raise ValueError("line %d: %r is not a number"
                                         % (lineno, tok)) from None
                else:
                    keys.append(tok)
            if len(keys) != 2 or opts["leg"] is None or opts["total"] is None:
                raise ValueError("line %d: shuttle <K1> <K2> leg=<s> total=<s> "
                                 "[pause=<s>]" % lineno)
            try:
                legs = shuttle(keys[0], keys[1], opts["leg"], opts["total"],
                               opts["pause"])
            except ValueError as why:
                raise ValueError("line %d: %s" % (lineno, why)) from None
            for dt, chars, what in legs:
                events.append(Event(phase, index, t0 + dt, chars,
                                    "shuttle: " + what, lineno))
        else:
            chars = [_key(tok, lineno) for tok in tokens[2:]]
            events.append(Event(phase, index, t0, chars,
                                " ".join(tok.upper() for tok in tokens[2:]),
                                lineno))
    if not events:
        raise ValueError("%s: no events" % name)
    return events


class KeyScript:
    """A key source for `hw.stand.run`: `poll(now, stand)` once a sweep."""

    def __init__(self, events: list, name: str = "<script>"):
        self.events = list(events)
        self.name = name
        self._phase: str | None = None
        self._entry = 0
        self._t_entry = 0.0
        self._count: dict = {}
        self.queue: list = []
        self.last: Event | None = None
        self.fired = 0
        self.skipped = 0
        self.cancelled: str | None = None

    @classmethod
    def load(cls, path: str) -> "KeyScript":
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        return cls(parse(text, path), os.path.basename(path))

    @classmethod
    def from_text(cls, text: str, name: str = "<script>") -> "KeyScript":
        return cls(parse(text, name), name)

    # -- what the runner asks before the run ---------------------------------
    def presses(self, ch: str) -> bool:
        """True if any line sends this character."""
        return any(ch in ev.keys for ev in self.events)

    def banner(self) -> list:
        lines = ["SCRIPT %s: %d event%s -- any key pressed by hand cancels "
                 "the rest" % (self.name, len(self.events),
                               "" if len(self.events) == 1 else "s")]
        for ev in self.events:
            lines.append("     %-10s %+8.1f s  %s" % (ev.anchor, ev.t, ev.text))
        return lines

    # -- the sweep ---------------------------------------------------------
    def _see_phase(self, now: float, stand) -> None:
        name = stand.phase_name
        if name == self._phase:
            return
        # The phase that ended: whatever was still due for this entry is
        # skipped, and says so.
        if self._phase is not None:
            for ev in self.events:
                if (ev.phase == self._phase and ev.wants(self._entry)
                        and self._entry not in ev.outcome):
                    ev.outcome[self._entry] = (
                        "skipped: %s ended at +%.1f s"
                        % (self._phase, now - self._t_entry))
                    self.skipped += 1
        self._phase = name
        self._count[name] = self._count.get(name, 0) + 1
        self._entry = self._count[name]
        # The sequence's own clock for the phase, when it is this sweep's
        # (the trot's is `t_trot`, the sweep it started); else this sweep.
        t = float(getattr(stand, "t_trot", 0.0)) if name == "trot" else float(
            getattr(stand, "t_phase", 0.0))
        self._t_entry = t if now - 2.0 <= t <= now + 1e-6 else float(now)

    def poll(self, now: float, stand) -> str | None:
        """The character to press this sweep, or None."""
        self._see_phase(now, stand)
        if self.cancelled is not None:
            return None
        for ev in self.events:
            if (ev.phase != self._phase or not ev.wants(self._entry)
                    or self._entry in ev.outcome):
                continue
            if now - self._t_entry >= ev.t:
                ev.outcome[self._entry] = float(now)
                self.queue.extend(ev.keys)
                self.fired += 1
                self.last = ev
        if self.queue:
            return self.queue.pop(0)
        return None

    def cancel(self, why: str) -> str:
        """The operator took over: nothing more goes out."""
        if self.cancelled is None:
            self.cancelled = why
            self.queue = []
        return ("script CANCELLED (%s): %d of %d events had fired; the keys "
                "are yours" % (why, self.fired, len(self.events)))

    def describe_last(self) -> str:
        ev = self.last
        if ev is None:
            return ""
        return "%s %+.1f s: %s" % (ev.anchor, ev.t, ev.text)

    # -- the status line and the report ------------------------------------
    @property
    def done(self) -> bool:
        return all(ev.outcome for ev in self.events) or bool(
            self.cancelled is not None)

    def status(self, now: float, stand) -> str:
        if self.cancelled is not None:
            return "script: CANCELLED (%s)" % self.cancelled
        here = [ev for ev in self.events
                if ev.phase == self._phase and ev.wants(self._entry)
                and self._entry not in ev.outcome]
        if here:
            ev = min(here, key=lambda e: e.t)
            return ("script: next %s in %.1f s (%s %+.1f s)"
                    % (ev.text, max(0.0, ev.t - (now - self._t_entry)),
                       ev.anchor, ev.t))
        later = [ev for ev in self.events if not ev.outcome]
        if later:
            return "script: waiting for %s" % later[0].anchor
        return "script: done"

    def report(self) -> str:
        head = ("  script %s: %d events, %d fired, %d skipped%s"
                % (self.name, len(self.events), self.fired, self.skipped,
                   "" if self.cancelled is None else
                   ", CANCELLED (%s)" % self.cancelled))
        rows = []
        for ev in self.events:
            if not ev.outcome:
                how = ("never: %s not reached" % ev.anchor
                       if self.cancelled is None else "not reached")
            else:
                parts = []
                for entry, out in sorted(ev.outcome.items()):
                    tag = "" if ev.index is None else "[%d] " % entry
                    parts.append(tag + (out if isinstance(out, str)
                                        else "fired"))
                how = "; ".join(parts)
            rows.append("    %-10s %+8.1f s  %-34s %s"
                        % (ev.anchor, ev.t, ev.text, how))
        return "\n".join([head] + rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    script = KeyScript.load(argv[0])
    print("\n".join(script.banner()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
