"""Sway the trunk over planted feet: 1 in x, 2 in y, n nods, y shakes.

    V=/home/robot01/Documents/can_motor_control/.venv/bin/python
    $V -m hw.sway --fake --auto 1 --no-imu --track-stop 0   no robot
    $V -m hw.sway                                           on the robot
    $V -m hw.sway --sway-x 40 --sway-y 25 --sway-period 3
    $V -m hw.sway --sway-nod 4 --sway-shake 8

    limp -> settle -> crouch -> rise -> hold --key--> sway --key--> hold -> park

`hw.stand` from the normal crouch, SRB law, at `hw.stand`'s defaults, with
one more layer: `Sway`, below, through the hook `hw.stand.main` takes.  It
moves two targets and adds nothing else: the joint-space layer's, and the
SRB law's attitude setpoint (`law.BalanceLaw.offset_attitude`).

HOW THE TRUNK IS MOVED
    DOG5's body shift (`crawl_hw2.0/stand3_hold_hw.py`,
    `dog5_description/com_swing_test_hw.py`): the feet stay where they are
    and the TARGET moves -- the trunk moved by (d, R) sees its planted feet
    at R^T (f - d) in its own frame, through the IK:

        q_target = q_hold + IK(R^T (f - d)) - IK(f)

    f is the hold's foot sites at the commanded height, level trunk; q_hold
    is the angles latched on reaching HOLD, so (0, I) is exactly the hold and
    the first sweep of a sway is no step.  The rotation is about the trunk
    origin.

      1  x      d along x.  Only the joint layer holds xy (KP_XY is 0), so
      2  y      the joint layer is what moves the trunk.
      n  nod    R about y.  The SRB law's pitch setpoint moves with it, so
                the attitude loop drives the nod instead of fighting it.
      y  shake  R about z.  The yaw setpoint moves too, but the SRB yaw
                spring is off (KP_YAW 0, the damper only) -- the joint layer
                is what turns the trunk.

    The law's height loop is untouched throughout.  The tilt stop reads the
    MOVED setpoint while nodding.

    signal(t) = A env(t) sin(2 pi t / T).  env smoothsteps 0 -> 1 over the
    first period and back to 0 over one period after the stop, so neither
    end is a step, and the stop always lands on 0.

    Stiffness in x, y and yaw is the joint layer's `--kp-joint` (DOG5's 3.0
    by default).  `--no-joint-hold` leaves only the nod, which the SRB law
    can do alone; 1, 2 and y are refused.

KEYS
    1      from HOLD, sway in x (fore-aft).
    2      from HOLD, sway in y (left-right).
    n      from HOLD, nod (pitch).
    y      from HOLD, shake (yaw).
           Any of the four again: stop, back to 0 over one period.
    ENTER  REFUSED while swaying, or until the stop has come back to 0: the
           park is a joint ramp to the crouch and would drag a moved trunk.
    X      E-STOP.  From rise or hold it DROPS the robot.  Run supported.
"""
from __future__ import annotations

if __package__ in (None, ""):        # allow `python hw/sway.py` too
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "hw"

import numpy as np                   # noqa: E402

from sim import coordinates as C     # noqa: E402
from sim import params as P          # noqa: E402

from . import kinematics as HK       # noqa: E402
from . import stand as STAND         # noqa: E402
from .balance import controller as BCTRL  # noqa: E402
from .balance import posture as POSE  # noqa: E402
from .balance import swing as SWING  # noqa: E402

__all__ = ["main", "Sway", "MODES", "SWAY_X_MM", "SWAY_Y_MM", "SWAY_NOD_DEG",
           "SWAY_SHAKE_DEG", "SWAY_PERIOD_S"]

#: Amplitudes, trunk frame.  The nominal stance's feet are +-237 mm in x and
#: +-65 mm in y, so y has the far smaller margin.  A nod of 5 deg moves the
#: front and rear feet 21 mm in trunk z, opposite ways; a 6 deg shake moves
#: every foot 26 mm round the trunk.  Each is a flag.
SWAY_X_MM = 30.0
SWAY_Y_MM = 20.0
SWAY_NOD_DEG = 5.0
SWAY_SHAKE_DEG = 6.0

#: One sway cycle, seconds; the fade in and out are one period each.
SWAY_PERIOD_S = 2.0

#: key -> (name, kind, axis, needs the joint layer)
MODES = {
    "1": ("x", "shift", 0, True),
    "2": ("y", "shift", 1, True),
    "n": ("nod", "turn", 1, False),
    "y": ("shake", "turn", 2, True),
}


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class Sway:
    """The `hw.stand.main` hook: keys 1 / 2 / n / y, and the moving targets."""

    def __init__(self):
        self.amp = {"1": SWAY_X_MM * 1e-3, "2": SWAY_Y_MM * 1e-3,
                    "n": np.radians(SWAY_NOD_DEG),
                    "y": np.radians(SWAY_SHAKE_DEG)}
        self.period = SWAY_PERIOD_S
        self.mode: str | None = None
        self.t0 = 0.0
        self.t_stop: float | None = None
        self.env_stop = 0.0
        self.value = 0.0             # the signal now: m, or rad
        self.q_base = None           # the hold's q_hold, restored at the end
        self.f_b = None              # (4, 3) the hold's foot sites, trunk
        self.q_c = None              # IK at those sites
        self.ref0 = None             # what `status` measures from

    # -- hw.stand.main's hook ---------------------------------------------
    def add_arguments(self, ap) -> None:
        g = ap.add_argument_group("the sway",
                                  "from HOLD: 1 = x, 2 = y, n = nod, y = shake")
        g.add_argument("--sway-x", type=float, default=SWAY_X_MM,
                       metavar="MM", help="x amplitude, trunk frame")
        g.add_argument("--sway-y", type=float, default=SWAY_Y_MM,
                       metavar="MM", help="y amplitude, trunk frame")
        g.add_argument("--sway-nod", type=float, default=SWAY_NOD_DEG,
                       metavar="DEG", help="nod (pitch) amplitude")
        g.add_argument("--sway-shake", type=float, default=SWAY_SHAKE_DEG,
                       metavar="DEG", help="shake (yaw) amplitude")
        g.add_argument("--sway-period", type=float, default=SWAY_PERIOD_S,
                       metavar="SECONDS", help="one sway cycle; the fade in "
                                               "and out take one each")

    def configure(self, args) -> None:
        if not args.sway_period > 0.0:
            raise SystemExit("--sway-period must be positive")
        self.amp = {"1": args.sway_x * 1e-3, "2": args.sway_y * 1e-3,
                    "n": np.radians(args.sway_nod),
                    "y": np.radians(args.sway_shake)}
        self.period = float(args.sway_period)

    def banner(self, args) -> list:
        lines = ["SWAY in HOLD: 1 = x +-%.0f mm, 2 = y +-%.0f mm, n = nod "
                 "+-%.1f deg, y = shake +-%.1f deg; %.1f s a cycle; any sway "
                 "key stops it (fades to 0 over one cycle)"
                 % (1e3 * self.amp["1"], 1e3 * self.amp["2"],
                    np.degrees(self.amp["n"]), np.degrees(self.amp["y"]),
                    self.period)]
        if args.joint_hold:
            lines.append("     moves the joint layer's target (Kp %.1f N*m/rad "
                         "Kd %.2f N*m*s/rad) and, for n / y, the SRB attitude "
                         "setpoint; the height loop is untouched"
                         % (args.kp_joint, args.kd_joint))
        else:
            lines.append("     --no-joint-hold: NO joint layer -- only n (the "
                         "nod, the SRB law's own pitch) is allowed")
        return lines

    def key(self, ch: str, now: float, stand) -> str | None:
        ch = ch.lower()
        if ch not in MODES:
            return None
        if self.mode is not None:
            if self.t_stop is not None:
                return "sway %s already stopping" % MODES[self.mode][0]
            self.env_stop = self._envelope(now)
            self.t_stop = float(now)
            return ("sway %s: stopping, back to 0 over %.1f s"
                    % (MODES[self.mode][0], self.period))
        name, _, _, needs_joint = MODES[ch]
        if stand.phase_name != "hold":
            return "%s ignored: the sway starts from HOLD, not %s" % (
                ch, stand.phase_name)
        law = stand.balance
        joint = law.q_hold is not None and law.kp_joint > 0.0
        if needs_joint and not joint:
            return ("%s ignored: the %s moves the joint layer's target, and "
                    "the joint layer is off" % (ch, name))
        self.mode = ch
        self.t0 = float(now)
        self.t_stop = None
        self.value = 0.0
        if joint:
            self.q_base = law.q_hold.copy()
            self.f_b = SWING.rest_feet_b(law.h_lift, law.foot_xy)
            self.q_c = self._ik(self.f_b, C.unflat(self.q_base))
        else:
            self.q_base = self.f_b = self.q_c = None
        body = stand.body
        self.ref0 = None if body is None else (
            body.x_b[:, :2].mean(axis=0).copy(),
            np.array([body.roll, body.pitch, body.yaw]))
        unit = ("+-%.0f mm" % (1e3 * self.amp[ch]) if MODES[ch][1] == "shift"
                else "+-%.1f deg" % np.degrees(self.amp[ch]))
        return ("SWAY %s %s, %.1f s a cycle.  Any sway key stops."
                % (name, unit, self.period))

    def blocks_advance(self, stand) -> str | None:
        if self.mode is not None:
            return ("swaying -- press the sway key to stop, ENTER once it is "
                    "back at 0")
        return None

    def sweep(self, now: float, stand) -> None:
        if self.mode is None:
            return
        law = stand.balance
        if stand.phase_name != "hold":       # an E-stop path, never a demo
            self._finish(law)
            return
        if self.t_stop is not None and now - self.t_stop >= self.period:
            self._finish(law)
            stand.notice = "sway stopped -- back at the hold.  ENTER parks."
            return
        _, kind, axis, _ = MODES[self.mode]
        self.value = (self.amp[self.mode] * self._envelope(now)
                      * np.sin(2.0 * np.pi * (now - self.t0) / self.period))
        d = np.zeros(3)
        rpy = np.zeros(3)
        if kind == "shift":
            d[axis] = self.value
        else:
            rpy[axis] = self.value
            law.offset_attitude(*rpy)
        if self.q_c is not None:
            R = BCTRL.latched_attitude(*rpy)
            f_moved = (self.f_b - d) @ R          # R^T (f - d), row-wise
            law.q_hold = self.q_base + C.flat(self._ik(f_moved, self.q_c)
                                              - self.q_c)

    def status(self, now: float, stand) -> str:
        """The commanded signal, and what the robot MEASURES: the planted
        feet's mean xy in the trunk frame (moved the opposite way) for x / y,
        the IMU's pitch / yaw from where the sway began for n / y."""
        if self.mode is None:
            return "sway: off  (1 = x, 2 = y, n = nod, y = shake)"
        name, kind, axis, _ = MODES[self.mode]
        body = stand.body
        if kind == "shift":
            measured = ("" if self.ref0 is None or body is None else
                        "  measured %+5.1f mm" % (1e3 * -(
                            body.x_b[:, :2].mean(axis=0)[axis]
                            - self.ref0[0][axis])))
            said = "d %+5.1f mm" % (1e3 * self.value)
        else:
            now_rpy = (None if body is None
                       else np.array([body.roll, body.pitch, body.yaw]))
            measured = ("" if self.ref0 is None or now_rpy is None else
                        "  measured %+5.1f deg"
                        % np.degrees(now_rpy[axis] - self.ref0[1][axis]))
            said = "%+5.1f deg" % np.degrees(self.value)
        return ("sway %s  %s%s%s" % (name, said, measured,
                                     "  STOPPING" if self.t_stop is not None
                                     else ""))

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _ik(f_b, q_seed) -> np.ndarray:
        """(4, 3) joints for (4, 3) TRUNK-frame feet.  The hip frame is the
        trunk frame translated by HIP_OFFSET's xy, as `swing.rest_feet_b`."""
        p_hip = np.array(f_b, dtype=float)
        p_hip[:, :2] -= np.asarray(P.HIP_OFFSET, dtype=float)[:, :2]
        return HK.all_leg_ik(p_hip, q_seed=q_seed)

    def _envelope(self, now: float) -> float:
        if self.t_stop is None:
            return _smoothstep((now - self.t0) / self.period)
        return self.env_stop * (1.0 - _smoothstep((now - self.t_stop)
                                                  / self.period))

    def _finish(self, law) -> None:
        if self.q_base is not None:
            law.q_hold = self.q_base
        if MODES[self.mode][1] == "turn" and law.armed:
            law.offset_attitude(0.0, 0.0, 0.0)
        self.mode = None
        self.t_stop = None
        self.value = 0.0


def main(argv=None) -> int:
    """`hw.stand.main` from the normal crouch, SRB only, plus the sway."""
    return STAND.main(argv, crouch=POSE.NOMINAL, only_law="srb", hook=Sway())


if __name__ == "__main__":
    raise SystemExit(main())
