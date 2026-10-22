"""URRegisterArm — one UR5e follower of the dionysus rig, driven through RTDE registers.

The followers run PolyScope X with only RTDE (30004) and the REST Robot API (80)
open, so ``ur_rtde``'s control interface (script ports 30001-30003) cannot work.
Instead the vendor "Default program" (MotionGuide Live Control URCap) is played
on the arm, and it servos to whatever sits in its input registers:

    WRITE  input_double_register_24..29  joint targets [rad] (shoulder_pan..wrist_3)
           input_double_register_30      goal time [s]: linear ramp from the current
                                         setpoint to the new target over this long
           input_double_register_31      gripper command, vendor units 0 = open .. ~0.92
           speed_slider_mask/fraction    programmatic speed slider (scales all motion)
    READ   actual_q, actual_qd, actual_TCP_pose, robot_mode, safety_mode, runtime_state
           output_double_register_30     gripper position, vendor units

Verified on both followers by imit_ur5 (2026-08-04, re-verified 2026-08-10).

The program has **no velocity limit and no watchdog**, so this driver owns both:
a 125 Hz thread moves a reference toward the commanded target at most
``max_joint_vel`` rad/s, never lets it lead the measured joints by more than
``max_lead``, and holds when commands go stale.

**Known limitation — the program ignores slow targets.** Per joint, it only
servos while the register keeps moving fast enough, for long enough; otherwise
it holds wherever the arm is, however far away the register sits. Measured on
the left follower, 2026-10-19 (wrist_2, +0.03 rad ramps, registers = reference):

    slider 0.10: 0.25 s ramp followed (2 mrad short), 1 s followed (3 mrad short),
                 5 s (0.048 mrad/cycle) not at all
    slider 0.30: 0.25 s followed (1 mrad short), 1 s and 5 s not at all
    a ~6 mrad burst of fast register steps, then constant: not followed

So a joint whose part of a long synchronized move is small (wrist_2 +0.03 rad in
a 5 s transit) never moves, and every move ends a few mrad short; callers must
check where the arm actually got to. The cause is MotionGuide's soft joint
impedance (Kp 28 Nm/rad on the wrists, no integral) against joint friction. For
precise motion load ``rr servo`` on the pendant and drive it with
``program="rr servo"`` (:mod:`robots_realtime.rig.servo_program`): same
registers, a ``servoj`` position servo. Load ``Default program`` back on the
pendant afterwards; the vendor teleop expects it.

Every tick also checks the stop
latch (:mod:`robots_realtime.rig.stop`), the safety mode, the program run state
and a TCP/link floor above the table; any of those trips HOLD (setpoint =
``actual_q``) and latches until ``rr-ur-stop --reset``. Writing ``actual_q`` is a
true stop on this program: it ramps to where the arm already is.

Bring-up (``enable_motion=True``) follows the procedure proven on this rig:
REST pre-checks (Remote, Default program loaded and STOPPED, safety NORMAL), open
the input recipe (``IN_USE`` -> someone else owns the registers -> refuse), seed
regs 24-31 with the current state, Play over REST, then confirm PLAYING and that
the arm did not move. ``stop()`` hands the arm back the way the vendor setup
expects it: program stopped, neutral registers, the speed slider restored.

With ``enable_motion=False`` (the default) only an output recipe is opened:
nothing is ever written, so it is safe alongside live teleop. Commands are
validated and logged.

Joint/gripper convention on the bus (same as every robots_realtime robot):
``joint_pos`` = 6 joints + gripper **open fraction** (1 = open, 0 = closed).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from robots_realtime.rig import stop as rigstop

log = logging.getLogger(__name__)

N_JOINTS = 6
JOINT_CMD_BASE = 24
GOAL_TIME_REG = 30
GRIPPER_CMD_REG = 31
GRIPPER_STATE_REG = 30
RTDE_PORT = 30004
DEFAULT_PROGRAM = "Default program"
SERVO_PROGRAM = "rr servo"  # robots_realtime.rig.servo_program
SERVO_TOKEN_OUT = "output_int_register_26"  # rr servo publishes a per-run session token here
SERVO_TOKEN_IN = "input_int_register_24"  # and accepts targets only while a client echoes it here
SAFETY_OK = (1, 2)  # NORMAL, REDUCED
RUNTIME_PLAYING = 2

_OUT_NAMES = [
    "robot_mode", "safety_mode", "runtime_state",
    "actual_q", "actual_qd", "actual_TCP_pose", "tcp_offset", "actual_TCP_force",
    f"output_double_register_{GRIPPER_STATE_REG}",
]
# Program outputs 24..29: the vendor follower publishes its flange wrench there
# (flange frame, N / Nm); "rr servo" publishes the joint reference it tracks.
_OUT_PROGRAM_REGS = [f"output_double_register_{JOINT_CMD_BASE + i}" for i in range(N_JOINTS)]
_IN_REG_NAMES = [f"input_double_register_{i}" for i in range(JOINT_CMD_BASE, GRIPPER_CMD_REG + 1)]
_IN_NAMES = _IN_REG_NAMES + ["speed_slider_mask", "speed_slider_fraction"]
_IN_TYPES = ["DOUBLE"] * len(_IN_REG_NAMES) + ["UINT32", "DOUBLE"]


class URRegisterError(RuntimeError):
    """Bring-up refused or the link failed; the arm was not (or is no longer) driven."""


def _wrap_to_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class URRegisterArm:
    """One follower: 6 joints + gripper (7 DOF on the bus).

    Args:
        ip: Follower controller IP.
        name: Arm name in ``rig.yaml`` (``right`` / ``left``); also the log label.
        enable_motion: Open the input recipe, Play the program and stream targets.
            False (default) reads only.
        rate: Control/RTDE rate [Hz].
        goal_time: Register-30 ramp time [s]; default one control period.
        max_joint_vel: Setpoint speed cap [rad/s] per joint.
        max_lead: Max |setpoint - actual_q| per joint [rad] (no wind-up when the arm lags).
        speed_fraction: Speed slider written while driving (0..1).
        restore_speed_fraction: Speed slider written on hand-back (found value 0.30).
        cmd_timeout_s: A target older than this is replaced by "hold here".
        joint_lower / joint_upper: Hard limits; a target outside latches a stop.
        wrap_joints: Continuous joints wrapped to the branch nearest the setpoint.
        gripper_closed_cmd: Vendor command for "fully closed" (open fraction 0).
        gripper_open_state / gripper_closed_state: Vendor gripper readings at full
            open / empty full close, mapped to open fraction 1 / 0.
        program: The robot program that must be loaded (and STOPPED) for bring-up: the vendor
            ``Default program`` or ``rr servo``, loaded by a person on the pendant. The driver
            never loads programs itself; anything else is refused.
        rig_yaml: Rig model for the table floor (``None`` = default rig.yaml;
            ``""`` disables the floor check).
        play_on_start: Play the program over REST during bring-up (needs Remote mode).
        allow_attach_playing: Accept a program that is already PLAYING (off: refuse,
            someone may be using the rig).
        tcp_offset_tolerance: Max mismatch [m] between rig.yaml's TCP and RTDE
            ``tcp_offset`` before warning.
        extra_outputs: Further RTDE output fields to stream (e.g. ``target_q``,
            ``actual_current``, ``joint_control_output``, ``speed_scaling``,
            ``joint_temperatures``); each appears in the observations as
            ``rtde_<field>``. See UR's RTDE guide for the field list.
    """

    def __init__(
        self,
        ip: str,
        name: str = "",
        *,
        enable_motion: bool = False,
        rate: float = 125.0,
        goal_time: float | None = None,
        max_joint_vel: float = 0.25,
        max_lead: float = 0.05,
        speed_fraction: float = 0.10,
        restore_speed_fraction: float = 0.30,
        cmd_timeout_s: float = 0.3,
        joint_lower: list[float] | None = None,
        joint_upper: list[float] | None = None,
        wrap_joints: tuple[int, ...] | list[int] = (5,),
        gripper_closed_cmd: float = 0.92,
        gripper_open_state: float = 0.0118,
        gripper_closed_state: float = 0.90,
        rig_yaml: str | None = None,
        program: str = DEFAULT_PROGRAM,
        play_on_start: bool = True,
        allow_attach_playing: bool = False,
        tcp_offset_tolerance: float = 0.005,
        extra_outputs: list[str] | tuple[str, ...] = (),
        rtde_factory: Any = None,
        rest: Any = None,
    ) -> None:
        self.ip = ip
        self.name = name or ip
        self.enable_motion = bool(enable_motion)
        self.rate = float(rate)
        self.period = 1.0 / self.rate
        self.goal_time = float(goal_time) if goal_time is not None else self.period
        self.max_step = float(max_joint_vel) * self.period
        self.max_lead = float(max_lead)
        if not (0.0 < speed_fraction <= 1.0) or not (0.0 < restore_speed_fraction <= 1.0):
            raise ValueError("speed fractions must be in (0, 1]")
        self.speed_fraction = float(speed_fraction)
        self.restore_speed_fraction = float(restore_speed_fraction)
        self.cmd_timeout_s = float(cmd_timeout_s)
        lo = joint_lower if joint_lower is not None else [-2 * np.pi] * N_JOINTS
        hi = joint_upper if joint_upper is not None else [2 * np.pi] * N_JOINTS
        self.joint_lower = np.asarray(lo, dtype=np.float64).reshape(N_JOINTS)
        self.joint_upper = np.asarray(hi, dtype=np.float64).reshape(N_JOINTS)
        self.wrap_joints = tuple(int(j) for j in wrap_joints)
        self.gripper_closed_cmd = float(gripper_closed_cmd)
        self.gripper_open_state = float(gripper_open_state)
        self.gripper_closed_state = float(gripper_closed_state)
        self.program = str(program)
        self.play_on_start = bool(play_on_start)
        self.allow_attach_playing = bool(allow_attach_playing)
        self.tcp_offset_tolerance = float(tcp_offset_tolerance)
        self.extra_outputs = tuple(str(f) for f in extra_outputs)
        self._rtde_factory = rtde_factory
        self._rest = rest or _Rest(ip)

        # Floor model (world frame) from the rig, if available.
        self._floor = None
        if rig_yaml != "":
            try:
                from robots_realtime.rig.dionysus import load_rig

                rig = load_rig(rig_yaml or None)
                arm = rig.arms.get(self.name)
                if arm is not None:
                    self._floor = (
                        arm,
                        rig.surface_z + rig.clearance.get("tcp_above_surface", 0.005),
                        rig.surface_z + rig.clearance.get("links_above_surface", 0.04),
                    )
            except Exception as e:
                log.warning("[%s] no rig floor model (%s); floor check disabled", self.name, e)

        self._lock = threading.Lock()
        self._con = None
        self._setp = None
        self._state: dict[str, Any] | None = None
        self._target: np.ndarray | None = None  # 7: joints + open fraction
        self._target_t = 0.0
        self._setpoint: np.ndarray | None = None  # 6 joints actually written
        self._token = 0  # rr servo's session token, as last read
        self._gripper_cmd_raw = 0.0
        self._latched: str | None = None
        self._driving = False
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = {"ticks": 0, "clamped": 0, "floor_holds": 0, "stale_holds": 0}
        self._closed = False

        self._connect()
        if self.enable_motion:
            self._bring_up()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"URRegisterArm-{self.name}")
        self._thread.start()
        self._wait_first_state()

    # ── gripper units ─────────────────────────────────────────────────────

    def _open_fraction(self, raw: float) -> float:
        span = self.gripper_closed_state - self.gripper_open_state
        return float(np.clip((self.gripper_closed_state - raw) / span, 0.0, 1.0))

    def _raw_command(self, open_fraction: float) -> float:
        return float(np.clip((1.0 - float(open_fraction)) * self.gripper_closed_cmd, 0.0, self.gripper_closed_cmd))

    # ── connection / bring-up ─────────────────────────────────────────────

    def _connect(self) -> None:
        if self._rtde_factory is not None:
            con = self._rtde_factory(self.ip, RTDE_PORT)
        else:
            import rtde.rtde as rtde

            con = rtde.RTDE(self.ip, RTDE_PORT)
        con.connect()
        con.get_controller_version()
        servo = self.program == SERVO_PROGRAM
        outputs = _OUT_NAMES + _OUT_PROGRAM_REGS + ([SERVO_TOKEN_OUT] if servo else []) + list(self.extra_outputs)
        if not con.send_output_setup(outputs, frequency=self.rate):
            con.disconnect()
            raise URRegisterError(f"[{self.name}] RTDE output setup rejected by {self.ip}")
        if self.enable_motion:
            in_names, in_types = _IN_NAMES, _IN_TYPES
            if servo:
                in_names, in_types = in_names + [SERVO_TOKEN_IN], in_types + ["INT32"]
            setp = con.send_input_setup(in_names, in_types)
            if setp is None:
                con.disconnect()
                raise URRegisterError(
                    f"[{self.name}] input registers 24-31 / speed slider are IN USE on {self.ip} "
                    "(recording_studio 'Robot Arms' on, or another client) — refusing to drive"
                )
            for nm in _IN_NAMES:
                setattr(setp, nm, 0.0)
            setattr(setp, "speed_slider_mask", 1)
            if servo:
                setattr(setp, SERVO_TOKEN_IN, 0)
            self._setp = setp
        if not con.send_start():
            con.disconnect()
            raise URRegisterError(f"[{self.name}] RTDE send_start refused by {self.ip}")
        self._con = con
        log.info("[%s] RTDE connected to %s (motion=%s)", self.name, self.ip, self.enable_motion)

    def _receive(self) -> dict[str, Any]:
        s = self._con.receive()
        if s is None:
            raise URRegisterError(f"[{self.name}] RTDE stream ended")
        raw_grip = float(getattr(s, f"output_double_register_{GRIPPER_STATE_REG}"))
        self._token = int(getattr(s, SERVO_TOKEN_OUT, 0) or 0)  # echoed on every write (rr servo only)
        return {
            "t": time.time(),
            "robot_mode": int(s.robot_mode),
            "safety_mode": int(s.safety_mode),
            "runtime_state": int(s.runtime_state),
            "q": np.asarray(s.actual_q, dtype=np.float64),
            "qd": np.asarray(s.actual_qd, dtype=np.float64),
            "tcp_pose": np.asarray(s.actual_TCP_pose, dtype=np.float64),
            "tcp_offset": np.asarray(s.tcp_offset, dtype=np.float64),
            "gripper_raw": raw_grip,
            "tcp_force": np.asarray(getattr(s, "actual_TCP_force", None) or [0.0] * 6, dtype=np.float64),
            "program_out": np.array([float(getattr(s, n, 0.0) or 0.0) for n in _OUT_PROGRAM_REGS]),
            "extra": {f: getattr(s, f, None) for f in self.extra_outputs},
        }

    def _write(self, joints: np.ndarray, gripper_raw: float, goal_time: float | None = None,
               slider: float | None = None) -> None:
        if self._setp is None:
            return
        for i in range(N_JOINTS):
            setattr(self._setp, f"input_double_register_{JOINT_CMD_BASE + i}", float(joints[i]))
        setattr(self._setp, f"input_double_register_{GOAL_TIME_REG}", float(goal_time or self.goal_time))
        setattr(self._setp, f"input_double_register_{GRIPPER_CMD_REG}", float(gripper_raw))
        setattr(self._setp, "speed_slider_mask", 1)
        setattr(self._setp, "speed_slider_fraction", float(slider if slider is not None else self.speed_fraction))
        if self.program == SERVO_PROGRAM:
            setattr(self._setp, SERVO_TOKEN_IN, self._token)
        self._con.send(self._setp)

    def _bring_up(self) -> None:
        latch = rigstop.latch_active()
        if latch:
            raise URRegisterError(f"[{self.name}] stop latch is set ({latch.get('reason')}); run `rr-ur-stop --reset`")
        st = self._rest.status()
        problems = []
        if st.get("control_mode") != "REMOTE":
            problems.append(f"control mode {st.get('control_mode')} (needs REMOTE on the pendant)")
        if st.get("safety_mode") != "NORMAL":
            problems.append(f"safety mode {st.get('safety_mode')}")
        if st.get("loaded_program") != self.program:
            problems.append(f"loaded program {st.get('loaded_program')!r} (needs {self.program!r})")
        playing = st.get("program_state") == "PLAYING"
        if playing and not self.allow_attach_playing:
            problems.append("program already PLAYING (someone may be using the rig)")
        if problems:
            self._disconnect()
            raise URRegisterError(f"[{self.name}] refusing to drive {self.ip}: " + "; ".join(problems))

        s = self._receive()
        self._check_tcp_offset(s)
        # Seed: the program's first target must be where the arm already is.
        for _ in range(5):
            self._write(s["q"], s["gripper_raw"], goal_time=self.goal_time)
            s = self._receive()
        q0 = s["q"].copy()
        self._setpoint = q0.copy()
        self._gripper_cmd_raw = s["gripper_raw"]
        self._target = np.concatenate([q0, [self._open_fraction(s["gripper_raw"])]])
        self._target_t = time.time()

        if not playing:
            if not self.play_on_start:
                raise URRegisterError(f"[{self.name}] program STOPPED and play_on_start=False")
            reply = self._rest.play()
            log.info("[%s] REST play -> %s", self.name, reply)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            self._write(q0, self._gripper_cmd_raw)
            s = self._receive()
            if s["runtime_state"] == RUNTIME_PLAYING:
                break
        else:
            self._rest.stop()
            self._disconnect()
            raise URRegisterError(f"[{self.name}] program did not reach PLAYING (runtime_state={s['runtime_state']})")
        # Did Play move the arm? (stale registers would show up as a jump here)
        t_end = time.time() + 0.3
        while time.time() < t_end:
            self._write(q0, self._gripper_cmd_raw)
            s = self._receive()
        drift = float(np.abs(s["q"] - q0).max())
        if drift > 0.005:
            rigstop.latch_set(f"{self.name}: arm moved {drift:.4f} rad on Play", f"URRegisterArm[{self.name}]")
            self._rest.stop()
            raise URRegisterError(f"[{self.name}] arm drifted {drift:.4f} rad after Play — stopped and latched")
        self._driving = True
        log.warning("[%s] DRIVING %s: speed slider %.2f, vel cap %.2f rad/s", self.name, self.ip,
                    self.speed_fraction, self.max_step * self.rate)

    def _check_tcp_offset(self, s: dict) -> None:
        if self._floor is None:
            return
        arm = self._floor[0]
        ctrl = s["tcp_offset"][:3]
        if np.linalg.norm(ctrl) < 1e-6:
            return
        diff = float(np.linalg.norm(ctrl - arm.tcp_offset))
        if diff > self.tcp_offset_tolerance:
            log.warning("[%s] rig.yaml tcp_offset %s differs from controller %s by %.1f mm",
                        self.name, arm.tcp_offset.round(4).tolist(), ctrl.round(4).tolist(), diff * 1e3)

    def _wait_first_state(self, timeout: float = 3.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._state is not None:
                    return
            time.sleep(0.01)
        raise URRegisterError(f"[{self.name}] no RTDE state within {timeout:.1f}s")

    # ── control loop ──────────────────────────────────────────────────────

    def _trip(self, reason: str) -> None:
        """HOLD + latch. Called from the loop thread."""
        if self._latched is None:
            self._latched = reason
            log.error("[%s] STOP: %s", self.name, reason)
            rigstop.latch_set(reason, f"URRegisterArm[{self.name}]")
            if self._driving:
                threading.Thread(target=self._rest.stop, daemon=True).start()

    def _floor_ok(self, q: np.ndarray, ref: np.ndarray) -> bool:
        """True if ``q`` keeps TCP/links above the table, or is no lower than ``ref``."""
        if self._floor is None:
            return True
        arm, tcp_floor, link_floor = self._floor

        def margin(qq):
            tcp_z = arm.tcp_world(qq)[2, 3]
            link_z = arm.link_points_world(qq)[2:-1, 2].min()  # skip base + shoulder, TCP separate
            return min(tcp_z - tcp_floor, link_z - link_floor)

        m = margin(q)
        return m >= 0.0 or m >= margin(ref) - 1e-6

    def _next_setpoint(self, s: dict) -> np.ndarray:
        q = s["q"]
        sp = self._setpoint if self._setpoint is not None else q.copy()
        tgt = self._target
        now = time.time()
        if self._latched is not None or tgt is None:
            return q.copy()  # HOLD where the arm is
        if now - self._target_t > self.cmd_timeout_s:
            self._stats["stale_holds"] += 1
            return sp.copy()
        goal = tgt[:N_JOINTS].copy()
        for j in self.wrap_joints:
            goal[j] = sp[j] + _wrap_to_pi(goal[j] - sp[j])
        if np.any(goal < self.joint_lower) or np.any(goal > self.joint_upper):
            self._trip(f"{self.name}: target outside joint limits: {np.round(goal, 3).tolist()}")
            return q.copy()
        step = np.clip(goal - sp, -self.max_step, self.max_step)
        if np.any(np.abs(step) < np.abs(goal - sp) - 1e-12):
            self._stats["clamped"] += 1
        nxt = sp + step
        nxt = np.clip(nxt, q - self.max_lead, q + self.max_lead)
        if not self._floor_ok(nxt, sp):
            self._stats["floor_holds"] += 1
            return sp.copy()
        return nxt

    def _loop(self) -> None:
        try:
            while not self._stop_evt.is_set():
                s = self._receive()
                with self._lock:
                    self._state = s
                if not self.enable_motion:
                    continue
                latch = rigstop.latch_active()  # every tick: one read on tmpfs
                if latch and self._latched is None:
                    self._trip(f"latch: {latch.get('reason')}")
                if s["safety_mode"] not in SAFETY_OK:
                    self._trip(f"{self.name}: safety_mode={s['safety_mode']}")
                if self._driving and s["runtime_state"] != RUNTIME_PLAYING and self._latched is None:
                    self._trip(f"{self.name}: program no longer PLAYING (runtime_state={s['runtime_state']})")
                with self._lock:
                    nxt = self._next_setpoint(s)
                    if self._latched is None and self._target is not None:
                        self._gripper_cmd_raw = self._raw_command(self._target[N_JOINTS])
                    elif self._latched is not None:
                        self._gripper_cmd_raw = s["gripper_raw"]
                    self._setpoint = nxt
                    self._stats["ticks"] += 1
                self._write(nxt, self._gripper_cmd_raw)
        except Exception as e:
            log.exception("[%s] control loop crashed", self.name)
            try:
                self._trip(f"{self.name}: control loop crashed: {e}")
                with self._lock:
                    s = self._state
                if s is not None:
                    self._write(s["q"], s["gripper_raw"])
            except Exception:
                pass

    # ── robot protocol ────────────────────────────────────────────────────

    def num_dofs(self) -> int:
        return N_JOINTS + 1

    def get_joint_pos(self) -> np.ndarray:
        with self._lock:
            s = self._state
        if s is None:
            return np.zeros(N_JOINTS + 1)
        return np.concatenate([s["q"], [self._open_fraction(s["gripper_raw"])]])

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        cmd = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        if cmd.shape[0] != N_JOINTS + 1 or not np.all(np.isfinite(cmd)):
            with self._lock:
                self._trip(f"{self.name}: malformed command {cmd.tolist()}")
            return
        if not self.enable_motion:
            if self._stats["ticks"] % 125 == 0:
                log.info("[%s] DRY-RUN cmd %s", self.name, np.round(cmd, 4).tolist())
            self._stats["ticks"] += 1
            return
        with self._lock:
            if self._latched is not None:
                return
            self._target = cmd
            self._target_t = time.time()

    def get_observations(self) -> dict[str, Any]:
        with self._lock:
            s = self._state
            setpoint = None if self._setpoint is None else self._setpoint.copy()
            latched = self._latched
        if s is None:
            return {"joint_pos": np.zeros(N_JOINTS + 1)}
        grip = self._open_fraction(s["gripper_raw"])
        return {
            "joint_pos": np.concatenate([s["q"], [grip]]),
            "joint_vel": s["qd"].copy(),
            "tcp_pose": s["tcp_pose"].copy(),  # controller base frame, rotvec
            "tcp_offset": s["tcp_offset"].copy(),
            "gripper_pos": np.array([grip]),
            "gripper_raw": np.array([s["gripper_raw"]]),
            "setpoint": setpoint if setpoint is not None else s["q"].copy(),
            "safety_mode": np.array([s["safety_mode"]]),
            "runtime_state": np.array([s["runtime_state"]]),
            "driving": np.array([1 if (self._driving and latched is None) else 0]),
            "latched": np.array([0 if latched is None else 1]),
            "state_t": np.array([s["t"]]),
            "tcp_force": s["tcp_force"].copy(),  # RTDE actual_TCP_force: TCP wrench, base frame
            **self._program_outputs(s),
            **{f"rtde_{k}": np.atleast_1d(np.asarray(v, dtype=np.float64))
               for k, v in s["extra"].items() if v is not None},
        }

    def _program_outputs(self, s: dict) -> dict[str, np.ndarray]:
        """Program registers 24..29, named for the program that writes them.

        Output registers keep their last value after a program stops, so they
        are only reported while it is PLAYING.
        """
        if s["runtime_state"] != RUNTIME_PLAYING:
            return {}
        if self.program == SERVO_PROGRAM:
            return {"servo_reference": s["program_out"].copy()}
        return {"flange_wrench": s["program_out"].copy()}  # vendor follower: flange frame

    @property
    def latched_reason(self) -> str | None:
        return self._latched

    def stats(self) -> dict:
        return dict(self._stats)

    # ── shutdown / hand-back ──────────────────────────────────────────────

    def stop(self) -> None:
        """HOLD, stop the program, write neutral registers, restore the slider."""
        if self._closed:
            return
        self._closed = True
        was_driving = self._driving
        if was_driving and self._latched is None:
            self._latched = "driver stop()"
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if self.enable_motion and self._con is not None:
                s = self._receive()
                self._write(s["q"], s["gripper_raw"])  # HOLD first
                if was_driving:
                    ok, msg = self._rest.stop()
                    log.info("[%s] REST stop -> %s %s", self.name, ok, msg)
                for _ in range(3):
                    s = self._receive()
                    self._write(s["q"], s["gripper_raw"], goal_time=0.007, slider=self.restore_speed_fraction)
                log.info("[%s] handed back: neutral registers, speed slider %.2f", self.name,
                         self.restore_speed_fraction)
        except Exception:
            log.exception("[%s] hand-back failed — check the arm (rr-ur-stop --status)", self.name)
        finally:
            self._driving = False
            self._disconnect()
            log.info("[%s] stats %s", self.name, self._stats)

    close = stop

    def _disconnect(self) -> None:
        con, self._con = self._con, None
        if con is not None:
            try:
                con.send_pause()
            except Exception:
                pass
            try:
                con.disconnect()
            except Exception:
                pass


class _Rest:
    """Minimal REST client (status / play / stop) over :mod:`robots_realtime.rig.stop`."""

    def __init__(self, ip: str) -> None:
        self.ip = ip

    def status(self) -> dict:
        return rigstop.arm_status(self.ip)

    def play(self) -> dict:
        reply = rigstop._rest(self.ip, "PUT", "/program/v1/state", {"action": "play"}, timeout=15.0)
        if reply.get("state") != "PLAYING":
            raise URRegisterError(f"{self.ip}: PLAY did not start the program: {reply}")
        return reply

    def stop(self) -> tuple[bool, str]:
        return rigstop.program_stop(self.ip)


def load_from_rig(arm: str, rig_yaml: str | Path | None = None, **overrides: Any) -> URRegisterArm:
    """Build a driver for ``arm`` (``right``/``left``) from ``rig.yaml``."""
    from robots_realtime.rig.dionysus import load_rig

    rig = load_rig(rig_yaml)
    a = rig.arms[arm]
    kwargs: dict[str, Any] = {
        "name": arm,
        "joint_lower": a.joint_lower.tolist(),
        "joint_upper": a.joint_upper.tolist(),
        "rig_yaml": str(rig_yaml) if rig_yaml else None,
    }
    kwargs.update(overrides)
    return URRegisterArm(a.ip, **kwargs)


__all__ = ["URRegisterArm", "URRegisterError", "load_from_rig"]
