"""URLeaderArm — one UR3e leader of the 747 rig, read over RTDE, optionally fed haptics.

The leaders run the vendor "Default program" (MotionGuide Live Control URCap, the
program recording_studio uses): force-guided **admittance** from the UR3e's own
F/T sensor. A person moves the handle; nothing commands the leader's joints. This
driver uses the same interface recording_studio does:

    READ   actual_q, actual_qd                     joints [rad], [rad/s]
           actual_digital_input_bits               bit 16 tool DI0 = grab (handle button),
                                                   bit 0 std DI0 = hold (leader frozen while high)
           tool_analog_input0                      trigger [V]; Scale's gripper command is V / 13
           output_double_register_24..29           leader flange wrench (flange frame, N / Nm)
           output_double_register_30               trigger / 13, as published by the program
           runtime_state, safety_mode, robot_mode, actual_TCP_force
    WRITE  input_double_register_24..29            follower flange wrench -> the program's
           (``haptic_feedback=True`` only)          contact damping (``use_contact_force_damping``):
                                                   it raises damping and clamps pushes into contact

Writing needs the leader's input registers, which recording_studio owns while its
"Start Robot Arms" is on (a second writer gets ``IN_USE``). Read-only use is always
safe alongside it.

Program lifecycle (``manage_program=True``): bring-up checks over REST (Remote
control, safety NORMAL, ``Default program`` loaded, not already PLAYING), seeds the
haptic registers with zeros, then Play; ``close()`` stops the program. On Play the
vendor leader zeroes its F/T sensor and may ``movej`` the elbow/wrist out of a
singular range, so keep hands off the handle while it starts.

Bus convention (same as the followers): gripper = **open fraction** (1 open,
0 closed). With the defaults the follower's register-31 command equals Scale's
``trigger / 13`` exactly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots_realtime.rig import stop as rigstop

log = logging.getLogger(__name__)

N_JOINTS = 6
RTDE_PORT = 30004
DEFAULT_PROGRAM = "Default program"
RUNTIME_PLAYING = 2
GRAB_BIT = 16  # tool DI0
HOLD_BIT = 0  # standard DI0
WRENCH_REGS = [f"output_double_register_{24 + i}" for i in range(6)]
TRIGGER_REG = "output_double_register_30"
HAPTIC_REGS = [f"input_double_register_{24 + i}" for i in range(6)]

_OUT_NAMES = [
    "robot_mode", "safety_mode", "runtime_state",
    "actual_q", "actual_qd", "actual_TCP_pose", "actual_TCP_force",
    "actual_digital_input_bits", "tool_analog_input0",
    *WRENCH_REGS, TRIGGER_REG,
]


class URLeaderError(RuntimeError):
    """Bring-up refused or the RTDE link failed."""


@dataclass
class LeaderState:
    t: float
    q: np.ndarray
    qd: np.ndarray
    grab: bool
    hold: bool
    trigger_volts: float
    trigger: float  # trigger_volts / trigger_divisor, Scale's gripper command
    flange_wrench: np.ndarray  # vendor program output; zeros unless PLAYING
    tcp_force: np.ndarray
    tcp_pose: np.ndarray
    robot_mode: int
    safety_mode: int
    runtime_state: int
    extra: dict


class URLeaderArm:
    """One UR3e leader.

    Args:
        ip: Leader controller IP (left 192.168.0.10, right 192.168.0.12).
        name: Log label.
        rate: RTDE output rate [Hz] (500 = recording_studio's).
        haptic_feedback: Open the input recipe for registers 24..29 and accept
            :meth:`write_haptic`. Off: read-only.
        manage_program: Play the leader program on start and stop it on close.
        allow_attach_playing: Accept a program that is already PLAYING when
            ``manage_program`` is on (off: refuse).
        trigger_divisor: Trigger volts per unit gripper command (vendor: 13).
        gripper_closed_cmd: Follower command meaning "fully closed" (0.92), used
            to express the trigger as an open fraction.
        haptic_limit: Per-component clamp on the written wrench [N, Nm].
        extra_outputs: Further RTDE output fields, reported in ``LeaderState.extra``.
    """

    def __init__(
        self,
        ip: str,
        name: str = "",
        *,
        rate: float = 500.0,
        haptic_feedback: bool = False,
        manage_program: bool = False,
        allow_attach_playing: bool = False,
        trigger_divisor: float = 13.0,
        gripper_closed_cmd: float = 0.92,
        haptic_limit: float = 60.0,
        extra_outputs: list[str] | tuple[str, ...] = (),
        rtde_factory: Any = None,
        rest: Any = None,
    ) -> None:
        self.ip = ip
        self.name = name or ip
        self.rate = float(rate)
        self.haptic_feedback = bool(haptic_feedback)
        self.manage_program = bool(manage_program)
        self.allow_attach_playing = bool(allow_attach_playing)
        self.trigger_divisor = float(trigger_divisor)
        self.gripper_closed_cmd = float(gripper_closed_cmd)
        self.haptic_limit = float(haptic_limit)
        self.extra_outputs = tuple(str(f) for f in extra_outputs)
        self._rtde_factory = rtde_factory
        self._rest = rest or _Rest(ip)
        self._lock = threading.Lock()
        self._con = None
        self._setp = None
        self._state: LeaderState | None = None
        self._haptic = np.zeros(6)
        self._played = False
        self._closed = False
        self._stop_evt = threading.Event()

        self._connect()
        if self.manage_program:
            self._bring_up()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"URLeaderArm-{self.name}")
        self._thread.start()
        self._wait_first_state()

    # ── connection ────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self._rtde_factory is not None:
            con = self._rtde_factory(self.ip, RTDE_PORT)
        else:
            import rtde.rtde as rtde

            con = rtde.RTDE(self.ip, RTDE_PORT)
        con.connect()
        con.get_controller_version()
        if not con.send_output_setup(_OUT_NAMES + list(self.extra_outputs), frequency=self.rate):
            con.disconnect()
            raise URLeaderError(f"[{self.name}] RTDE output setup rejected by {self.ip}")
        if self.haptic_feedback:
            setp = con.send_input_setup(HAPTIC_REGS, ["DOUBLE"] * 6)
            if setp is None:
                con.disconnect()
                raise URLeaderError(
                    f"[{self.name}] leader input registers 24-29 are IN USE on {self.ip} "
                    "(recording_studio 'Robot Arms' on, or another client); run without haptic_feedback "
                    "or turn the other writer off"
                )
            for n in HAPTIC_REGS:
                setattr(setp, n, 0.0)
            self._setp = setp
        if not con.send_start():
            con.disconnect()
            raise URLeaderError(f"[{self.name}] RTDE send_start refused by {self.ip}")
        self._con = con
        if self._setp is not None:
            con.send(self._setp)  # stale register values would change the leader's feel
        log.info("[%s] RTDE connected to %s (haptics=%s)", self.name, self.ip, self.haptic_feedback)

    def _bring_up(self) -> None:
        latch = rigstop.latch_active()
        if latch:
            raise URLeaderError(f"[{self.name}] stop latch is set ({latch.get('reason')}); run `rr-ur-stop --reset`")
        st = self._rest.status()
        problems = []
        if st.get("control_mode") != "REMOTE":
            problems.append(f"control mode {st.get('control_mode')} (needs REMOTE on the pendant)")
        if st.get("safety_mode") != "NORMAL":
            problems.append(f"safety mode {st.get('safety_mode')}")
        if st.get("loaded_program") != DEFAULT_PROGRAM:
            problems.append(f"loaded program {st.get('loaded_program')!r} (needs {DEFAULT_PROGRAM!r})")
        playing = st.get("program_state") == "PLAYING"
        if playing and not self.allow_attach_playing:
            problems.append("program already PLAYING (someone may be using the rig)")
        if problems:
            self._disconnect()
            raise URLeaderError(f"[{self.name}] refusing to start {self.ip}: " + "; ".join(problems))
        if not playing:
            log.warning("[%s] playing the leader program: hands off the handle (F/T zeroing)", self.name)
            reply = self._rest.play()
            log.info("[%s] REST play -> %s", self.name, reply)
            self._played = True

    def _receive(self) -> LeaderState:
        s = self._con.receive()
        if s is None:
            raise URLeaderError(f"[{self.name}] RTDE stream ended")
        bits = int(getattr(s, "actual_digital_input_bits", 0) or 0)
        volts = float(getattr(s, "tool_analog_input0", 0.0) or 0.0)
        runtime = int(s.runtime_state)
        wrench = np.array([float(getattr(s, n, 0.0) or 0.0) for n in WRENCH_REGS])
        return LeaderState(
            t=time.time(),
            q=np.asarray(s.actual_q, dtype=np.float64),
            qd=np.asarray(s.actual_qd, dtype=np.float64),
            grab=bool(bits >> GRAB_BIT & 1),
            hold=bool(bits >> HOLD_BIT & 1),
            trigger_volts=volts,
            trigger=volts / self.trigger_divisor,
            flange_wrench=wrench if runtime == RUNTIME_PLAYING else np.zeros(6),
            tcp_force=np.asarray(getattr(s, "actual_TCP_force", None) or [0.0] * 6, dtype=np.float64),
            tcp_pose=np.asarray(getattr(s, "actual_TCP_pose", None) or [0.0] * 6, dtype=np.float64),
            robot_mode=int(s.robot_mode),
            safety_mode=int(s.safety_mode),
            runtime_state=runtime,
            extra={f: getattr(s, f, None) for f in self.extra_outputs},
        )

    def _wait_first_state(self, timeout: float = 3.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._state is not None:
                    return
            time.sleep(0.01)
        raise URLeaderError(f"[{self.name}] no RTDE state within {timeout:.1f}s")

    def _loop(self) -> None:
        try:
            while not self._stop_evt.is_set():
                st = self._receive()
                with self._lock:
                    self._state = st
                    haptic = self._haptic.copy()
                if self._setp is not None:
                    for n, v in zip(HAPTIC_REGS, haptic):
                        setattr(self._setp, n, float(v))
                    self._con.send(self._setp)
        except Exception:
            if not self._stop_evt.is_set():
                log.exception("[%s] RTDE loop crashed", self.name)

    # ── API ───────────────────────────────────────────────────────────────

    def read(self) -> LeaderState:
        with self._lock:
            if self._state is None:
                raise URLeaderError(f"[{self.name}] no state yet")
            return self._state

    def open_fraction(self, st: LeaderState | None = None) -> float:
        """Trigger as the bus gripper value (1 open, 0 closed)."""
        st = st or self.read()
        return float(np.clip(1.0 - st.trigger / self.gripper_closed_cmd, 0.0, 1.0))

    def write_haptic(self, wrench) -> None:
        """Follower flange wrench to send to the leader program (needs ``haptic_feedback``)."""
        if self._setp is None:
            return
        w = np.asarray(wrench, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(w)):
            w = np.zeros(6)
        with self._lock:
            self._haptic = np.clip(w, -self.haptic_limit, self.haptic_limit)

    def get_observations(self) -> dict[str, Any]:
        st = self.read()
        return {
            "joint_pos": np.concatenate([st.q, [self.open_fraction(st)]]),
            "joint_vel": st.qd.copy(),
            "grab": np.array([int(st.grab)]),
            "hold": np.array([int(st.hold)]),
            "trigger_volts": np.array([st.trigger_volts]),
            "trigger": np.array([st.trigger]),
            "flange_wrench": st.flange_wrench.copy(),
            "tcp_force": st.tcp_force.copy(),
            "tcp_pose": st.tcp_pose.copy(),
            "safety_mode": np.array([st.safety_mode]),
            "runtime_state": np.array([st.runtime_state]),
            "state_t": np.array([st.t]),
            **{f"rtde_{k}": np.atleast_1d(np.asarray(v, dtype=np.float64))
               for k, v in st.extra.items() if v is not None},
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.write_haptic(np.zeros(6))
        time.sleep(3.0 / self.rate)  # let the loop write the zeros
        self._stop_evt.set()
        self._thread.join(timeout=1.0)
        if self._played:
            ok, msg = self._rest.stop()
            log.info("[%s] REST stop -> %s %s", self.name, ok, msg)
        self._disconnect()

    stop = close

    def _disconnect(self) -> None:
        con, self._con = self._con, None
        if con is not None:
            for call in (con.send_pause, con.disconnect):
                try:
                    call()
                except Exception:
                    pass


class _Rest:
    def __init__(self, ip: str) -> None:
        self.ip = ip

    def status(self) -> dict:
        return rigstop.arm_status(self.ip)

    def play(self) -> dict:
        reply = rigstop._rest(self.ip, "PUT", "/program/v1/state", {"action": "play"}, timeout=15.0)
        if reply.get("state") != "PLAYING":
            raise URLeaderError(f"{self.ip}: PLAY did not start the program: {reply}")
        return reply

    def stop(self) -> tuple[bool, str]:
        return rigstop.program_stop(self.ip)


__all__ = ["LeaderState", "URLeaderArm", "URLeaderError"]
