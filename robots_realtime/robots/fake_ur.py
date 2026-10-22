"""Simulated dionysus followers and leaders for hardware-free rehearsals.

``FakeURRegisterArm`` is the real :class:`URRegisterArm` — same bring-up, rate
limiting, floor, latch and hand-back code — talking to :class:`FakeURController`,
an in-process stand-in for one PolyScope X controller (RTDE + REST): it plays a
program that ramps to register targets, reports a gripper that follows its
command, and starts at the rig's home pose. Like the vendor program it has a
per-joint deadband on register *changes*: a joint whose register moved less
than ``deadband`` since the last write holds where it is (measured on the rig
2026-10-19: 0.048 mrad/cycle is ignored, 0.24 mrad/cycle is followed).

Use it through ``configs/ur747_bimanual/teleop_fake.yaml`` (with
``rig_fake.yaml``, whose IPs are not routable, and a separate stop latch via
``DIONYSUS_STOP_LATCH``) so a rehearsal can never reach the real arms.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from robots_realtime.robots.ur_register_arm import URRegisterArm


class FakeURController:
    """RTDE + REST stand-in for one follower."""

    def __init__(self, q0, rate: float = 125.0, slider_scale: bool = True, deadband: float = 1.5e-4) -> None:
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.track = self.q.copy()  # where each joint is servoing to
        self.deadband = float(deadband)
        self.regs = {i: 0.0 for i in range(24, 32)}
        self.regs.update({24 + i: float(v) for i, v in enumerate(self.q)})
        self.slider = (0, 1.0)
        self.playing = False
        self.grip = 0.0118
        self.period = 1.0 / rate
        self.slider_scale = slider_scale
        self._last = time.time()

    # RTDE
    def __call__(self, ip, port):
        return self

    def connect(self):
        pass

    def get_controller_version(self):
        return (10, 12, 1, 0)

    def send_output_setup(self, names, types=(), frequency=125):
        self.period = 1.0 / float(frequency)
        return True

    def send_input_setup(self, names, types=()):
        return SimpleNamespace(**{n: 0.0 for n in names})

    def send_start(self):
        return True

    def send_pause(self):
        return True

    def disconnect(self):
        pass

    def send(self, setp):
        for j in range(6):
            new = float(getattr(setp, f"input_double_register_{24 + j}"))
            self.track[j] = new if abs(new - self.regs[24 + j]) >= self.deadband else self.q[j]
        for i in range(24, 32):
            self.regs[i] = float(getattr(setp, f"input_double_register_{i}"))
        self.slider = (int(setp.speed_slider_mask), float(setp.speed_slider_fraction))

    def receive(self):
        now = time.time()
        sleep = self.period - (now - self._last)
        if sleep > 0:
            time.sleep(sleep)
        dt = time.time() - self._last
        self._last = time.time()
        if self.playing:
            tgt = self.track
            vmax = 1.0 * (self.slider[1] if self.slider_scale and self.slider[0] else 1.0)
            self.q = self.q + np.clip(tgt - self.q, -vmax * dt, vmax * dt)
            goal = self.regs[31] * (0.90 / 0.92)
            self.grip += float(np.clip(goal - self.grip, -2.0 * dt, 2.0 * dt))
        return SimpleNamespace(
            robot_mode=7, safety_mode=1, runtime_state=2 if self.playing else 1,
            actual_q=self.q.tolist(), actual_qd=[0.0] * 6, actual_TCP_pose=[0.0] * 6,
            tcp_offset=[-0.0021, 0.0003, 0.1558, 0.0, 0.0, 0.0], output_double_register_30=self.grip,
        )

    # REST
    def status(self):
        return {"control_mode": "REMOTE", "safety_mode": "NORMAL", "loaded_program": "Default program",
                "program_state": "PLAYING" if self.playing else "STOPPED"}

    def play(self):
        self.playing = True
        self.track = self.q.copy()
        return {"state": "PLAYING"}

    def stop(self):
        self.playing = False
        return True, "STOPPED"


def FakeURRegisterArm(name: str, rig_yaml: str | None = None, deadband: float = 1.5e-4,
                      **kwargs) -> URRegisterArm:
    """A URRegisterArm on a FakeURController at the rig's home pose.

    ``deadband`` emulates the vendor follower ignoring slow register changes; use 0
    to emulate a servoj program ("rr servo")."""
    from robots_realtime.rig.dionysus import load_rig

    rig = load_rig(rig_yaml)
    arm = rig.arms[name]
    fake = FakeURController(arm.home_q, deadband=deadband)
    kwargs.setdefault("enable_motion", True)
    return URRegisterArm(arm.ip, name=name, rtde_factory=fake, rest=fake, rig_yaml=rig_yaml, **kwargs)


class FakeURLeaderController:
    """RTDE + REST stand-in for one UR3e leader: a hand moving the handle.

    Joints sweep smoothly around ``q0``; the grab button is held; the trigger
    ramps between open and closed. Records the haptic registers written to it.
    """

    def __init__(self, q0, amplitude: float = 0.15, period_s: float = 6.0, rate: float = 500.0) -> None:
        self.q0 = np.asarray(q0, dtype=np.float64).copy()
        self.amplitude = float(amplitude)
        self.period_s = float(period_s)
        self.period = 1.0 / rate
        self.t0 = time.time()
        self._last = self.t0
        self.playing = False
        self.haptic = np.zeros(6)

    def __call__(self, ip, port):
        return self

    def connect(self):
        pass

    def get_controller_version(self):
        return (10, 12, 1, 0)

    def send_output_setup(self, names, types=(), frequency=500):
        self.period = 1.0 / float(frequency)
        return True

    def send_input_setup(self, names, types=()):
        return SimpleNamespace(**{n: 0.0 for n in names})

    def send_start(self):
        return True

    def send_pause(self):
        return True

    def disconnect(self):
        pass

    def send(self, setp):
        self.haptic = np.array([float(getattr(setp, f"input_double_register_{24 + i}")) for i in range(6)])

    def receive(self):
        sleep = self.period - (time.time() - self._last)
        if sleep > 0:
            time.sleep(sleep)
        self._last = time.time()
        phase = 2.0 * np.pi * (self._last - self.t0) / self.period_s
        q = self.q0 + self.amplitude * np.sin(phase + np.arange(6) * 0.7) * np.array([1, 0.5, 0.5, 1, 1, 1])
        volts = 6.0 * (1.0 - np.cos(phase))  # 0 .. 12 V
        return SimpleNamespace(
            robot_mode=7, safety_mode=1, runtime_state=2 if self.playing else 1,
            actual_q=q.tolist(), actual_qd=[0.0] * 6, actual_TCP_pose=[0.0] * 6, actual_TCP_force=[0.0] * 6,
            actual_digital_input_bits=1 << 16, tool_analog_input0=float(volts),
            **{f"output_double_register_{24 + i}": 0.0 for i in range(6)},
            output_double_register_30=float(volts) / 13.0,
        )

    def status(self):
        return {"control_mode": "REMOTE", "safety_mode": "NORMAL", "loaded_program": "Default program",
                "program_state": "PLAYING" if self.playing else "STOPPED"}

    def play(self):
        self.playing = True
        return {"state": "PLAYING"}

    def stop(self):
        self.playing = False
        return True, "STOPPED"


def FakeURLeaderArm(follower: str, rig_yaml: str | None = None, **kwargs):
    """A URLeaderArm on a FakeURLeaderController sweeping around ``follower``'s home pose."""
    from robots_realtime.rig.dionysus import load_rig
    from robots_realtime.robots.ur_leader_arm import URLeaderArm

    rig = load_rig(rig_yaml)
    fake = FakeURLeaderController(rig.arms[follower].home_q)
    kwargs.setdefault("manage_program", True)
    return URLeaderArm(f"fake-leader-{follower}", name=f"{follower}_leader", rtde_factory=fake, rest=fake, **kwargs)


__all__ = ["FakeURController", "FakeURLeaderArm", "FakeURLeaderController", "FakeURRegisterArm"]
