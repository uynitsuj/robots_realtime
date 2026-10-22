"""Simulated dionysus follower for hardware-free rehearsals.

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
            tcp_offset=[-0.0021, 0.0003, 0.1746, 0.0, 0.0, 0.0], output_double_register_30=self.grip,
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


def FakeURRegisterArm(name: str, rig_yaml: str | None = None, **kwargs) -> URRegisterArm:
    """A URRegisterArm on a FakeURController at the rig's home pose."""
    from robots_realtime.rig.dionysus import load_rig

    rig = load_rig(rig_yaml)
    arm = rig.arms[name]
    fake = FakeURController(arm.home_q)
    kwargs.setdefault("enable_motion", True)
    return URRegisterArm(arm.ip, name=name, rtde_factory=fake, rest=fake, rig_yaml=rig_yaml, **kwargs)


__all__ = ["FakeURController", "FakeURRegisterArm"]
