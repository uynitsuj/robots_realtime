"""Hardware-free tests for the dionysus rig layer: calibration, URRegisterArm's
safety logic against a fake UR controller, the stop latch, and the RGB-D recorder.

    .venv/bin/python -m pytest tests/test_dionysus_rig.py -q
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from robots_realtime.rig import dionysus
from robots_realtime.rig import stop as rigstop
from robots_realtime.robots import ur_register_arm as ura



@pytest.fixture(autouse=True)
def _latch(tmp_path, monkeypatch):
    monkeypatch.setattr(rigstop, "LATCH_PATH", tmp_path / "latch")
    yield


# ── calibration / kinematics ──────────────────────────────────────────────────


def test_rig_loads_and_home_points_down():
    rig = dionysus.load_rig()
    assert [a.name for a in rig.ordered_arms()] == ["right", "left"]
    for arm in rig.arms.values():
        T = arm.tcp_world(arm.home_q)
        assert T[2, 3] > 0.3  # above the table
        assert T[:3, 2] @ np.array([0, 0, -1.0]) > 0.99  # approach axis points down
    bases = [a.world_T_base[:3, 3] for a in rig.ordered_arms()]
    assert abs(np.linalg.norm(bases[0] - bases[1]) - 0.388) < 0.01
    # Optical frame: +z forward. The top camera looks down at the table.
    top = rig.cameras["top"].T
    assert top[:3, 2] @ np.array([0, 0, -1.0]) > 0.9


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = dionysus.quat_xyzw_to_matrix(q)
        w = dionysus.matrix_to_quat_wxyz(R)
        R2 = dionysus.quat_xyzw_to_matrix([w[1], w[2], w[3], w[0]])
        assert np.allclose(R, R2, atol=1e-9)


def test_wrist_camera_follows_arm():
    rig = dionysus.load_rig()
    q = {a.name: a.home_q for a in rig.arms.values()}
    T0 = rig.cameras["right_wrist"].world_pose(rig.arms, q)
    q2 = dict(q)
    q2["right"] = q["right"] + np.array([0.2, 0, 0, 0, 0, 0])
    T1 = rig.cameras["right_wrist"].world_pose(rig.arms, q2)
    assert np.linalg.norm(T0[:3, 3] - T1[:3, 3]) > 0.05
    # Hand-eye: the camera stays fixed relative to the flange.
    arm = rig.arms["right"]
    rel0 = np.linalg.inv(arm.flange_world(q["right"])) @ T0
    rel1 = np.linalg.inv(arm.flange_world(q2["right"])) @ T1
    assert np.allclose(rel0, rel1, atol=1e-9)


# ── fake UR controller ────────────────────────────────────────────────────────


class FakeUR:
    """RTDE + REST stand-in: a program that jumps to register targets, except a
    joint whose register changed less than ``deadband`` since the last write
    holds where it is (the vendor program does that; see fake_ur.py). With
    ``loaded="rr servo"`` it is that program instead: no deadband, but targets
    count only while the client echoes the session ``token``."""

    def __init__(self, q0, *, in_use=False, control_mode="REMOTE", program_state="STOPPED", deadband=1.5e-4,
                 loaded="Default program", token=4242):
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.track = self.q.copy()
        self.deadband = deadband
        self.loaded = loaded
        self.token = token
        self.token_echo = None
        self.regs = {i: 0.0 for i in range(24, 32)}
        self.regs.update({24 + i: float(v) for i, v in enumerate(self.q)})
        self.slider = (0, 0.0)
        self.playing = program_state == "PLAYING"
        self.in_use = in_use
        self.control_mode = control_mode
        self.safety = 1
        self.grip = 0.0118
        self.rest_calls: list[str] = []
        self.sent = 0

    # RTDE
    def __call__(self, ip, port):
        return self

    def connect(self):
        pass

    def get_controller_version(self):
        return (10, 12, 1, 0)

    def send_output_setup(self, names, types=(), frequency=125):
        return True

    def send_input_setup(self, names, types=()):
        if self.in_use:
            return None
        return SimpleNamespace(**{n: 0.0 for n in names})

    def send_start(self):
        return True

    def send_pause(self):
        return True

    def disconnect(self):
        pass

    def send(self, setp):
        self.sent += 1
        self.token_echo = getattr(setp, "input_int_register_24", None)
        for j in range(6):
            new = float(getattr(setp, f"input_double_register_{24 + j}"))
            if self.loaded == "rr servo":
                self.track[j] = new if self.token_echo == self.token else self.track[j]
            else:
                self.track[j] = new if abs(new - self.regs[24 + j]) >= self.deadband else self.q[j]
        for i in range(24, 32):
            self.regs[i] = float(getattr(setp, f"input_double_register_{i}"))
        self.slider = (int(setp.speed_slider_mask), float(setp.speed_slider_fraction))

    def receive(self):
        time.sleep(0.002)
        if self.playing:
            self.q = self.track.copy()
            self.grip = self.regs[31] * (0.90 / 0.92)
        return SimpleNamespace(
            robot_mode=7, safety_mode=self.safety, runtime_state=2 if self.playing else 1,
            actual_q=self.q.tolist(), actual_qd=[0.0] * 6, actual_TCP_pose=[0.0] * 6,
            tcp_offset=[-0.0021, 0.0003, 0.1746, 0, 0, 0], output_double_register_30=self.grip,
            output_int_register_26=self.token if self.playing else 0,
        )

    # REST
    def status(self):
        return {"control_mode": self.control_mode, "safety_mode": "NORMAL" if self.safety == 1 else "PROTECTIVE",
                "loaded_program": self.loaded, "program_state": "PLAYING" if self.playing else "STOPPED"}

    def play(self):
        self.rest_calls.append("play")
        self.playing = True
        self.track = self.q.copy()
        return {"state": "PLAYING"}

    def stop(self):
        self.rest_calls.append("stop")
        self.playing = False
        return True, "STOPPED"


def _arm(fake, **kw):
    rig = dionysus.load_rig()
    a = rig.arms["right"]
    kw.setdefault("rate", 250.0)
    return ura.URRegisterArm(a.ip, name="right", rtde_factory=fake, rest=fake, **kw)


def _wait(pred, timeout=2.0):
    t = time.time() + timeout
    while time.time() < t:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_read_only_never_writes():
    rig = dionysus.load_rig()
    fake = FakeUR(rig.arms["right"].home_q)
    arm = _arm(fake, enable_motion=False)
    obs = arm.get_observations()
    assert np.allclose(obs["joint_pos"][:6], rig.arms["right"].home_q)
    arm.command_joint_pos(np.zeros(7))
    time.sleep(0.05)
    arm.stop()
    assert fake.sent == 0 and fake.rest_calls == []


@pytest.mark.parametrize("kw,match", [
    ({"in_use": True}, "IN USE"),
    ({"control_mode": "LOCAL"}, "REMOTE"),
    ({"program_state": "PLAYING"}, "PLAYING"),
])
def test_bring_up_refusals(kw, match):
    rig = dionysus.load_rig()
    fake = FakeUR(rig.arms["right"].home_q, **kw)
    with pytest.raises(ura.URRegisterError, match=match):
        _arm(fake, enable_motion=True)
    assert "play" not in fake.rest_calls


def test_motion_is_rate_limited_and_handed_back():
    rig = dionysus.load_rig()
    q0 = rig.arms["right"].home_q
    fake = FakeUR(q0)
    arm = _arm(fake, enable_motion=True, max_joint_vel=0.25, max_lead=0.5)
    assert fake.rest_calls == ["play"]
    tgt = np.concatenate([q0 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05]), [0.0]])  # close gripper
    t0 = time.time()
    for _ in range(40):
        arm.command_joint_pos(tgt)
        time.sleep(0.01)
    elapsed = time.time() - t0
    moved = abs(fake.q[5] - q0[5])
    assert moved <= 0.25 * (elapsed + 0.05) + 1e-6  # never faster than the cap
    assert moved > 0.02
    assert fake.regs[31] > 0.8  # gripper commanded closed (vendor units)
    arm.stop()
    assert fake.rest_calls[-1] == "stop"
    assert fake.slider == (1, 0.30)  # slider restored on hand-back
    assert abs(fake.regs[30] - 0.007) < 1e-9


def test_program_must_be_the_expected_one():
    rig = dionysus.load_rig()
    fake = FakeUR(rig.arms["right"].home_q)  # Default program loaded
    with pytest.raises(ura.URRegisterError, match="needs 'rr servo'"):
        _arm(fake, enable_motion=True, program="rr servo")
    assert "play" not in fake.rest_calls


def test_rr_servo_gets_its_token_echoed():
    rig = dionysus.load_rig()
    fake = FakeUR(rig.arms["right"].home_q, loaded="rr servo", token=777)
    arm = _arm(fake, enable_motion=True, program="rr servo")
    assert _wait(lambda: fake.token_echo == 777)
    arm.stop()
    plain = FakeUR(rig.arms["right"].home_q)
    arm = _arm(plain, enable_motion=True)
    time.sleep(0.05)
    assert plain.token_echo is None  # the vendor program's recipe is unchanged
    arm.stop()


@pytest.mark.parametrize("program", [
    pytest.param("Default program", marks=pytest.mark.xfail(
        strict=True, reason="the vendor program ignores slow targets (URRegisterArm docstring)")),
    "rr servo",
])
def test_slow_targets_still_converge(program):
    """The vendor program ignores slowly changing registers: a joint whose target
    creeps (small joints in a long synchronized move) never moves. rr servo follows."""
    rig = dionysus.load_rig()
    q0 = rig.arms["right"].home_q
    fake = FakeUR(q0, loaded=program)
    arm = _arm(fake, enable_motion=True, max_joint_vel=0.15, program=program)
    goal = q0 + np.array([0.0, 0.0, -0.027, 0.0, 0.031, 0.0])
    t0 = time.time()
    while time.time() - t0 < 4.0:  # 0.03 rad over 4 s: ~0.06 mrad/tick, below the deadband
        a = min(1.0, (time.time() - t0) / 4.0)
        arm.command_joint_pos(np.r_[q0 + a * (goal - q0), 1.0])
        time.sleep(0.01)
    for _ in range(100):
        arm.command_joint_pos(np.r_[goal, 1.0])
        time.sleep(0.01)
    assert np.abs(fake.q - goal).max() < 0.0025
    held = fake.q.copy()
    for _ in range(50):  # within tolerance: registers freeze, the arm stays put
        arm.command_joint_pos(np.r_[goal, 1.0])
        time.sleep(0.01)
    assert np.abs(fake.q - held).max() < 1e-6
    arm.stop()


def test_latch_holds_and_stops_program():
    rig = dionysus.load_rig()
    q0 = rig.arms["right"].home_q
    fake = FakeUR(q0)
    arm = _arm(fake, enable_motion=True, max_lead=0.5)
    tgt = np.concatenate([q0 + 0.3, [1.0]])
    arm.command_joint_pos(tgt)
    time.sleep(0.05)
    rigstop.latch_set("test", "pytest")
    assert _wait(lambda: arm.latched_reason is not None)
    assert _wait(lambda: "stop" in fake.rest_calls)
    q_at = fake.q.copy()
    arm.command_joint_pos(tgt)  # ignored once latched
    time.sleep(0.1)
    assert np.allclose(fake.q, q_at, atol=1e-9) or not fake.playing
    arm.stop()


def test_stale_command_holds():
    rig = dionysus.load_rig()
    q0 = rig.arms["right"].home_q
    fake = FakeUR(q0)
    arm = _arm(fake, enable_motion=True, cmd_timeout_s=0.05, max_lead=0.5)
    arm.command_joint_pos(np.concatenate([q0 + np.array([0, 0, 0, 0, 0, 1.0]), [1.0]]))
    time.sleep(0.3)
    moved = abs(fake.q[5] - q0[5])
    assert moved < 0.25 * 0.05 + 0.02  # only moved while the command was fresh
    arm.stop()


def test_joint_limit_trips():
    rig = dionysus.load_rig()
    q0 = rig.arms["right"].home_q
    fake = FakeUR(q0)
    arm = _arm(fake, enable_motion=True, joint_lower=[-3.2] * 6, joint_upper=[3.2] * 6)
    bad = np.concatenate([q0, [1.0]])
    bad[1] = -5.0
    arm.command_joint_pos(bad)
    assert _wait(lambda: arm.latched_reason is not None)
    assert "joint limits" in arm.latched_reason
    arm.stop()


def test_floor_blocks_descending_below_table():
    rig = dionysus.load_rig()
    a = rig.arms["right"]
    q0 = a.home_q
    fake = FakeUR(q0)
    arm = _arm(fake, enable_motion=True, max_joint_vel=2.0, max_lead=1.0)
    # A shoulder/elbow move that would put the TCP under the table (the bases are
    # tilted 45 deg, so search rather than guess the sign).
    tgt = None
    rng = np.random.default_rng(1)
    for _ in range(5000):
        q = q0 + rng.uniform(-1.0, 1.0, size=6) * np.array([0.6, 1.5, 1.5, 1.0, 0.5, 0.0])
        if a.tcp_world(q)[2, 3] < rig.surface_z - 0.05:
            tgt = np.concatenate([q, [1.0]])
            break
    assert tgt is not None, "no joint move found that goes below the table"
    t_end = time.time() + 3.0
    while time.time() < t_end and arm.stats()["floor_holds"] < 20:
        arm.command_joint_pos(tgt)
        time.sleep(0.005)
    assert a.tcp_world(fake.q)[2, 3] >= rig.surface_z + 0.004
    assert arm.stats()["floor_holds"] > 0
    arm.stop()


def test_rgbd_writer(tmp_path):
    from robots_realtime.runtime.recording import RgbdWriter

    w = RgbdWriter(fps=10, chunk=4)
    w.open(str(tmp_path), "cam_top")
    K = np.eye(3)
    for i in range(10):
        w.write("rgb", time.time(), {
            "images": {"rgb": np.full((64, 96, 3), i * 10, dtype=np.uint8)},
            "depth_data": np.full((64, 96), 0.5, dtype=np.float32),
            "intrinsics": {"left": {"intrinsics_matrix": K}},
        })
    w.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "cam_top-images-rgb.mp4" in names and "cam_top-camera_info.json" in names
    chunks = sorted(tmp_path.glob("cam_top-depth-*.npz"))
    assert len(chunks) == 3
    d = np.load(chunks[0])
    assert d["depth"].dtype == np.uint16 and int(d["depth"][0, 0, 0]) == 500
