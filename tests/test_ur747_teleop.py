"""Hardware-free tests for the 747 teleop stack: the UR3e leader driver, the
leader agent, the follower's extra outputs and the session configs.

    .venv/bin/python -m pytest tests/test_ur747_teleop.py -q
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from robots_realtime.robots import ur_leader_arm as ula
from robots_realtime.robots.fake_ur import FakeURController, FakeURLeaderController

ROOT = Path(__file__).resolve().parents[1]
FAKE_RIG = str(ROOT / "configs/ur747_bimanual/rig_fake.yaml")


@pytest.fixture(autouse=True)
def _private_latch(tmp_path, monkeypatch):
    from robots_realtime.rig import stop as rigstop

    monkeypatch.setattr(rigstop, "LATCH_PATH", tmp_path / "latch", raising=False)
    monkeypatch.setenv("DIONYSUS_STOP_LATCH", str(tmp_path / "latch"))


def _leader(fake, **kw):
    return ula.URLeaderArm("fake", name="t_leader", rtde_factory=fake, rest=fake, **kw)


def test_leader_reads_scale_interface():
    fake = FakeURLeaderController(np.zeros(6))
    arm = _leader(fake)
    try:
        st = arm.read()
        assert st.q.shape == (6,) and st.grab and not st.hold
        assert st.trigger == pytest.approx(st.trigger_volts / 13.0)
        obs = arm.get_observations()
        assert obs["joint_pos"].shape == (7,)
        # Open fraction maps back to Scale's register-31 command: (1 - open) * 0.92 == volts / 13.
        cmd = (1.0 - obs["joint_pos"][6]) * 0.92
        assert cmd == pytest.approx(min(st.trigger, 0.92), abs=1e-9)
        assert not fake.playing  # read-only never plays
    finally:
        arm.close()


def test_leader_haptics_written_clamped_and_zeroed_on_close():
    fake = FakeURLeaderController(np.zeros(6))
    arm = _leader(fake, haptic_feedback=True, manage_program=True, haptic_limit=10.0)
    try:
        assert fake.playing
        arm.write_haptic([5.0, -50.0, 0.0, 0.1, 0.2, np.nan])
        time.sleep(0.02)
        assert np.allclose(fake.haptic, 0.0)  # a NaN wrench is replaced by zeros
        arm.write_haptic([5.0, -50.0, 0.0, 0.1, 0.2, 0.3])
        time.sleep(0.02)
        assert np.allclose(fake.haptic, [5.0, -10.0, 0.0, 0.1, 0.2, 0.3])
    finally:
        arm.close()
    assert np.allclose(fake.haptic, 0.0) and not fake.playing


def test_leader_refuses_when_program_already_playing():
    fake = FakeURLeaderController(np.zeros(6))
    fake.playing = True
    with pytest.raises(ula.URLeaderError, match="already PLAYING"):
        _leader(fake, manage_program=True)


def test_agent_forwards_leader_and_follower_wrench(tmp_path, monkeypatch):
    from robots_realtime.agents.teleoperation import ur_leader_agent as mod

    fake = FakeURLeaderController(np.full(6, 0.3))
    leader = _leader(fake, haptic_feedback=True)
    monkeypatch.setattr(mod, "_instantiate", lambda path: leader)
    agent = mod.URLeaderAgent("unused.yaml", robot_name="follower", haptic_scale=0.5)
    try:
        wrench = np.array([2.0, 0.0, -4.0, 0.0, 0.0, 0.2])
        action = agent.act({"follower": {"flange_wrench": wrench}})
        assert action["pos"].shape == (7,) and "_state" in action
        assert np.allclose(action["pos"][:6], leader.read().q, atol=0.05)
        time.sleep(0.02)
        assert np.allclose(fake.haptic, 0.5 * wrench)
        # No follower state yet -> zero feedback, never a stale wrench.
        agent.act({})
        time.sleep(0.02)
        assert np.allclose(fake.haptic, 0.0)
    finally:
        agent.close()


def test_agent_grab_gating(monkeypatch):
    from robots_realtime.agents.teleoperation import ur_leader_agent as mod

    leader = _leader(FakeURLeaderController(np.zeros(6)))
    monkeypatch.setattr(mod, "_instantiate", lambda path: leader)
    agent = mod.URLeaderAgent("unused.yaml", follow="grab")
    try:
        monkeypatch.setattr(leader, "get_observations",
                            lambda: {"joint_pos": np.zeros(7), "grab": np.array([0])})
        assert "pos" not in agent.act({})
        monkeypatch.setattr(leader, "get_observations",
                            lambda: {"joint_pos": np.zeros(7), "grab": np.array([1])})
        assert "pos" in agent.act({})
    finally:
        agent.close()


def test_follower_reports_program_outputs_and_extra_fields():
    from robots_realtime.rig.dionysus import load_rig
    from robots_realtime.robots.ur_register_arm import URRegisterArm

    rig = load_rig(FAKE_RIG)
    fake = FakeURController(rig.arms["left"].home_q)
    arm = URRegisterArm(rig.arms["left"].ip, name="left", rtde_factory=fake, rest=fake, rig_yaml=FAKE_RIG,
                        enable_motion=True, rate=500.0, extra_outputs=["target_q"])
    try:
        obs = arm.get_observations()
        assert obs["tcp_force"].shape == (6,)
        assert "flange_wrench" in obs  # vendor program, PLAYING
        assert "servo_reference" not in obs
    finally:
        arm.stop()


@pytest.mark.parametrize("path", sorted((ROOT / "configs/ur747_bimanual").glob("*.yaml")))
def test_session_configs_are_safe(path):
    cfg = yaml.safe_load(path.read_text())
    if "nodes" not in cfg:
        return  # rig.yaml / rig_fake.yaml
    for node in cfg["nodes"]:
        if node["type"] == "RobotNode":
            # The RobotNode default parks at all-zero joints on shutdown: never on a UR.
            assert node.get("shutdown_joint_pos", "missing") is None, node["name"]
            assert node.get("startup_joint_pos", "missing") is None, node["name"]
            assert (ROOT / node["robot_config"]).is_file()
        if node["type"] == "AgentNode":
            assert (ROOT / node["agent_kwargs"]["robot_config"]).is_file()
    if "fake" in path.name:
        for node in cfg["nodes"]:
            ref = node.get("robot_config") or node.get("agent_kwargs", {}).get("robot_config")
            if ref:
                assert "fake" in ref, f"{path.name} must only use simulated arms, got {ref}"


@pytest.mark.parametrize("path", sorted((ROOT / "robot_configs/ur747").glob("*readonly*.yaml")))
def test_readonly_profiles_never_write(path):
    cfg = yaml.safe_load(path.read_text())
    assert cfg.get("enable_motion", False) is False
    assert cfg.get("manage_program", False) is False
    assert cfg.get("haptic_feedback", False) is False


def test_follower_aligns_slowly_then_tracks_fast_and_realigns_after_stale():
    from robots_realtime.rig.dionysus import load_rig
    from robots_realtime.robots.ur_register_arm import URRegisterArm

    rig = load_rig(FAKE_RIG)
    q0 = rig.arms["left"].home_q
    fake = FakeURController(q0, rate=500.0)
    fake.track = q0.copy()
    arm = URRegisterArm(rig.arms["left"].ip, name="left", rtde_factory=fake, rest=fake, rig_yaml="",
                        enable_motion=True, rate=500.0, max_joint_vel=3.0, max_lead=0.35,
                        align_joint_vel=0.3, align_tolerance=0.05, cmd_timeout_s=0.1, speed_fraction=1.0)
    try:
        target = np.concatenate([q0 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]), [1.0]])
        sp_prev, t_prev, speeds = None, None, []
        t_start = time.time()
        t_end = t_start + 4.0
        while time.time() < t_end:
            arm.command_joint_pos(target)
            obs = arm.get_observations()
            if sp_prev is not None and not obs["aligned"][0]:
                dt = obs["state_t"][0] - t_prev
                if dt > 0:
                    speeds.append(np.abs(obs["setpoint"] - sp_prev).max() / dt)
            sp_prev, t_prev = obs["setpoint"].copy(), obs["state_t"][0]
            if obs["aligned"][0]:
                break
            time.sleep(0.004)
        took = time.time() - t_start
        assert arm.get_observations()["aligned"][0] == 1
        # 0.45 rad to cover at <= 0.3 rad/s: at least 1.5 s (3 rad/s uncapped would take 0.15 s).
        assert took > 1.4
        # setpoint and state_t are read from neighbouring ticks, so single samples jitter.
        assert speeds and np.percentile(speeds, 95) < 0.3 * 1.1
        # A stale gap re-arms the slow phase.
        time.sleep(0.25)
        assert arm.get_observations()["aligned"][0] == 0
    finally:
        arm.stop()
