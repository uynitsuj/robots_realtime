"""Hardware-free tests for the 747 sys-id trajectories and their playback agent.

    .venv/bin/python -m pytest tests/test_ur747_sysid.py -q
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from robots_realtime.rig.dionysus import load_rig
from robots_realtime.sysid import trajectories as st

ROOT = Path(__file__).resolve().parents[1]
RIG = str(ROOT / "configs/ur747_bimanual/rig.yaml")
FAKE_RIG = str(ROOT / "configs/ur747_bimanual/rig_fake.yaml")


@pytest.fixture(autouse=True)
def _private_latch(tmp_path, monkeypatch):
    from robots_realtime.rig import stop as rigstop

    monkeypatch.setattr(rigstop, "LATCH_PATH", tmp_path / "latch", raising=False)


def _setup(side):
    rig = load_rig(RIG)
    return rig, rig.arms[side], st.load_envelope(side), st.Limits()


@pytest.mark.parametrize("side", ["left", "right"])
def test_generated_trajectories_are_safe(side):
    rig, arm, env, lim = _setup(side)
    trajs = [
        st.static_holds(side, arm, env, lim, n_poses=12),
        st.friction_sweeps(side, arm, env, lim, joints=(0, 3), speeds=(0.05, 0.4)),
        st.excitation(side, arm, env, lim, duration_s=20.0),
    ]
    for tr in trajs:
        assert st.check(tr.q, arm, env, lim, rig.surface_z, stride=2) == [], tr.kind
        v = np.abs(np.diff(tr.q, axis=0)).max() / st.DT
        assert v <= lim.vmax + 1e-6
    excite = trajs[2]
    assert np.allclose(excite.q[0], excite.q[-1], atol=1e-6)  # starts and ends at home
    assert np.abs(np.diff(excite.q[:3], axis=0)).max() < 1e-5  # zero speed at the ends
    assert trajs[0].meta["n_poses"] >= 10


def test_validator_rejects_unsafe_paths():
    rig, arm, env, lim = _setup("left")
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    far = home.copy()
    far[1] = env.q_lo[1] - 0.3  # outside the observed shoulder range
    assert st.check_pose(far, arm, env, lim)
    fast = st.min_jerk(home, home + 0.3, vpeak=3.0, amax=50.0, min_t=0.05)
    assert any("speed" in m for m in st.check(fast, arm, env, lim))


def test_trajectory_roundtrip(tmp_path):
    rig, arm, env, lim = _setup("right")
    tr = st.excitation("right", arm, env, lim, duration_s=10.0, seed=3)
    back = st.Trajectory.load(tr.save(tmp_path / "x.npz"))
    assert back.arm == "right" and back.kind == "excite" and np.allclose(back.q, tr.q)
    assert back.phase_names == tr.phase_names


def _short_static(tmp_path, side="left"):
    rig = load_rig(FAKE_RIG)
    env, lim = st.load_envelope(side), st.Limits()
    tr = st.static_holds(side, rig.arms[side], env, lim, n_poses=2, settle_s=0.2, record_s=0.3)
    return str(tr.save(tmp_path / f"{side}_static.npz"))


def test_agent_plays_trajectory_on_fake_follower(tmp_path):
    from robots_realtime.agents.sysid.trajectory_agent import DONE, SysidTrajectoryAgent
    from robots_realtime.robots.fake_ur import FakeURRegisterArm

    path = _short_static(tmp_path)
    arm = FakeURRegisterArm("left", rig_yaml=FAKE_RIG, rate=500.0, max_joint_vel=1.0, max_lead=0.25,
                            speed_fraction=1.0, deadband=0.0)  # "rr servo": no vendor deadband
    agent = SysidTrajectoryAgent([path], "left", robot_name="f", settle_s=0.2, pause_s=0.2,
                                 rig_yaml=FAKE_RIG)
    records, errs = [], []
    try:
        t_end = time.time() + 60.0
        while time.time() < t_end and agent.state != DONE:
            obs = arm.get_observations()
            action = agent.act({"f": obs})
            assert agent.state_name != "aborted", agent.reason
            arm.command_joint_pos(action["pos"])
            records.append(action["_record"])
            if agent.state_name == "run":
                errs.append(np.abs(action["pos"][:6] - obs["joint_pos"][:6]).max())
            time.sleep(0.002)
    finally:
        arm.stop()
    assert agent.state == DONE
    # recording rises once (one episode) and falls once
    edges = np.flatnonzero(np.diff(np.asarray(records, dtype=int)) != 0)
    assert len(edges) == 2
    assert errs and max(errs) < 0.15


def test_agent_aborts_on_latch(tmp_path):
    from robots_realtime.agents.sysid.trajectory_agent import ABORTED, SysidTrajectoryAgent

    rig = load_rig(FAKE_RIG)
    q = np.clip(rig.arms["left"].home_q, st.load_envelope("left").q_lo, st.load_envelope("left").q_hi)
    agent = SysidTrajectoryAgent([_short_static(tmp_path)], "left", robot_name="f", rig_yaml=FAKE_RIG)
    obs = {"joint_pos": np.concatenate([q, [1.0]]), "latched": np.array([0]), "safety_mode": np.array([1])}
    agent.act({"f": obs})
    obs["latched"] = np.array([1])
    out = agent.act({"f": obs})
    assert agent.state == ABORTED and out["_record"] is False
    assert np.allclose(out["pos"][:6], q)


def test_agent_refuses_mismatched_arm(tmp_path):
    from robots_realtime.agents.sysid.trajectory_agent import SysidTrajectoryAgent

    with pytest.raises(ValueError, match="left arm"):
        SysidTrajectoryAgent([_short_static(tmp_path, "left")], "right", rig_yaml=FAKE_RIG)


def test_regressor_gravity_matches_energy_gradient():
    from robots_realtime.rig.dionysus import fk

    rig, arm, env, lim = _setup("left")
    q0 = arm.home_q + np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.4])
    Y = st.regressor(np.repeat(q0[None], 40, axis=0), arm, stride=1).reshape(-1, 6, 60)[5]
    theta = np.zeros(60)
    theta[50] = 1.0  # 1 kg point mass at the flange origin
    U = lambda q: 9.81 * (arm.world_T_base @ fk(q))[2, 3]  # noqa: E731
    grad = np.array([(U(q0 + 1e-6 * e) - U(q0 - 1e-6 * e)) / 2e-6 for e in np.eye(6)])
    assert np.allclose(Y @ theta, grad, atol=1e-6)


def test_widened_set_covers_recorded_region_and_excites():
    rig, arm, env, lim = _setup("right")
    static = st.static_holds("right", arm, env, lim, n_poses=20)
    excite = st.excitation_set("right", arm, env, lim, n_centers=2, candidates=2)
    assert len(excite) == 2
    for tr in [static, *excite]:
        assert st.check(tr.q, arm, env, lim, rig.surface_z, stride=4) == []
        assert np.allclose(tr.q[0], tr.q[-1], atol=1e-6)  # every trajectory starts and ends at home
    for tr in excite:
        qe = tr.q[tr.phase == tr.phase_names.index("excite")]
        assert np.abs(np.diff(qe, axis=0)).max() / st.DT >= 0.3
    report = st.coverage("right", arm, [static, *excite])
    assert report["tcp_cells_covered"] > 0.25 and report["excitation_regressor_rank"] >= 30
