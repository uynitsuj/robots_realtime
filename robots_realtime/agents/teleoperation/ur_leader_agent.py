"""URLeaderAgent — a UR3e leader as the teleop input for one UR5e follower.

Reproduces recording_studio's ``leader_follower_position_teleop`` +
``wrench_teleop_feedback`` steps for one pair:

- follower target = leader ``actual_q`` 1:1 (UR3e and UR5e share joint order),
  gripper = trigger / 13 (as the bus open fraction, see
  :class:`~robots_realtime.robots.ur_leader_arm.URLeaderArm`);
- optional haptics: the follower's flange wrench (``flange_wrench`` in its
  ``joint_state``, published by the vendor follower program) is written to the
  leader's registers 24..29, scaled by ``haptic_scale``.

The leader's full state (grab / hold buttons, trigger, its own wrench) is
published as ``{node}/state`` and recorded with the session.

Gating (``follow``): ``always`` streams the leader every tick, as Scale's stack
does (the admittance leader only moves while its grab button is held anyway);
``grab`` sends targets only while grab is held, so the follower holds when the
handle is released.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.ur_leader_arm import N_JOINTS

log = logging.getLogger(__name__)


def _instantiate(config_path: str):
    from robots_realtime.runtime.environment.robot_node import _instantiate_from_target_yaml

    return _instantiate_from_target_yaml(config_path)


class URLeaderAgent(Agent):
    """Leader → follower command for one pair.

    Args:
        robot_config: ``_target_`` YAML of the leader (``URLeaderArm`` or a fake).
        robot_name: ``state_topics`` key of the follower's ``joint_state`` (haptics source).
        follow: ``always`` (Scale) or ``grab``.
        haptic_scale: Gain on the follower wrench written to the leader (0 disables
            the writes even when the leader opened its input registers).
        joint_offsets: Added to the leader joints before commanding [rad]
            (zero for the 747 rig, whose pairs share joint zeros).
    """

    def __init__(
        self,
        robot_config: str,
        robot_name: str = "follower",
        follow: str = "always",
        haptic_scale: float = 1.0,
        joint_offsets: list[float] | None = None,
    ) -> None:
        if follow not in ("always", "grab"):
            raise ValueError(f"follow must be 'always' or 'grab', got {follow!r}")
        self.robot_name = robot_name
        self.follow = follow
        self.haptic_scale = float(haptic_scale)
        self.joint_offsets = np.asarray(joint_offsets or [0.0] * N_JOINTS, dtype=np.float64)
        self.leader = _instantiate(robot_config)
        self._haptic_on = bool(getattr(self.leader, "haptic_feedback", False)) and self.haptic_scale > 0.0

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        leader_obs = self.leader.get_observations()
        if self._haptic_on:
            follower = obs.get(self.robot_name)
            wrench = None if follower is None else follower.get("flange_wrench")
            self.leader.write_haptic(np.zeros(6) if wrench is None else self.haptic_scale * np.asarray(wrench))
        action: dict[str, Any] = {"_state": leader_obs}
        if self.follow == "grab" and not int(leader_obs["grab"][0]):
            return action
        pos = np.asarray(leader_obs["joint_pos"], dtype=np.float64).copy()
        pos[:N_JOINTS] += self.joint_offsets
        action["pos"] = pos.astype(np.float32)
        return action

    def action_spec(self) -> dict:
        from dm_env.specs import Array

        return {self.robot_name: {"pos": Array(shape=(N_JOINTS + 1,), dtype=np.float32)}}

    def reset(self) -> None:
        pass

    def close(self) -> None:
        if hasattr(self.leader, "close"):
            self.leader.close()


__all__ = ["URLeaderAgent"]
