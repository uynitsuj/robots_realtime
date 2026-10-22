"""AgentNode — node wrapper for any Agent.

Bridges the Agent protocol (act / reset / close) onto the ZMQ bus.

Supports three loop modes:
    flat_out          — runs as fast as possible; use for hardware-paced agents
                        (e.g. GelloLeaderAgent paced by serial read)
    fixed_rate        — polls at poll_freq Hz (e.g. viser IK at 100 Hz)
    subscriber_driven — blocks on incoming observations; use for learned policies

Agent can be injected at construction (programmatic usage) or instantiated
from a dotted class path in YAML (via agent_class / agent_kwargs).

Action format (returned by agent.act()):
    {"pos": array}                         — single arm, published as joint_pos
    {"left": {"pos": array}, ...}          — multi arm; single non-_ key → joint_pos,
                                             multiple keys → {key}_pos each
    arm_key set                            — extract action[arm_key]["pos"] → joint_pos
    "_record" key                          — forwarded as record signal
    "_state" key                           — published (and recorded) as {name}/state

Published topics:
    ``{name}/joint_pos``     — single-arm command
    ``{name}/{key}_pos``     — per-arm commands (multi-arm policies)
    ``{name}/record``        — forwarded record signal

Subscribed topics: state_topics.values() + image_topics.values()
"""

from __future__ import annotations

import importlib
import time

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole


class AgentNode(Node):
    """Wraps any Agent and bridges it onto the ZMQ bus.

    Args:
        agent:           Pre-built agent. If None, built from agent_class in setup().
        name:            Node name on the bus.
        agent_class:     Dotted import path, e.g.
                         "robots_realtime.agents.teleoperation.gello_leader_agent:GelloLeaderAgent".
                         Used when loading from YAML.
        agent_kwargs:    Keyword arguments forwarded to agent_class().
        loop_mode:       "flat_out" | "fixed_rate" | "subscriber_driven".
        poll_freq:       Hz for fixed_rate mode (or timeout in subscriber_driven).
        publish_freq:    Optional ZMQ send rate cap (Hz).
        state_topics:    {obs_key: bus_topic} — joint state inputs.
        image_topics:    {obs_key: bus_topic} — image inputs.
        arm_key:         If set, extract action[arm_key]["pos"] and publish as joint_pos.
                         Useful for agents that always return a multi-arm dict but are
                         deployed per-arm (e.g. GelloLeaderAgent).
        normalize_gripper: If True, map the last element of pos from raw degrees to [0,1].
        gripper_open_deg:  Raw degrees corresponding to gripper fully open (1.0).
        gripper_closed_deg: Raw degrees corresponding to gripper fully closed (0.0).
        writer:          Optional Writer injected at construction for recording.
    """

    role = NodeRole.CONTROLLER
    published_topics: list[str] = ["joint_pos", "record", "state"]

    def __init__(
        self,
        agent=None,
        name: str = "agent",
        agent_class: str | None = None,
        agent_kwargs: dict | None = None,
        loop_mode: str = "subscriber_driven",
        poll_freq: float | None = None,
        publish_freq: float | None = None,
        state_topics: dict[str, str] | None = None,
        image_topics: dict[str, str] | None = None,
        arm_key: str | None = None,
        normalize_gripper: bool = False,
        gripper_open_deg: float = 85.0,
        gripper_closed_deg: float = 5.0,
        writer=None,
        **kwargs,
    ) -> None:
        self._state_topics = state_topics or {}
        self._image_topics = image_topics or {}
        self.subscribed_topics = (
            list(self._state_topics.values()) + list(self._image_topics.values())
        )

        if loop_mode == "subscriber_driven":
            self.subscriber_driven = True
            self.poll_freq = poll_freq
        elif loop_mode == "fixed_rate":
            self.subscriber_driven = False
            self.poll_freq = poll_freq
        elif loop_mode == "flat_out":
            self.subscriber_driven = False
            self.poll_freq = None
        else:
            raise ValueError(f"Unknown loop_mode: {loop_mode!r}")

        if publish_freq is not None:
            self.publish_freq = publish_freq

        super().__init__(name=name, writer=writer, **kwargs)

        self._agent = agent
        self._agent_class = agent_class
        self._agent_kwargs = agent_kwargs or {}
        self._arm_key = arm_key
        self._normalize_gripper = normalize_gripper
        self._gripper_open_deg = gripper_open_deg
        self._gripper_closed_deg = gripper_closed_deg

    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self._agent is None:
            if self._agent_class is None:
                raise RuntimeError(
                    f"[{self.name}] AgentNode requires 'agent' or 'agent_class'"
                )
            self._agent = self._build_agent()
        if hasattr(self._agent, "reset"):
            self._agent.reset()

    def _build_agent(self):
        ref = self._agent_class
        if ":" not in ref:
            raise ValueError(
                f"agent_class must be 'module.path:ClassName', got {ref!r}"
            )
        module_path, cls_name = ref.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)(**self._agent_kwargs)

    def step(self) -> None:
        obs: dict = {"timestamp": time.time()}
        for obs_key, topic in self._state_topics.items():
            data = self.get_latest(topic)
            if data is not None:
                obs[obs_key] = data
        for obs_key, topic in self._image_topics.items():
            data = self.get_latest(topic)
            if data is not None:
                obs[obs_key] = data

        _t_act = time.perf_counter()
        action = self._agent.act(obs)
        self._perf.record("act_ms", (time.perf_counter() - _t_act) * 1e3)
        ts = time.time()

        if "_record" in action:
            self.publish("record", {"record": bool(action["_record"])}, ts=ts)

        # Optional action-chunk snapshot for visualization consumers (e.g.
        # ViserMonitorNode rendering predicted end-effector positions). The
        # agent sets this under "_chunk" — keep it off the joint-command path.
        chunk = action.get("_chunk")
        if chunk is not None:
            self.publish("chunk", chunk, ts=ts)

        # Optional agent-side state (e.g. a teleop leader's buttons, trigger and
        # wrench) published and recorded as {name}/state.
        state = action.get("_state")
        if state is not None:
            self.publish("state", state, ts=ts)

        # Optional preprocessed-image snapshots — the exact frames fed to the
        # policy. Mirrored on image/{label} so downstream viewers (viser
        # monitor) can subscribe without reprocessing raw cameras themselves.
        # record=False: raw frames are already being written to MP4 by each
        # CameraNode's AsyncMp4Writer; recording them again here would route
        # 3× VGA-sized arrays through McapWriter's JSON fallback on the agent's
        # hot path (~45 ms/step → drops a 30 Hz loop to ~15 Hz during record).
        images = action.get("_images")
        if images:
            for label, img in images.items():
                self.publish(
                    f"image/{label}",
                    {"images": {"rgb": img}},
                    ts=ts,
                    record=False,
                )

        self._publish_commands(action, ts)

    def _publish_commands(self, action: dict, ts: float) -> None:
        if self._arm_key is not None:
            arm_action = action.get(self._arm_key)
            if arm_action is not None:
                pos = arm_action["pos"] if isinstance(arm_action, dict) else arm_action
                self.publish(
                    "joint_pos",
                    {"joint_pos": self._process_pos(pos)},
                    ts=ts,
                )
        elif "pos" in action:
            self.publish(
                "joint_pos",
                {"joint_pos": self._process_pos(action["pos"])},
                ts=ts,
            )
        else:
            arm_keys = [k for k in action if not k.startswith("_")]
            if len(arm_keys) == 1:
                arm_action = action[arm_keys[0]]
                if isinstance(arm_action, dict) and "pos" in arm_action:
                    self.publish(
                        "joint_pos",
                        {"joint_pos": self._process_pos(arm_action["pos"])},
                        ts=ts,
                    )
            else:
                for key in arm_keys:
                    arm_action = action[key]
                    if isinstance(arm_action, dict) and "pos" in arm_action:
                        self.publish(
                            f"{key}_pos",
                            {"joint_pos": self._process_pos(arm_action["pos"])},
                            ts=ts,
                        )

    def _process_pos(self, pos) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32)
        if self._normalize_gripper and len(pos) > 6:
            span = self._gripper_open_deg - self._gripper_closed_deg
            pos = pos.copy()
            pos[-1] = float(np.clip((pos[-1] - self._gripper_closed_deg) / span, 0.0, 1.0))
        return pos

    def cleanup(self) -> None:
        if self._agent is not None and hasattr(self._agent, "close"):
            self._agent.close()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name": params["name"],
            "agent_class": params.get("agent_class"),
            "agent_kwargs": params.get("agent_kwargs") or {},
            "loop_mode": params.get("loop_mode", "subscriber_driven"),
            "poll_freq": params.get("poll_freq"),
            "publish_freq": params.get("publish_freq"),
            "state_topics": params.get("state_topics"),
            "image_topics": params.get("image_topics"),
            "arm_key": params.get("arm_key"),
            "normalize_gripper": params.get("normalize_gripper", False),
            "gripper_open_deg": params.get("gripper_open_deg", 85.0),
            "gripper_closed_deg": params.get("gripper_closed_deg", 5.0),
        }
