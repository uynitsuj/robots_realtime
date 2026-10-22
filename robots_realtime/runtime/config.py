"""YAML session loader — maps node type strings to Node classes.

Session YAML schema:
    version: "1"
    session:
      save_root: recordings
      auto_record_duration: 10.0   # optional
      record_topic: null            # optional, e.g. "gello_left/record"
    nodes:
      - type: DummyGelloNode
        name: gello_left
        arm: left
        ...
      - type: MujocoSimNode
        name: yam_sim
        cmd_topics:
          left: gello_left/joint_pos
          right: gello_right/joint_pos
        ...

Alternatively, pass a Python module dotted path (no .yaml extension) that
exports a make_session() function — for backward compatibility.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any


# ── Node registry ─────────────────────────────────────────────────────────────

_NODE_REGISTRY: dict[str, str] = {
    "AgentNode":        "robots_realtime.runtime.agent_node:AgentNode",
    "RobotNode":        "robots_realtime.runtime.environment.robot_node:RobotNode",
    "CameraNode":       "robots_realtime.runtime.environment.camera_node:CameraNode",
    "MujocoSimNode":    "robots_realtime.runtime.sim.mujoco_sim_node:MujocoSimNode",
    "OrbbecRigNode":    "robots_realtime.runtime.environment.orbbec_rig_node:OrbbecRigNode",
    "ViserTeleopNode":  "robots_realtime.runtime.viser_teleop_node:ViserTeleopNode",
    "ViserMonitorNode": "robots_realtime.runtime.viser_monitor_node:ViserMonitorNode",
}


def _resolve_node_cls(type_name: str):
    """Resolve a node class from the registry or a dotted module path."""
    if type_name in _NODE_REGISTRY:
        ref = _NODE_REGISTRY[type_name]
    elif ":" in type_name:
        ref = type_name
    else:
        raise ValueError(
            f"Unknown node type '{type_name}'. "
            f"Known types: {list(_NODE_REGISTRY.keys())}"
        )
    module_path, cls_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


# ── Writer factory ────────────────────────────────────────────────────────────

def _make_writer_for_node(node_cls, node_params: dict):
    """Return an appropriate Writer for the given node type.

    A node may pick its writer explicitly with ``writer: mcap | mp4 | rgbd | null``.
    """
    from robots_realtime.runtime.recording import McapWriter, AsyncMp4Writer, NullWriter, RgbdWriter
    from robots_realtime.runtime.node import NodeRole

    explicit = node_params.get("writer")
    if explicit is not None:
        fps = float(node_params.get("fps", 30.0))
        writers = {
            "mcap": McapWriter,
            "mp4": lambda: AsyncMp4Writer(fps=fps),
            "rgbd": lambda: RgbdWriter(fps=fps),
            "null": NullWriter,
        }
        if explicit not in writers:
            raise ValueError(f"unknown writer {explicit!r}; expected one of {sorted(writers)}")
        return writers[explicit]()

    # MujocoSimNode manages its own writers — give it NullWriter
    type_name = node_params.get("type", "")
    if type_name == "MujocoSimNode":
        return NullWriter()

    # Check role
    role = getattr(node_cls, "role", NodeRole.ROBOT)
    if role == NodeRole.SENSOR:
        fps = float(node_params.get("fps", 30.0))
        return AsyncMp4Writer(fps=fps)

    return McapWriter()


# ── YAML loader ───────────────────────────────────────────────────────────────

def load_session(path: str) -> "Session":
    """Load a Session from a YAML config file or Python module path.

    Args:
        path: Path to a .yaml file, or a dotted Python module path
              containing make_session() (legacy).
    """
    # Python module path (legacy / advanced)
    if not path.endswith(".yaml") and not path.endswith(".yml") and os.path.sep not in path:
        # Try as a Python module first
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, "make_session"):
                return mod.make_session()
        except (ImportError, ModuleNotFoundError):
            pass

    # YAML file
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Session config not found: {yaml_path}")

    return _load_from_yaml(yaml_path)


def _resolve_policy_name(save_root_template: str, nodes_cfg: list[dict]) -> str:
    """Replace {policy_name} in save_root by querying the OpenPI server metadata."""
    for node in nodes_cfg:
        if node.get("type") != "AgentNode":
            continue
        kw = node.get("agent_kwargs", {})
        ip = kw.get("ip", "127.0.0.1")
        port = kw.get("port")
        if port is None:
            continue
        try:
            import websockets.sync.client  # noqa: PLC0415
            from openpi_client import msgpack_numpy  # noqa: PLC0415

            uri = f"ws://{ip}:{port}"
            logging.info("Querying policy name from %s ...", uri)
            with websockets.sync.client.connect(uri, compression=None, max_size=None) as ws:
                metadata = msgpack_numpy.unpackb(ws.recv())
            policy_name = metadata.get("policy_name")
            if policy_name:
                resolved = save_root_template.format(policy_name=policy_name)
                logging.info("Resolved save_root to: %s", resolved)
                return resolved
            logging.warning("Server at %s has no policy_name in metadata", uri)
        except Exception as exc:
            logging.warning("Could not query policy name from %s:%s: %s", ip, port, exc)
    return save_root_template


def _load_from_yaml(yaml_path: Path) -> "Session":
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load YAML session configs. "
                          "Install it with: pip install pyyaml") from e

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("version") != "1":
        raise ValueError(f"Unsupported session config version: {cfg.get('version')}")

    session_cfg: dict = cfg.get("session", {})
    save_root: str = session_cfg.get("save_root", "recordings")
    auto_record_duration: float | None = session_cfg.get("auto_record_duration")
    record_topic: str | None = session_cfg.get("record_topic")
    start_paused: bool = bool(session_cfg.get("start_paused", False))
    record_on_unpause: bool = bool(session_cfg.get("record_on_unpause", False))
    episode_timeout: float | None = session_cfg.get("episode_timeout")

    nodes_cfg: list[dict] = cfg.get("nodes", [])

    if "{policy_name}" in save_root:
        save_root = _resolve_policy_name(save_root, nodes_cfg)
    nodes = []

    for node_params in nodes_cfg:
        node_params = dict(node_params)  # copy to avoid mutation
        type_name: str = node_params.pop("type")
        node_cls = _resolve_node_cls(type_name)

        # Build constructor kwargs via classmethod
        kwargs = node_cls.build_kwargs({**node_params, "type": type_name})

        # Inject writer
        node_params_with_type = {**node_params, "type": type_name}
        writer = _make_writer_for_node(node_cls, node_params_with_type)
        kwargs["writer"] = writer

        try:
            node = node_cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate {type_name} with kwargs {kwargs}: {e}"
            ) from e

        nodes.append(node)

    from robots_realtime.runtime.session import Session

    return Session(
        nodes=nodes,
        save_root=save_root,
        record_topic=record_topic,
        auto_record_duration=auto_record_duration,
        start_paused=start_paused,
        record_on_unpause=record_on_unpause,
        episode_timeout=episode_timeout,
    )
