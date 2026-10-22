"""Rig model for the dionysus bimanual UR5e cell: calibration, kinematics, frames.

One world frame for everything: the ``checkerboard`` frame of the uksang
calibration (origin at the board's first internal corner, +Z up from the table).
Every pose this module hands out is a 4x4 ``world_T_x`` in that frame.

Sources (copied into ``configs/ur747_bimanual/calib/`` so this package does not
depend on the calibration checkout):

- ``{left,right}_follower_arm_handeye.yaml``: ``checkerboard -> <arm>_base`` (the
  UR controller base, i.e. the frame of RTDE ``actual_TCP_pose``) and
  ``<arm>_tool0 -> <arm>_wrist_camera_link``.
- ``{top,bottom}_camera.yaml``: ``checkerboard -> <cam>_camera_link``.

Each record is ``parent_T_child`` (translation in metres, quaternion xyzw). The
``*_camera_link`` frames are rotated from the OpenCV optical frame by
``T_LINK_OPTICAL`` (zero translation), so a camera pose in the OpenCV
convention (+z forward, +x right, +y down) is ``world_T_link @ T_LINK_OPTICAL``.

Kinematics are the UR5e manufacturer DH constants: ``fk(q)`` is the flange
(``tool0``) in the controller base frame, the same frame the hand-eye solve used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── UR5e kinematics ───────────────────────────────────────────────────────────

UR5E_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
UR5E_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
UR5E_ALPHA = np.array([math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0])

#: OpenCV optical frame expressed in the ``*_camera_link`` frame.
T_LINK_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_frames(q) -> list[np.ndarray]:
    """Base-frame poses of the 6 DH frames (index 5 is the flange)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)[:6]
    T = np.eye(4)
    out = []
    for i in range(6):
        T = T @ _dh(float(q[i]), UR5E_D[i], UR5E_A[i], UR5E_ALPHA[i])
        out.append(T.copy())
    return out


def fk(q) -> np.ndarray:
    """6 joint angles -> 4x4 flange (``tool0``) pose in the controller base frame."""
    return fk_frames(q)[-1]


# ── Small SE(3) helpers (numpy only; this runs inside realtime nodes) ─────────


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def pose7_wxyz(T: np.ndarray) -> np.ndarray:
    """4x4 -> [x, y, z, qw, qx, qy, qz]."""
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:3, 3], matrix_to_quat_wxyz(T[:3, :3])])


def rotvec_to_matrix(rv) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def record_to_matrix(rec: dict) -> np.ndarray:
    """``{translation:{x,y,z}, rotation:{x,y,z,w}}`` -> 4x4 parent_T_child."""
    t, r = rec["translation"], rec["rotation"]
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix([r["x"], r["y"], r["z"], r["w"]])
    T[:3, 3] = [float(t["x"]), float(t["y"]), float(t["z"])]
    return T


def _find_transform(path: Path, parent: str, child: str) -> np.ndarray:
    data = yaml.safe_load(path.read_text())
    for rec in data.get("transforms", []):
        if rec.get("parent_frame") == parent and rec.get("child_frame") == child:
            return record_to_matrix(rec)
    raise KeyError(f"{path}: no transform {parent} -> {child}")


# ── Rig model ─────────────────────────────────────────────────────────────────


@dataclass
class ArmModel:
    name: str  # "right" / "left"
    arm_id: int  # arm index, 0..N-1
    ip: str
    world_T_base: np.ndarray
    tcp_offset: np.ndarray  # flange -> TCP, flange frame [m]
    home_q: np.ndarray  # 6 joints
    home_gripper: float  # vendor gripper units (0 open .. ~0.92 closed)
    joint_lower: np.ndarray
    joint_upper: np.ndarray

    def flange_world(self, q) -> np.ndarray:
        return self.world_T_base @ fk(q)

    def tcp_world(self, q) -> np.ndarray:
        T = self.flange_world(q)
        out = T.copy()
        out[:3, 3] = T[:3, 3] + T[:3, :3] @ self.tcp_offset
        return out

    def link_points_world(self, q) -> np.ndarray:
        """Points along the arm (base, DH frame origins, TCP) for clearance checks."""
        frames = fk_frames(q)
        pts = [self.world_T_base[:3, 3]]
        for T in frames:
            pts.append((self.world_T_base @ T)[:3, 3])
        pts.append(self.tcp_world(q)[:3, 3])
        return np.asarray(pts)


@dataclass
class CameraModel:
    name: str
    serial: str
    mount: str  # "fixed" or an arm name ("right"/"left")
    T: np.ndarray  # fixed: world_T_optical; wrist: tool0_T_optical

    def world_pose(self, arms: dict[str, ArmModel], q_by_arm: dict[str, Any]) -> np.ndarray | None:
        if self.mount == "fixed":
            return self.T
        arm = arms.get(self.mount)
        q = q_by_arm.get(self.mount)
        if arm is None or q is None:
            return None
        return arm.flange_world(np.asarray(q, dtype=np.float64)[:6]) @ self.T


@dataclass
class Rig:
    root: Path
    surface_z: float
    arms: dict[str, ArmModel]
    cameras: dict[str, CameraModel]
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    clearance: dict[str, float] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def arm_by_id(self, arm_id: int) -> ArmModel:
        for a in self.arms.values():
            if a.arm_id == arm_id:
                return a
        raise KeyError(f"no arm with arm_id={arm_id}")

    def ordered_arms(self) -> list[ArmModel]:
        return sorted(self.arms.values(), key=lambda a: a.arm_id)

    def to_wire(self) -> dict:
        """Static rig description as plain JSON-able types."""
        return {
            "world_frame": "checkerboard",
            "surface_z": float(self.surface_z),
            "workspace_min": self.workspace_min.astype(float).tolist(),
            "workspace_max": self.workspace_max.astype(float).tolist(),
            "clearance": dict(self.clearance),
            "arms": {
                a.name: {
                    "arm_id": a.arm_id,
                    "ip": a.ip,
                    "world_T_base": a.world_T_base.astype(float).tolist(),
                    "tcp_offset": a.tcp_offset.astype(float).tolist(),
                    "home_q": a.home_q.astype(float).tolist(),
                    "home_gripper": float(a.home_gripper),
                    "joint_lower": a.joint_lower.astype(float).tolist(),
                    "joint_upper": a.joint_upper.astype(float).tolist(),
                }
                for a in self.arms.values()
            },
            "cameras": {
                c.name: {"serial": c.serial, "mount": c.mount, "T": c.T.astype(float).tolist()}
                for c in self.cameras.values()
            },
        }


DEFAULT_RIG_YAML = Path(__file__).resolve().parents[2] / "configs" / "ur747_bimanual" / "rig.yaml"


def load_rig(path: str | Path | None = None) -> Rig:
    """Load ``rig.yaml`` (default: ``configs/ur747_bimanual/rig.yaml``)."""
    path = Path(path) if path is not None else DEFAULT_RIG_YAML
    cfg = yaml.safe_load(path.read_text())
    root = path.parent

    arms: dict[str, ArmModel] = {}
    for name, a in cfg["arms"].items():
        calib = root / a["calib"]
        world_T_base = _find_transform(calib, "checkerboard", a["base_frame"])
        lim = a.get("joint_limits", [-2 * math.pi, 2 * math.pi])
        lower = np.asarray(a.get("joint_lower", [lim[0]] * 6), dtype=np.float64)
        upper = np.asarray(a.get("joint_upper", [lim[1]] * 6), dtype=np.float64)
        arms[name] = ArmModel(
            name=name,
            arm_id=int(a["arm_id"]),
            ip=str(a["ip"]),
            world_T_base=world_T_base,
            tcp_offset=np.asarray(a["tcp_offset"], dtype=np.float64),
            home_q=np.asarray(a["home_q"], dtype=np.float64),
            home_gripper=float(a.get("home_gripper", 0.0)),
            joint_lower=lower,
            joint_upper=upper,
        )
    ids = sorted(a.arm_id for a in arms.values())
    if ids != list(range(len(ids))):
        raise ValueError(f"{path}: arm_id values must be 0..N-1, got {ids}")

    cameras: dict[str, CameraModel] = {}
    for name, c in cfg.get("cameras", {}).items():
        calib = root / c["calib"]
        mount = str(c.get("mount", "fixed"))
        parent = "checkerboard" if mount == "fixed" else c["parent_frame"]
        T_link = _find_transform(calib, parent, c["link_frame"])
        cameras[name] = CameraModel(
            name=name, serial=str(c["serial"]), mount=mount, T=T_link @ T_LINK_OPTICAL,
        )

    ws = cfg.get("workspace", {})
    return Rig(
        root=root,
        surface_z=float(cfg.get("surface_z", 0.0)),
        arms=arms,
        cameras=cameras,
        workspace_min=np.asarray(ws.get("min", [-1.0, -1.0, -0.05]), dtype=np.float64),
        workspace_max=np.asarray(ws.get("max", [1.5, 1.5, 1.2]), dtype=np.float64),
        clearance={k: float(v) for k, v in (cfg.get("clearance") or {}).items()},
        raw=cfg,
    )


__all__ = [
    "ArmModel",
    "CameraModel",
    "DEFAULT_RIG_YAML",
    "Rig",
    "T_LINK_OPTICAL",
    "fk",
    "fk_frames",
    "load_rig",
    "matrix_to_quat_wxyz",
    "pose7_wxyz",
    "quat_xyzw_to_matrix",
    "record_to_matrix",
    "rotvec_to_matrix",
]
