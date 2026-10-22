"""FakeRgbdCamera — ray-cast RGB-D of a synthetic table scene from a rig camera.

For hardware-free rehearsals. Renders, from the calibrated pose and nominal
intrinsics of one rig camera: the table plane (grey checker) at ``surface_z``,
a red cube and a blue bowl-ish box at configurable world positions. Depth is
exact z-depth in metres, aligned to RGB. Wrist cameras render from the arm's
home pose (the fake has no joint feed).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData


def _ray_box(o, d, lo, hi):
    """Slab test for many rays (o: 3, d: N x 3) against one AABB -> t (inf on miss)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmax.dtype.type(0)) & (tmin <= tmax) & (tmin > 0)
    return np.where(hit, tmin, np.inf)


class FakeRgbdCamera:
    def __init__(
        self,
        camera: str,
        serial: str = "",
        width: int = 848,
        height: int = 530,
        fps: int = 15,
        rig_yaml: str | None = None,
        cube_xy: tuple[float, float] = (0.05, 0.42),
        cube_size: float = 0.04,
        bowl_xy: tuple[float, float] = (0.10, 0.25),
        **_: Any,
    ) -> None:
        from robots_realtime.rig.dionysus import load_rig, matrix_to_quat_wxyz

        rig = load_rig(rig_yaml or None)
        cam = rig.cameras[camera]
        self.name, self.serial = camera, serial or cam.serial
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        q_home = {a.name: a.home_q for a in rig.arms.values()}
        self.T = cam.world_pose(rig.arms, q_home)
        sx = self.width / 848.0
        sy = self.height / 530.0
        self.K = np.array([[410.9 * sx, 0, 424.0 * sx], [0, 410.9 * sy, 265.0 * sy], [0, 0, 1.0]])
        self.intrinsic_data = {"left": {"intrinsics_matrix": self.K, "distortion_coefficients": [0.0] * 8,
                                        "distortion_model": "none", "width": self.width, "height": self.height}}
        self.extrinsics = {"position": self.T[:3, 3].tolist(), "wxyz": matrix_to_quat_wxyz(self.T[:3, :3]).tolist(),
                           "pose_mat": self.T.tolist(),
                           "parent_frame": "world" if cam.mount == "fixed" else f"{cam.mount}_tool0"}
        z0 = rig.surface_z
        h = cube_size
        self.objects = [
            (np.array([cube_xy[0] - h / 2, cube_xy[1] - h / 2, z0]), np.array([cube_xy[0] + h / 2, cube_xy[1] + h / 2, z0 + h]),
             np.array([200, 30, 30], np.uint8)),
            (np.array([bowl_xy[0] - 0.07, bowl_xy[1] - 0.07, z0]), np.array([bowl_xy[0] + 0.07, bowl_xy[1] + 0.07, z0 + 0.05]),
             np.array([40, 70, 200], np.uint8)),
        ]
        self.z0 = z0
        self._rgb, self._depth = self._render()
        self._period = 1.0 / self.fps
        self._last = 0.0

    def _render(self):
        H, W = self.height, self.width
        v, u = np.mgrid[0:H, 0:W]
        rays_c = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u, float)],
                          axis=-1).reshape(-1, 3)
        R, o = self.T[:3, :3], self.T[:3, 3]
        d = rays_c @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t_plane = (self.z0 - o[2]) / d[:, 2]
        t_plane = np.where(t_plane > 0, t_plane, np.inf)
        p = o + d * t_plane[:, None]
        checker = ((np.floor(p[:, 0] / 0.05) + np.floor(p[:, 1] / 0.05)) % 2).astype(np.uint8)
        rgb = np.where(checker[:, None] == 1, np.array([170, 165, 160], np.uint8), np.array([140, 135, 130], np.uint8))
        t_best = t_plane.copy()
        for lo, hi, col in self.objects:
            t = _ray_box(o, d, lo, hi)
            m = t < t_best
            rgb[m] = col
            t_best = np.where(m, t, t_best)
        depth = np.where(np.isfinite(t_best), t_best, 0.0).astype(np.float32)  # rays_c has z=1 -> t is z-depth
        depth[depth > 6.0] = 0.0
        return rgb.reshape(H, W, 3), depth.reshape(H, W)

    def read(self) -> CameraData:
        wait = self._period - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        data = CameraData(images={"rgb": self._rgb.copy()}, timestamp=self._last * 1000.0)
        data.depth_data = self._depth.copy()  # type: ignore[attr-defined]
        return data

    def read_calibration_data_intrinsics(self) -> dict[str, Any]:
        return self.intrinsic_data

    def get_camera_info(self) -> dict[str, Any]:
        return {"device_id": self.serial, "name": self.name, "width": self.width, "height": self.height,
                "fps": self.fps, "fake": True}

    def stop(self) -> None:
        pass


__all__ = ["FakeRgbdCamera"]
