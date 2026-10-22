"""CameraNode — wraps any CameraDriver and publishes frames to the bus.

Hardware timestamp from the driver (RealSense / ZED SDK) is used directly,
giving sub-millisecond accurate per-frame timestamps for post-hoc alignment.

poll_freq is None by default: the driver's blocking read() call paces the loop
at the hardware frame rate.  Set poll_freq only for drivers (e.g. bare OpenCV)
where read() returns immediately and you want an explicit rate cap.

Optional ``publish_resize`` shrinks frames before they hit the bus so consumers
(e.g. an OpenPI policy that resizes to 224×224 anyway) don't pay full-VGA
serialization + TCP cost. The on-disk MP4 keeps the full-resolution frame —
only the bus payload is downsized. Two modes match AsyncDiffusionAgent's
``image_preprocess``: ``center_crop`` (crop to min(H,W) square then resize)
and ``pad`` (resize-with-pad / letterbox).

Optional ``publish_fov_crop`` (fraction in ``(0, 1]``) artificially narrows the
field of view by center-cropping the frame *before* the resize. This is a
deployment-only knob: it simulates a narrower-FOV camera so the policy input
matches the FOV the model was trained on, without touching the optics. Like
``publish_resize`` it only affects the bus payload — the on-disk recording
keeps the full FOV — so it composes with either resize mode.
"""

from __future__ import annotations

import importlib
import time

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole
from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver


_CAMERA_DRIVER_REGISTRY: dict[str, str] = {
    "ZedCamera":        "robots_realtime.sensors.cameras.zed_camera:ZedCamera",
    "OpenCVCamera":     "robots_realtime.sensors.cameras.opencv_camera:OpencvCamera",
    "RealSenseCamera":  "robots_realtime.sensors.cameras.realsense_camera:RealSenseCamera",
    "OrbbecCamera":     "robots_realtime.sensors.cameras.orbbec_camera:OrbbecCamera",
}

_NODE_ONLY_KEYS = {
    "name", "type", "poll_freq",
    "publish_resize", "publish_resize_mode", "publish_fov_crop",
    "publish_freq", "writer",
}


def _center_fov_crop(img: np.ndarray, frac: float) -> np.ndarray:
    """Center-crop to ``frac`` of each dimension to simulate a narrower FOV.

    ``frac`` is the fraction of width/height kept (``(0, 1]``); smaller = tighter
    FOV / more zoom. ``frac >= 1.0`` is a no-op. The crop is symmetric about the
    image center so the principal point stays centered.
    """
    if frac >= 1.0:
        return img
    h, w = img.shape[:2]
    ch = max(1, round(h * frac))
    cw = max(1, round(w * frac))
    h0 = (h - ch) // 2
    w0 = (w - cw) // 2
    return img[h0:h0 + ch, w0:w0 + cw]


def _center_crop_and_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop to the largest square that fits, then resize to (target_h, target_w).

    Mirrors ``AsyncDiffusionAgent._center_crop_and_resize`` so frames published
    with mode=center_crop are bit-identical to what the policy would produce
    if it received the full-res frame and ran its own preprocessing.
    """
    from openpi_client.image_tools import resize_with_pad  # noqa: PLC0415

    h, w = img.shape[:2]
    side = min(h, w)
    h0 = (h - side) // 2
    w0 = (w - side) // 2
    cropped = img[h0:h0 + side, w0:w0 + side]
    return resize_with_pad(cropped, target_h, target_w)


def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    from openpi_client.image_tools import resize_with_pad  # noqa: PLC0415
    return resize_with_pad(img, target_h, target_w)


_RESIZE_MODES = {
    "center_crop": _center_crop_and_resize,
    "pad": _resize_with_pad,
}


def _instantiate_camera_driver(spec: dict) -> CameraDriver:
    """Instantiate a camera driver from a spec dict (driver name + kwargs)."""
    driver_name: str = spec["driver"]
    if driver_name in _CAMERA_DRIVER_REGISTRY:
        ref = _CAMERA_DRIVER_REGISTRY[driver_name]
    elif ":" in driver_name:
        ref = driver_name
    else:
        raise ValueError(
            f"Unknown camera driver '{driver_name}'. "
            f"Known drivers: {list(_CAMERA_DRIVER_REGISTRY.keys())}"
        )
    module_path, cls_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    kwargs = {k: v for k, v in spec.items() if k != "driver"}
    return getattr(mod, cls_name)(**kwargs)


class CameraNode(Node):
    """Publish camera frames from any CameraDriver onto the bus.

    Published topics:
        ``{name}/rgb``    — dict with ``frame`` (H,W,3 uint8) and ``ts`` float
        ``{name}/info``   — camera info dict (published once on setup)

    Optionally also publishes:
        ``{name}/depth``  — if driver provides it in CameraData.other_sensors
        ``{name}/imu``    — if driver provides IMUData

    Args:
        driver:    Camera driver implementing read() -> CameraData.
        name:      Node name on the bus.
        poll_freq: Optional rate cap for drivers where read() is non-blocking.
        writer:    Optional Writer injected at construction for recording.
    """

    role = NodeRole.SENSOR
    published_topics: list[str] = ["rgb"]
    poll_freq: float | None = None

    def __init__(
        self,
        driver: CameraDriver | None = None,
        name: str = "camera",
        poll_freq: float | None = None,
        publish_freq: float | None = None,
        writer=None,
        _driver_spec: dict | None = None,
        publish_resize: tuple[int, int] | list[int] | None = None,
        publish_resize_mode: str = "center_crop",
        publish_fov_crop: float = 1.0,
        **kwargs,
    ) -> None:
        # publish_freq caps the bus rate only; the writer still records every frame.
        if publish_freq is not None:
            self.publish_freq = publish_freq
        super().__init__(name=name, writer=writer, **kwargs)
        self._driver = driver
        self._driver_spec = _driver_spec
        self.poll_freq = poll_freq

        if not (0.0 < publish_fov_crop <= 1.0):
            raise ValueError(
                f"[{name}] publish_fov_crop must be in (0, 1], got {publish_fov_crop!r}"
            )
        self._publish_fov_crop = float(publish_fov_crop)

        if publish_resize is not None:
            if publish_resize_mode not in _RESIZE_MODES:
                raise ValueError(
                    f"[{name}] publish_resize_mode must be one of "
                    f"{sorted(_RESIZE_MODES)}, got {publish_resize_mode!r}"
                )
            h, w = publish_resize
            self._publish_resize: tuple[int, int] | None = (int(h), int(w))
            self._publish_resize_fn = _RESIZE_MODES[publish_resize_mode]
        else:
            self._publish_resize = None
            self._publish_resize_fn = None

    def setup(self) -> None:
        if self._driver is None:
            if self._driver_spec is None:
                raise RuntimeError(
                    f"[{self.name}] CameraNode.driver is None — inject a camera driver before starting."
                )
            self._driver = _instantiate_camera_driver(self._driver_spec)

    def step(self) -> None:
        data: CameraData = self._driver.read()

        # Hardware timestamp from driver (ms) → seconds
        ts = data.timestamp / 1000.0 if data.timestamp else time.time()

        # Publish one consolidated message that matches the format agents and
        # visualization code already expect (same shape as old CameraNode._get_latest_data).
        msg: dict = {"images": data.images, "timestamp": ts}

        # Depth: support both other_sensors["depth"] (standard) and the
        # dynamic depth_data attribute that ZedCamera sets directly.
        # (No ``or`` chaining: truth-testing an ndarray raises.)
        depth = (data.other_sensors or {}).get("depth")
        if depth is None:
            depth = getattr(data, "depth_data", None)
        if depth is not None:
            msg["depth_data"] = depth

        # Intrinsics and extrinsics from the driver if available.
        intrinsics = getattr(self._driver, "intrinsic_data", None)
        if intrinsics is not None:
            msg["intrinsics"] = intrinsics
        extrinsics = getattr(self._driver, "extrinsics", None)
        if extrinsics is not None:
            msg["extrinsics"] = extrinsics

        fov_crop = self._publish_fov_crop
        if self._publish_resize is None and fov_crop >= 1.0:
            self.publish("rgb", msg, ts=ts)
        else:
            # Bus payload: optionally FOV-cropped then resized RGB only — depth
            # and intrinsics would need geometric rescaling to stay consistent
            # with the new pixel grid, so they're dropped from the bus version.
            # The disk recording (record_data=msg) keeps everything at full
            # resolution and full FOV.
            def _to_bus(img: np.ndarray) -> np.ndarray:
                if fov_crop < 1.0:
                    img = _center_fov_crop(img, fov_crop)
                if self._publish_resize is not None:
                    target_h, target_w = self._publish_resize
                    img = self._publish_resize_fn(img, target_h, target_w)
                return img

            bus_msg: dict = {
                "images": {k: _to_bus(img) for k, img in data.images.items()},
                "timestamp": ts,
            }
            if extrinsics is not None:
                bus_msg["extrinsics"] = extrinsics  # pose is resolution-invariant
            self.publish("rgb", bus_msg, ts=ts, record_data=msg)

        if data.imu_data is not None:
            imu = data.imu_data
            self.publish("imu", {
                "accel": imu.acceleration,
                "gyro": imu.gyroscope,
                "ts": imu.timestamp,
            }, ts=ts)

    def cleanup(self) -> None:
        if hasattr(self._driver, "stop"):
            self._driver.stop()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        kwargs: dict = {
            "name": params["name"],
            "poll_freq": params.get("poll_freq"),
            "publish_resize": params.get("publish_resize"),
            "publish_resize_mode": params.get("publish_resize_mode", "center_crop"),
            "publish_fov_crop": params.get("publish_fov_crop", 1.0),
            "publish_freq": params.get("publish_freq"),
        }
        if "driver" in params:
            driver_kwargs = {k: v for k, v in params.items() if k not in _NODE_ONLY_KEYS}
            kwargs["_driver_spec"] = driver_kwargs
        return kwargs
