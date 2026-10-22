"""OrbbecCamera — Orbbec Gemini 305g (GMSL2) RGB-D driver.

Streams COLOR + DEPTH with depth aligned to colour (software ``AlignFilter``), so
``depth_data[v, u]`` is the metric z-depth of ``rgb[v, u]`` and both share the
colour intrinsics. That is the frame the rig calibration was solved in
(848x530, Default work mode).

Work modes: these units only expose a DEPTH sensor in the depth work modes
("Default", "High Accuracy", ...). "Dual Color Streams" — what recording_studio
and imit_ur5 use for training images — has no depth, so the driver switches the
device to ``work_mode`` when needed. The switch reboots the camera (~3.5 s);
recording_studio / imit_ur5 switch it back themselves on their next start.

GMSL access needs ``recording_studio.service`` running with all its capture
stages OFF: with the service stopped the cameras are "not found", with a stage
streaming they are "busy".

Output (``read()``):
    images      {"rgb": HxWx3 uint8, RGB order}
    depth_data  HxW float32 metres, 0 = invalid
    timestamp   host time [ms] at frame arrival
Attributes consumed by CameraNode: ``intrinsic_data`` (the robots_realtime shape,
keyed ``"left"`` for compatibility) and ``extrinsics`` (from the rig model).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData

log = logging.getLogger(__name__)

WORK_MODE_SETTLE_S = 3.5
_DEPTH_MIN_M = 0.03
_DEPTH_MAX_M = 6.0


class OrbbecCameraError(RuntimeError):
    pass


def _sdk():
    try:
        import pyorbbecsdk as ob
    except ImportError as e:  # pragma: no cover - depends on the Jetson build
        raise OrbbecCameraError(
            "pyorbbecsdk is not importable in this venv. Build it from ~/Desktop/pyorbbecsdk "
            "for this interpreter (pybind11==2.12.0) and copy the .so + sdk/lib/arm64/* into site-packages."
        ) from e
    return ob


def _serials(ctx) -> list[str]:
    """Serials of every connected Orbbec, read without opening any device.

    Opening a device (``get_device_by_index``) claims it; with one CameraNode per
    camera in separate processes, every process opening every device collides
    ("Property not found to register callback").
    """
    lst = ctx.query_devices()
    out = []
    for i in range(lst.get_count()):
        try:
            out.append(lst.get_device_serial_number_by_index(i))
        except Exception:  # noqa: BLE001
            continue
    return out


def _enumerate(ob, ctx) -> dict[str, Any]:
    """serial -> opened device, for every connected Orbbec (diagnostics only).
    ``ctx`` must outlive the returned devices."""
    lst = ctx.query_devices()
    out: dict[str, Any] = {}
    for i in range(lst.get_count()):
        dev = lst.get_device_by_index(i)
        try:
            out[dev.get_device_info().get_serial_number()] = dev
        except Exception:  # noqa: BLE001
            continue
    return out


def _frame_to_rgb(ob, frame) -> np.ndarray:
    import cv2

    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    buf = np.frombuffer(frame.get_data(), dtype=np.uint8)
    F = ob.OBFormat
    if fmt == F.RGB:
        return buf.reshape(h, w, 3).copy()
    if fmt == F.BGR:
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    if fmt == F.MJPG:
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OrbbecCameraError("MJPG decode failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt == F.YUYV:
        return cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUYV)
    if hasattr(F, "NV12") and fmt == F.NV12:
        return cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2RGB_NV12)
    raise OrbbecCameraError(f"unsupported colour format {fmt} ({w}x{h})")


class OrbbecCamera:
    """One Orbbec camera opened by serial.

    Args:
        serial: Device serial (e.g. ``CV3H46000036``).
        camera: Rig camera name (``top``/``bottom``/``right_wrist``/``left_wrist``);
            selects the extrinsics from the rig model. (``name`` is accepted too;
            CameraNode keeps ``name`` for itself, so session YAMLs use ``camera``.)
        width/height/fps: Colour and depth profile (848x530@30 is the calibrated one).
        enable_depth: Stream DEPTH and align it to colour.
        work_mode: Depth work mode to switch into when the device is elsewhere.
        rig_yaml: Rig model for ``extrinsics`` (``None`` = default rig.yaml,
            ``""`` = none).
        timeout_ms: ``wait_for_frames`` timeout.
        context: A shared ``pyorbbecsdk.Context`` (several cameras in one process
            should share one); default creates one per camera.
    """

    def __init__(
        self,
        serial: str,
        camera: str = "",
        name: str = "",
        width: int = 848,
        height: int = 530,
        fps: int = 30,
        enable_depth: bool = True,
        work_mode: str = "Default",
        rig_yaml: str | None = None,
        timeout_ms: int = 500,
        context: Any = None,
        **_: Any,
    ) -> None:
        self.serial = str(serial)
        self.name = camera or name or self.serial
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.enable_depth = bool(enable_depth)
        self.work_mode = work_mode
        self.timeout_ms = int(timeout_ms)
        self.intrinsic_data: dict | None = None
        self.extrinsics: dict | None = None
        self._depth_scale_m: float | None = None
        self._last_ts = 0.0

        ob = self._ob = _sdk()
        self._ctx = context if context is not None else ob.Context()  # must outlive the device handles
        try:
            self._ctx.set_logger_level(ob.OBLogLevel.WARNING)
        except Exception:
            pass
        dev = self._open_device()
        if self.enable_depth:
            dev = self._ensure_work_mode(dev)
        self._dev = dev
        self._pipeline = ob.Pipeline(dev)
        cfg = ob.Config()
        self._color_profile = self._pick(ob.OBSensorType.COLOR_SENSOR, (ob.OBFormat.RGB, ob.OBFormat.MJPG,
                                                                          ob.OBFormat.YUYV))
        cfg.enable_stream(self._color_profile)
        self._align = None
        if self.enable_depth:
            depth_profile = self._pick(ob.OBSensorType.DEPTH_SENSOR, (ob.OBFormat.Y16,))
            cfg.enable_stream(depth_profile)
            cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            try:
                self._pipeline.enable_frame_sync()
            except Exception as e:
                log.warning("[%s] frame sync unavailable: %s", self.name, e)
            self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        try:
            self._pipeline.start(cfg)
        except Exception as e:
            raise OrbbecCameraError(
                f"[{self.name}] pipeline start failed (camera busy? recording_studio stage on?): {e}"
            ) from e
        self.intrinsic_data = self.read_calibration_data_intrinsics()
        self.extrinsics = self._load_extrinsics(rig_yaml)
        log.info("[%s] Orbbec %s streaming %dx%d@%d depth=%s mode=%s", self.name, self.serial,
                 self.width, self.height, self.fps, self.enable_depth, self.work_mode)

    # ── setup ─────────────────────────────────────────────────────────────

    def _open_device(self, wait_s: float = 25.0):
        """Open this camera by serial, waiting out reboots of any camera.

        Another process switching *its* camera's work mode reboots that camera,
        and while it re-enumerates the device list here is transiently short or
        wrong; so a missing serial is retried before it is an error.
        """
        deadline = time.time() + wait_s
        serials: list[str] = []
        while True:
            serials = _serials(self._ctx)
            if self.serial in serials:
                try:
                    return self._ctx.query_devices().get_device_by_serial_number(self.serial)
                except Exception as e:  # noqa: BLE001 - mid-reboot; retry
                    log.info("[%s] open failed (%s); retrying", self.name, e)
            if time.time() > deadline:
                raise OrbbecCameraError(
                    f"[{self.name}] Orbbec {self.serial} not found (have {sorted(serials)}); "
                    "is recording_studio.service running with its camera stage OFF?"
                )
            time.sleep(0.5)

    def _ensure_work_mode(self, dev):
        try:
            cur = dev.get_depth_work_mode()
            cur_name = getattr(cur, "name", str(cur))
        except Exception as e:
            raise OrbbecCameraError(f"[{self.name}] cannot query depth work mode: {e}") from e
        if cur_name == self.work_mode:
            return dev
        log.warning("[%s] switching depth work mode %r -> %r (camera reboots ~%.1fs)",
                    self.name, cur_name, self.work_mode, WORK_MODE_SETTLE_S)
        dev.set_depth_work_mode(self.work_mode)
        time.sleep(WORK_MODE_SETTLE_S)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                dev = self._open_device()
                if getattr(dev.get_depth_work_mode(), "name", "") == self.work_mode:
                    return dev
            except OrbbecCameraError:
                pass
            time.sleep(0.5)
        raise OrbbecCameraError(f"[{self.name}] did not come back in work mode {self.work_mode!r}")

    def _pick(self, sensor_type, formats):
        plist = self._pipeline.get_stream_profile_list(sensor_type)
        for fmt in formats:
            try:
                p = plist.get_video_stream_profile(self.width, self.height, fmt, self.fps)
                if p is not None:
                    return p
            except Exception:
                continue
        raise OrbbecCameraError(
            f"[{self.name}] no {sensor_type} profile {self.width}x{self.height}@{self.fps} in {formats}"
        )

    def _load_extrinsics(self, rig_yaml: str | None) -> dict | None:
        if rig_yaml == "":
            return None
        try:
            from robots_realtime.rig.dionysus import load_rig, matrix_to_quat_wxyz

            rig = load_rig(rig_yaml or None)
            cam = rig.cameras.get(self.name)
            if cam is None:
                return None
            T = cam.T
            return {
                "position": T[:3, 3].tolist(),
                "wxyz": matrix_to_quat_wxyz(T[:3, :3]).tolist(),
                "pose_mat": T.tolist(),
                # fixed: world_T_optical; wrist: <arm>_tool0_T_optical
                "parent_frame": "world" if cam.mount == "fixed" else f"{cam.mount}_tool0",
            }
        except Exception as e:
            log.warning("[%s] no rig extrinsics: %s", self.name, e)
            return None

    # ── CameraDriver protocol ─────────────────────────────────────────────

    def read(self) -> CameraData:
        ob = self._ob
        deadline = time.time() + 5.0
        while True:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
            if frames is not None:
                if self._align is not None:
                    frames = self._align.process(frames)
                    if frames is not None and hasattr(frames, "as_frame_set"):
                        try:
                            frames = frames.as_frame_set()
                        except Exception:
                            pass
                color = frames.get_color_frame() if frames is not None else None
                depth = frames.get_depth_frame() if (frames is not None and self.enable_depth) else None
                if color is not None and (depth is not None or not self.enable_depth):
                    break
            if time.time() > deadline:
                raise OrbbecCameraError(f"[{self.name}] no frames for 5 s")
        ts_ms = time.time() * 1000.0
        rgb = _frame_to_rgb(ob, color)
        result = CameraData(images={"rgb": rgb}, timestamp=ts_ms)
        if depth is not None:
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width())
            d = raw.astype(np.float32) * (float(depth.get_depth_scale()) / 1000.0)  # mm -> m
            d[(d < _DEPTH_MIN_M) | (d > _DEPTH_MAX_M)] = 0.0
            result.depth_data = d  # type: ignore[attr-defined]
        else:
            result.depth_data = None  # type: ignore[attr-defined]
        self._last_ts = ts_ms
        return result

    def read_calibration_data_intrinsics(self) -> dict[str, Any]:
        intr = self._color_profile.get_intrinsic()
        K = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = []
        model = "rational_polynomial"
        try:
            dd = self._color_profile.get_distortion()
            # OpenCV rational order: k1, k2, p1, p2, k3, k4, k5, k6
            dist = [dd.k1, dd.k2, dd.p1, dd.p2, dd.k3, dd.k4, dd.k5, dd.k6]
        except Exception:
            model = "none"
        block = {
            "intrinsics_matrix": K,
            "distortion_coefficients": [float(v) for v in dist],
            "distortion_model": model,
            "width": int(intr.width),
            "height": int(intr.height),
        }
        return {"left": block}

    def get_camera_info(self) -> dict[str, Any]:
        return {
            "device_id": self.serial,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "depth": self.enable_depth,
            "work_mode": self.work_mode,
        }

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"OrbbecCamera({self.name}:{self.serial})"


def prepare(serials: list[str], work_mode: str = "Default") -> dict[str, str]:
    """Switch the given cameras into ``work_mode`` one at a time (each reboots ~3.5 s).

    Run before a session so its camera nodes (one process each) find every
    camera already in the right mode and nothing reboots while they open.
    """
    ob = _sdk()
    ctx = ob.Context()
    out: dict[str, str] = {}
    for serial in serials:
        lst = ctx.query_devices()
        if serial not in _serials(ctx):
            out[serial] = "not found"
            continue
        dev = lst.get_device_by_serial_number(serial)
        cur = getattr(dev.get_depth_work_mode(), "name", "?")
        if cur == work_mode:
            out[serial] = f"already {work_mode}"
            continue
        dev.set_depth_work_mode(work_mode)
        del dev
        time.sleep(WORK_MODE_SETTLE_S)
        deadline = time.time() + 15.0
        while time.time() < deadline and serial not in _serials(ctx):
            time.sleep(0.5)
        out[serial] = f"{cur} -> {work_mode}"
    return out


def list_devices() -> list[dict]:
    """Serials, names and current work modes of every connected Orbbec (diagnostic)."""
    ob = _sdk()
    ctx = ob.Context()
    out = []
    for serial, dev in _enumerate(ob, ctx).items():
        info = dev.get_device_info()
        try:
            mode = dev.get_depth_work_mode().name
        except Exception:
            mode = "?"
        out.append({"serial": serial, "name": info.get_name(), "work_mode": mode})
    return out


__all__ = ["OrbbecCamera", "OrbbecCameraError", "list_devices", "prepare"]
