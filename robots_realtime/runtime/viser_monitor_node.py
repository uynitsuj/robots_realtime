"""ViserMonitorNode — pure-observation visualization node.

Subscribes to joint-state and camera-RGB topics, renders:
  * one ``viser.extras.ViserUrdf`` per arm, configured from the latest joint_pos
  * one viser GUI image panel per camera, resized to a small preview size

No IK, no command publishing, no agent — this node exists only to let a human
see what the hardware is doing from any browser pointed at its port.

Auto-opens the viser URL in a browser when a display is detected. Prefers
``chromium`` / ``google-chrome`` kiosk mode for a fullscreen, chromeless
window that we can gracefully kill on ``cleanup()``. Falls back to
``webbrowser.open()`` (standard tab) if no Chromium-family browser is
installed, and skips entirely on headless machines (no ``DISPLAY`` /
``WAYLAND_DISPLAY``).

YAML example::

    - type: ViserMonitorNode
      name: viz
      port: 8080
      viz_freq: 20
      preview_size: [320, 240]
      auto_open_browser: true
      fullscreen: true
      urdfs:
        yam_left:
          path: dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf
          state_topic: yam_left/joint_state
          flip_joints: true      # YAM motor order is reversed vs URDF joint order
          extrinsic:             # optional; omit for an identity pose
            position: [0.0, 0.0, 0.0]
            rotation: [1.0, 0.0, 0.0, 0.0]   # wxyz
        yam_right:
          path: dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf
          state_topic: yam_right/joint_state
          flip_joints: true
          extrinsic:
            position: [0.0, 0.5, 0.0]
            rotation: [1.0, 0.0, 0.0, 0.0]
      image_topics:
        top:   camera_top/rgb
        left:  camera_left/rgb
        right: camera_right/rgb
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import time
import webbrowser
from typing import Any

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole
from robots_realtime.sensors.cameras.camera_utils import resize_with_pad

logger = logging.getLogger(__name__)


# Gripper presets — poses and mesh paths scraped from the i2rt MJCF files.
# We render three static sub-meshes (shell + two tips) attached to the arm's
# attach_link. The two tips each sit inside a sub-frame whose position is
# updated each step from the bus message's ``gripper_pos`` (normalized [0, 1]
# where 0 = closed, 1 = open per i2rt's JointMapper convention).
#
# To add a new gripper type, scrape the same fields from its MJCF file in
# dependencies/i2rt/i2rt/robot_models/gripper/<type>/<type>.xml.
_GRIPPER_I2RT_ROOT = "dependencies/i2rt/i2rt/robot_models/gripper"

GRIPPER_PRESETS: dict[str, dict] = {
    "linear_4310": {
        "shell_stl":        f"{_GRIPPER_I2RT_ROOT}/linear_4310/assets/gripper.stl",
        "tip_left_stl":     f"{_GRIPPER_I2RT_ROOT}/linear_4310/assets/tip_left.stl",
        "tip_right_stl":    f"{_GRIPPER_I2RT_ROOT}/linear_4310/assets/tip_right.stl",
        # Gripper-body frame pose relative to the arm's attach_link (URDF link_6).
        #
        # There's no clean algebraic derivation because the arm MJCF and arm URDF
        # disagree on link_6's orientation — URDF has joint6 rpy=(-π/2, 0, 0),
        # MJCF uses a 120° off-axis quat. The combine_arm_and_gripper_xml pipeline
        # doesn't apply to URDF-based rendering. So these defaults are
        # empirical: 180° about the joint6 axis (local Z) with zero offset.
        # Tune `body_offset_pos` and `body_offset_quat_wxyz` from YAML per arm
        # to dial in the final visual match.
        "body_offset_pos":        (0.0, 0.0, 0.0),
        "body_offset_quat_wxyz":  (0, 0.7071068, 0.7071068, 0 ),
        # Shell mesh pose in the gripper-body frame.
        "shell_pos":        (-0.014, -0.0463995, 0.0731),
        "shell_quat_wxyz":  (1.0, 0.0, 0.0, 0.0),
        # Tip-left body pose in gripper-body frame.
        "tip_left_body_pos":      (-0.0238981, 0.0450619, -0.0545599),
        "tip_left_body_quat_wxyz":(0.499998, -0.5, -0.5, -0.500002),
        # Tip-left mesh pose in tip-left body frame (before slide).
        "tip_left_mesh_pos":       (0.129783, 0.00999321, -0.0914614),
        "tip_left_mesh_quat_wxyz": (0.499998, 0.5, 0.500002, 0.5),
        # Tip-right body pose in gripper-body frame.
        "tip_right_body_pos":      (0.0238981, -0.0450619, -0.0545599),
        "tip_right_body_quat_wxyz":(0.707105, 0.707108, 0.0, 0.0),
        # Tip-right mesh pose in tip-right body frame.
        "tip_right_mesh_pos":       (-0.0379932, 0.129783, 0.00133753),
        "tip_right_mesh_quat_wxyz": (0.707105, -0.707108, 0.0, 0.0),
        # Slide motion: joint7/joint8 translate along local axis with range [0, 0.0475] m.
        "slide_axis": (0.0, 0.0, -1.0),
        "slide_range_m": 0.0475,
    },
}


class ViserMonitorNode(Node):
    """Read-only visualization — URDF overlays and camera panels via viser."""

    role = NodeRole.SENSOR
    poll_freq: float | None = None
    subscriber_driven: bool = False

    def __init__(
        self,
        name: str = "viz",
        port: int = 8080,
        urdfs: dict[str, dict] | None = None,
        image_topics: dict[str, str] | None = None,
        viz_freq: float = 20.0,
        preview_size: tuple[int, int] = (224, 224),
        # How to shape image thumbnails. Use ``center_crop`` + a square
        # ``preview_size`` to display exactly what the policy consumes
        # (matches AsyncDiffusionAgent's ``image_preprocess`` path).
        image_preprocess: str = "pad",
        auto_open_browser: bool = True,
        fullscreen: bool = True,
        # Initial 3D camera view — "+x is the direction the arms point", "+z is up".
        # Default places the viewer ~1.5 m behind the robots, slightly elevated,
        # looking at the approximate workspace centre.
        initial_camera_position: tuple[float, float, float] = (-1.3, 0.3, 0.9),
        initial_camera_look_at: tuple[float, float, float] = (0.45, 0.3, 0.3),
        up_axis: str = "+z",
        # Optional chunk-prediction visualization: subscribe to the agent's
        # `chunk` topic and render N end-effector frames per arm along the
        # predicted trajectory. Each URDF spec opts in via `chunk_arm_key:
        # left|right` matching the chunk payload.
        chunk_topic: str | None = None,
        n_chunk_frames: int = 10,
        chunk_frame_axes_length: float = 0.03,
        chunk_frame_axes_radius: float = 0.0025,
        # Optional anchor-steering visualization: subscribe to the SteeringAgent's
        # `steering` topic and render the K candidate hand paths (selected in green),
        # the bin / hole / active anchor, and the current phase.
        steering_topic: str | None = None,
        # Optional static point clouds (e.g. the depth snapshot the anchors were derived from):
        # list of {path: <.npz with points (N,3) float32 world/cell frame + colors (N,3) uint8>,
        #          name?: str, point_size?: float, visible?: bool}. Loaded once at start.
        point_clouds: list[dict] | None = None,
        # Operator controls for the anchor-steering agent, published on ``<name>/steer_ctrl`` (the AgentNode's
        # ``control_topics`` feeds them to SteeringAgent.set_controls): steering may keep computing/drawing
        # candidates while the robot executes the plain policy sample, and the guidance strength is live.
        steer_control: bool = False,
        steer_control_defaults: dict | None = None,
        writer=None,
        **kwargs,
    ) -> None:
        self._point_clouds_spec = list(point_clouds or [])
        self._cloud_handles: dict = {}
        self._steer_control = bool(steer_control)
        self._steer_ctrl = {"steer_enabled": True, "lam_scale": 0.0}
        self._steer_ctrl.update(steer_control_defaults or {})
        self._steer_ctrl_dirty = True
        self._steer_ctrl_last_pub = 0.0
        self._urdfs_spec = urdfs or {}
        self._image_topics = image_topics or {}
        self._chunk_topic = chunk_topic
        self._steering_topic = steering_topic
        self._steer_handles: dict = {}
        self._steer_gui = None
        self._steer_seen = False
        self._n_chunk_frames = int(max(0, n_chunk_frames))
        self._chunk_axes_length = float(chunk_frame_axes_length)
        self._chunk_axes_radius = float(chunk_frame_axes_radius)
        self.subscribed_topics = (
            [spec["state_topic"] for spec in self._urdfs_spec.values() if "state_topic" in spec]
            + list(self._image_topics.values())
            + ([self._chunk_topic] if self._chunk_topic else [])
            + ([self._steering_topic] if self._steering_topic else [])
        )
        # poll_freq drives how often the URDF/image GUI updates — lower is cheaper.
        self.poll_freq = float(viz_freq)
        super().__init__(name=name, writer=writer, **kwargs)

        self._port = int(port)
        # preview_size is (width, height) — user convention. Converted to
        # (height, width) when calling resize_with_pad which takes H, W.
        if len(preview_size) != 2:
            raise ValueError(f"preview_size must be (width, height); got {preview_size!r}")
        self._preview_w = int(preview_size[0])
        self._preview_h = int(preview_size[1])
        if image_preprocess not in ("pad", "center_crop"):
            raise ValueError(
                f"image_preprocess must be 'pad' or 'center_crop'; got {image_preprocess!r}"
            )
        self._image_preprocess = image_preprocess
        self._auto_open_browser = bool(auto_open_browser)
        self._fullscreen = bool(fullscreen)
        self._initial_camera_position = tuple(float(v) for v in initial_camera_position)
        self._initial_camera_look_at = tuple(float(v) for v in initial_camera_look_at)
        self._up_axis = up_axis

        # Initialised in setup()
        self._server: Any = None
        self._urdf_vis: dict[str, Any] = {}
        self._urdfs: dict[str, Any] = {}        # raw yourdfpy.URDF, for FK to drive grippers + chunks
        self._gripper_state: dict[str, dict] = {}
        self._image_handles: dict[str, Any] = {}
        # chunk frame handles: arm_key -> list of viser frame handles (one per
        # downsampled chunk step). Pre-allocated in setup() so step() only has
        # to update positions/quats — no viser create/destroy per tick.
        self._chunk_frames: dict[str, list] = {}
        self._chunk_tip_offset_T: dict[str, np.ndarray] = {}
        self._browser_proc: subprocess.Popen | None = None

    # Surfaced on the session TUI (see tui.py:_endpoints_text).
    @property
    def web_endpoints(self) -> list[str]:
        return [f"viser: http://localhost:{self._port}"]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        import viser  # noqa: PLC0415
        import viser.extras  # noqa: PLC0415
        import viser.transforms as vtf  # noqa: PLC0415
        import yourdfpy  # noqa: PLC0415
        self._vtf = vtf

        self._server = viser.ViserServer(port=self._port)
        logger.info("[%s] viser server listening on http://localhost:%d", self.name, self._port)

        # World-up direction — determines the natural axis for the orbit
        # controls (e.g. rolling around pitches, not yaws).
        try:
            self._server.scene.set_up_direction(self._up_axis)
        except Exception as exc:
            logger.debug("[%s] set_up_direction(%r) failed: %s", self.name, self._up_axis, exc)

        # Set the initial camera pose for every client that connects.
        init_pos = np.asarray(self._initial_camera_position, dtype=np.float32)
        init_target = np.asarray(self._initial_camera_look_at, dtype=np.float32)

        @self._server.on_client_connect
        def _set_initial_view(client) -> None:  # noqa: ANN001 — viser client type
            client.camera.position = init_pos
            client.camera.look_at = init_target

        for arm_key, spec in self._urdfs_spec.items():
            urdf_path = os.path.abspath(os.path.expanduser(spec["path"]))
            if not os.path.isfile(urdf_path):
                raise FileNotFoundError(f"[{self.name}] URDF for {arm_key!r} not found at {urdf_path}")
            mesh_dir = spec.get("mesh_dir")
            if mesh_dir is not None:
                mesh_dir = os.path.abspath(os.path.expanduser(mesh_dir))
            else:
                # Default sibling "assets/" next to the URDF — matches i2rt layout.
                default_mesh = os.path.join(os.path.dirname(urdf_path), "assets")
                if os.path.isdir(default_mesh):
                    mesh_dir = default_mesh

            urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir) if mesh_dir else yourdfpy.URDF.load(urdf_path)

            root = spec.get("root_node_name", f"/{arm_key}")
            frame = self._server.scene.add_frame(root, show_axes=bool(spec.get("show_axes", False)))
            extrinsic = spec.get("extrinsic")
            if extrinsic is not None:
                if "position" in extrinsic:
                    frame.position = np.asarray(extrinsic["position"], dtype=np.float32)
                if "rotation" in extrinsic:
                    frame.wxyz = np.asarray(extrinsic["rotation"], dtype=np.float32)

            urdf_kwargs: dict = {"root_node_name": root}
            if "mesh_color" in spec and spec["mesh_color"] is not None:
                urdf_kwargs["mesh_color_override"] = tuple(spec["mesh_color"])
            urdf_vis = viser.extras.ViserUrdf(self._server, urdf, **urdf_kwargs)

            opacity = spec.get("opacity")
            if opacity is not None:
                for mesh in urdf_vis._meshes:
                    try:
                        mesh.opacity = float(opacity)
                    except Exception:
                        pass

            self._urdf_vis[arm_key] = urdf_vis
            self._urdfs[arm_key] = urdf
            logger.info("[%s] URDF loaded: %s (root=%s, meshes=%d)", self.name, arm_key, root, len(urdf_vis._meshes))

            gripper_spec = spec.get("gripper")
            if gripper_spec is not None:
                self._add_gripper(arm_key, root, gripper_spec)

            # Pre-allocate chunk prediction frames for this arm if it opted in.
            if self._chunk_topic and spec.get("chunk_arm_key") and self._n_chunk_frames > 0:
                self._init_chunk_frames(arm_key, root, spec)

        for i, spec in enumerate(self._point_clouds_spec):
            path = spec.get("path")
            name = str(spec.get("name") or f"cloud_{i}")
            try:
                z = np.load(path)
                pts = np.asarray(z["points"], dtype=np.float32)
                cols = np.asarray(z["colors"], dtype=np.uint8) if "colors" in z else np.full((len(pts), 3), 200, np.uint8)
                h = self._server.scene.add_point_cloud(f"/clouds/{name}", pts, cols,
                                                       point_size=float(spec.get("point_size", 0.004)),
                                                       point_shape="circle", precision="float16")
                h.visible = bool(spec.get("visible", True))
                self._cloud_handles[name] = h
                cb = self._server.gui.add_checkbox(f"cloud: {name}", h.visible)

                def _mk(handle):
                    def _cb(ev):
                        handle.visible = ev.target.value
                    return _cb
                cb.on_update(_mk(h))
                print(f"[{self.name}] point cloud '{name}': {len(pts)} pts from {path}", flush=True)
            except Exception as exc:
                print(f"[{self.name}] point cloud '{name}' failed ({path}): {type(exc).__name__}: {exc}", flush=True)

        if self._steer_control:
            with self._server.gui.add_folder("steering control"):
                cb_ctrl = self._server.gui.add_checkbox("steering controls the robot", bool(self._steer_ctrl["steer_enabled"]))
                sl_lam = self._server.gui.add_slider("guidance strength (λ scale)", 0.0, 1.0, 0.05, float(self._steer_ctrl["lam_scale"]))
                self._server.gui.add_markdown(
                    "Off = the agent still asks for K candidates, ranks and draws them, but the robot executes the "
                    "plain (unsteered) sample. λ scale 0 = selection only; > 0 adds gradient guidance toward the target.")

            @cb_ctrl.on_update
            def _(_ev):
                self._steer_ctrl["steer_enabled"] = bool(cb_ctrl.value)
                self._steer_ctrl_dirty = True

            @sl_lam.on_update
            def _(_ev):
                self._steer_ctrl["lam_scale"] = float(sl_lam.value)
                self._steer_ctrl_dirty = True

        if self._auto_open_browser:
            self._open_browser()

    def _add_gripper(self, arm_key: str, arm_root: str, gripper_spec: dict) -> None:
        """Attach a gripper's 3 meshes + 2 animated tip frames under the arm's attach_link.

        The root gripper frame is placed at the arm's attach_link pose via FK each
        ``step()``; the shell is static under it, and each tip sits inside a
        translation-only sub-frame whose position is driven by ``gripper_pos``.
        """
        import trimesh  # noqa: PLC0415 — dependency of viser anyway

        gtype = gripper_spec.get("type", "linear_4310")
        if gtype not in GRIPPER_PRESETS:
            logger.warning(
                "[%s] unknown gripper type %r; supported: %s",
                self.name, gtype, sorted(GRIPPER_PRESETS.keys()),
            )
            return
        preset = dict(GRIPPER_PRESETS[gtype])   # shallow copy so we don't mutate the shared preset
        # YAML may override the body-offset (attach_link → gripper body frame)
        # empirically — URDF/MJCF frame conventions for "link_6" disagree and
        # the preset's default may still look slightly off. Tweak these two
        # fields in the YAML to align the shell visually with the hardware.
        if "body_offset_pos" in gripper_spec:
            preset["body_offset_pos"] = tuple(float(v) for v in gripper_spec["body_offset_pos"])
        if "body_offset_quat_wxyz" in gripper_spec:
            preset["body_offset_quat_wxyz"] = tuple(float(v) for v in gripper_spec["body_offset_quat_wxyz"])
        attach_link = gripper_spec.get("attach_link", "link_6")
        if attach_link not in self._urdfs[arm_key].link_map:
            logger.warning("[%s] attach_link %r not in URDF for %s", self.name, attach_link, arm_key)
            return

        def _load_mesh(rel_path: str):
            full = os.path.abspath(os.path.expanduser(rel_path))
            if not os.path.isfile(full):
                raise FileNotFoundError(f"gripper mesh not found: {full}")
            return trimesh.load(full, force="mesh")

        # IMPORTANT: parent the gripper under the arm's root frame so it inherits
        # the arm's extrinsic (e.g. the right arm's +Y offset). A sibling path
        # would detach the gripper from the arm's world pose and make it float.
        gripper_root_path = f"{arm_root}/gripper"
        gripper_root = self._server.scene.add_frame(gripper_root_path, show_axes=False)

        # Static shell mesh.
        shell_mesh = _load_mesh(preset["shell_stl"])
        self._server.scene.add_mesh_trimesh(
            f"{gripper_root_path}/shell",
            shell_mesh,
            position=np.asarray(preset["shell_pos"], dtype=np.float32),
            wxyz=np.asarray(preset["shell_quat_wxyz"], dtype=np.float32),
        )

        # Tip frames (body) + slide sub-frames + meshes.
        tip_handles: dict[str, Any] = {}
        for side in ("left", "right"):
            body_pos = np.asarray(preset[f"tip_{side}_body_pos"], dtype=np.float32)
            body_quat = np.asarray(preset[f"tip_{side}_body_quat_wxyz"], dtype=np.float32)
            mesh_pos = np.asarray(preset[f"tip_{side}_mesh_pos"], dtype=np.float32)
            mesh_quat = np.asarray(preset[f"tip_{side}_mesh_quat_wxyz"], dtype=np.float32)

            body_path = f"{gripper_root_path}/tip_{side}"
            slide_path = f"{body_path}/slide"
            mesh_path = f"{slide_path}/mesh"

            body_frame = self._server.scene.add_frame(body_path, show_axes=False)
            body_frame.position = body_pos
            body_frame.wxyz = body_quat

            slide_frame = self._server.scene.add_frame(slide_path, show_axes=False)
            slide_frame.position = np.zeros(3, dtype=np.float32)

            tip_mesh = _load_mesh(preset[f"tip_{side}_stl"])
            self._server.scene.add_mesh_trimesh(
                mesh_path, tip_mesh, position=mesh_pos, wxyz=mesh_quat,
            )
            tip_handles[side] = slide_frame

        # Precompute the 4x4 body-offset transform (attach_link → gripper-body
        # frame) so step() only needs a single matmul with the link FK pose.
        body_R = self._vtf.SO3(
            np.asarray(preset["body_offset_quat_wxyz"], dtype=np.float64)
        ).as_matrix()
        body_T = np.eye(4, dtype=np.float64)
        body_T[:3, :3] = body_R
        body_T[:3, 3] = np.asarray(preset["body_offset_pos"], dtype=np.float64)

        self._gripper_state[arm_key] = {
            "urdf": self._urdfs[arm_key],
            "attach_link": attach_link,
            "root_frame": gripper_root,
            "tip_slide_frames": tip_handles,
            "slide_axis": np.asarray(preset["slide_axis"], dtype=np.float32),
            "slide_range_m": float(preset["slide_range_m"]),
            "body_offset_T": body_T,
        }
        logger.info("[%s] gripper attached: %s / %s @ %s", self.name, arm_key, gtype, attach_link)

    def _init_chunk_frames(self, arm_key: str, arm_root: str, spec: dict) -> None:
        """Allocate ``n_chunk_frames`` viser frame handles per arm for chunk predictions.

        Frames are children of the arm's root frame (so they inherit the same
        extrinsic as the arm itself). We update their `position`/`wxyz` each
        tick from FK of downsampled chunk actions — no create/destroy per step.
        """
        # Ramp down axes_length across chunk index so "further into the future"
        # is visually smaller — quick feedback that later predictions are less
        # trusted without needing a separate legend.
        handles = []
        base_len = self._chunk_axes_length
        for i in range(self._n_chunk_frames):
            # linear ramp from 1.0 → 0.35 across i ∈ [0, N-1]
            shrink = 1.0 - 0.65 * (i / max(1, self._n_chunk_frames - 1))
            h = self._server.scene.add_frame(
                f"{arm_root}/chunk_{i}",
                show_axes=True,
                axes_length=base_len * shrink,
                axes_radius=self._chunk_axes_radius * shrink,
            )
            # Start hidden until first chunk message lands.
            h.visible = False
            handles.append(h)
        self._chunk_frames[arm_key] = handles

        # Store the tip-of-gripper offset in the attach_link frame for this arm.
        # If a gripper is configured, we want the predicted frame to land at the
        # grasp site (≈ 13.5 cm past link_6 along the gripper's -Z) rather than
        # at link_6 itself. Keep identity if no gripper for that arm.
        offset_T = np.eye(4, dtype=np.float64)
        gs = self._gripper_state.get(arm_key)
        if gs is not None:
            # body_offset_T takes attach_link → gripper body; additionally
            # offset along gripper -Z by a representative grasp-site length.
            grasp_offset = np.eye(4, dtype=np.float64)
            grasp_offset[:3, 3] = np.array([0.0, 0.0, -0.1347])   # linear_4310 grasp_site
            offset_T = gs["body_offset_T"] @ grasp_offset
        self._chunk_tip_offset_T[arm_key] = offset_T

        logger.info(
            "[%s] chunk prediction viz: arm=%s chunk_key=%s n_frames=%d",
            self.name, arm_key, spec.get("chunk_arm_key"), self._n_chunk_frames,
        )

    def _update_steering(self, msg: dict) -> None:
        """Render the SteeringAgent snapshot: candidate hand paths (cell frame == this scene's
        world frame), the bin/hole/active anchor spheres, and a phase read-out in the GUI.

        Everything is drawn under /steering; nodes are addressed by a stable name so a re-add
        overwrites in place, and paths that disappear when K shrinks or steering stops are removed.
        """
        import numpy as _np
        want: set[str] = set()

        steered = bool(msg.get("steered", False))
        paths = msg.get("paths")
        sel = int(msg.get("selected", 0))
        if steered and paths is not None:
            paths = _np.asarray(paths, dtype=_np.float32)        # (K, H, 3)
            K = paths.shape[0]
            scores = msg.get("scores")
            # colour by rank of the score (closest approach to the target): best = bright green, worst = grey-blue
            if scores is not None and len(scores) == K:
                order = _np.argsort(_np.asarray(scores, dtype=float))
                rank = _np.empty(K, dtype=float); rank[order] = _np.linspace(0.0, 1.0, K) if K > 1 else 0.0
            else:
                rank = _np.ones(K); rank[sel] = 0.0
            good, bad = _np.array([30, 220, 30], float), _np.array([95, 95, 120], float)
            for k in range(K):
                name = f"/steering/cand/{k:02d}"
                want.add(name)
                is_sel = (k == sel)
                col = tuple(int(c) for c in (good * (1 - rank[k]) + bad * rank[k]))
                self._steer_handles[name] = self._server.scene.add_spline_catmull_rom(
                    name, points=paths[k], line_width=6.0 if is_sel else 1.5, color=col)
            want.add("/steering/selected_end")                   # endpoint of the EXECUTED chunk (green: steered, orange: plain)
            executed_is_steered = bool(msg.get("steer_enabled", True))
            self._steer_handles["/steering/selected_end"] = self._server.scene.add_icosphere(
                "/steering/selected_end", radius=0.012, color=(30, 220, 30) if executed_is_steered else (255, 140, 0),
                position=tuple(paths[sel, -1].astype(float)))

        # raw VLM/depth points (small grey) linked to the offset targets they produce
        for key in ("bin", "hole"):
            raw, tgt = msg.get(f"{key}_point"), msg.get(key)
            if raw is not None:
                name = f"/steering/{key}_point"
                want.add(name)
                self._steer_handles[name] = self._server.scene.add_icosphere(
                    name, radius=0.009, color=(200, 200, 200), position=tuple(_np.asarray(raw, dtype=float)))
                if tgt is not None:
                    lname = f"/steering/{key}_link"
                    want.add(lname)
                    seg = _np.stack([_np.asarray(raw, dtype=_np.float32), _np.asarray(tgt, dtype=_np.float32)])
                    self._steer_handles[lname] = self._server.scene.add_spline_catmull_rom(
                        lname, points=seg, line_width=1.0, color=(255, 140, 0) if key == "bin" else (0, 200, 255))

        for key, col, rad in (("anchor", (235, 30, 30), 0.022), ("bin", (255, 140, 0), 0.016), ("hole", (0, 200, 255), 0.016)):
            p = msg.get(key)
            if p is not None:
                name = f"/steering/{key}"
                want.add(name)
                self._steer_handles[name] = self._server.scene.add_icosphere(
                    name, radius=rad, color=col, position=tuple(_np.asarray(p, dtype=float)))

        # wrist-camera gaze (SteeringAgent gaze feature): per arm a magenta ray from the camera along its optical
        # axis and a small magenta sphere on the point it should look at
        gaze = msg.get("gaze") or {}
        cams, axes = msg.get("gaze_cam") or {}, msg.get("gaze_axis") or {}
        for arm, tgt in gaze.items():
            tname = f"/steering/gaze/{arm}/target"
            want.add(tname)
            self._steer_handles[tname] = self._server.scene.add_icosphere(
                tname, radius=0.012, color=(230, 60, 230), position=tuple(_np.asarray(tgt, dtype=float)))
            if arm in cams and arm in axes:
                c = _np.asarray(cams[arm], dtype=_np.float32); a = _np.asarray(axes[arm], dtype=_np.float32)
                rname = f"/steering/gaze/{arm}/ray"
                want.add(rname)
                self._steer_handles[rname] = self._server.scene.add_spline_catmull_rom(
                    rname, points=_np.stack([c, c + 0.4 * a]), line_width=2.5, color=(230, 60, 230))

        for name in list(self._steer_handles):                    # drop stale nodes (K shrank / steering stopped)
            if name not in want:
                try:
                    self._steer_handles.pop(name).remove()
                except Exception:
                    pass

        if self._steer_gui is None:
            try:
                self._steer_gui = self._server.gui.add_text("steering", initial_value="", disabled=True)
                self._server.gui.add_markdown(
                    "**steering overlay** — paths: the K candidate chunks' hand paths, coloured by rank of closest approach "
                    "to the target (green = best … grey = worst); the thick one with the green end-sphere is the executed chunk. "
                    "Spheres: red = active target of the phase; orange = bin target; cyan = hole (release) target; small grey "
                    "= the raw VLM/depth point each target is offset from (thin line). Magenta: wrist-camera gaze — the ray is "
                    "the camera's optical axis, the sphere the point it should look at (shake / gripper primitives are named "
                    "in the text field).")
            except Exception:
                self._steer_gui = None
        if self._steer_gui is not None:
            phase = msg.get("phase", "?")
            if msg.get("primitive"):
                phase = f"{phase} [{msg['primitive']}]"
            robot = "STEERED" if bool(msg.get("steer_enabled", True)) else "PLAIN sample (steering off)"
            if steered:
                self._steer_gui.value = (f"{phase}  arm={msg.get('arm','?')}  exec k={sel} best k={int(msg.get('best', sel))}/{int(msg.get('K', 0))}  "
                                         f"closest {100 * float(msg.get('d_best', 0.0)):.1f} cm (spread {100 * float(msg.get('spread', 0.0)):.1f})  "
                                         f"λ={float(msg.get('lam_scale', 0.0)):.2f}  robot: {robot}")
            else:
                self._steer_gui.value = f"{phase}  (no target this phase)  robot: {robot}"

    def _update_chunk_frames(self, chunk_msg: dict) -> None:
        """Per-tick: run FK on downsampled chunk actions, update frame handles."""
        for arm_key, spec in self._urdfs_spec.items():
            chunk_key = spec.get("chunk_arm_key")
            if chunk_key is None or arm_key not in self._chunk_frames:
                continue
            arm_chunk = chunk_msg.get(chunk_key)
            if arm_chunk is None:
                continue
            arm_chunk = np.asarray(arm_chunk)
            if arm_chunk.ndim != 2 or arm_chunk.shape[1] < 6:
                continue

            urdf = self._urdfs[arm_key]
            handles = self._chunk_frames[arm_key]
            attach_link = spec.get("gripper", {}).get("attach_link", "link_6")
            tip_offset = self._chunk_tip_offset_T[arm_key]

            # Downsample to N evenly-spaced indices across whatever the agent
            # sent (the server chunk length can be anywhere from 1 to 30+).
            n_avail = arm_chunk.shape[0]
            n_show = min(len(handles), n_avail)
            if n_show <= 0:
                for h in handles:
                    h.visible = False
                continue
            idx = np.linspace(0, n_avail - 1, n_show).astype(int)

            # Save current URDF cfg so we can restore after predictive FK —
            # otherwise the arm URDF would visually snap to the last chunk
            # action instead of reflecting the live joint_state.
            saved_cfg = None
            try:
                saved_cfg = np.asarray(urdf.cfg, dtype=np.float64).copy()
            except Exception:
                pass

            for slot, action_idx in enumerate(idx):
                cfg = np.asarray(arm_chunk[action_idx][:6], dtype=np.float64)
                if spec.get("flip_joints", True):
                    cfg = np.flip(cfg)
                try:
                    urdf.update_cfg(cfg)
                    T = urdf.get_transform(attach_link) @ tip_offset
                except Exception:
                    handles[slot].visible = False
                    continue
                handles[slot].position = T[:3, 3].astype(np.float32)
                handles[slot].wxyz = self._vtf.SO3.from_matrix(T[:3, :3]).wxyz.astype(np.float32)
                handles[slot].visible = True

            # Any unused slots (e.g. chunk got shorter near drain) → hide.
            for slot in range(n_show, len(handles)):
                handles[slot].visible = False

            # Restore live cfg so the URDF mesh keeps reflecting real joint state.
            if saved_cfg is not None:
                try:
                    urdf.update_cfg(saved_cfg)
                except Exception:
                    pass

    def step(self) -> None:
        # Update URDF configs from the latest joint state.
        for arm_key, spec in self._urdfs_spec.items():
            topic = spec.get("state_topic")
            if topic is None:
                continue
            data = self.get_latest(topic)
            if data is None:
                continue
            jp = data.get("joint_pos")
            if jp is None:
                continue
            cfg = np.asarray(jp, dtype=np.float64)
            if spec.get("flip_joints", True):
                cfg = np.flip(cfg)
            try:
                # ViserUrdf.update_cfg accepts whatever length matches the URDF's
                # actuated joints. If the bus publishes more entries (e.g. gripper),
                # trim to the URDF's joint count.
                expected = len(self._urdf_vis[arm_key]._urdf.actuated_joint_names)  # type: ignore[attr-defined]
                trimmed_cfg = cfg[:expected]
                self._urdf_vis[arm_key].update_cfg(trimmed_cfg)
                # Drive the raw yourdfpy URDF too so get_transform() reflects the
                # current pose when the gripper FK below asks for attach_link.
                if arm_key in self._urdfs:
                    self._urdfs[arm_key].update_cfg(trimmed_cfg)
            except Exception as exc:
                logger.debug("[%s] URDF update for %s failed: %s", self.name, arm_key, exc)

            # If a gripper is attached, place its root frame at the arm's
            # attach_link pose (FK), then animate the tip slides from gripper_pos.
            gs = self._gripper_state.get(arm_key)
            if gs is not None:
                try:
                    # Compose attach_link FK (URDF convention) with the MJCF's
                    # gripper-body offset so our gripper-root frame lands at the
                    # body frame the shell / tip offsets are defined in.
                    T_fk = gs["urdf"].get_transform(gs["attach_link"])
                    T_world_body = T_fk @ gs["body_offset_T"]
                    gs["root_frame"].position = T_world_body[:3, 3].astype(np.float32)
                    gs["root_frame"].wxyz = self._vtf.SO3.from_matrix(T_world_body[:3, :3]).wxyz.astype(np.float32)
                except Exception as exc:
                    logger.debug("[%s] FK for gripper %s failed: %s", self.name, arm_key, exc)

                gp_raw = data.get("gripper_pos")
                if gp_raw is not None:
                    gp = float(np.asarray(gp_raw).reshape(-1)[0])
                    gp = max(0.0, min(1.0, gp))   # i2rt command space: 0 closed, 1 open
                    slide_m = gp * gs["slide_range_m"]
                    offset = (gs["slide_axis"] * slide_m).astype(np.float32)
                    for frame in gs["tip_slide_frames"].values():
                        frame.position = offset

        # Update chunk prediction frames (if enabled).
        if self._chunk_topic and self._chunk_frames:
            chunk_msg = self.get_latest(self._chunk_topic)
            if chunk_msg is not None:
                self._update_chunk_frames(chunk_msg)

        # Publish the operator's steering controls (on change + 2 s heartbeat so a late agent catches up).
        if self._steer_control:
            now = time.monotonic()
            if self._steer_ctrl_dirty or now - self._steer_ctrl_last_pub > 2.0:
                try:
                    self.publish("steer_ctrl", dict(self._steer_ctrl), record=False)
                    self._steer_ctrl_dirty, self._steer_ctrl_last_pub = False, now
                except Exception as exc:
                    logger.debug("[%s] steer_ctrl publish failed: %s", self.name, exc)

        # Update anchor-steering overlay (if enabled).
        if self._steering_topic:
            steer_msg = self.get_latest(self._steering_topic)
            if steer_msg is not None:
                try:
                    if not self._steer_seen:
                        self._steer_seen = True
                        # stdout: node stdout is what the session TUI shows (node loggers go to the temp log dir)
                        print(f"[{self.name}] steering overlay: first message (phase={steer_msg.get('phase')}, "
                              f"K={steer_msg.get('K')}, steered={steer_msg.get('steered')})", flush=True)
                    self._update_steering(steer_msg)
                except Exception as exc:
                    if not getattr(self, "_steer_err_printed", False):
                        self._steer_err_printed = True
                        print(f"[{self.name}] steering overlay update failed: {type(exc).__name__}: {exc}", flush=True)
                    logger.debug("[%s] steering overlay update failed: %s", self.name, exc)

        # Update camera panels.
        for label, topic in self._image_topics.items():
            msg = self.get_latest(topic)
            if msg is None:
                continue
            img = self._extract_rgb(msg)
            if img is None:
                continue
            # resize_with_pad takes (image, height, width) — not (w, h). The
            # preview_size kwarg is (width, height) for ergonomic YAML config,
            # so we transpose here.
            if self._image_preprocess == "center_crop":
                h, w = img.shape[:2]
                side = min(h, w)
                h0 = (h - side) // 2
                w0 = (w - side) // 2
                img = img[h0:h0 + side, w0:w0 + side]
            thumb = resize_with_pad(img, self._preview_h, self._preview_w)
            if label not in self._image_handles:
                self._image_handles[label] = self._server.gui.add_image(thumb, label=label)
            else:
                self._image_handles[label].image = thumb

    def cleanup(self) -> None:
        # Close the browser we spawned (best-effort).
        if self._browser_proc is not None:
            try:
                self._browser_proc.terminate()
                try:
                    self._browser_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._browser_proc.kill()
            except Exception as exc:
                logger.debug("[%s] browser terminate failed: %s", self.name, exc)
            self._browser_proc = None

        # Stop the viser server.
        if self._server is not None:
            try:
                self._server.stop()
            except Exception as exc:
                logger.debug("[%s] viser stop failed: %s", self.name, exc)
            self._server = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_rgb(msg: dict) -> np.ndarray | None:
        """CameraNode publishes ``{"images": {"rgb": ndarray}, ...}``; also
        tolerate older ``{"frame": ndarray}`` and ``{"rgb": ndarray}`` shapes."""
        if not isinstance(msg, dict):
            return None
        images = msg.get("images")
        if isinstance(images, dict):
            arr = images.get("rgb")
            if arr is None and images:
                arr = next(iter(images.values()))
            if arr is not None:
                return np.asarray(arr)
        for key in ("frame", "rgb"):
            arr = msg.get(key)
            if arr is not None:
                return np.asarray(arr)
        return None

    @staticmethod
    def _active_local_graphical_session_for_uid(uid: int) -> dict[str, str] | None:
        """Return env overrides for the current user's active local X/Wayland session, or None.

        Uses ``loginctl`` to find a session where:
          * ``State=active`` (currently in front of the user)
          * ``Remote=no``   (it's a local seat, not SSH)
          * ``Type`` is ``x11`` or ``wayland``
          * ``User`` matches ``uid`` (so we have read access to its Xauthority)

        This is far more reliable than ``/sys/class/drm/*/status`` — DRM reports
        "disconnected" when the monitor is in DPMS standby or the driver hasn't
        polled recently, giving false negatives even when a display is plugged in.
        A real graphical session in loginctl implies a real physical display.

        Returns the env dict needed to spawn a child process into that session
        (DISPLAY, WAYLAND_DISPLAY, XAUTHORITY, XDG_RUNTIME_DIR as applicable),
        or None if no such session exists.
        """
        if shutil.which("loginctl") is None:
            return None
        try:
            out = subprocess.check_output(
                ["loginctl", "list-sessions", "--no-legend"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode("utf-8", "replace")
        except Exception:
            return None

        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            sid = parts[0]
            try:
                props_out = subprocess.check_output(
                    [
                        "loginctl",
                        "show-session",
                        sid,
                        "--property=Type",
                        "--property=State",
                        "--property=Remote",
                        "--property=User",
                        "--property=Display",
                    ],
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                ).decode("utf-8", "replace")
            except Exception:
                continue
            props: dict[str, str] = {}
            for prop_line in props_out.splitlines():
                if "=" in prop_line:
                    k, v = prop_line.split("=", 1)
                    props[k.strip()] = v.strip()

            if props.get("State") != "active":
                continue
            if props.get("Remote") == "yes":
                continue
            stype = props.get("Type")
            if stype not in ("x11", "wayland"):
                continue
            try:
                session_uid = int(props.get("User", "-1"))
            except ValueError:
                continue
            if session_uid != uid:
                continue

            env_overrides: dict[str, str] = {}
            runtime_dir = f"/run/user/{uid}"
            if os.path.isdir(runtime_dir):
                env_overrides["XDG_RUNTIME_DIR"] = runtime_dir

            if stype == "x11":
                # Prefer the DISPLAY the session advertised (e.g. ":1"); fall back
                # to any local X socket under /tmp/.X11-unix.
                disp = props.get("Display")
                if not disp:
                    for sock in sorted(glob.glob("/tmp/.X11-unix/X*")):
                        num = sock.rsplit("X", 1)[-1]
                        if num.isdigit():
                            disp = f":{num}"
                            break
                if not disp:
                    continue
                env_overrides["DISPLAY"] = disp
                # gdm writes the user's X cookies to /run/user/<uid>/gdm/Xauthority;
                # fall back to the user's $HOME/.Xauthority (pre-systemd).
                for candidate in (
                    f"/run/user/{uid}/gdm/Xauthority",
                    os.path.expanduser(f"~{os.getlogin()}/.Xauthority"),
                ):
                    if os.path.isfile(candidate):
                        env_overrides["XAUTHORITY"] = candidate
                        break
            else:  # wayland
                for sock in sorted(glob.glob(f"{runtime_dir}/wayland-*")):
                    if sock.endswith(".lock"):
                        continue
                    env_overrides["WAYLAND_DISPLAY"] = os.path.basename(sock)
                    break
                if "WAYLAND_DISPLAY" not in env_overrides:
                    continue

            return env_overrides
        return None

    @staticmethod
    def _fallback_display_from_sockets() -> dict[str, str]:
        """Best-effort DISPLAY discovery when loginctl isn't available or finds nothing.

        Just points at the first X socket or Wayland socket we can find —
        works when the current user has a running session on the local seat
        but loginctl isn't exposing it cleanly.
        """
        for sock in sorted(glob.glob("/tmp/.X11-unix/X*")):
            num = sock.rsplit("X", 1)[-1]
            if num.isdigit():
                return {"DISPLAY": f":{num}"}
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.geteuid()}"
        if os.path.isdir(runtime_dir):
            for sock in sorted(glob.glob(f"{runtime_dir}/wayland-*")):
                if sock.endswith(".lock"):
                    continue
                return {"WAYLAND_DISPLAY": os.path.basename(sock), "XDG_RUNTIME_DIR": runtime_dir}
        return {}

    def _open_browser(self) -> None:
        url = f"http://localhost:{self._port}"

        # Only auto-launch when BOTH of these are true:
        #   1. This terminal has no DISPLAY / WAYLAND_DISPLAY set (we're in SSH
        #      or a bare tty) — because if the user is sitting at the machine
        #      running the session from a graphical terminal, chromium --kiosk
        #      would cover the terminal and kiosk mode traps input, leaving
        #      them no way to stop the session.
        #   2. A local graphical session (monitor + logged-in desktop) exists
        #      for our uid — so there's actually a physical display to pop up on.
        this_terminal_is_graphical = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if this_terminal_is_graphical:
            logger.info(
                "[%s] launched from a graphical terminal (DISPLAY set) — "
                "skipping auto-browser so the console stays reachable. Visit %s",
                self.name, url,
            )
            return

        env_overrides = self._active_local_graphical_session_for_uid(os.geteuid())
        source = "loginctl"
        if not env_overrides:
            env_overrides = self._fallback_display_from_sockets()
            source = "socket_probe"
        if not env_overrides:
            logger.info(
                "[%s] no local graphical session found for uid=%d — skipping auto-browser; visit %s",
                self.name, os.geteuid(), url,
            )
            return

        display_repr = (
            env_overrides.get("DISPLAY") or env_overrides.get("WAYLAND_DISPLAY") or "?"
        )

        # A display is targetable. Prefer a Chromium-family browser in kiosk
        # mode so the window is fullscreen and we can kill the process on
        # cleanup. Firefox also has a --kiosk flag. Fall back to the stdlib
        # webbrowser module if none of those are installed.
        kiosk_candidates: list[list[str]] = []
        if self._fullscreen:
            for browser in (
                "chromium",
                "chromium-browser",
                "google-chrome",
                "google-chrome-stable",
                "firefox",
            ):
                bin_path = shutil.which(browser)
                if bin_path is None:
                    continue
                kiosk_candidates.append([bin_path, "--kiosk", url])

        child_env = os.environ.copy()
        child_env.update(env_overrides)

        for cmd in kiosk_candidates:
            try:
                self._browser_proc = subprocess.Popen(
                    cmd,
                    env=child_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.info(
                    "[%s] opened %s in kiosk mode (via %s, display=%s) -> %s",
                    self.name, os.path.basename(cmd[0]), source, display_repr, url,
                )
                return
            except Exception as exc:
                logger.warning("[%s] failed to spawn %s: %s", self.name, cmd[0], exc)

        # Fallback: regular tab (can't close on exit, but at least opens).
        try:
            webbrowser.open(url)
            logger.info("[%s] opened default browser (fallback) -> %s", self.name, url)
        except Exception as exc:
            logger.warning("[%s] could not open browser at %s: %s", self.name, url, exc)

    # ------------------------------------------------------------------ #
    # YAML wiring
    # ------------------------------------------------------------------ #

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name":                    params["name"],
            "port":                    params.get("port", 8080),
            "urdfs":                   params.get("urdfs") or {},
            "image_topics":            params.get("image_topics") or {},
            "viz_freq":                params.get("viz_freq", 20.0),
            "preview_size":            tuple(params.get("preview_size", (240, 180))),
            "image_preprocess":        params.get("image_preprocess", "pad"),
            "auto_open_browser":       params.get("auto_open_browser", True),
            "fullscreen":              params.get("fullscreen", True),
            "initial_camera_position": tuple(params.get("initial_camera_position", (-1.3, 0.3, 0.9))),
            "initial_camera_look_at":  tuple(params.get("initial_camera_look_at",  (0.45, 0.3, 0.3))),
            "up_axis":                 params.get("up_axis", "+z"),
            "chunk_topic":             params.get("chunk_topic"),
            "steering_topic":          params.get("steering_topic"),
            "point_clouds":            params.get("point_clouds") or [],
            "steer_control":           bool(params.get("steer_control", False)),
            "steer_control_defaults":  params.get("steer_control_defaults") or {},
            "n_chunk_frames":          int(params.get("n_chunk_frames", 10)),
            "chunk_frame_axes_length": float(params.get("chunk_frame_axes_length", 0.03)),
            "chunk_frame_axes_radius": float(params.get("chunk_frame_axes_radius", 0.0025)),
        }
