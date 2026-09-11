"""Viser joint-slider agent for bringing up a single YAM arm.

One slider per joint (radians, bounded by the robot config's ``joint_limits``)
plus a normalized gripper slider. :class:`~robots_realtime.runtime.viser_teleop_node.ViserTeleopNode`
calls :meth:`act` at ``ik_freq`` and publishes the returned joint vector to a
``RobotNode``. No IK, no gizmos — this is the "does every joint move?" test.

Scene contents
--------------
* Opaque URDF  — the *commanded* pose (slew-limited slider targets).
* Ghost URDF   — the *measured* pose fed back from ``{arm}/joint_state``.
* GUI folders  — Safety (arm / sync / speed), Joints (sliders), Measured
                 (live readouts in radians), Status.

Safety behaviour
----------------
* **Disarmed = silent.** While the "Armed" checkbox is off, :meth:`act` returns
  ``{}`` and ViserTeleopNode publishes nothing, so the RobotNode never issues a
  command and the arm stays in i2rt's compliant gravity-comp mode.
* **Arming syncs first.** Ticking "Armed" snaps every joint slider onto the
  measured pose before the first command goes out, so arming never moves the
  arm. The first published vector equals the measured state; only a subsequent
  slider drag produces motion.
* **Slew limiting + smoothing.** The slider target is first passed through a
  first-order filter (``smoothing_tau_s``, live slider) and the command then
  chases that at no more than ``max_joint_speed`` rad/s (live slider). A slider
  yanked across its range therefore produces one bounded, rounded sweep — the
  arm never steps to a spot.
* **Gripper never jumps either.** On sync the gripper slider is set from the
  measured raw gripper angle via ``gripper_raw_limits`` (the calibrated stops
  live inside the follower and are not on the bus, so this is the YAM's
  nominal closed/open range), so the first command holds the gripper too.
* **Limits.** Slider bounds come from the robot config, and the RobotNode's
  ``MotorChainRobot`` clips again on its side.

Conventions (same as YamPyrokiViserAgent)
-----------------------------------------
* Bus joint order is i2rt motor order (joint1 … joint6). The YAM URDF declares
  its joints reversed, so poses are ``np.flip``-ed before ``update_cfg``.
* The 7th command element is the *normalized* gripper in ``[0, 1]``
  (0 = closed, 1 = open); the follower remaps it onto calibrated limits. The
  measured ``gripper_pos`` on the bus is raw radians and is shown as such.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import viser
import viser.extras
import yaml
import yourdfpy
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent

ARM_DOF = 6

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_YAM_DIR = os.path.join(_REPO_ROOT, "dependencies", "i2rt", "i2rt", "robot_models", "arm", "yam")

# Fallback if no robot_config is given: robot_configs/yam/xdof_hq/left.yaml.
_DEFAULT_LIMITS = [
    [-2.09, 3.14],
    [0.0, 3.14],
    [0.05, 3.14],
    [-1.35, 1.35],
    [-1.50, 1.50],
    [-2.00, 2.00],
]


def _load_yam_urdf() -> yourdfpy.URDF:
    """Fresh URDF instance (each ViserUrdf overlay mutates its own copy)."""
    return yourdfpy.URDF.load(os.path.join(_YAM_DIR, "yam.urdf"), mesh_dir=os.path.join(_YAM_DIR, "assets"))


class YamJointSliderAgent(Agent):
    """Per-joint slider control of one YAM arm through Viser.

    Args:
        viser_port:        Port for this agent's Viser server.
        robot_config:      Path to the RobotNode's ``_target_`` YAML. Used for
                           ``joint_limits`` (slider bounds) and ``gripper_index``
                           (whether a gripper slider exists). Relative paths are
                           resolved against the CWD, then the repo root.
        joint_limits:      Explicit ``[[lo, hi], ...]`` override (radians).
        has_gripper:       Explicit override for the gripper slider.
        max_joint_speed:   Cap on commanded joint speed (rad/s); live slider.
                           0 disables the limiter.
        max_gripper_speed: Same for the normalized gripper (units/s).
        smoothing_tau_s:   First-order filter on the slider target (s); live
                           slider. 0 disables it.
        gripper_initial:   Initial normalized gripper command (1 = open); only
                           used before the first sync.
        gripper_raw_limits: ``[closed_rad, open_rad]`` used to turn the measured
                           raw gripper angle into a normalized slider value on
                           sync. None disables gripper syncing.
        armed:             Start with command output enabled. Default False.
        viz_period_s:      Ghost / readout refresh period.
    """

    def __init__(
        self,
        viser_port: int = 8080,
        robot_config: Optional[str] = None,
        joint_limits: Optional[List[List[float]]] = None,
        has_gripper: Optional[bool] = None,
        max_joint_speed: float = 0.5,
        max_gripper_speed: float = 1.0,
        smoothing_tau_s: float = 0.1,
        gripper_initial: float = 1.0,
        gripper_raw_limits: Optional[List[float]] = (0.10, -2.83),
        armed: bool = False,
        viz_period_s: float = 0.05,
        viser_server: Optional["viser.ViserServer"] = None,
    ) -> None:
        cfg_limits, cfg_gripper = self._read_robot_config(robot_config)
        limits = joint_limits if joint_limits is not None else (cfg_limits or _DEFAULT_LIMITS)
        self.joint_limits = np.asarray(limits, dtype=np.float64)
        assert self.joint_limits.shape == (ARM_DOF, 2), f"expected {ARM_DOF} joint limits, got {self.joint_limits.shape}"
        self.has_gripper = bool(cfg_gripper if has_gripper is None else has_gripper)
        self.n_cmd = ARM_DOF + (1 if self.has_gripper else 0)

        self._max_gripper_speed = float(max_gripper_speed)
        self._smoothing_tau_init = float(smoothing_tau_s)
        self._gripper_initial = float(np.clip(gripper_initial, 0.0, 1.0))
        self._gripper_raw_limits = None if gripper_raw_limits is None else tuple(float(v) for v in gripper_raw_limits)
        self._goal_filtered: Optional[np.ndarray] = None
        self._viz_period = float(viz_period_s)

        self._armed = bool(armed)
        # When arming, hold output until the sliders have been snapped onto the
        # measured pose (needs at least one observation).
        self._sync_pending = True
        self._cmd: Optional[np.ndarray] = None  # last published vector
        self._last_act_t: Optional[float] = None
        self._lock = threading.Lock()

        self.obs: Optional[Dict[str, Any]] = None
        self._running = True

        self.viser_server = viser_server if viser_server is not None else viser.ViserServer(port=viser_port)
        self._setup_scene()
        self._setup_gui(max_joint_speed)

        self._vis_thread = threading.Thread(target=self._vis_loop, name="yam_slider_vis", daemon=True)
        self._vis_thread.start()

    # ── Config ────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_robot_config(path: Optional[str]):
        if not path:
            return None, True
        candidates = [path, os.path.join(_REPO_ROOT, path)]
        for p in candidates:
            if os.path.exists(p):
                with open(p) as f:
                    cfg = yaml.safe_load(f) or {}
                return cfg.get("joint_limits"), cfg.get("gripper_index") is not None
        raise FileNotFoundError(f"robot_config not found: {path!r} (tried {candidates})")

    # ── Scene / GUI ───────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        self.viser_server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.viser_server.scene.add_frame("/base", show_axes=True, axes_length=0.1, axes_radius=0.004)
        self.urdf_cmd = viser.extras.ViserUrdf(self.viser_server, _load_yam_urdf(), root_node_name="/base")
        self.viser_server.scene.add_frame("/base_real", show_axes=False)
        self.urdf_real = viser.extras.ViserUrdf(
            self.viser_server, _load_yam_urdf(), root_node_name="/base_real", mesh_color_override=(0.8, 0.5, 0.5)
        )
        for mesh in self.urdf_real._meshes:
            mesh.opacity = 0.3  # type: ignore[attr-defined]
        self.urdf_cmd.update_cfg(np.zeros(ARM_DOF))
        self.urdf_real.update_cfg(np.zeros(ARM_DOF))

    def _setup_gui(self, max_joint_speed: float) -> None:
        gui = self.viser_server.gui
        gui.add_markdown(
            "**YAM joint-slider test.** Solid arm = commanded, red ghost = measured. "
            "Tick *Armed* to start sending commands (sliders snap to the measured pose first)."
        )

        with gui.add_folder("Safety"):
            self.armed_cb = gui.add_checkbox("Armed (send commands)", initial_value=self._armed)
            self.sync_btn = gui.add_button("Sync sliders to robot")
            self.speed_slider = gui.add_slider(
                "Max joint speed (rad/s)", min=0.0, max=3.0, step=0.05, initial_value=float(max_joint_speed)
            )
            self.smoothing_slider = gui.add_slider(
                "Smoothing tau (s)", min=0.0, max=0.5, step=0.01, initial_value=self._smoothing_tau_init
            )
            self.status_md = gui.add_markdown("Status: waiting for joint state…")

        with gui.add_folder("Joints (rad)"):
            self.joint_sliders: List[viser.GuiInputHandle] = []
            for i in range(ARM_DOF):
                lo, hi = self.joint_limits[i]
                init = float(np.clip(0.0, lo, hi))
                self.joint_sliders.append(
                    gui.add_slider(f"joint{i + 1}", min=float(lo), max=float(hi), step=0.005, initial_value=init)
                )
            self.gripper_slider: Optional[viser.GuiInputHandle] = None
            if self.has_gripper:
                self.gripper_slider = gui.add_slider(
                    "gripper (0=closed, 1=open)", min=0.0, max=1.0, step=0.01, initial_value=self._gripper_initial
                )
            self.zero_btn = gui.add_button("Sliders → zero pose (arm joints only)")

        with gui.add_folder("Measured (rad)"):
            self.meas_numbers: List[viser.GuiInputHandle] = [
                gui.add_number(f"joint{i + 1}", 0.0, disabled=True, step=0.001) for i in range(ARM_DOF)
            ]
            self.meas_gripper: Optional[viser.GuiInputHandle] = (
                gui.add_number("gripper (raw rad)", 0.0, disabled=True, step=0.001) if self.has_gripper else None
            )
            self.track_err = gui.add_number("max |cmd − meas| (rad)", 0.0, disabled=True, step=0.001)

        @self.armed_cb.on_update
        def _(_) -> None:
            self.set_armed(self.armed_cb.value)

        @self.sync_btn.on_click
        def _(_) -> None:
            with self._lock:
                self._sync_pending = True

        @self.zero_btn.on_click
        def _(_) -> None:
            for i, s in enumerate(self.joint_sliders):
                lo, hi = self.joint_limits[i]
                s.value = float(np.clip(0.0, lo, hi))

    # ── State helpers ─────────────────────────────────────────────────────────

    def _arm_obs(self) -> Optional[Dict[str, Any]]:
        obs = self.obs
        if not isinstance(obs, dict):
            return None
        for key, val in obs.items():
            if isinstance(val, dict) and "joint_pos" in val:
                return val
        return None

    def _measured_joints(self) -> Optional[np.ndarray]:
        arm = self._arm_obs()
        if arm is None:
            return None
        jp = np.asarray(arm["joint_pos"], dtype=np.float64).ravel()
        return jp[:ARM_DOF] if jp.size >= ARM_DOF else None

    def _measured_gripper_raw(self) -> Optional[float]:
        arm = self._arm_obs()
        if arm is None:
            return None
        g = arm.get("gripper_pos")
        if g is None:
            return None
        return float(np.asarray(g, dtype=np.float64).ravel()[0])

    def _measured_gripper_norm(self) -> Optional[float]:
        raw = self._measured_gripper_raw()
        if raw is None or self._gripper_raw_limits is None:
            return None
        closed, opened = self._gripper_raw_limits
        if abs(opened - closed) < 1e-6:
            return None
        return float(np.clip((raw - closed) / (opened - closed), 0.0, 1.0))

    def _slider_goal(self) -> np.ndarray:
        goal = np.array([float(s.value) for s in self.joint_sliders], dtype=np.float64)
        goal = np.clip(goal, self.joint_limits[:, 0], self.joint_limits[:, 1])
        if self.gripper_slider is not None:
            goal = np.concatenate([goal, [float(np.clip(self.gripper_slider.value, 0.0, 1.0))]])
        return goal

    def _sync_sliders_to_state(self) -> bool:
        """Snap the joint sliders onto the measured pose. True if state was available."""
        meas = self._measured_joints()
        if meas is None:
            return False
        clipped = np.clip(meas, self.joint_limits[:, 0], self.joint_limits[:, 1])
        for s, v in zip(self.joint_sliders, clipped):
            s.value = float(v)
        # Restart the slew + filter from the measured pose so the first command
        # is a no-op. The gripper slider is set from the measured raw angle via
        # gripper_raw_limits (nominal YAM range) so it holds too.
        if self.gripper_slider is not None:
            g_norm = self._measured_gripper_norm()
            if g_norm is not None:
                self.gripper_slider.value = g_norm
            self._cmd = np.concatenate([meas, [float(self.gripper_slider.value)]])
        else:
            self._cmd = meas.copy()
        self._goal_filtered = self._cmd.copy()
        return True

    # ── Visualization loop ────────────────────────────────────────────────────

    def _vis_loop(self) -> None:
        while self._running:
            meas = self._measured_joints()
            if meas is not None:
                self.urdf_real.update_cfg(np.flip(meas))
                for h, v in zip(self.meas_numbers, meas):
                    h.value = float(v)
                g = self._measured_gripper_raw()
                if self.meas_gripper is not None and g is not None:
                    self.meas_gripper.value = g
            cmd = self._cmd
            if cmd is not None:
                self.urdf_cmd.update_cfg(np.flip(cmd[:ARM_DOF]))
                if meas is not None:
                    self.track_err.value = float(np.abs(cmd[:ARM_DOF] - meas).max())
            elif meas is None:
                self.urdf_cmd.update_cfg(np.flip(self._slider_goal()[:ARM_DOF]))
            else:
                # Disarmed: show the sliders' intent, ghost shows reality.
                self.urdf_cmd.update_cfg(np.flip(self._slider_goal()[:ARM_DOF]))

            if meas is None:
                status = "Status: **waiting for joint state** (is the RobotNode up?)"
            elif not self._armed:
                status = "Status: **disarmed** — no commands sent; arm is compliant"
            elif self._sync_pending:
                status = "Status: armed, syncing sliders to measured pose…"
            else:
                status = "Status: **ARMED — sliders drive the arm**"
            if self.status_md.content != status:
                self.status_md.content = status
            time.sleep(self._viz_period)

    # ── Agent interface ───────────────────────────────────────────────────────

    def set_armed(self, armed: bool) -> None:
        armed = bool(armed)
        with self._lock:
            if armed and not self._armed:
                self._sync_pending = True  # never move on arm: snap sliders first
            self._armed = armed
            if not armed:
                self._cmd = None
                self._goal_filtered = None
                self._last_act_t = None
        if self.armed_cb.value != armed:
            self.armed_cb.value = armed

    def act(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        self.obs = obs

        with self._lock:
            if not self._armed:
                return {}  # publish nothing — RobotNode issues no command
            if self._sync_pending:
                if not self._sync_sliders_to_state():
                    return {}  # no state yet; keep silent
                self._sync_pending = False
                self._last_act_t = time.monotonic()
                return {"pos": self._cmd.astype(np.float32)}

            now = time.monotonic()
            dt = 0.01 if self._last_act_t is None else float(np.clip(now - self._last_act_t, 1e-4, 0.05))
            self._last_act_t = now

            goal = self._slider_goal()
            cur = self._cmd if self._cmd is not None else goal.copy()

            # Filter the goal, then rate-limit the approach to it (same order as
            # YamPyrokiViserAgent so the two knobs stay independent).
            tau = float(self.smoothing_slider.value)
            smoothed = self._goal_filtered
            if smoothed is None or tau <= 0.0:
                smoothed = goal.copy()
            else:
                smoothed = smoothed + (dt / (tau + dt)) * (goal - smoothed)
            self._goal_filtered = smoothed

            v_max = float(self.speed_slider.value)
            if v_max <= 0.0:
                cur = smoothed.copy()
            else:
                limit = np.full(goal.shape, v_max * dt)
                if self.gripper_slider is not None:
                    limit[-1] = self._max_gripper_speed * dt if self._max_gripper_speed > 0 else np.inf
                cur = cur + np.clip(smoothed - cur, -limit, limit)
            self._cmd = cur
            return {"pos": cur.astype(np.float32)}

    def action_spec(self) -> Dict[str, Array]:
        return {"pos": Array(shape=(self.n_cmd,), dtype=np.float32)}

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._running = False
        try:
            self.viser_server.stop()
        except Exception:
            pass


__all__ = ["YamJointSliderAgent"]
