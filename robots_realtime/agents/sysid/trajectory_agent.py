"""SysidTrajectoryAgent — play validated sys-id trajectories on one 747 follower.

For each trajectory file (``robots_realtime.sysid.trajectories``), in order:

1. wait for the follower's ``joint_state``;
2. plan a minimum-jerk approach from the measured pose to the trajectory's first
   sample (<= ``approach_vel``) and validate it against the same envelope, speed and
   floor checks as the trajectory itself (refuse if the arm starts outside);
3. approach, settle, then play the trajectory on wall-clock time (interpolated, so
   loop jitter does not stretch it), with ``_record`` high: each trajectory is its own
   episode;
4. pause ``pause_s`` and continue with the next file.

Aborts to a hold at the measured pose (and stops recording) when the follower latches,
reports a non-NORMAL safety mode, or tracks worse than ``max_track_error`` after the
first second of motion. The gripper is held where it was.

Published state (``{node}/state``, recorded): file index, phase index, the commanded
joints as sent, and the agent state.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from robots_realtime.rig.dionysus import load_rig
from robots_realtime.sysid.trajectories import DT, Limits, Trajectory, check, load_envelope, min_jerk

log = logging.getLogger(__name__)

WAIT, APPROACH, SETTLE, RUN, PAUSE, DONE, ABORTED = range(7)
STATE_NAMES = ("wait", "approach", "settle", "run", "pause", "done", "aborted")


class SysidTrajectoryAgent:
    """Args:
    trajectories: Trajectory ``.npz`` files, played in order.
    arm: ``left`` / ``right`` (rig and envelope lookup; must match the files).
    robot_name: ``state_topics`` key of the follower's ``joint_state``.
    approach_vel: Peak joint speed of the approach move [rad/s].
    settle_s / pause_s: Hold before each trajectory / between trajectories.
    max_track_error: Abort when max |command - measured| exceeds this [rad].
    record: Raise ``_record`` while each trajectory plays.
    rig_yaml / envelope: Overrides for the rig model and envelope file.
    """

    def __init__(self, trajectories: list[str], arm: str, robot_name: str = "follower",
                 approach_vel: float = 0.2, settle_s: float = 2.0, pause_s: float = 3.0,
                 max_track_error: float = 0.15, record: bool = True, rig_yaml: str | None = None,
                 envelope: str | None = None) -> None:
        self.trajs = [Trajectory.load(p) for p in trajectories]
        for p, tr in zip(trajectories, self.trajs):
            if tr.arm != arm:
                raise ValueError(f"{p} is for the {tr.arm} arm, not {arm}")
        rig = load_rig(rig_yaml)
        self.arm_model = rig.arms[arm]
        self.surface_z = rig.surface_z
        self.env = load_envelope(arm, envelope)
        self.limits = Limits()
        for p, tr in zip(trajectories, self.trajs):  # re-validate: files may have been edited
            bad = check(tr.q, self.arm_model, self.env, self.limits, self.surface_z)
            if bad:
                raise ValueError(f"{p} fails validation: {bad[:3]}")
        self.robot_name = robot_name
        self.approach_vel = float(approach_vel)
        self.settle_s = float(settle_s)
        self.pause_s = float(pause_s)
        self.max_track_error = float(max_track_error)
        self.record = bool(record)
        self.state = WAIT
        self.index = 0
        self.reason = ""
        self._path: np.ndarray | None = None
        self._t0 = 0.0
        self._hold: np.ndarray | None = None
        self._gripper = 1.0

    # ── helpers ───────────────────────────────────────────────────────────

    def _sample(self, path: np.ndarray, elapsed: float) -> tuple[np.ndarray, bool]:
        x = elapsed / DT
        i = int(np.floor(x))
        if i >= len(path) - 1:
            return path[-1].copy(), True
        f = x - i
        return (1 - f) * path[i] + f * path[i + 1], False

    def _phase(self, elapsed: float) -> int:
        tr = self.trajs[self.index]
        return int(tr.phase[min(int(elapsed / DT), len(tr.phase) - 1)])

    def _abort(self, reason: str, q_meas: np.ndarray) -> None:
        self.state = ABORTED
        self.reason = reason
        self._hold = q_meas.copy()
        log.error("[sysid] ABORT: %s", reason)

    def _start_approach(self, q_meas: np.ndarray, now: float) -> None:
        start = self.trajs[self.index].q[0]
        path = min_jerk(q_meas, start, self.approach_vel, self.limits.amax)
        bad = check(path, self.arm_model, self.env, self.limits, self.surface_z)
        if bad:
            self._abort(f"approach to trajectory {self.index} leaves the envelope: {bad[0]}", q_meas)
            return
        self._path, self._t0, self.state = path, now, APPROACH

    # ── agent protocol ────────────────────────────────────────────────────

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        follower = obs.get(self.robot_name)
        if follower is None:
            return {"_state": self._state_msg(None)}
        q_meas = np.asarray(follower["joint_pos"], dtype=np.float64)[:6]
        if self.state == WAIT:
            self._gripper = float(np.asarray(follower["joint_pos"])[6])
            self._start_approach(q_meas, now)

        latched = int(np.asarray(follower.get("latched", [0]))[0])
        safety = int(np.asarray(follower.get("safety_mode", [1]))[0])
        if self.state not in (DONE, ABORTED):
            if latched:
                self._abort("follower latched", q_meas)
            elif safety not in (1, 2):
                self._abort(f"safety_mode {safety}", q_meas)

        cmd = q_meas.copy()
        recording = False
        if self.state == APPROACH:
            cmd, finished = self._sample(self._path, now - self._t0)
            if finished:
                self.state, self._t0 = SETTLE, now
        elif self.state == SETTLE:
            cmd = self.trajs[self.index].q[0].copy()
            if now - self._t0 >= self.settle_s:
                self._path, self._t0, self.state = self.trajs[self.index].q, now, RUN
                log.info("[sysid] playing %s %s (%.0f s)", self.trajs[self.index].arm,
                         self.trajs[self.index].kind, self.trajs[self.index].duration)
        elif self.state == RUN:
            elapsed = now - self._t0
            cmd, finished = self._sample(self._path, elapsed)
            recording = self.record
            err = float(np.abs(cmd - q_meas).max())
            if elapsed > 1.0 and err > self.max_track_error:
                self._abort(f"tracking error {err:.3f} rad > {self.max_track_error}", q_meas)
                cmd, recording = q_meas.copy(), False
            elif finished:
                self.state, self._t0 = PAUSE, now
                self._hold = cmd.copy()
        elif self.state == PAUSE:
            cmd = self._hold.copy()
            if now - self._t0 >= self.pause_s:
                self.index += 1
                if self.index >= len(self.trajs):
                    self.state = DONE
                    log.info("[sysid] all %d trajectories done", len(self.trajs))
                else:
                    self._start_approach(q_meas, now)
        if self.state in (DONE, ABORTED):
            cmd = self._hold if self._hold is not None else q_meas
        return {"pos": np.concatenate([cmd, [self._gripper]]).astype(np.float64),
                "_record": recording, "_state": self._state_msg(cmd)}

    def _state_msg(self, cmd) -> dict[str, Any]:
        phase = self._phase(time.time() - self._t0) if self.state == RUN else -1
        return {"state": np.array([self.state]), "trajectory": np.array([self.index]),
                "phase": np.array([phase]),
                "cmd_q": np.zeros(6) if cmd is None else np.asarray(cmd, dtype=np.float64)}

    @property
    def state_name(self) -> str:
        return STATE_NAMES[self.state]

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


__all__ = ["SysidTrajectoryAgent"]
