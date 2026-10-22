"""Node base class and ProcessHost.

A Node is the unit of independent execution in the ZMQ node graph.  Each node:
  - Owns one piece of hardware (or a sim backend, or agent logic)
  - Runs its own loop — either flat-out (poll_freq=None) or at a fixed rate
  - Publishes state to the bus AND writes to its injected Writer at every call
  - Optionally subscribes to commands

ProcessHost spawns a Node in its own OS process and exposes a thin REQ/REP
control socket so the Session can start, stop, and query it.  The control loop
handles START_RECORDING / STOP_RECORDING while node.run() is executing.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path

import zmq

from robots_realtime.runtime.perf import PerfStats
from robots_realtime.runtime.transport.message_bus import DEFAULT_PUB_PORT, DEFAULT_SUB_PORT
from robots_realtime.runtime.transport.publisher import Publisher
from robots_realtime.runtime.transport.subscriber import Subscriber


# ── NodeRole ──────────────────────────────────────────────────────────────────


class NodeRole(Enum):
    CONTROLLER = auto()
    ROBOT      = auto()
    SENSOR     = auto()
    EVENT      = auto()


# ── Node ──────────────────────────────────────────────────────────────────────


class Node(ABC):
    """Abstract base for all nodes.

    Subclasses declare:
        name            : str               unique node name on the bus
        role            : NodeRole          semantic role (default ROBOT)
        published_topics: list[str]         topic suffixes this node produces
        subscribed_topics: list[str]        full topic strings this node consumes
        poll_freq       : float | None      inner loop rate; None = flat-out
        publish_freq    : float | None      ZMQ send rate; None = every step
        subscriber_driven: bool             if True, block on sub instead of sleeping
    """

    name: str = ""
    role: NodeRole = NodeRole.ROBOT
    published_topics: list[str] = []
    subscribed_topics: list[str] = []
    poll_freq: float | None = None
    publish_freq: float | None = None
    subscriber_driven: bool = False

    _step_count: int = 0
    _step_hz: float = 0.0
    _last_stats_t: float = 0.0
    _stats_interval: float = 1.0

    def __init__(
        self,
        name: str | None = None,
        writer=None,
        pub_host: str = "127.0.0.1",
        pub_port: int = DEFAULT_PUB_PORT,
        sub_host: str = "127.0.0.1",
        sub_port: int = DEFAULT_SUB_PORT,
    ) -> None:
        if name is not None:
            self.name = name
        assert self.name, "Node.name must be set"

        self._pub_host = pub_host
        self._pub_port = pub_port
        self._sub_host = sub_host
        self._sub_port = sub_port

        # Injected writer — stored for pickling; passed to Publisher in run()
        self._writer = writer  # Writer | None

        self._publisher: Publisher | None = None
        self._subscriber: Subscriber | None = None
        # Per-loop perf accumulator. Subclasses call self._perf.record(label, ms)
        # inside step(); the base loop records loop period + step() duration and
        # flushes a summary once per interval via _tick().
        self._perf = PerfStats(self.name)
        self._stop = False
        self._recording: bool = False
        # Session-level gate toggled via `session.toggle_pause()` (space in TUI).
        # Base behaviour is a no-op; RobotNode overrides to stop issuing joint
        # commands while paused so the motors hold their last pose.
        self._paused: bool = False

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """Open hardware handles, allocate resources."""

    @abstractmethod
    def step(self) -> None:
        """Called at poll_freq (or flat-out).  Read hardware, publish."""

    def cleanup(self) -> None:
        """Release hardware handles.  Called after the loop exits."""

    @property
    def web_endpoints(self) -> list[str]:
        """Return human-readable localhost URLs this node exposes (e.g. viser)."""
        return []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self, save_dir: str) -> None:
        """Open the writer and start recording."""
        if self._writer is not None:
            self._writer.open(save_dir, self.name)
        self._recording = True

    def stop_recording(self) -> str:
        """Close the writer and stop recording.  Returns the output path."""
        self._recording = False
        if self._writer is not None and self._writer.is_open:
            return self._writer.close()
        return ""

    # ------------------------------------------------------------------
    # Pause / resume — session-level gate driven by the TUI (space key)
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Called when the session enters pause. Default: set the flag only.

        RobotNode overrides this to stop issuing joint commands so the motors
        hold their last pose. Other node types are free to ignore it.
        """
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------
    # Transport helpers (available inside step())
    # ------------------------------------------------------------------

    def publish(
        self,
        topic_suffix: str,
        data: dict,
        ts: float | None = None,
        record: bool = True,
        record_data: dict | None = None,
    ) -> bool:
        """Publish data on ``"{self.name}/{topic_suffix}"``.

        Also writes to self._writer at every call (full poll rate), and sends
        on the ZMQ bus throttled by publish_freq. Pass ``record=False`` to
        skip the writer for this call (the message still goes on the bus).
        Pass ``record_data`` to write a different (e.g. full-fidelity) payload
        to disk than the one shipped on the wire.
        """
        assert self._publisher is not None, "publish() called before run()"
        return self._publisher.publish(
            topic_suffix, data, ts=ts, record=record, record_data=record_data
        )

    def get_latest(self, topic: str) -> dict | None:
        """Return latest data dict for a subscribed topic, or None."""
        assert self._subscriber is not None, "get_latest() called before run()"
        return self._subscriber.get_data(topic)

    def get_timestamp(self, topic: str) -> float | None:
        assert self._subscriber is not None
        return self._subscriber.get_timestamp(topic)

    # ------------------------------------------------------------------
    # YAML config classmethod
    # ------------------------------------------------------------------

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        """Build constructor kwargs from a YAML params dict.

        Subclasses should override this to extract their specific parameters.
        Default implementation returns {"name": params["name"]}.
        """
        return {"name": params["name"]}

    # ------------------------------------------------------------------
    # Main loop (called by ProcessHost worker)
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._publisher = Publisher(
            node_name=self.name,
            writer=self._writer,
            publish_freq=self.publish_freq,
            host=self._pub_host,
            port=self._pub_port,
        )
        if self.subscribed_topics:
            self._subscriber = Subscriber(
                topics=self.subscribed_topics,
                host=self._sub_host,
                port=self._sub_port,
            )

        self.setup()

        self._step_count = 0
        self._last_stats_t = time.perf_counter()
        self._perf.reset()

        try:
            if self.subscriber_driven:
                self._run_subscriber_driven()
            elif self.poll_freq is None:
                self._run_flat_out()
            else:
                self._run_fixed_rate()
        finally:
            self.cleanup()
            if self._publisher:
                self._publisher.close()
            if self._subscriber:
                self._subscriber.close()

    def _tick(self) -> None:
        """Count one step and publish step_hz once per _stats_interval seconds."""
        self._step_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_stats_t
        if elapsed >= self._stats_interval:
            self._step_hz = self._step_count / elapsed
            self._step_count = 0
            self._last_stats_t = now
            if self._publisher is not None:
                self.publish("_step_hz", {"step_hz": self._step_hz})
        self._perf.maybe_log()

    def _timed_step(self, prev_loop_t: float | None) -> float:
        """Run step() once, recording loop period and step() duration (ms).

        Returns the perf_counter timestamp taken just before step() so the
        caller can thread it back in as ``prev_loop_t`` next iteration.
        """
        t0 = time.perf_counter()
        if prev_loop_t is not None:
            self._perf.record("loop_ms", (t0 - prev_loop_t) * 1e3)
        self.step()
        self._perf.record("step_ms", (time.perf_counter() - t0) * 1e3)
        return t0

    def _run_flat_out(self) -> None:
        prev = None
        while not self._stop:
            prev = self._timed_step(prev)
            self._tick()

    def _run_fixed_rate(self) -> None:
        period = 1.0 / self.poll_freq  # type: ignore[operator]
        next_t = time.perf_counter()
        prev = None
        while not self._stop:
            prev = self._timed_step(prev)
            self._tick()
            next_t += period
            now = time.perf_counter()
            if next_t < now - period:
                next_t = now
            remaining = next_t - now
            if remaining > 3e-4:
                time.sleep(remaining - 1e-4)

    def _run_subscriber_driven(self) -> None:
        """Block on incoming messages; call step() for each batch received."""
        assert self._subscriber is not None
        # If poll_freq is also set, use it as the timeout for the blocking poll.
        timeout_ms = int(1000.0 / self.poll_freq) if self.poll_freq else 50
        prev = None
        while not self._stop:
            self._subscriber.drain_one(timeout_ms=timeout_ms)
            self._subscriber.drain()   # consume any burst that arrived
            prev = self._timed_step(prev)
            self._tick()

    def stop(self) -> None:
        self._stop = True


# ── ProcessHost ───────────────────────────────────────────────────────────────

_CTRL_READY = b"READY"
_CTRL_STOP  = b"STOP"
_CTRL_OK    = b"OK"


def _host_worker(
    node: Node,
    ctrl_addr: str,
    ready_event: mp.Event,
    log_path: Path | None = None,
) -> None:
    """Entry point for the subprocess spawned by ProcessHost.

    Control loop runs in a background thread while node.run() executes in the
    main thread.  The control loop handles:
        START               — signal main thread to start node.run()
        STOP                — call node.stop(), break out of control loop
        START_RECORDING:<d> — call node.start_recording(d)
        STOP_RECORDING      — call node.stop_recording()
    """
    # Detach from the parent's terminal: redirect stdin so that
    # libraries using input() / readline don't get the TUI's setcbreak stdin.
    sys.stdin = open(os.devnull, "r")

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(log_path, "w", buffering=1)
        sys.stdout = _log_file
        sys.stderr = _log_file

    import logging as _logging
    _logging.basicConfig(
        stream=sys.stderr,
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ctx = zmq.Context()
    ctrl = ctx.socket(zmq.REP)
    ctrl.bind(ctrl_addr)

    # Signal that the process is alive and the control socket is bound.
    ready_event.set()

    # Event to signal the main thread that it should start node.run()
    start_event = threading.Event()
    # Event to signal that we should stop (set by STOP command)
    stop_event = threading.Event()

    def _watch_ctrl():
        """Persistent control loop — handles multiple commands."""
        while True:
            try:
                msg = ctrl.recv()
            except zmq.ZMQError:
                break

            if msg == b"START":
                ctrl.send(_CTRL_OK)
                start_event.set()
            elif msg == _CTRL_STOP:
                ctrl.send(_CTRL_OK)
                node.stop()
                stop_event.set()
                break
            elif msg.startswith(b"START_RECORDING:"):
                save_dir = msg[len(b"START_RECORDING:"):].decode()
                try:
                    node.start_recording(save_dir)
                except Exception as e:
                    pass  # best-effort; don't crash the control loop
                ctrl.send(_CTRL_OK)
            elif msg == b"STOP_RECORDING":
                try:
                    node.stop_recording()
                except Exception:
                    pass
                ctrl.send(_CTRL_OK)
            elif msg == b"PAUSE":
                try:
                    node.pause()
                except Exception:
                    pass
                ctrl.send(_CTRL_OK)
            elif msg == b"RESUME":
                try:
                    node.resume()
                except Exception:
                    pass
                ctrl.send(_CTRL_OK)
            else:
                # Unknown command — send OK to unblock the requester
                ctrl.send(_CTRL_OK)

    ctrl_thread = threading.Thread(target=_watch_ctrl, daemon=True, name="CtrlWatcher")
    ctrl_thread.start()

    # SIGTERM reaches every node (launchers signal the whole process group).
    # Stop the loop instead of dying, so ``cleanup()`` runs: hardware drivers
    # rely on it to hand the robot back (e.g. URRegisterArm stops the program
    # and restores the speed slider).
    import signal as _signal

    def _graceful(*_):
        node.stop()
        stop_event.set()
        start_event.set()

    try:
        _signal.signal(_signal.SIGTERM, _graceful)
    except (OSError, ValueError):
        pass

    # Block main thread until START arrives
    start_event.wait()

    if not stop_event.is_set():
        node.run()

    # Wait for the control thread to finish
    ctrl_thread.join(timeout=2.0)
    ctx.destroy(linger=0)


class ProcessHost:
    """Manages a Node running in a dedicated subprocess.

    Usage:
        host = ProcessHost(my_node)
        host.start()           # spawns subprocess, waits for ready
        host.send_start()      # tells the node to begin its loop
        ...
        host.start_recording(save_dir)  # delegate recording to node
        host.stop_recording()           # stop recording in node
        host.stop()            # sends STOP, waits for clean exit
    """

    def __init__(self, node: Node, ctrl_port: int | None = None) -> None:
        self._node = node
        self._ctrl_port = ctrl_port or _find_free_port()
        self._ctrl_addr = f"tcp://127.0.0.1:{self._ctrl_port}"
        self._proc: mp.Process | None = None
        self._ctx = zmq.Context.instance()
        self._ctrl: zmq.Socket | None = None

    def start(self, timeout: float = 10.0, log_path: Path | None = None) -> None:
        """Spawn subprocess and wait until its control socket is bound."""
        ready = mp.Event()
        self._proc = mp.Process(
            target=_host_worker,
            args=(self._node, self._ctrl_addr, ready, log_path),
            daemon=True,
            name=f"Node-{self._node.name}",
        )
        self._proc.start()
        if not ready.wait(timeout):
            raise RuntimeError(f"ProcessHost for '{self._node.name}' timed out on start")
        self._ctrl = self._ctx.socket(zmq.REQ)
        self._ctrl.connect(self._ctrl_addr)

    def send_start(self) -> None:
        """Tell the node subprocess to begin its loop."""
        assert self._ctrl is not None
        self._ctrl.send(b"START")
        self._ctrl.recv()

    def start_recording(self, save_dir: str) -> None:
        """Tell the node subprocess to start recording into save_dir."""
        assert self._ctrl is not None
        self._ctrl.send(f"START_RECORDING:{save_dir}".encode())
        self._ctrl.recv()

    def stop_recording(self) -> str:
        """Tell the node subprocess to stop recording.  Returns empty string."""
        assert self._ctrl is not None
        self._ctrl.send(b"STOP_RECORDING")
        self._ctrl.recv()
        return ""

    def pause(self) -> None:
        """Tell the node subprocess to enter its paused state."""
        assert self._ctrl is not None
        self._ctrl.send(b"PAUSE")
        self._ctrl.recv()

    def resume(self) -> None:
        """Tell the node subprocess to exit its paused state."""
        assert self._ctrl is not None
        self._ctrl.send(b"RESUME")
        self._ctrl.recv()

    def stop(self, timeout: float = 8.0) -> None:
        # 8 s: a robot driver's cleanup (hold, REST program stop, neutral
        # registers) must finish before the node is killed.
        if self._ctrl is not None:
            self._ctrl.setsockopt(zmq.RCVTIMEO, 2000)  # 2 s receive timeout
            self._ctrl.send(_CTRL_STOP)
            try:
                self._ctrl.recv()
            except zmq.ZMQError:
                pass  # timeout or error — proceed to kill
            self._ctrl.close(linger=0)
            self._ctrl = None
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.kill()  # SIGKILL — cleanup already had its chance
            self._proc = None

    @property
    def node_name(self) -> str:
        return self._node.name

    @property
    def node_names(self) -> list[str]:
        return [self._node.name]

    @property
    def video_node_names(self) -> list[str]:
        """Node names whose writer is video-based (kept for Session compatibility)."""
        return []


def _find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]
