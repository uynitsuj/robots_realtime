# robots_realtime

A research codebase for real-time robot teleoperation, data collection, and policy deployment.

### Why robots_realtime?
- **Unified Pipeline:** Collect data in simulation or on real hardware platforms, and deploy learned policies with the same infrastructure.
- **Modular Stack:** Switch between GELLO leader arms, IK gizmos, Franka or I2RT YAM robot hardware via runtime YAML configs.
- **High Frequency:** Built with ZeroMQ nodes for asynchronous, low-latency real-time control.

<table>
<tr>
<td><img src="media/real_yams_rr.gif" width="360"></td>
<td><img src="media/franka_realtime2.gif" width="360"></td>
</tr>
<tr>
<td><img src="media/yam_active_leader_dagger.gif" width="360"></td>
<td><img src="media/rr_vr_support.gif" width="360"></td>
</tr>
</table>

To build your own YAM active leader arms refer to: [lerobot_teleoperator_yamactiveleader](https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader)

## Other Documentation
[Architecture & recording format](docs/architecture.md) 

[Extending (new agents, robots, cameras)](docs/extending.md) 

[VR streaming MuJoCo sim to Quest](docs/vr_streaming.md)

[747 bimanual UR teleop on dionysus (UR3e leaders, UR5e followers, Orbbec cameras)](docs/ur747_bimanual.md)

---

## Installation

```bash
git clone --recurse-submodules https://github.com/uynitsuj/robots_realtime.git
cd robots_realtime
# if already cloned, or some of the submodules are incompletely cloned, run
git submodule update --init --recursive
uv venv --python 3.11 && uv pip install -e .
```

---

## Usage / Quickstart
### I2RT YAM Configuration
If using real-world I2RT YAM arms, configure YAM arms CAN chain according to instructions from the [I2RT repo](https://github.com/i2rt-robotics/i2rt)

### Run a teleop session with YAM Followers and YAM Leaders
#### Session Configuration
```bash
uv run rr-session configs/yam/yam_bimanual_yam_leader.yaml
```

### Run a teleop session with YAM Followers and custom [3d printed active leaders](https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader)
```bash
uv run rr-session configs/yam/yam_bimanual_gello_teleop.yaml
```

### Run a teleop session in sim using [3d printed leaders](https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader)

```bash
uv run rr-session configs/yam/yam_sim_gello_teleop.yaml
```
Upon running any of the above configs, you should see the terminal populate with a rich TUI session:
```
╭─────────────────────────────── robots_realtime ────────────────────────────────╮
│   NODE                STATUS             HZ    TOPICS                          │
│   gello_left          ● live          255.8    joint_pos                       │
│   gello_right         ● live          255.8    joint_pos                       │
│   yam                 ● live           29.6    left_state, right_state         │
│ http://localhost:8765  (viser)  http://localhost:8012  (vr)                    │
│ ────────────────────────────────────────────────────────────────────────────── │
│ ○  idle                                      [r] record  [d] discard  [q] quit │
│ ────────────────────────────────────────────────────────────────────────────── │
│ [yam] ╭────── viser (listening *:8765) ───────╮                                │
│ [yam] │             ╷                         │                                │
│ [yam] │   HTTP      │ http://localhost:8765   │                                │
│ [yam] │   Websocket │ ws://localhost:8765     │                                │
│ [yam] │             ╵                         │                                │
│ [yam] ╰───────────────────────────────────────╯                                │
│   logs: /tmp/rr_logs_7hhz62am                                                  │
╰────────────────────────────────────────────────────────────────────────────────╯
```
Look under `/configs` for other existing configs

### Replay an episode

```bash
uv run rr-replay recordings/20260323/episode_175805_0473b1bc/
```

Opens a Viser viewer at `http://localhost:8080`. For sim episodes you get two modes: **qpos** (exact, restores recorded state) and **physics** (re-simulates from actions). For real data, you get viser visualization of joint angles replayed on urdfs and other sensor streams (e.g. rgb).

