# 747 bimanual UR teleop (dionysus)

robots_realtime sessions for the "747" rig on dionysus (Jetson AGX Orin): two UR3e leaders
(192.168.0.10 left, .12 right) and two UR5e followers (.11 left, .13 right), PolyScope X 10.12,
Robotiq 2F-85 grippers, four Orbbec Gemini 305g cameras over GMSL2. The hardware interfaces are the
ones Scale AI's recording_studio (trainmybot v3.19.1) and UR's AI Trainer bundle use, so data from
both stacks is directly comparable.

**Status:** built and tested offline (fake controllers); the cameras were streamed live with these
settings (2026-10-22). No session in this branch has driven the real arms yet. Run `monitor_readonly*.yaml` first, then a gentle teleop with someone at the e-stop.

## What matches recording_studio

| | recording_studio (Scale) | this stack |
| --- | --- | --- |
| Leader program | vendor `Default program` (MotionGuide admittance; moves only while the grab button is held) | same, played/stopped over REST by `URLeaderArm` (`manage_program`) |
| Follower program | vendor `Default program` (joint impedance, Kp 600/600/300/28/28/28 Nm/rad, no integral) | same (`*_follower_vendor.yaml`), or `rr servo` (`servoj`) |
| Command path | follower input regs 24-29 = leader `actual_q` 1:1, reg 30 goal time 0.007 s, reg 31 gripper | same; `URRegisterArm` adds a setpoint rate limit, lead clamp, stale-command hold and stop latch |
| Loop rate | 500 Hz | 500 Hz (RTDE, agent and robot nodes) |
| Gripper | follower reg 31 = leader trigger volts / 13 | same value: leader open fraction `1 - (V/13)/0.92`, follower maps back by `x 0.92` |
| Wrench feedback | follower flange wrench (out regs 24-29) -> leader in regs 24-29 | same (`haptic_feedback`, `haptic_scale`) |
| Cameras | Dual Color Streams, LEFT colour image, 848x480; top 60 Hz, others 30 Hz | same (`OrbbecRigNode`, `stream: color_left`, per-camera `fps`) |
| Recorded state | joint_states, tcp_pose, tcp_wrench, tool_flange_wrench, commands, grab | follower `joint_state` (joints, gripper, TCP pose and force, flange wrench, setpoint, RTDE extras), leader `state` (joints, grab, hold, trigger, wrench), commands, camera RGB + intrinsics |

Not reproduced: forwarding the leader's DI bits to follower bits 64/65 (the vendor follower re-zeroes
its F/T sensor on a grab release), and recording_studio's collision checker heartbeat (input int 26,
off in this deployment). Recording format is robots_realtime's (MCAP per node, MP4 + camera_info per
camera), not trainmybot's MCAP topic layout.

## Sessions (`configs/ur747_bimanual/`)

| Config | Leaders | Followers | Writes to arms |
| --- | --- | --- | --- |
| `monitor_readonly_nocams.yaml` | read-only | read-only | nothing; safe any time |
| `monitor_readonly.yaml` | read-only | read-only | nothing; needs the cameras free |
| `teleop_vendor_gentle.yaml` | vendor, no haptics | vendor, 1 rad/s cap, 0.1 rad lead | yes |
| `teleop_vendor.yaml` | vendor + wrench feedback | vendor, Scale-like limits | yes |
| `teleop_vendor_left.yaml` | left pair of the above | | yes |
| `teleop_rr_servo.yaml` | vendor, no haptics | `rr servo` (servoj) | yes |
| `teleop_fake.yaml` | simulated | simulated (real driver logic) | never (unroutable IPs) |

```bash
uv run rr-session configs/ur747_bimanual/monitor_readonly_nocams.yaml
DIONYSUS_STOP_LATCH=/dev/shm/ur747_stop_fake uv run rr-session configs/ur747_bimanual/teleop_fake.yaml
```

## Controller types and every knob

### Followers: `URRegisterArm` (`robot_configs/ur747/*_follower_*.yaml`)

Two robot programs share one register contract (in doubles 24-29 joint target, 30 goal time,
31 gripper):

- **vendor `Default program`** (`program: Default program`): what Scale runs. Soft joint impedance
  via `direct_torque`: `tau = Kp*e + clamp(Kd*e_dot, 0.2*tau_max)`, Kp = tau_max/[0.25,0.25,0.5,1,1,1].
  Gains are fixed inside the URCap. Measured behaviour: moves end 1-6 mrad short, and slow targets
  (e.g. 0.03 rad over 5 s on a wrist) are not followed at all, because friction beats Kp x error.
  Good for teleop, poor for precise planning. Publishes its flange wrench on out regs 24-29.
- **`rr servo`** (`program: rr servo`): a `servoj` position servo built from each arm's own
  `Default program` (keeps TCP, payload, mounting and the Robotiq functions). Robot-side safety:
  session token, RTDE watchdog, reference walked at <= `vmax`, targets more than `max_jump` away
  refused. Build-time knobs (`rr-ur-servo-program build`): `gain` (100-2000, default 300),
  `lookahead` (0.03-0.2 s, default 0.1), `vmax` (rad/s), `max_jump` (rad), `watchdog_hz`.
  Upload stores it next to `Default program` without loading or playing it; a person loads it on
  the pendant.

Runtime knobs (both programs):

| Key | Meaning |
| --- | --- |
| `enable_motion` | false = output recipe only, nothing written (dry run) |
| `rate` | RTDE and control rate [Hz] |
| `goal_time` | reg 30 ramp time per target [s] (vendor only; Scale 0.007) |
| `max_joint_vel` | setpoint speed cap [rad/s] |
| `max_lead` | max setpoint - actual_q per joint [rad] |
| `align_joint_vel`, `align_tolerance` | slow alignment onto the commanded pose at start and after stale gaps (recording_studio's `reconciliate`) |
| `speed_fraction` / `restore_speed_fraction` | speed slider while driving / on hand-back (scales servoj, probably not direct_torque) |
| `cmd_timeout_s` | older commands become "hold here" |
| `joint_lower` / `joint_upper` | a target outside latches a stop |
| `gripper_closed_cmd`, `gripper_open_state`, `gripper_closed_state` | gripper unit mapping |
| `rig_yaml` | rig model for the table floor guard (`""` disables) |
| `play_on_start`, `allow_attach_playing` | program lifecycle |
| `extra_outputs` | any RTDE output fields, published as `rtde_<field>` (configs stream `target_q/qd/qdd`, `target_current`, `target_moment`, `actual_current`, `joint_control_output`, `joint_temperatures`, `speed_scaling`, `target_speed_fraction`) |

Observations (`arm_<side>_follower/joint_state`): `joint_pos` (6 + gripper open fraction),
`joint_vel`, `aligned`, `tcp_pose`, `tcp_force` (RTDE `actual_TCP_force`), `flange_wrench` (vendor, while
PLAYING) or `servo_reference` (rr servo), `setpoint`, `gripper_raw`, `safety_mode`,
`runtime_state`, `driving`, `latched`, `rtde_*`.

### Leaders: `URLeaderArm` (`robot_configs/ur747/*_leader_*.yaml`)

| Key | Meaning |
| --- | --- |
| `manage_program` | Play the leader program on start (after REST checks), Stop on exit |
| `haptic_feedback` | own leader input regs 24-29 and write the follower wrench (needs recording_studio's arms off) |
| `haptic_limit` | per-component clamp on the written wrench |
| `trigger_divisor`, `gripper_closed_cmd` | trigger volts -> gripper command (13, 0.92) |
| `extra_outputs` | further RTDE fields, published as `rtde_<field>` |

Agent (`URLeaderAgent`): `follow: always | grab`, `haptic_scale`, `joint_offsets`.

The leader's admittance parameters (mass, damping, deadband) are inside the vendor URCap and not
adjustable here. A custom leader program (e.g. alvin's "ur747 leader" in `~alvinz/UR_747_teleop`,
with tunable admittance, position tracking and penetration-based force feedback) would be the next
step for leader-side control; it is not part of this branch.

## Safeguards compared with recording_studio

Sources: the MotionGuide Live Control URCap scripts (`leader.script`, `follower.script`,
`collision_detection.script`, `collision_liveness.script` in `~/ai-trainer-0.2.8`), the settings in
each arm's deployed `Default program` (read 2026-10-22), and recording_studio's binaries.

**Robot-side (inside the vendor programs; kept whenever a session uses the vendor programs):**

| Safeguard | Where | Active on 747 |
| --- | --- | --- |
| Joint velocity damping above 180/150/150/180/180/190 deg/s, torque clamped to joint ratings, damping clamped to 0.2 x rating | follower | yes |
| Leader admittance capped at 1.5 m/s and 180 deg/s; singularity handling; virtual walls at joint limits | leader | yes |
| Leader moves only while the grab button is held; hold input freezes it; F/T re-zeroed on release | leader | yes |
| Leader contact damping from the follower wrench (quadratic up to 20 N, removes push into contact) | leader | yes (needs `haptic_feedback`) |
| Leader in-script self-collision check, predictive (only blocks motion that gets closer) | leader | yes (`check_self_collision_in_script`) |
| External collision checker gate: hold on its flag or a stale heartbeat (fail-closed) | both | **off** (`collision_checker_active = False`); nothing to send |
| On Play: leader `movej` (0.2 rad/s) of elbow/wrist2 into the working range | leader | yes |
| Path-divergence `movej` at 0.2 rad/s | follower | Modbus mode only: in register (RTDE) mode the program assumes the path has converged |

**PC-side:**

| Safeguard | recording_studio | this stack |
| --- | --- | --- |
| Refuse unless safety mode NORMAL | yes | yes, plus Remote control, expected program loaded and STOPPED, not already PLAYING, registers not IN_USE |
| Slow start alignment of follower onto leader | `start_mode: reconciliate` | `align_joint_vel` (0.3 rad/s; 0.2 gentle) until within 0.05 rad, re-armed after any stale gap |
| Protective-stop handling | UI unlock / restart safety | refuses and latches; unlock on the pendant |
| Start confirmation | `CONFIRM_SAFETY` UI step | none (start is the operator running the session) |
| Setpoint rate limit, lead clamp, stale-command hold, joint-limit trip, NaN/shape trip | not observed | yes |
| Table floor guard (TCP and link frame origins above the table) | not observed | yes (`rig.yaml` clearance) |
| Stop latch every tick + out-of-band REST stop from any shell | not observed | yes (`rr-ur-stop`) |
| Did-Play-move-the-arm check | not observed | yes (latches on > 5 mrad) |

**Still missing in both (consider before long sessions):**

- **Follower-follower collision**: nothing checks the two followers against each other (Scale's
  external checker is disabled; the leader's self-collision check covers only the leader).
  `rig.yaml` has `inter_arm: 0.12`, but no driver enforces it yet.
- **Process death mid-motion**: the vendor follower has no watchdog in register mode; if the PC
  process dies it holds its last target (same as recording_studio). The hardware e-stop and
  `rr-ur-stop` are the backstops. `rr servo` adds an RTDE watchdog.
- **Leader DI forwarding**: recording_studio forwards the leader's grab/hold to follower bits 64/65,
  so the follower re-zeroes its F/T sensor on release. Not done here, so the follower wrench (and the
  leader's contact damping from it) can carry offset drift over a long session.
- **`rr servo` followers** do not have the vendor follower's velocity damping or torque clamps:
  protection is `servoj`, the program's `vmax`/`max_jump`, the PC-side limits and UR's safety
  configuration.

## Bring-up checklist

1. `rr-ur-stop --status`: latch clear, arms NORMAL.
2. recording_studio: "Start Robot Arms" OFF (it owns the input registers otherwise) and its camera
   stage OFF. Keep the service running: the GMSL cameras are not found without it.
3. Pendants: Remote control, the right program loaded (`Default program`, or `rr servo` on the
   followers for `teleop_rr_servo.yaml`), STOPPED.
4. `monitor_readonly_nocams.yaml`: leaders and followers stream, joint angles sane.
5. `teleop_vendor_gentle.yaml` with someone at the e-stop, hands off the leaders while they start.
6. Exit with `q` / Ctrl-C: followers hold, programs stop, registers neutral, slider restored.
   `rr-ur-stop` (no args) latches and REST-stops both followers from any shell.
7. Before handing back to recording_studio: `Default program` loaded on all four arms.

## Code map

- `robots_realtime/robots/ur_register_arm.py`: follower driver (ported from
  KeplerC/controllers@dionysus-ur5e; adds wrench outputs and RTDE pass-through).
- `robots_realtime/robots/ur_leader_arm.py`: leader driver.
- `robots_realtime/agents/teleoperation/ur_leader_agent.py`: leader -> follower agent.
- `robots_realtime/robots/fake_ur.py`: simulated followers and leaders.
- `robots_realtime/sensors/cameras/orbbec_camera.py`, `runtime/environment/orbbec_rig_node.py`:
  cameras (`stream: color | color_left | color_right`, per-camera overrides).
- `robots_realtime/rig/`: rig model (`configs/ur747_bimanual/rig.yaml`, run_002 calibration, 155.8 mm
  2F-85 tool point), stop latch, `rr servo` builder.
- Tests: `tests/test_ur747_teleop.py`, `tests/test_dionysus_rig.py`, `tests/test_servo_program.py`.

## Camera bindings on dionysus

`pyorbbecsdk` is not on PyPI for this platform. Build it for the session's Python from the shared
source (SDK 2.8.6, the same as recording_studio's):

```bash
rsync -a --exclude build --exclude .git ~/Desktop/pyorbbecsdk/ ~/justinyu/build/pyorbbecsdk-src/
uv pip install "pybind11==2.12.0" "pybind11-global==2.12.0"
cd ~/justinyu/build/pyorbbecsdk-src && mkdir -p build-cp311 && cd build-cp311
cmake .. -DPython3_EXECUTABLE=$VENV/bin/python -Dpybind11_DIR=$($VENV/bin/python -m pybind11 --cmakedir) -DCMAKE_BUILD_TYPE=Release
make -j8
SP=$($VENV/bin/python -c "import site; print(site.getsitepackages()[0])")
cp pyorbbecsdk.cpython-311-aarch64-linux-gnu.so $SP/
cp -a ../sdk/lib/arm64/libOrbbecSDK.so* ../sdk/lib/arm64/OrbbecSDKConfig.xml ../sdk/lib/arm64/extensions $SP/
```

Checked 2026-10-22: all four cameras open in one process in ~6 s and stream left colour 848x480
(top ~60 Hz, others ~30 Hz) with the same intrinsics as recording_studio's CameraInfo
(top fx/fy/cx/cy 410.9/411.0/415.8/241.8). The SDK logs "depth frame processor not found" and
"RTT too large" in Dual Color mode; both are harmless.
