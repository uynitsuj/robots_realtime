"""Viser playback of 747 sys-id trajectories in the calibrated r2s2r station scene.

Visual check before running anything on the rig: the moving arm plays each trajectory
(the other holds its rig home pose), with the gripper's TCP trail, the shrunk TCP / link
bounds the validator enforces and the 5 cm sys-id floor. The station scene (calibrated
UR5e + 2F-85 mounts, run_002) comes from an r2s2r checkout with its assets fetched.

    uv run python scripts/sysid/view_trajectories.py --traj-dir sysid/trajectories --r2s2r ~/r2s2r
    # open http://localhost:8080
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

ap = argparse.ArgumentParser()
ap.add_argument("--traj-dir", required=True)
ap.add_argument("--r2s2r", required=True)
ap.add_argument("--rr", default=str(Path(__file__).resolve().parents[2]), help="robots_realtime checkout")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()
sys.path.insert(0, str(Path(args.r2s2r) / "src"))
sys.path.insert(0, str(Path(args.rr)))

from r2s2r.scene import ARM_JOINTS, TABLE_Z, make_spec  # noqa: E402
from r2s2r.station_calibration import load as load_station  # noqa: E402
from robots_realtime.rig.dionysus import load_rig  # noqa: E402
from robots_realtime.sysid.trajectories import DT, Trajectory, load_envelope  # noqa: E402

args.traj_dir = str(Path(args.traj_dir).resolve())
os.chdir(args.r2s2r)  # r2s2r resolves its assets relative to the repo
spec = make_spec("colored_box", station="run_002")
for body in spec.bodies:  # hide the task cube
    if body.name == "object":
        for g in body.geoms:
            g.rgba = [0, 0, 0, 0]
model = spec.compile()
data = mujoco.MjData(model)
rig = load_rig(str(Path(args.rr) / "configs/ur747_bimanual/rig.yaml"))
world_board = load_station("run_002").world_from_board(TABLE_Z)
addr = {s: [model.joint(f"{s}/{n}").qposadr[0] for n in ARM_JOINTS] for s in ("left", "right")}
pinch = {s: model.site(f"{s}/gripper/pinch").id for s in ("left", "right")}


def set_arms(q_by_side):
    for s, q in q_by_side.items():
        data.qpos[addr[s]] = q
    mujoco.mj_forward(model, data)


trajs = {Path(p).stem: Trajectory.load(p) for p in sorted(glob.glob(f"{args.traj_dir}/*.npz"))}
if not trajs:
    raise SystemExit(f"no trajectories in {args.traj_dir}")
home = {s: rig.arms[s].home_q.copy() for s in ("left", "right")}

server = viser.ViserServer(port=args.port)
server.scene.set_up_direction("+z")

# ── static geometry and per-geom handles ──────────────────────────────────────
set_arms(home)
handles = []
for g in range(model.ngeom):
    rgba = model.geom_rgba[g].copy()
    if model.geom_matid[g] >= 0:
        rgba = model.mat_rgba[model.geom_matid[g]].copy()
    if model.geom_group[g] >= 3 or rgba[3] < 0.05:
        continue
    gt, size = model.geom_type[g], model.geom_size[g]
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        va, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts, faces = model.mesh_vert[va:va + nv], model.mesh_face[fa:fa + nf]
    elif gt == mujoco.mjtGeom.mjGEOM_BOX:
        m = trimesh.creation.box(extents=2 * size)
        verts, faces = m.vertices, m.faces
    elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
        m = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=24)
        verts, faces = m.vertices, m.faces
    elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
        m = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[12, 12])
        verts, faces = m.vertices, m.faces
    elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        m = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        verts, faces = m.vertices, m.faces
    elif gt == mujoco.mjtGeom.mjGEOM_PLANE:
        m = trimesh.creation.box(extents=[3.0, 3.0, 0.002])
        verts, faces = m.vertices, m.faces
    else:
        continue
    color = tuple(int(c * 255) for c in rgba[:3])
    h = server.scene.add_mesh_simple(f"/scene/g{g}", vertices=np.asarray(verts, np.float32),
                                     faces=np.asarray(faces, np.uint32), color=color,
                                     opacity=float(rgba[3]) if rgba[3] < 1 else None)
    handles.append((g, h))


def update_geoms():
    q = np.empty(4)
    for g, h in handles:
        mujoco.mju_mat2Quat(q, data.geom_xmat[g])
        h.position = data.geom_xpos[g].copy()
        h.wxyz = q.copy()


# ── envelope boxes (shrunk bounds the validator uses), world frame ────────────
env_handles = {}
for s, col in (("left", (40, 170, 255)), ("right", (255, 80, 200))):
    env = load_envelope(s)
    for kind, lo, hi in (("tcp", env.tcp_lo, env.tcp_hi), ("links", env.link_lo, env.link_hi)):
        lo_w, hi_w = (world_board @ np.r_[lo, 1])[:3], (world_board @ np.r_[hi, 1])[:3]
        h = server.scene.add_box(f"/envelope/{s}_{kind}", color=col, dimensions=tuple(hi_w - lo_w),
                                 position=tuple((lo_w + hi_w) / 2), wireframe=True)
        env_handles[(s, kind)] = h
table_top = world_board[2, 3]
server.scene.add_box("/floor_limit", color=(255, 170, 0), dimensions=(1.4, 1.6, 0.0005),
                     position=(0.1, 0.0, table_top + 0.05), opacity=0.15)

# ── precompute TCP trails (10 Hz) ─────────────────────────────────────────────
trails = {}
for name, tr in trajs.items():
    other = "right" if tr.arm == "left" else "left"
    pts = []
    for q in tr.q[:: int(0.1 / DT)]:
        set_arms({tr.arm: q, other: home[other]})
        pts.append(data.site_xpos[pinch[tr.arm]].copy())
    trails[name] = np.asarray(pts)

# ── GUI ───────────────────────────────────────────────────────────────────────
with server.gui.add_folder("Playback"):
    g_traj = server.gui.add_dropdown("Trajectory", tuple(trajs))
    g_play = server.gui.add_checkbox("Play", True)
    g_speed = server.gui.add_slider("Speed (x)", min=0.25, max=20.0, step=0.25, initial_value=4.0)
    g_time = server.gui.add_slider("Time (s)", min=0.0, max=1.0, step=0.1, initial_value=0.0)
with server.gui.add_folder("Show"):
    g_trail = server.gui.add_checkbox("TCP trail", True)
    g_env_tcp = server.gui.add_checkbox("TCP bounds", True)
    g_env_links = server.gui.add_checkbox("Link bounds", False)
g_info = server.gui.add_markdown("")

trail_handle = None
current = None


def select(name):
    global trail_handle, current
    current = name
    tr = trajs[name]
    g_time.max = round(tr.duration, 1)
    g_time.value = 0.0
    if trail_handle is not None:
        trail_handle.remove()
    pts = trails[name]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    col = (40, 170, 255) if tr.arm == "left" else (255, 80, 200)
    trail_handle = server.scene.add_line_segments("/trail", segs, colors=col, thickness=0.004)
    trail_handle.visible = g_trail.value


g_traj.on_update(lambda _: select(g_traj.value))
g_trail.on_update(lambda _: setattr(trail_handle, "visible", g_trail.value))
select(g_traj.value)

print(f"viser on http://localhost:{args.port}  ({len(trajs)} trajectories)", flush=True)
last = time.time()
while True:
    now = time.time()
    dt, last = now - last, now
    tr = trajs[current]
    if g_play.value:
        t = g_time.value + dt * g_speed.value
        g_time.value = round(t % tr.duration, 2)
    i = min(int(g_time.value / DT), len(tr.q) - 1)
    other = "right" if tr.arm == "left" else "left"
    set_arms({tr.arm: tr.q[i], other: home[other]})
    update_geoms()
    for (s, kind), h in env_handles.items():
        h.visible = s == tr.arm and (g_env_tcp.value if kind == "tcp" else g_env_links.value)
    v = np.abs(tr.q[min(i + 1, len(tr.q) - 1)] - tr.q[i]).max() / DT
    g_info.content = (f"**{current}** ({tr.kind}, {tr.arm} arm) — phase **{tr.phase_names[tr.phase[i]]}**  \n"
                      f"t = {g_time.value:.1f} / {tr.duration:.1f} s · peak joint speed now {v:.2f} rad/s  \n"
                      f"q = {np.round(tr.q[i], 3).tolist()}")
    time.sleep(1 / 30)
