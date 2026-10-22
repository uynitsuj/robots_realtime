"""rr-ur-snapshot — calibrated RGB-D captures from the dionysus rig (no rr-session).

Read-only on the arms (RTDE output recipe only) and opens the Orbbec cameras in
their depth work mode. Needs ``recording_studio.service`` up with its camera
stage OFF. Each capture lands in ``<out>/<timestamp>/``::

    <cam>/rgb.png            colour (RGB, 848x530)
    <cam>/depth.npy          float32 metres, aligned to rgb, 0 = invalid
    <cam>/depth_mm.png       uint16 millimetres (lossless, for viewers)
    <cam>/K.json             3x3 K, distortion (rational, 8), size, serial
    <cam>/T_world_cam.npy    4x4 camera->world (OpenCV optical; world = checkerboard)
    <cam>/overlay.png        rgb + projected TCPs, world axes, workspace box (calibration check)
    robot.json               per arm: q, gripper, TCP in world, controller TCP pose
    meta.json                rig.yaml snapshot, surface_z, time

Usage::

    rr-ur-snapshot --list                     # cameras (serial, work mode) + arm states
    rr-ur-snapshot --prepare                  # all cameras -> depth work mode (before a session)
    rr-ur-snapshot --once                     # one capture of every camera
    rr-ur-snapshot --every 2 --count 10       # 10 captures, 2 s apart
    rr-ur-snapshot --once --plane             # + fit the table plane in the top camera's depth
    rr-ur-snapshot --once --export-rss-ref ../../../RoboSimStudio/examples/live/ref
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from robots_realtime.rig.dionysus import load_rig, pose7_wxyz

DEFAULT_OUT = Path("/home/lab/eric/data/captures")


def _project(K: np.ndarray, world_T_cam: np.ndarray, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points -> pixel coords (N,2) and a mask of points in front of the camera."""
    cam_T_world = np.linalg.inv(world_T_cam)
    pc = pts_w @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
    front = pc[:, 2] > 1e-3
    uv = np.full((len(pts_w), 2), np.nan)
    uv[front, 0] = K[0, 0] * pc[front, 0] / pc[front, 2] + K[0, 2]
    uv[front, 1] = K[1, 1] * pc[front, 1] / pc[front, 2] + K[1, 2]
    return uv, front


def _overlay(rgb: np.ndarray, K: np.ndarray, T: np.ndarray, rig, q_by_arm: dict) -> np.ndarray:
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    # World axes at the origin.
    axes = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np.float64)
    uv, ok = _project(K, T, axes)
    if ok.all():
        o = tuple(int(v) for v in uv[0])
        for i, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.line(img, o, tuple(int(v) for v in uv[i]), col, 2)
    # Workspace box edges.
    lo, hi = rig.workspace_min, rig.workspace_max
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    uv, ok = _project(K, T, corners)
    for a, b in edges:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), (220, 140, 60), 1)
    # TCP of each arm (+ its approach axis).
    for name, q in q_by_arm.items():
        arm = rig.arms[name]
        Tt = arm.tcp_world(q)
        pts = np.array([Tt[:3, 3], Tt[:3, 3] - 0.05 * Tt[:3, 2]])
        uv, ok = _project(K, T, pts)
        if ok[0] and np.all(np.isfinite(uv[0])):
            c = tuple(int(v) for v in uv[0])
            cv2.circle(img, c, 6, (0, 255, 255), 2)
            cv2.putText(img, f"{name} tcp", (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if ok[1]:
                cv2.line(img, c, tuple(int(v) for v in uv[1]), (0, 200, 255), 1)
    return img


def fit_plane(depth: np.ndarray, K: np.ndarray, T: np.ndarray, roi: tuple[float, float, float, float] | None = None
              ) -> dict:
    """RANSAC-free robust plane fit of the depth cloud (world frame) -> height + tilt + residual."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:2, 0:w:2]
    d = depth[::2, ::2]
    m = d > 0
    if roi is not None:
        x0, y0, x1, y1 = roi
        m &= (uu >= x0 * w) & (uu <= x1 * w) & (vv >= y0 * h) & (vv <= y1 * h)
    z = d[m]
    pc = np.stack([(uu[m] - K[0, 2]) * z / K[0, 0], (vv[m] - K[1, 2]) * z / K[1, 1], z], axis=1)
    pw = pc @ T[:3, :3].T + T[:3, 3]
    # The table is the lowest dense horizontal layer: keep points near the median low height.
    zw = pw[:, 2]
    lowband = zw < np.percentile(zw, 40) + 0.02
    P = pw[lowband]
    for _ in range(3):
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
        res = P[:, 2] - A @ coef
        keep = np.abs(res) < max(3 * np.std(res), 0.002)
        P = P[keep]
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    res = P[:, 2] - A @ coef
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    center = P.mean(axis=0)
    return {
        "surface_z_at_centroid": float(coef[0] * center[0] + coef[1] * center[1] + coef[2]),
        "tilt_deg": float(np.degrees(np.arccos(abs(n[2])))),
        "rms_m": float(np.sqrt(np.mean(res**2))),
        "points": int(len(P)),
        "centroid": center.tolist(),
    }


def _open_cameras(rig, names: list[str], depth: bool):
    import pyorbbecsdk as ob

    from robots_realtime.sensors.cameras.orbbec_camera import OrbbecCamera

    ctx = ob.Context()
    cams = {}
    for n in names:
        c = rig.cameras[n]
        cams[n] = OrbbecCamera(c.serial, camera=n, enable_depth=depth, context=ctx)
    return ctx, cams


def _open_arms(rig):
    from robots_realtime.robots.ur_register_arm import URRegisterArm

    arms = {}
    for name, a in rig.arms.items():
        try:
            arms[name] = URRegisterArm(a.ip, name=name, enable_motion=False, rate=125.0)
        except Exception as e:
            print(f"[rr-ur-snapshot] {name} arm state unavailable: {e}", file=sys.stderr)
    return arms


def capture(rig, cams: dict, arms: dict, out_root: Path, plane: bool = False) -> Path:
    import cv2

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = out_root / ts
    out.mkdir(parents=True, exist_ok=True)
    q_by_arm, robot = {}, {}
    for name, arm in arms.items():
        o = arm.get_observations()
        q = np.asarray(o["joint_pos"], dtype=np.float64)
        q_by_arm[name] = q[:6]
        robot[name] = {
            "q": q[:6].tolist(),
            "gripper_open_fraction": float(q[6]),
            "gripper_raw": float(np.asarray(o.get("gripper_raw", [np.nan]))[0]),
            "tcp_world_xyz_wxyz": pose7_wxyz(rig.arms[name].tcp_world(q[:6])).tolist(),
            "controller_tcp_pose_base_rotvec": np.asarray(o.get("tcp_pose", [np.nan] * 6)).tolist(),
            "safety_mode": int(np.asarray(o.get("safety_mode", [-1]))[0]),
        }
    (out / "robot.json").write_text(json.dumps(robot, indent=2))
    report = {}
    for name, cam in cams.items():
        data = cam.read()
        rgb = data.images["rgb"]
        depth = getattr(data, "depth_data", None)
        intr = cam.intrinsic_data["left"]
        K = np.asarray(intr["intrinsics_matrix"], dtype=np.float64)
        T = rig.cameras[name].world_pose(rig.arms, q_by_arm)
        d = out / name
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if depth is not None:
            np.save(d / "depth.npy", depth.astype(np.float32))
            cv2.imwrite(str(d / "depth_mm.png"), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
        (d / "K.json").write_text(json.dumps({
            "K": K.tolist(), "distortion": intr["distortion_coefficients"], "model": intr["distortion_model"],
            "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "serial": cam.serial,
        }, indent=2))
        if T is not None:
            np.save(d / "T_world_cam.npy", T)
            cv2.imwrite(str(d / "overlay.png"), _overlay(rgb, K, T, rig, q_by_arm))
            if plane and depth is not None and rig.cameras[name].mount == "fixed":
                report[name] = fit_plane(depth, K, T)
    meta = {"time": ts, "surface_z": rig.surface_z, "rig": rig.raw, "plane_fit": report}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if report:
        for n, r in report.items():
            print(f"[rr-ur-snapshot] plane {n}: z={r['surface_z_at_centroid']:+.4f} m  tilt={r['tilt_deg']:.2f} deg  "
                  f"rms={r['rms_m'] * 1000:.1f} mm  (rig.yaml surface_z={rig.surface_z})")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rr-ur-snapshot", description=__doc__.split("\n\n")[0])
    p.add_argument("--rig", default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cameras", default="", help="comma list (default: all)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--prepare", action="store_true",
                   help="switch every rig camera into the depth work mode (one at a time), then exit")
    p.add_argument("--once", action="store_true")
    p.add_argument("--every", type=float, default=0.0)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--no-arms", action="store_true")
    p.add_argument("--plane", action="store_true", help="fit the table plane in fixed cameras' depth")
    p.add_argument("--export-rss-ref", default="", help="copy each fixed camera's rgb.png here (RSS live page)")
    args = p.parse_args(argv)
    rig = load_rig(args.rig)

    if args.prepare:
        from robots_realtime.sensors.cameras.orbbec_camera import prepare

        for serial, what in prepare([c.serial for c in rig.cameras.values()]).items():
            print(f"camera {serial}: {what}")
        return 0

    if args.list:
        from robots_realtime.sensors.cameras.orbbec_camera import list_devices

        for d in list_devices():
            known = [n for n, c in rig.cameras.items() if c.serial == d["serial"]]
            print(f"camera {d['serial']} {d['name']} work_mode={d['work_mode']} rig={known or '-'}")
        for name, arm in ({} if args.no_arms else _open_arms(rig)).items():
            o = arm.get_observations()
            print(f"arm {name}: q={np.round(o['joint_pos'][:6], 4).tolist()} grip={o['joint_pos'][6]:.3f} "
                  f"safety={int(o['safety_mode'][0])} runtime={int(o['runtime_state'][0])}")
            arm.stop()
        return 0

    names = [n for n in (args.cameras.split(",") if args.cameras else rig.cameras) if n]
    ctx, cams = _open_cameras(rig, names, depth=not args.no_depth)
    arms = {} if args.no_arms else _open_arms(rig)
    try:
        n = 1 if args.once else max(1, args.count)
        for i in range(n):
            out = capture(rig, cams, arms, Path(args.out), plane=args.plane)
            print(f"[rr-ur-snapshot] {out}")
            if args.export_rss_ref:
                ref = Path(args.export_rss_ref)
                ref.mkdir(parents=True, exist_ok=True)
                for name in cams:
                    if rig.cameras[name].mount == "fixed" and (out / name / "rgb.png").exists():
                        shutil.copy(out / name / "rgb.png", ref / f"{out.name}_{name}.png")
                print(f"[rr-ur-snapshot] exported fixed-camera stills to {ref}")
            if i + 1 < n and args.every > 0:
                time.sleep(args.every)
    finally:
        for c in cams.values():
            c.stop()
        for a in arms.values():
            a.stop()
        del ctx
    return 0


if __name__ == "__main__":
    sys.exit(main())
