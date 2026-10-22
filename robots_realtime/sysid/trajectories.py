"""Safe sys-id trajectories for one 747 follower: static holds, friction sweeps, excitation.

Every trajectory is a dense joint path (500 Hz) with phase labels, generated around
the arm's rig home pose and validated sample by sample before anything is sent:

- **Observed envelope** (``configs/ur747_bimanual/sysid/envelope.yaml``): the joint,
  TCP and DH-link boxes each arm actually visited in 316 teleop episodes, shrunk by
  ``q_margin`` / ``pos_margin``. Sys-id only goes where operators already went.
- Joint speed and acceleration caps (peak speed stays under the "rr servo" program's
  1 rad/s reference limit).
- Floor: TCP and link frame origins above the table by ``tcp_floor`` / ``link_floor``
  (no contact is intended during free-space identification).

Coverage: static poses and excitation centers are drawn from **recorded** joint
configurations (``recorded_poses.npz``, a farthest-point sample of every pose the arm
visited in those episodes), so they span the visited region and were each visited by an
operator. Every trajectory starts and ends at the rig home pose.

The painting side is covered by the right arm's observed box (its link y never
exceeded 0.925 m in the board frame, below the 0.935 m keep-out).

    uv run rr-ur-sysid-traj --arm left --kind all --out sysid/trajectories
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from robots_realtime.rig.dionysus import ArmModel, load_rig

RATE = 500.0
DT = 1.0 / RATE
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVELOPE = ROOT / "configs" / "ur747_bimanual" / "sysid" / "envelope.yaml"
DEFAULT_POSES = ROOT / "configs" / "ur747_bimanual" / "sysid" / "recorded_poses.npz"


@dataclass
class Envelope:
    q_lo: np.ndarray
    q_hi: np.ndarray
    tcp_lo: np.ndarray
    tcp_hi: np.ndarray
    link_lo: np.ndarray
    link_hi: np.ndarray


def load_envelope(side: str, path: str | Path | None = None, q_margin: float = 0.05,
                  pos_margin: float = 0.03) -> Envelope:
    """The observed envelope of ``side``, shrunk by the margins."""
    e = yaml.safe_load(Path(path or DEFAULT_ENVELOPE).read_text())[side]
    a = lambda k: np.asarray(e[k], dtype=np.float64)  # noqa: E731
    return Envelope(a("q_min") + q_margin, a("q_max") - q_margin,
                    a("tcp_min") + pos_margin, a("tcp_max") - pos_margin,
                    a("links_min") + pos_margin, a("links_max") - pos_margin)


@dataclass
class Limits:
    vmax: float = 0.8  # rad/s, any joint
    amax: float = 2.0  # rad/s^2, any joint
    tcp_floor: float = 0.05  # m above the table (surface_z)
    link_floor: float = 0.06


@dataclass
class Trajectory:
    """Joint path at ``RATE`` with an integer phase label per sample."""

    arm: str
    kind: str
    q: np.ndarray  # (N, 6)
    phase: np.ndarray  # (N,) index into phase_names
    phase_names: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.q)) * DT

    @property
    def duration(self) -> float:
        return len(self.q) * DT

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=self.q, phase=self.phase, rate=RATE,
                            info=json.dumps({"arm": self.arm, "kind": self.kind,
                                             "phase_names": self.phase_names, "meta": self.meta}))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        d = np.load(path)
        if float(d["rate"]) != RATE:
            raise ValueError(f"{path}: rate {float(d['rate'])} != {RATE}")
        info = json.loads(str(d["info"]))
        return cls(info["arm"], info["kind"], d["q"], d["phase"], info["phase_names"], info["meta"])


# ── validation ────────────────────────────────────────────────────────────────


def check_pose(q, arm: ArmModel, env: Envelope, limits: Limits, surface_z: float = 0.0) -> list[str]:
    q = np.asarray(q, dtype=np.float64)
    out = []
    if np.any(q < env.q_lo) or np.any(q > env.q_hi):
        bad = np.flatnonzero((q < env.q_lo) | (q > env.q_hi))
        out.append(f"joints {bad.tolist()} outside the observed joint box")
    tcp = arm.tcp_world(q)[:3, 3]
    links = arm.link_points_world(q)[1:-1]
    if np.any(tcp < env.tcp_lo) or np.any(tcp > env.tcp_hi):
        out.append(f"TCP {np.round(tcp, 3).tolist()} outside the observed TCP box")
    if np.any(links < env.link_lo) or np.any(links > env.link_hi):
        out.append("a link frame origin is outside the observed link box")
    if tcp[2] < surface_z + limits.tcp_floor:
        out.append(f"TCP z {tcp[2]:.3f} below the floor")
    if links[:, 2].min() < surface_z + limits.link_floor:
        out.append(f"link z {links[:, 2].min():.3f} below the floor")
    return out


def check(traj_q: np.ndarray, arm: ArmModel, env: Envelope, limits: Limits, surface_z: float = 0.0,
          stride: int = 5) -> list[str]:
    """Every violation of the path (empty = safe). Kinematics checked every ``stride`` samples."""
    q = np.asarray(traj_q, dtype=np.float64)
    out = []
    if len(q) > 1:
        v = np.diff(q, axis=0) / DT
        if np.abs(v).max() > limits.vmax * 1.001:
            out.append(f"joint speed {np.abs(v).max():.3f} rad/s > {limits.vmax}")
        if len(q) > 2:
            a = np.diff(v, axis=0) / DT
            if np.abs(a).max() > limits.amax * 1.001:
                out.append(f"joint acceleration {np.abs(a).max():.3f} rad/s^2 > {limits.amax}")
    idx = list(range(0, len(q), stride))
    if idx[-1] != len(q) - 1:
        idx.append(len(q) - 1)
    for i in idx:
        for msg in check_pose(q[i], arm, env, limits, surface_z):
            out.append(f"t={i * DT:.2f}s: {msg}")
            if len(out) > 20:
                return out
    return out


# ── primitives ────────────────────────────────────────────────────────────────


def min_jerk(q0, q1, vpeak: float, amax: float = 2.0, min_t: float = 0.5) -> np.ndarray:
    """Rest-to-rest minimum-jerk move whose peak joint speed is <= vpeak, accel <= amax."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    d = float(np.abs(q1 - q0).max())
    # min-jerk peaks: v = 1.875 d / T, a = 5.774 d / T^2
    T = max(min_t, 1.875 * d / vpeak, np.sqrt(5.774 * d / amax))
    n = max(2, int(np.ceil(T * RATE)))
    s = np.linspace(0.0, 1.0, n)
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    return q0 + np.outer(s, q1 - q0)


def stroke(q0, j: int, distance: float, v: float, a: float) -> np.ndarray:
    """Trapezoidal-speed single-joint move with smoothed (raised-cosine) acceleration."""
    sign = np.sign(distance)
    distance = abs(distance)
    t_acc = 2.0 * v / a  # raised-cosine ramp: peak accel a reaches speed v in 2v/a
    d_acc = 0.5 * v * t_acc
    if 2 * d_acc > distance:  # cannot reach v: scale down
        v = np.sqrt(distance * a / 2.0)
        t_acc = 2.0 * v / a
        d_acc = 0.5 * v * t_acc
    t_flat = (distance - 2 * d_acc) / v
    T = 2 * t_acc + t_flat
    t = np.arange(0.0, T + DT, DT)
    vel = np.where(t < t_acc, v * 0.5 * (1 - np.cos(np.pi * t / t_acc)),
                   np.where(t < t_acc + t_flat, v,
                            v * 0.5 * (1 + np.cos(np.pi * np.clip(t - t_acc - t_flat, 0, t_acc) / t_acc))))
    pos = np.concatenate([[0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * DT)])
    pos *= distance / pos[-1]
    out = np.repeat(np.asarray(q0, dtype=np.float64)[None], len(t), axis=0)
    out[:, j] += sign * pos
    return out


class _Builder:
    def __init__(self, q0):
        self.q = [np.asarray(q0, dtype=np.float64)[None]]
        self.phase = [np.array([0])]
        self.names = ["start"]

    @property
    def last(self) -> np.ndarray:
        return self.q[-1][-1]

    def add(self, path: np.ndarray, name: str) -> None:
        if name not in self.names:
            self.names.append(name)
        k = self.names.index(name)
        self.q.append(np.asarray(path)[1:] if len(path) > 1 else np.asarray(path))
        self.phase.append(np.full(len(self.q[-1]), k))

    def hold(self, seconds: float, name: str) -> None:
        self.add(np.repeat(self.last[None], int(seconds * RATE) + 1, axis=0), name)

    def build(self, arm: str, kind: str, meta: dict) -> Trajectory:
        return Trajectory(arm, kind, np.concatenate(self.q), np.concatenate(self.phase), self.names, meta)


# ── recorded poses ────────────────────────────────────────────────────────────


def farthest_points(points: np.ndarray, k: int, start: np.ndarray | None = None) -> np.ndarray:
    """Indices of ``k`` points chosen greedily to be far from each other (max-min distance)."""
    pts = np.asarray(points, dtype=np.float64)
    d = np.full(len(pts), np.inf) if start is None else np.abs(pts - start).max(1)
    first = int(np.argmax(d)) if start is not None else 0
    chosen = [first]
    d = np.minimum(d, np.abs(pts - pts[first]).max(1))
    for _ in range(min(k, len(pts)) - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.abs(pts - pts[i]).max(1))
    return np.asarray(chosen)


def build_recorded_poses(export_dir: str | Path, out: str | Path = DEFAULT_POSES, stride: int = 100,
                         k: int = 3000) -> Path:
    """Farthest-point sample of every recorded follower pose (``export_episode.py`` NPZs)."""
    import glob

    arrays = {}
    for side in ("left", "right"):
        files = sorted(glob.glob(f"{export_dir}/*_{side}.npz"))
        q = np.concatenate([np.load(f)["actual_q"][::stride] for f in files])
        arrays[side] = q[farthest_points(q, k)].astype(np.float32)
        arrays[f"{side}_episodes"] = np.array(len(files))
    out = Path(out)
    np.savez_compressed(out, **arrays)
    return out


def load_recorded_poses(side: str, arm: ArmModel, env: Envelope, limits: Limits, surface_z: float = 0.0,
                        path: str | Path | None = None) -> np.ndarray:
    """Recorded poses of ``side`` that pass the (shrunk) envelope and floor checks."""
    q = np.load(path or DEFAULT_POSES)[side].astype(np.float64)
    return np.asarray([x for x in q if not check_pose(x, arm, env, limits, surface_z)])


def _route(start, waypoints, arm, env, limits, surface_z, move_vel, home) -> list[np.ndarray] | None:
    """Validated min-jerk legs from ``start`` through ``waypoints``; via ``home`` if direct fails."""
    for via in ([], [home]):
        legs, cur, ok = [], start, True
        for wp in [*via, *waypoints]:
            leg = min_jerk(cur, wp, move_vel, limits.amax)
            if check(leg, arm, env, limits, surface_z, stride=10):
                ok = False
                break
            legs.append(leg)
            cur = wp
        if ok:
            return legs
    return None


# ── generators ────────────────────────────────────────────────────────────────


def static_holds(side: str, arm: ArmModel, env: Envelope, limits: Limits, n_poses: int = 40,
                 settle_s: float = 2.0, record_s: float = 2.0, move_vel: float = 0.25,
                 reverse_every: int = 3, overshoot: float = 0.04, surface_z: float = 0.0,
                 poses_path: str | Path | None = None) -> Trajectory:
    """Hold ``n_poses`` recorded poses spread over the visited region (gravity, payload,
    static friction). Every ``reverse_every``-th pose is approached from the opposite side
    (overshoot, then back) to expose the static friction band."""
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    library = load_recorded_poses(side, arm, env, limits, surface_z, poses_path)
    poses = library[farthest_points(library, 3 * n_poses, start=home)]
    order, rest, cur = [], list(range(len(poses))), home
    while rest:  # greedy nearest neighbour keeps moves short
        k = min(rest, key=lambda i: np.abs(poses[i] - cur).max())
        order.append(k)
        rest.remove(k)
        cur = poses[k]
    b = _Builder(home)
    held = 0
    for k in order:
        if held >= n_poses:
            break
        target = poses[k]
        route = [target]
        if reverse_every and held % reverse_every == reverse_every - 1:
            beyond = np.clip(target + overshoot * np.sign(target - b.last), env.q_lo, env.q_hi)
            route = [beyond, target]
        legs = _route(b.last, route, arm, env, limits, surface_z, move_vel, home)
        if legs is None:
            continue
        for leg in legs:
            b.add(leg, "move")
        b.hold(settle_s, "settle")
        b.hold(record_s, "hold")
        held += 1
    back = _route(b.last, [home], arm, env, limits, surface_z, move_vel, home)
    for leg in back or [min_jerk(b.last, home, move_vel, limits.amax)]:
        b.add(leg, "move")
    return b.build(side, "static", {"n_poses": held, "settle_s": settle_s, "record_s": record_s,
                                    "reverse_every": reverse_every, "poses": "recorded, farthest-point"})


def friction_sweeps(side: str, arm: ArmModel, env: Envelope, limits: Limits,
                    joints=(0, 1, 2, 3, 4, 5), speeds=(0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8),
                    half_stroke: float = 0.6, slow_stroke: float = 0.6, accel: float = 1.5,
                    dwell_s: float = 0.5, move_vel: float = 0.25, surface_z: float = 0.0) -> Trajectory:
    """Constant-speed strokes, one joint at a time (others at home), both directions.

    Strokes use the largest symmetric range around home (<= ``half_stroke``) that stays in
    the envelope; speeds below 0.1 rad/s use at most ``slow_stroke`` of it to bound time.
    Phase names carry the speed actually reached."""
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    b = _Builder(home)
    used = {}
    for j in joints:
        A = half_stroke
        while A > 0.05:
            probe = np.repeat(home[None], 61, axis=0)
            probe[:, j] += np.linspace(-A, A, 61)
            if all(not check_pose(q, arm, env, limits, surface_z) for q in probe):
                break
            A *= 0.9
        if A <= 0.05:
            continue
        used[j] = round(A, 3)
        lo = home.copy()
        lo[j] -= A
        b.add(min_jerk(b.last, lo, move_vel, limits.amax), "move")
        for v_req in speeds:
            dist = 2 * A if v_req >= 0.1 else min(2 * A, slow_stroke)
            reachable = float(np.sqrt(dist * accel / 2.0))
            v = round(min(v_req, limits.vmax, reachable), 3)
            b.add(stroke(b.last, j, dist, v, accel), f"j{j} +{v:g}")
            b.hold(dwell_s, "dwell")
            b.add(stroke(b.last, j, -dist, v, accel), f"j{j} -{v:g}")
            b.hold(dwell_s, "dwell")
        b.add(min_jerk(b.last, home, move_vel, limits.amax), "move")
    return b.build(side, "friction", {"half_stroke": used, "speeds": list(speeds), "accel": accel})


def _fourier(center, room, rng, duration_s, base_hz, harmonics, fade_s, vmax, amax):
    t = np.arange(0.0, duration_s, DT)
    w = np.ones_like(t)
    k = t < fade_s
    w[k] = 0.5 * (1 - np.cos(np.pi * t[k] / fade_s))
    k = t > duration_s - fade_s
    w[k] = 0.5 * (1 - np.cos(np.pi * (duration_s - t[k]) / fade_s))
    f = base_hz * np.arange(1, harmonics + 1)
    a = rng.normal(size=(6, harmonics)) / np.arange(1, harmonics + 1)
    ph = rng.uniform(0, 2 * np.pi, size=(6, harmonics))
    shape = np.einsum("jk,jkt->tj", a, np.sin(2 * np.pi * f[None, :, None] * t[None, None, :] + ph[..., None]))
    shape *= w[:, None]
    vel = np.gradient(shape, DT, axis=0)
    acc = np.gradient(vel, DT, axis=0)
    scale = np.minimum.reduce([vmax / np.abs(vel).max(0), amax / np.abs(acc).max(0), room / np.abs(shape).max(0)])
    return center + shape * scale


def excitation(side: str, arm: ArmModel, env: Envelope, limits: Limits, center=None, duration_s: float = 60.0,
               base_hz: float = 0.05, harmonics: int = 5, fade_s: float = 4.0, seed: int = 0,
               vmax: float = 0.7, amax: float = 1.8, move_vel: float = 0.25,
               surface_z: float = 0.0) -> Trajectory | None:
    """Band-limited Fourier excitation of all joints around ``center`` (default home),
    framed by validated moves from and back to home. Amplitude per joint is the largest the
    speed/accel caps and the joint box allow, then shrunk until the envelope check passes."""
    rng = np.random.default_rng(seed)
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    center = home if center is None else np.asarray(center, dtype=np.float64)
    room = np.maximum(np.minimum(center - env.q_lo, env.q_hi - center), 1e-3)
    q = _fourier(center, room, rng, duration_s, base_hz, harmonics, fade_s, vmax, amax)
    lim = Limits(vmax, amax, limits.tcp_floor, limits.link_floor)
    g = 1.0
    for _ in range(25):
        qq = center + g * (q - center)
        if not check(qq, arm, env, lim, surface_z, stride=10):
            break
        g *= 0.88
    else:
        return None
    to_center = _route(home, [center], arm, env, limits, surface_z, move_vel, home)
    from_center = _route(center, [home], arm, env, limits, surface_z, move_vel, home)
    if to_center is None or from_center is None:
        return None
    b = _Builder(home)
    for leg in to_center:
        b.add(leg, "move")
    b.hold(1.0, "settle")
    b.add(qq, "excite")
    b.hold(1.0, "settle")
    for leg in from_center:
        b.add(leg, "move")
    tr = b.build(side, "excite", {"seed": seed, "base_hz": base_hz, "harmonics": harmonics,
                                  "duration_s": duration_s, "gain": round(g, 3),
                                  "center": np.round(center, 4).tolist(),
                                  "amplitude_rad": np.round(np.abs(qq - center).max(0), 3).tolist()})
    return tr


# ── inertial regressor (for excitation selection and coverage reports) ─────────


def _skew_rows(v):
    """(N,3) -> (N,3,3) cross-product matrices."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1), np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], 1)


def regressor(q: np.ndarray, arm: ArmModel, g_world=(0.0, 0.0, -9.81), stride: int = 10) -> np.ndarray:
    """Joint-torque regressor Y with tau = Y @ theta, theta = 10 inertial parameters per link
    (m, m*c, I about the link origin, in the DH link frame); (6*N, 60).

    Velocities and accelerations of every link come from finite differences of the
    rig-model kinematics along the path, so gravity uses the real 45-degree mount."""
    from robots_realtime.rig.dionysus import fk_frames

    q = np.asarray(q, dtype=np.float64)
    frames = np.array([[arm.world_T_base @ T for T in fk_frames(x)] for x in q])  # (N,6,4,4)
    R, o = frames[..., :3, :3], frames[..., :3, 3]
    dt = DT
    dR = np.gradient(R, dt, axis=0)
    W = np.einsum("nlij,nlkj->nlik", dR, R)  # dR R^T, skew of omega
    w = np.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], -1)  # (N,6,3) world
    wd = np.gradient(w, dt, axis=0)
    a = np.gradient(np.gradient(o, dt, axis=0), dt, axis=0) - np.asarray(g_world)
    sel = slice(2, len(q) - 2, stride)
    R, o, w, wd, a = R[sel], o[sel], w[sel], wd[sel], a[sel]
    base = arm.world_T_base
    z_axes = np.concatenate([np.broadcast_to(base[:3, 2], (len(R), 1, 3)), R[:, :5, :, 2]], 1)
    p_axes = np.concatenate([np.broadcast_to(base[:3, 3], (len(R), 1, 3)), o[:, :5]], 1)
    n = len(R)
    Y = np.zeros((n, 6, 60))
    for i in range(6):
        Ri = R[:, i]
        wl = np.einsum("nji,nj->ni", Ri, w[:, i])
        wdl = np.einsum("nji,nj->ni", Ri, wd[:, i])
        al = np.einsum("nji,nj->ni", Ri, a[:, i])
        Sw, Swd, Sa = _skew_rows(wl), _skew_rows(wdl), _skew_rows(al)
        # Force F = m a + wd x h + w x (w x h); moment N = I wd + w x (I w) + h x a  (link frame)
        AF = np.zeros((n, 3, 10))
        AN = np.zeros((n, 3, 10))
        AF[:, :, 0] = al
        AF[:, :, 1:4] = Swd + Sw @ Sw
        AN[:, :, 1:4] = -Sa
        # I(x) = [[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]] @ x, params order Ixx Ixy Ixz Iyy Iyz Izz
        def lin(x):
            out = np.zeros((n, 3, 6))
            out[:, 0, 0], out[:, 0, 1], out[:, 0, 2] = x[:, 0], x[:, 1], x[:, 2]
            out[:, 1, 1], out[:, 1, 3], out[:, 1, 4] = x[:, 0], x[:, 1], x[:, 2]
            out[:, 2, 2], out[:, 2, 4], out[:, 2, 5] = x[:, 0], x[:, 1], x[:, 2]
            return out
        AN[:, :, 4:10] = lin(wdl) + Sw @ lin(wl)
        F = np.einsum("nij,njk->nik", Ri, AF)
        N = np.einsum("nij,njk->nik", Ri, AN)
        for j in range(i + 1):  # joint j moves link i
            r = o[:, i] - p_axes[:, j]
            moment = N + np.cross(r[:, :, None], F, axisa=1, axisb=1, axisc=1)
            Y[:, j, 10 * i:10 * i + 10] = np.einsum("ni,nik->nk", z_axes[:, j], moment)
    return Y.reshape(-1, 60)


def condition(Y: np.ndarray, rank: int | None = None) -> tuple[float, int]:
    """Condition number over the identifiable subspace (top ``rank`` singular values)."""
    s = np.linalg.svd(Y / max(1.0, np.abs(Y).max()), compute_uv=False)
    if rank is None:
        rank = int((s > s[0] * 1e-6).sum())
    return float(s[0] / s[rank - 1]), rank


def excitation_set(side: str, arm: ArmModel, env: Envelope, limits: Limits, n_centers: int = 5,
                   candidates: int = 6, min_amplitude: float = 0.12, min_peak_speed: float = 0.3,
                   surface_z: float = 0.0,
                   poses_path: str | Path | None = None, **kw) -> list[Trajectory]:
    """``n_centers`` excitations at recorded poses spread over the visited region (the first
    is home); per center the candidate seed with the best-conditioned inertial regressor.
    Centers whose excitation the envelope shrinks below ``min_amplitude`` (median over joints,
    rad) or ``min_peak_speed`` (rad/s, excitation phase) are skipped: near the bounds the motion
    would be too small or too slow to identify the inertial terms."""
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    library = load_recorded_poses(side, arm, env, limits, surface_z, poses_path)
    # centers need room to move: prefer recorded poses away from the joint-box faces
    margin = np.minimum(library - env.q_lo, env.q_hi - library).min(1)
    roomy = library[margin > 0.2] if (margin > 0.2).sum() >= 4 * n_centers else library
    centers = [home] + list(roomy[farthest_points(roomy, 4 * n_centers, start=home)])
    out, rank = [], None
    for center in centers:
        if len(out) >= n_centers:
            break
        best = None
        for c in range(candidates):
            tr = excitation(side, arm, env, limits, center=center, seed=100 * len(out) + c,
                            surface_z=surface_z, **kw)
            if tr is None or np.median(tr.meta["amplitude_rad"]) < min_amplitude:
                continue
            qe = tr.q[tr.phase == tr.phase_names.index("excite")]
            if np.abs(np.diff(qe, axis=0)).max() / DT < min_peak_speed:
                continue
            Y = regressor(tr.q[tr.phase == tr.phase_names.index("excite")], arm)
            if rank is None:
                _, rank = condition(Y)
            cond, _ = condition(Y, rank)
            if best is None or cond < best[0]:
                best = (cond, tr)
        if best is not None:
            best[1].meta["regressor_condition"] = round(best[0], 1)
            out.append(best[1])
    return out


# ── coverage ──────────────────────────────────────────────────────────────────


def coverage(side: str, arm: ArmModel, trajs: list[Trajectory], poses_path: str | Path | None = None,
             cell: float = 0.05) -> dict:
    """How much of the recorded region the trajectories visit (5 cm TCP cells, joint spans)
    and how well the whole set conditions the inertial regressor."""
    rec = np.load(poses_path or DEFAULT_POSES)[side].astype(np.float64)
    q = np.concatenate([t.q[::25] for t in trajs])
    cells = lambda qs: {tuple(np.floor(arm.tcp_world(x)[:3, 3] / cell).astype(int)) for x in qs}  # noqa: E731
    rc, tc = cells(rec), cells(q)
    span_rec = np.percentile(rec, 99, 0) - np.percentile(rec, 1, 0)
    ex = [t for t in trajs if t.kind == "excite"]
    cond = None
    if ex:
        Y = np.concatenate([regressor(t.q[t.phase == t.phase_names.index("excite")], arm) for t in ex])
        cond, rank = condition(Y)
    return {"tcp_cells_covered": round(len(rc & tc) / len(rc), 3), "tcp_cells_recorded": len(rc),
            "joint_span_ratio": np.round((q.max(0) - q.min(0)) / span_rec, 2).tolist(),
            "excitation_regressor_condition": None if cond is None else round(cond, 1),
            "excitation_regressor_rank": None if cond is None else rank}


# ── CLI ───────────────────────────────────────────────────────────────────────


def generate(side: str, kind: str, out: Path, rig_yaml: str | None = None, envelope: str | None = None,
             poses: str | None = None, n_centers: int = 5) -> list[Path]:
    rig = load_rig(rig_yaml)
    arm = rig.arms[side]
    env = load_envelope(side, envelope)
    lim = Limits()
    jobs = []
    if kind in ("static", "all"):
        jobs.append(("static", static_holds(side, arm, env, lim, surface_z=rig.surface_z, poses_path=poses)))
    if kind in ("friction", "all"):
        jobs.append(("friction", friction_sweeps(side, arm, env, lim, surface_z=rig.surface_z)))
    if kind in ("excite", "all"):
        ex = excitation_set(side, arm, env, lim, n_centers=n_centers, surface_z=rig.surface_z, poses_path=poses)
        for i, tr in enumerate(ex):
            tr.meta["held_out"] = i == len(ex) - 1
            jobs.append((f"excite_c{i}", tr))
    paths = []
    for name, tr in jobs:
        bad = check(tr.q, arm, env, lim, rig.surface_z)
        if bad:
            raise RuntimeError(f"{side} {name} failed validation: {bad[:3]}")
        v = np.abs(np.diff(tr.q, axis=0)).max() / DT
        p = tr.save(out / f"{side}_{name}.npz")
        extra = f", regressor cond {tr.meta['regressor_condition']}" if "regressor_condition" in tr.meta else ""
        print(f"{p}: {tr.duration:6.1f} s, peak joint speed {v:.2f} rad/s{extra}")
        paths.append(p)
    report = coverage(side, arm, [tr for _, tr in jobs], poses)
    (out / f"{side}_coverage.json").write_text(json.dumps(report, indent=1))
    print(f"{side} coverage: {report}")
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("left", "right"))
    ap.add_argument("--kind", choices=("static", "friction", "excite", "all"), default="all")
    ap.add_argument("--out", type=Path, default=Path("sysid/trajectories"))
    ap.add_argument("--rig", default=None)
    ap.add_argument("--envelope", default=None)
    ap.add_argument("--poses", default=None, help="recorded pose library (default: configs/.../recorded_poses.npz)")
    ap.add_argument("--centers", type=int, default=5, help="excitation centers per arm")
    ap.add_argument("--build-poses", metavar="EXPORT_DIR",
                    help="rebuild the recorded pose library from export_episode.py NPZs and exit")
    a = ap.parse_args(argv)
    if a.build_poses:
        print(build_recorded_poses(a.build_poses, a.poses or DEFAULT_POSES))
        return
    if not a.arm:
        ap.error("--arm is required")
    generate(a.arm, a.kind, a.out, a.rig, a.envelope, a.poses, a.centers)


if __name__ == "__main__":
    main()
