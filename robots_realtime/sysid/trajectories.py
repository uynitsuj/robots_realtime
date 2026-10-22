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


# ── generators ────────────────────────────────────────────────────────────────


def static_holds(side: str, arm: ArmModel, env: Envelope, limits: Limits, n_poses: int = 40,
                 settle_s: float = 2.0, record_s: float = 2.0, move_vel: float = 0.25,
                 reverse_every: int = 3, overshoot: float = 0.04, seed: int = 0,
                 surface_z: float = 0.0) -> Trajectory:
    """Hold ``n_poses`` poses (gravity, payload, static friction). Every ``reverse_every``-th
    pose is approached from the opposite side (overshoot, then back) to expose the static
    friction band."""
    rng = np.random.default_rng(seed)
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    span = np.array([0.6, 0.5, 0.6, 0.8, 0.8, 1.0])
    poses = []
    for _ in range(20000):
        q = np.clip(home + rng.uniform(-span, span), env.q_lo, env.q_hi)
        if not check_pose(q, arm, env, limits, surface_z):
            poses.append(q)
        if len(poses) >= n_poses:
            break
    order, rest, cur = [], list(range(len(poses))), home
    while rest:  # greedy nearest neighbour keeps moves short
        k = min(rest, key=lambda i: np.abs(poses[i] - cur).max())
        order.append(k)
        rest.remove(k)
        cur = poses[k]
    def safe(*waypoints) -> list[np.ndarray] | None:
        """Min-jerk legs through ``waypoints`` from the current pose, or None if any leg leaves the envelope."""
        legs, cur = [], b.last
        for wp in waypoints:
            leg = min_jerk(cur, wp, move_vel, limits.amax)
            if check(leg, arm, env, limits, surface_z, stride=10):
                return None
            legs.append(leg)
            cur = wp
        return legs

    b = _Builder(home)
    held = 0
    for k in order:
        target = poses[k]
        route = [target]
        if reverse_every and held % reverse_every == reverse_every - 1:
            beyond = np.clip(target + overshoot * np.sign(target - b.last), env.q_lo, env.q_hi)
            route = [beyond, target]
        legs = safe(*route) or safe(home, *route)  # direct, else through home
        if legs is None:
            continue
        for leg in legs:
            b.add(leg, "move")
        b.hold(settle_s, "settle")
        b.hold(record_s, "hold")
        held += 1
    b.add(min_jerk(b.last, home, move_vel, limits.amax), "move")
    return b.build(side, "static", {"n_poses": held, "seed": seed, "settle_s": settle_s,
                                    "record_s": record_s, "reverse_every": reverse_every})


def friction_sweeps(side: str, arm: ArmModel, env: Envelope, limits: Limits,
                    joints=(0, 1, 2, 3, 4, 5), speeds=(0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8),
                    half_stroke: float = 0.35, accel: float = 1.0, dwell_s: float = 0.5,
                    move_vel: float = 0.25, surface_z: float = 0.0) -> Trajectory:
    """Constant-speed strokes, one joint at a time (others at home), both directions."""
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    b = _Builder(home)
    used = {}
    for j in joints:
        # Largest symmetric stroke around home that stays in the envelope (checked densely).
        A = half_stroke
        while A > 0.05:
            probe = np.repeat(home[None], 41, axis=0)
            probe[:, j] += np.linspace(-A, A, 41)
            if all(not check_pose(q, arm, env, limits, surface_z) for q in probe):
                break
            A *= 0.85
        if A <= 0.05:
            continue
        used[j] = round(A, 3)
        lo = home.copy()
        lo[j] -= A
        b.add(min_jerk(b.last, lo, move_vel, limits.amax), "move")
        reachable = float(np.sqrt(2 * A * accel / 2.0))  # peak speed a 2A stroke can reach
        for v in sorted({round(min(v, limits.vmax, reachable), 3) for v in speeds}):
            b.add(stroke(b.last, j, 2 * A, v, accel), f"j{j} +{v:g}")
            b.hold(dwell_s, "dwell")
            b.add(stroke(b.last, j, -2 * A, v, accel), f"j{j} -{v:g}")
            b.hold(dwell_s, "dwell")
        b.add(min_jerk(b.last, home, move_vel, limits.amax), "move")
    return b.build(side, "friction", {"half_stroke": used, "speeds": list(speeds), "accel": accel})


def excitation(side: str, arm: ArmModel, env: Envelope, limits: Limits, duration_s: float = 40.0,
               base_hz: float = 0.1, harmonics: int = 5, fade_s: float = 3.0, seed: int = 0,
               vmax: float = 0.6, amax: float = 1.5, surface_z: float = 0.0) -> Trajectory:
    """Band-limited Fourier excitation of all six joints around home (inertial parameters)."""
    rng = np.random.default_rng(seed)
    home = np.clip(arm.home_q, env.q_lo, env.q_hi)
    t = np.arange(0.0, duration_s, DT)
    w = np.ones_like(t)  # raised-cosine fade in/out: zero speed at both ends
    k = t < fade_s
    w[k] = 0.5 * (1 - np.cos(np.pi * t[k] / fade_s))
    k = t > duration_s - fade_s
    w[k] = 0.5 * (1 - np.cos(np.pi * (duration_s - t[k]) / fade_s))
    f = base_hz * np.arange(1, harmonics + 1)
    a = rng.normal(size=(6, harmonics)) / np.arange(1, harmonics + 1)
    ph = rng.uniform(0, 2 * np.pi, size=(6, harmonics))
    shape = np.einsum("jk,jkt->tj", a, np.sin(2 * np.pi * f[None, :, None] * t[None, None, :] + ph[..., None]))
    shape *= w[:, None]
    # Per-joint scale for the speed/accel caps, then shrink until the envelope check passes.
    vel = np.gradient(shape, DT, axis=0)
    acc = np.gradient(vel, DT, axis=0)
    scale = np.minimum(vmax / np.abs(vel).max(0), amax / np.abs(acc).max(0))
    room = np.minimum(home - env.q_lo, env.q_hi - home)
    scale = np.minimum(scale, room / np.abs(shape).max(0))
    g = 1.0
    for _ in range(30):
        q = home + g * shape * scale
        if not check(q, arm, env, Limits(vmax, amax, limits.tcp_floor, limits.link_floor), surface_z, stride=10):
            break
        g *= 0.85
    else:
        raise RuntimeError("could not fit an excitation inside the envelope")
    b = _Builder(home)
    b.add(q, "excite")
    return b.build(side, "excite", {"seed": seed, "base_hz": base_hz, "harmonics": harmonics,
                                    "duration_s": duration_s, "gain": round(g, 3),
                                    "amplitude_rad": np.round(np.abs(q - home).max(0), 3).tolist()})


# ── CLI ───────────────────────────────────────────────────────────────────────


def generate(side: str, kind: str, out: Path, rig_yaml: str | None = None, envelope: str | None = None,
             excite_seeds=(0, 1, 2, 3)) -> list[Path]:
    rig = load_rig(rig_yaml)
    arm = rig.arms[side]
    env = load_envelope(side, envelope)
    lim = Limits()
    jobs = []
    if kind in ("static", "all"):
        jobs.append(("static", static_holds(side, arm, env, lim, surface_z=rig.surface_z)))
    if kind in ("friction", "all"):
        jobs.append(("friction", friction_sweeps(side, arm, env, lim, surface_z=rig.surface_z)))
    if kind in ("excite", "all"):
        for s in excite_seeds:
            tr = excitation(side, arm, env, lim, seed=s, surface_z=rig.surface_z)
            tr.meta["held_out"] = s == excite_seeds[-1]
            jobs.append((f"excite_s{s}", tr))
    paths = []
    for name, tr in jobs:
        bad = check(tr.q, arm, env, lim, rig.surface_z)
        if bad:
            raise RuntimeError(f"{side} {name} failed validation: {bad[:3]}")
        v = np.abs(np.diff(tr.q, axis=0)).max() / DT
        p = tr.save(out / f"{side}_{name}.npz")
        print(f"{p}: {tr.duration:6.1f} s, peak joint speed {v:.2f} rad/s, {len(tr.phase_names)} phases")
        paths.append(p)
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("left", "right"), required=True)
    ap.add_argument("--kind", choices=("static", "friction", "excite", "all"), default="all")
    ap.add_argument("--out", type=Path, default=Path("sysid/trajectories"))
    ap.add_argument("--rig", default=None)
    ap.add_argument("--envelope", default=None)
    a = ap.parse_args(argv)
    generate(a.arm, a.kind, a.out, a.rig, a.envelope)


if __name__ == "__main__":
    main()
