"""Stop latch + out-of-band program stop for the dionysus rig.

Two layers live here; both are stdlib-only so they work from any interpreter on
the machine (any project's venv, this package's venv, a bare ``python3``):

**Latch** (``/dev/shm/dionysus_rig_stop``). Any component that decides to stop
sets it; every :class:`~robots_realtime.robots.ur_register_arm.URRegisterArm`
checks it on every tick and, once set, holds ``actual_q`` and ignores all further
commands. Other control clients on the rig can check it too. It is a file so a process that
is wedged or dead cannot hold it hostage, and a person can set it from a shell.
Clearing it (``rr-ur-stop --reset``) never resumes motion by itself: the drivers
need a fresh seed-then-play to move again.

**Program stop** (REST ``PUT /universal-robots/robot-api/program/v1/state
{"action": "stop"}``). Needs no RTDE input ownership, so it still works when the
driver, the bus or a client is hung. On the vendor "Default program" a stopped program
leaves the arm holding where it is. Requires the arm in Remote control mode,
which is the operating mode for this stack. The hardware e-stop stays the
primary stop; this is the software path behind it.

CLI::

    rr-ur-stop               # latch + REST stop both followers
    rr-ur-stop --status        # latch, program state, safety mode per arm
    rr-ur-stop --reset         # clear the latch (asks; arms must be NORMAL and STOPPED)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LATCH_PATH = Path(os.environ.get("DIONYSUS_STOP_LATCH", "/dev/shm/dionysus_rig_stop"))
ROBOT_API = "/universal-robots/robot-api"
DEFAULT_FOLLOWERS = {"right": "192.168.0.13", "left": "192.168.0.11"}


# ── Latch ─────────────────────────────────────────────────────────────────────


def latch_set(reason: str, source: str = "") -> None:
    """Set the stop latch (idempotent; the first reason is kept)."""
    if LATCH_PATH.exists():
        return
    payload = {"t": time.time(), "reason": reason, "source": source, "pid": os.getpid()}
    tmp = LATCH_PATH.with_name(f".{LATCH_PATH.name}.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, LATCH_PATH)
    except OSError:
        # Fall back to a direct write; a partial file still reads as "latched".
        LATCH_PATH.write_text(json.dumps(payload))


def latch_active() -> dict | None:
    """The latch payload if set, else None. Cheap enough to call every tick."""
    try:
        raw = LATCH_PATH.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        return {"reason": "latch unreadable", "source": "?"}
    try:
        return json.loads(raw) if raw else {"reason": "latched", "source": "?"}
    except ValueError:
        return {"reason": "latched", "source": "?"}


def latch_clear() -> None:
    try:
        LATCH_PATH.unlink()
    except FileNotFoundError:
        pass


# ── REST ──────────────────────────────────────────────────────────────────────


def _rest(ip: str, method: str, path: str, payload: dict | None = None, timeout: float = 1.5) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(f"http://{ip}{ROBOT_API}{path}", data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data) if data else {}


def program_stop(ip: str, timeout: float = 1.5) -> tuple[bool, str]:
    """REST program stop on one arm. Returns (ok, message); never raises."""
    try:
        reply = _rest(ip, "PUT", "/program/v1/state", {"action": "stop"}, timeout=timeout)
        return True, str(reply.get("state", reply))
    except HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    except (URLError, OSError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}"


def arm_status(ip: str, timeout: float = 1.5) -> dict:
    out: dict = {"ip": ip}
    for key, path in (
        ("program_state", "/program/v1/state"),
        ("loaded_program", "/program/v1/loaded"),
        ("control_mode", "/system/v1/controlmode"),
        ("safety_mode", "/robotstate/v1/safetymode"),
        ("robot_mode", "/robotstate/v1/robotmode"),
    ):
        try:
            r = _rest(ip, "GET", path, timeout=timeout)
            out[key] = r.get("state") or r.get("name") or r.get("mode") or r
        except Exception as e:
            out[key] = f"? ({type(e).__name__})"
    return out


def emergency_stop(ips: dict[str, str] | list[str], reason: str, source: str = "",
                   timeout: float = 1.5) -> dict[str, tuple[bool, str]]:
    """Latch first (fast, local), then REST-stop every arm in parallel."""
    latch_set(reason, source)
    targets = ips if isinstance(ips, dict) else {ip: ip for ip in ips}
    results: dict[str, tuple[bool, str]] = {}

    def _one(name: str, ip: str) -> None:
        results[name] = program_stop(ip, timeout=timeout)

    threads = [threading.Thread(target=_one, args=(n, ip), daemon=True) for n, ip in targets.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 0.5)
    for name in targets:
        results.setdefault(name, (False, "timed out"))
    return results


def follower_ips(rig_yaml: str | None = None) -> dict[str, str]:
    """Follower IPs from rig.yaml when available, else the known defaults."""
    try:
        import yaml

        from robots_realtime.rig.dionysus import DEFAULT_RIG_YAML

        cfg = yaml.safe_load(Path(rig_yaml or DEFAULT_RIG_YAML).read_text())
        return {name: str(a["ip"]) for name, a in cfg["arms"].items()}
    except Exception:
        return dict(DEFAULT_FOLLOWERS)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rig-stop", description=__doc__.split("\n\n")[0])
    p.add_argument("--status", action="store_true", help="print latch + per-arm program/safety state")
    p.add_argument("--reset", action="store_true", help="clear the latch (arms must be NORMAL and STOPPED)")
    p.add_argument("--yes", action="store_true", help="do not ask before --reset")
    p.add_argument("--rig", default=None, help="rig.yaml (for follower IPs)")
    p.add_argument("--reason", default="rig-stop CLI")
    args = p.parse_args(argv)
    ips = follower_ips(args.rig)

    if args.status:
        latch = latch_active()
        print(f"latch: {'SET ' + json.dumps(latch) if latch else 'clear'}  ({LATCH_PATH})")
        for name, ip in ips.items():
            print(f"{name:6s} {json.dumps(arm_status(ip))}")
        return 0

    if args.reset:
        states = {name: arm_status(ip) for name, ip in ips.items()}
        bad = {n: s for n, s in states.items()
               if s.get("safety_mode") != "NORMAL" or s.get("program_state") not in ("STOPPED",)}
        for n, s in states.items():
            print(f"{n:6s} program={s.get('program_state')} safety={s.get('safety_mode')}")
        if bad:
            print("refusing to clear the latch: every arm must be safety NORMAL with its program STOPPED",
                  file=sys.stderr)
            return 2
        if not args.yes:
            ans = input("Clear the stop latch? Motion still needs a fresh seed-then-play. [y/N] ")
            if ans.strip().lower() != "y":
                print("latch kept")
                return 1
        latch_clear()
        print("latch cleared")
        return 0

    results = emergency_stop(ips, reason=args.reason, source="rig-stop CLI")
    ok = True
    for name, (good, msg) in results.items():
        ok &= good
        print(f"{name:6s} {'STOPPED' if good else 'FAILED'}: {msg}")
    print(f"latch SET at {LATCH_PATH}")
    if not ok:
        print("REST stop failed on at least one arm — use the pendant / hardware e-stop NOW.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
