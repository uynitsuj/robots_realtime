"""The "rr servo" robot program: a position servo for the dionysus followers.

Why not the vendor "Default program": its MotionGuide follower is a soft joint
impedance controller (``direct_torque``, Kp = 28 Nm/rad on the wrists, no
integral term). Joint friction then stops a joint wherever Kp·error drops below
it, and only a fast-moving target (its damping term) breaks it loose. Measured
on the left follower 2026-10-19: wrist_2 ignores a 0.03 rad target ramped over
5 s at every speed slider, and every move ends 2-6 mrad short. That is fine for
teleoperation and wrong for a planner.

This program keeps the vendor follower's register contract, so
:class:`~robots_realtime.robots.ur_register_arm.URRegisterArm` drives either one
the same way, and replaces the impedance loop with ``servoj`` (a stiff position
servo that converges on the target):

    in  double 24..29  joint target [rad]
    in  double 30      goal time [s] (the vendor's; ignored here)
    in  double 31      gripper command, 0 open .. 0.99 closed (x100 to the Robotiq,
                       exactly as the vendor program does)
    in  int    24      echo of this run's session token: targets count only then
    out double 24..29  the joint reference servoj is tracking
    out double 30      gripper position (``rq_current_pos_norm() * 0.01``, as vendor)
    out int    24      0 hold (no client), 1 tracking, 2 target refused (too far)
    out int    25      program version
    out int    26      session token, new on every Play

Robot-side safety, independent of the PC:
  * register values outlive the client that wrote them, so targets are ignored
    (the arm holds where it was at Play) until a client echoes this run's random
    session token: Play from the pendant with nobody connected never moves it;
  * the reference walks toward the target at most ``vmax`` rad/s, keeping the
    direction (the joint farthest from its target sets the pace), so a jump in
    the registers never becomes a jump of the arm;
  * a target farther than ``max_jump`` rad from the reference is refused (the arm
    holds) — wrongly seeded registers cannot move it;
  * the reference starts at the measured joints, so Play never moves the arm;
  * once a client has echoed the token for 0.5 s an RTDE watchdog stops the
    program if the echo is not updated at ``watchdog_hz`` (the PC died or hung).

The program is a copy of the arm's own ``Default program`` (.urpx JSON) whose
code is the Robotiq activation node plus one Script node: the application —
TCP, payload, mounting, the ``rq_*`` gripper functions — comes along unchanged.
Building is offline (``rr-ur-servo-program build``); ``rr-ur-servo-program upload --yes``
stores it on the arm next to ``Default program`` (it neither loads nor plays it).
A person loads it on the pendant for a run, the driver checks it is loaded
(``program: rr servo``), and a person loads ``Default program`` back afterwards
so the vendor teleop finds the arm as it left it.

The clone procedure is adapted from alvinz ``UR_747_teleop/ur747/servo_program.py``
(``clone_with_script``), which built the "ur747 servo" program on these arms.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from robots_realtime.rig import stop as rigstop

PROGRAM_NAME = "rr servo"
APPLICATION_NAME = "rr_servo"
VERSION = 1
MGLC_NODE = "ur-universal-robots-motionguide-live-control-program-node"
ROBOTIQ_ACTIVATE_NODE = "robotiq-gripper-activate-program"
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

SCRIPT_TEMPLATE = """\
# ---- rr servo v{version} (dionysus follower) ----
# Position servo on the vendor follower's register contract; see
# robots_realtime/rig/servo_program.py for the contract and the safety notes.
global gs_status = 0
global gs_token = floor(random() * 1000000) + 1

thread gs_grip_thread():
  # The vendor follower's gripper loop, unchanged.
  rq_set_speed_norm(100)
  local grip_norm = 0
  while True:
    grip_norm = read_input_float_register(31) * 100
    write_output_float_register(30, rq_current_pos_norm() * 0.01)
    if grip_norm > 99:
      grip_norm = 99
    end
    if grip_norm < 0:
      grip_norm = 0
    end
    rq_move_norm_multi_grippers(grip_norm)
  end
end

def gs_servo():
  local VMAX = {vmax}
  local MAX_JUMP = {max_jump}
  local dt = get_steptime()
  local q_ref = get_actual_joint_positions()
  local q_tgt = get_actual_joint_positions()
  local d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  local dmax = 0.0
  local scale = 0.0
  local live = 0
  local armed = False
  local i = 0
  write_output_integer_register(25, {version})
  write_output_integer_register(26, gs_token)
  while True:
    i = 0
    while i < 6:
      q_tgt[i] = read_input_float_register(24 + i)
      i = i + 1
    end
    if read_input_integer_register(24) == gs_token:
      live = live + 1
      if not armed and live * dt > 0.5:
        rtde_set_watchdog("input_int_register_24", {watchdog_hz}, "stop")
        armed = True
      end
      d = q_tgt - q_ref
      dmax = 0.0
      i = 0
      while i < 6:
        if norm(d[i]) > dmax:
          dmax = norm(d[i])
        end
        i = i + 1
      end
      if dmax > MAX_JUMP:
        gs_status = 2
      else:
        scale = 1.0
        if dmax > VMAX * dt:
          scale = VMAX * dt / dmax
        end
        q_ref = q_ref + d * scale
        gs_status = 1
      end
    else:
      live = 0
      gs_status = 0
    end
    i = 0
    while i < 6:
      write_output_float_register(24 + i, q_ref[i])
      i = i + 1
    end
    write_output_integer_register(24, gs_status)
    servoj(q_ref, t=dt, lookahead_time={lookahead}, gain={gain})
  end
end

gs_gthread = run gs_grip_thread()
gs_servo()
"""


def render_script(*, vmax: float = 0.5, max_jump: float = 0.1, watchdog_hz: float = 10.0,
                  lookahead: float = 0.1, gain: float = 300.0) -> str:
    if not 0.0 < vmax <= 1.0:
        raise ValueError("vmax must be in (0, 1] rad/s")
    if not 0.0 < max_jump <= 0.2:
        raise ValueError("max_jump must be in (0, 0.2] rad")
    if not 0.03 <= lookahead <= 0.2:
        raise ValueError("servoj lookahead_time must be in [0.03, 0.2] s")
    if not 100.0 <= gain <= 2000.0:
        raise ValueError("servoj gain must be in [100, 2000]")
    return SCRIPT_TEMPLATE.format(version=VERSION, vmax=f"{vmax:.4f}", max_jump=f"{max_jump:.4f}",
                                  watchdog_hz=f"{watchdog_hz:.1f}", lookahead=f"{lookahead:.3f}",
                                  gain=f"{gain:.1f}")


def _fresh_uuids(doc):
    """Give every non-zero UUID (values and keys) a new one, consistently, so the copy can't alias the source."""
    mapping: dict[str, str] = {}

    def sub(s: str) -> str:
        if not _UUID.match(s) or uuid.UUID(s).int == 0:
            return s
        return mapping.setdefault(s.lower(), str(uuid.uuid4()))

    def walk(x):
        if isinstance(x, dict):
            return {sub(k) if isinstance(k, str) else k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return sub(x) if isinstance(x, str) else x

    return walk(doc)


def clone_with_script(source: dict, script: str, *, name: str, label: str, description: str,
                      application_name: str) -> dict:
    """Copy of a vendor follower ``Default program`` whose code is the Robotiq activation + one Script node."""
    doc = _fresh_uuids(copy.deepcopy(source))
    program = doc["program"]
    root = program["programContent"]
    kids = {c["contributedNode"]["type"]: c for c in root["children"]}
    for t in ("ur-modules", "ur-functions", "ur-before-start", "ur-code"):
        if t not in kids:
            raise ValueError(f"source program has no {t} node")
    if kids["ur-modules"]["children"] or kids["ur-functions"]["children"]:
        raise ValueError("source program has modules/functions; not a vendor Default program layout")
    code = kids["ur-code"]
    types = [c["contributedNode"]["type"] for c in code["children"]]
    if MGLC_NODE not in types or not set(types) <= {MGLC_NODE, ROBOTIQ_ACTIVATE_NODE}:
        raise ValueError(f"unexpected ur-code children {types}; expected the vendor MotionGuide follower")
    activate = next((c for c in code["children"] if c["contributedNode"]["type"] == ROBOTIQ_ACTIVATE_NODE), None)
    if activate is None:
        raise ValueError("source has no Robotiq activation node (is this a follower?)")

    script_node = {
        "children": [],
        "contributedNode": {"type": "ur-script", "version": "0.0.1", "lockChildren": False, "allowsChildren": False,
                            "parameters": {"name": label, "expression": script, "valid": True}},
        "guid": str(uuid.uuid4()),
        "parentId": code["guid"],
        "programLabel": label,
    }
    kids["ur-before-start"]["children"] = []
    code["contributedNode"]["parameters"]["loopForever"] = False  # the script loops itself
    code["children"] = [activate, script_node]

    params = root["contributedNode"]["parameters"]
    history = params.get("symbolHistory", {})
    if isinstance(history.get("variables"), list):  # drop variables owned by removed nodes
        history["variables"] = [v for v in history["variables"]
                                if not (isinstance(v, dict) and "programNodeMetadata" in v)]
    params["name"] = name
    info = program["programInformation"]
    info["name"] = name
    info["description"] = f"{description}; application copied from {source['program']['programInformation']['name']!r}"
    doc["application"]["applicationInfo"]["name"] = application_name

    src = source["program"]["urscript"]["script"]
    m = re.search(rf'^\$ \d+ "{ROBOTIQ_ACTIVATE_NODE}"\n(.*?)(?=^\$ \d+ )', src, re.S | re.M)
    if not m:
        raise ValueError("could not find the Robotiq activation script in the source")
    ids = [str(uuid.UUID(int=0)), root["guid"], kids["ur-modules"]["guid"], kids["ur-functions"]["guid"],
           kids["ur-before-start"]["guid"], code["guid"], activate["guid"], script_node["guid"]]
    header = ['"ur-program"', '"ur-modules"', '"ur-functions"', '"ur-before-start"', '"ur-code"',
              f'"{ROBOTIQ_ACTIVATE_NODE}"', '"ur-script"']
    lines = []
    for n, h in enumerate(header, start=1):
        lines.append(f"$ {n} {h}")
        if h == f'"{ROBOTIQ_ACTIVATE_NODE}"':
            lines.append(m.group(1).rstrip("\n"))
    program["urscript"] = {"script": "\n".join(lines) + "\n" + script, "nodeIDList": ids}
    json.dumps(doc)  # must stay serializable
    return doc


def build_program(source: dict, **script_kw) -> dict:
    """``rr servo`` built from ``source`` (an arm's downloaded ``Default program``)."""
    return clone_with_script(source, render_script(**script_kw), name=PROGRAM_NAME,
                             label=f"rr servo v{VERSION}", description=f"rr servo v{VERSION} position servo",
                             application_name=APPLICATION_NAME)


# ── robot I/O (REST) ──────────────────────────────────────────────────────


def download_program(ip: str, name: str = "Default program") -> dict:
    url = f"http://{ip}{rigstop.ROBOT_API}/programs/v1/{quote(name, safe='')}"
    with urlopen(Request(url, method="GET"), timeout=15.0) as r:  # noqa: S310 - robot LAN
        return json.loads(r.read().decode("utf-8"))


def list_programs(ip: str) -> list[str]:
    return [p.get("name") for p in rigstop._rest(ip, "GET", "/programs/v1/", timeout=5.0).get("programs", [])]


def upload_program(ip: str, doc: dict, *, replace: bool = False) -> str:
    """Store ``doc`` on the arm (POST, or PUT to replace a program of the same name)."""
    name = f"{doc['program']['programInformation']['name']}.urpx"
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + json.dumps(doc).encode() \
        + f"\r\n--{boundary}--\r\n".encode()
    req = Request(f"http://{ip}{rigstop.ROBOT_API}/programs/v1/", data=body, method="PUT" if replace else "POST",
                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(req, timeout=30.0) as r:  # noqa: S310 - robot LAN
        return r.read().decode("utf-8", "replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rr-ur-servo-program", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the .urpx offline (downloads Default program unless --source)")
    b.add_argument("--arm", required=True, choices=("left", "right"))
    b.add_argument("--source", type=Path, help="the arm's Default program .urpx (default: GET it from the arm)")
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--script-only", type=Path, help="also write the bare URScript here (for the pendant)")
    b.add_argument("--vmax", type=float, default=0.5)
    b.add_argument("--max-jump", type=float, default=0.1)
    u = sub.add_parser("upload", help="store a built .urpx on the arm (does not load or play it)")
    u.add_argument("--arm", required=True, choices=("left", "right"))
    u.add_argument("--file", type=Path, required=True)
    u.add_argument("--replace", action="store_true", help="overwrite an existing program of the same name")
    u.add_argument("--yes", action="store_true", help="required: this writes to the robot")
    ls = sub.add_parser("list", help="programs stored on the arm")
    ls.add_argument("--arm", required=True, choices=("left", "right"))
    a = ap.parse_args(argv)
    ip = rigstop.follower_ips()[a.arm]

    if a.cmd == "list":
        for n in list_programs(ip):
            print(n)
        return 0
    if a.cmd == "build":
        source = json.loads(a.source.read_text()) if a.source else download_program(ip)
        doc = build_program(source, vmax=a.vmax, max_jump=a.max_jump)
        a.output.write_text(json.dumps(doc))
        print(f"wrote {a.output} ({PROGRAM_NAME!r} v{VERSION}, from {source['program']['programInformation']['name']!r})")
        if a.script_only:
            a.script_only.write_text(render_script(vmax=a.vmax, max_jump=a.max_jump))
            print(f"wrote {a.script_only}")
        return 0
    if not a.yes:
        print("refusing to write to the robot without --yes", file=sys.stderr)
        return 2
    doc = json.loads(a.file.read_text())
    if doc["program"]["programInformation"]["name"] != PROGRAM_NAME:
        print(f"{a.file} is not a {PROGRAM_NAME!r} program", file=sys.stderr)
        return 2
    present = PROGRAM_NAME in list_programs(ip)
    if present and not a.replace:
        print(f"{PROGRAM_NAME!r} already on {a.arm} ({ip}); pass --replace to overwrite", file=sys.stderr)
        return 2
    print(upload_program(ip, doc, replace=present))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
