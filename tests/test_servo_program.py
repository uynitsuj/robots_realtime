"""Offline checks of the "rr servo" robot program builder (no robot traffic)."""

import json
import re
import uuid

import pytest

from robots_realtime.rig import servo_program as sp

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _node(kind: str, guid: str, children=()):
    return {"children": list(children), "contributedNode": {"type": kind, "parameters": {}}, "guid": guid}


def _vendor_source() -> dict:
    """The shape of a follower's downloaded ``Default program`` (.urpx JSON)."""
    g = {k: str(uuid.uuid4()) for k in ("root", "mod", "fn", "before", "conf", "status", "code", "act", "mglc")}
    code = _node("ur-code", g["code"], [_node(sp.ROBOTIQ_ACTIVATE_NODE, g["act"]), _node(sp.MGLC_NODE, g["mglc"])])
    code["contributedNode"]["parameters"] = {"loopForever": False}
    root = _node("ur-program", g["root"], [
        _node("ur-modules", g["mod"]), _node("ur-functions", g["fn"]), _node("ur-before-start", g["before"]),
        _node("ur-configuration", g["conf"]), _node("ur-status", g["status"]), code])
    root["contributedNode"]["parameters"] = {"name": "Default program", "symbolHistory": {"variables": [
        {"name": "base"}, {"name": "wp1", "programNodeMetadata": {"id": g["mglc"]}}]}}
    script = ('$ 1 "ur-program"\n$ 2 "ur-modules"\n$ 3 "ur-functions"\n$ 4 "ur-before-start"\n$ 5 "ur-code"\n'
              f'$ 6 "{sp.ROBOTIQ_ACTIVATE_NODE}"\nrq_set_max_current_for_all_grippers()\n'
              'rq_activate_all_grippers(False)\n'
              f'$ 7 "{sp.MGLC_NODE}"\nmglc_follower()\n')
    return {
        "program": {"id": str(uuid.uuid4()), "programInformation": {"name": "Default program"},
                    "programContent": root,
                    "urscript": {"script": script, "nodeIDList": [str(uuid.UUID(int=0)), g["root"], g["mod"],
                                                                   g["fn"], g["before"], g["code"], g["act"],
                                                                   g["mglc"]]}},
        "application": {"id": str(uuid.uuid4()), "applicationInfo": {"name": "main_application"},
                        "applicationContent": {"tcp": g["root"]},
                        "urscript": {"script": "def rq_move_norm_multi_grippers(p):\nend\n"}},
    }


def test_build_replaces_motionguide_with_one_script_node():
    src = _vendor_source()
    doc = sp.build_program(src)
    prog = doc["program"]
    assert prog["programInformation"]["name"] == sp.PROGRAM_NAME
    assert prog["programContent"]["contributedNode"]["parameters"]["name"] == sp.PROGRAM_NAME
    assert doc["application"]["applicationInfo"]["name"] == sp.APPLICATION_NAME
    code = next(c for c in prog["programContent"]["children"] if c["contributedNode"]["type"] == "ur-code")
    assert [c["contributedNode"]["type"] for c in code["children"]] == [sp.ROBOTIQ_ACTIVATE_NODE, "ur-script"]
    script_node = code["children"][1]
    assert script_node["contributedNode"]["parameters"]["expression"] == sp.render_script()
    assert script_node["parentId"] == code["guid"]
    body = prog["urscript"]["script"]
    assert [ln for ln in body.split("\n") if ln.startswith("$ ")][-2:] == [
        f'$ 6 "{sp.ROBOTIQ_ACTIVATE_NODE}"', '$ 7 "ur-script"']
    assert "rq_activate_all_grippers(False)" in body and "mglc_follower" not in body
    assert prog["urscript"]["nodeIDList"][-1] == script_node["guid"]
    assert len(prog["urscript"]["nodeIDList"]) == 8
    # node-owned symbols of the removed MotionGuide node are dropped, the rest kept
    names = [v["name"] for v in prog["programContent"]["contributedNode"]["parameters"]["symbolHistory"]["variables"]]
    assert names == ["base"]
    # the application (Robotiq functions, TCP, payload) comes along unchanged
    assert doc["application"]["urscript"] == src["application"]["urscript"]


def test_build_shares_no_uuid_with_the_source_and_keeps_references():
    src = _vendor_source()
    doc = sp.build_program(src)
    old = set(_UUID.findall(json.dumps(src))) - {str(uuid.UUID(int=0))}
    new = set(_UUID.findall(json.dumps(doc)))
    assert not old & new
    # a reference to the root guid inside the application was remapped consistently
    assert doc["application"]["applicationContent"]["tcp"] == doc["program"]["programContent"]["guid"]
    assert src["program"]["programInformation"]["name"] == "Default program"  # source untouched


def test_build_refuses_programs_that_are_not_a_vendor_follower():
    src = _vendor_source()
    code = src["program"]["programContent"]["children"][-1]
    code["children"] = [c for c in code["children"] if c["contributedNode"]["type"] != sp.ROBOTIQ_ACTIVATE_NODE]
    with pytest.raises(ValueError, match="Robotiq activation"):
        sp.build_program(src)
    src = _vendor_source()
    src["program"]["programContent"]["children"][-1]["children"][1]["contributedNode"]["type"] = "ur-move"
    with pytest.raises(ValueError, match="unexpected ur-code children"):
        sp.build_program(src)


def test_script_has_the_robot_side_safety():
    s = sp.render_script(vmax=0.4, max_jump=0.08, watchdog_hz=10.0)
    assert "local VMAX = 0.4000" in s and "local MAX_JUMP = 0.0800" in s
    assert "if read_input_integer_register(24) == gs_token:" in s  # targets only with the session echo
    assert "write_output_integer_register(26, gs_token)" in s
    assert 'rtde_set_watchdog("input_int_register_24", 10.0, "stop")' in s
    assert "q_ref = get_actual_joint_positions()" in s  # Play holds where the arm is
    assert "servoj(q_ref, t=dt, lookahead_time=0.100, gain=300.0)" in s
    assert "rq_move_norm_multi_grippers(grip_norm)" in s and "write_output_float_register(30" in s
    # every def/thread/while/if block is closed (URScript has no other block openers here)
    lines = [ln.strip() for ln in s.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    openers = [ln for ln in lines if ln.endswith(":") and ln.split()[0] in ("def", "thread", "while", "if")]
    assert len(openers) == lines.count("end")


@pytest.mark.parametrize("kw", [{"vmax": 0.0}, {"vmax": 1.5}, {"max_jump": 0.5}, {"lookahead": 0.01},
                                {"gain": 50.0}])
def test_script_parameters_are_bounded(kw):
    with pytest.raises(ValueError):
        sp.render_script(**kw)
