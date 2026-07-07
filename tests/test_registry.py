import logging

import jsonschema
import pytest

from tools.base import Tool, ToolKind
from tools.registry import ToolRegistry


async def _noop(args):
    return "ok"


def make_tool(name: str, parameters: dict | None = None) -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters=parameters
        or {"type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"]},
        kind=ToolKind.READ,
        run=_noop,
    )


def test_register_and_get():
    r = ToolRegistry()
    tool = make_tool("read_file")
    r.register(tool)
    assert r.get("read_file") is tool
    assert [t.name for t in r.list()] == ["read_file"]


def test_get_unknown_raises_keyerror():
    r = ToolRegistry()
    with pytest.raises(KeyError):
        r.get("nope")


def test_collision_is_rejected_loudly_incumbent_wins(caplog):
    r = ToolRegistry()
    builtin = make_tool("read_file")
    shadow = make_tool("read_file")
    r.register(builtin)
    with caplog.at_level(logging.WARNING):
        r.register(shadow)
    # loud: a warning was logged...
    assert "already registered" in caplog.text
    # ...and the incumbent (builtin) is kept, not the duplicate.
    assert r.get("read_file") is builtin


def test_allowlist_hides_tool_from_wire_schemas():
    r = ToolRegistry(allowlist={"read_file"})
    r.register(make_tool("read_file"))
    r.register(make_tool("write_file"))
    names = [w["name"] for w in r.wire_schemas()]
    assert names == ["read_file"]  # write_file hidden from the model


def test_no_allowlist_exposes_everything():
    r = ToolRegistry()  # None = allow all
    r.register(make_tool("read_file"))
    r.register(make_tool("write_file"))
    assert {w["name"] for w in r.wire_schemas()} == {"read_file", "write_file"}


def test_validate_call_rejects_missing_required_field():
    r = ToolRegistry()
    r.register(make_tool("read_file"))
    with pytest.raises(jsonschema.ValidationError):
        r.validate_call("read_file", {})  # missing "path"


def test_validate_call_rejects_bad_typed_argument():
    r = ToolRegistry()
    r.register(make_tool("read_file"))
    with pytest.raises(jsonschema.ValidationError):
        r.validate_call("read_file", {"path": 123})  # path must be a string


def test_validate_call_rejects_non_allowlisted_tool():
    r = ToolRegistry(allowlist={"read_file"})
    r.register(make_tool("read_file"))
    r.register(make_tool("write_file"))
    # even a well-formed call to a hidden tool is refused before dispatch.
    with pytest.raises(ValueError):
        r.validate_call("write_file", {"path": "x"})


def test_validate_call_accepts_valid_args():
    r = ToolRegistry()
    r.register(make_tool("read_file"))
    r.validate_call("read_file", {"path": "x.txt"})  # no raise
