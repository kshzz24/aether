"""Adapting MCP tools into FORGE tools.

This is Phase 6's anti-corruption layer: past it, a server's tool is
indistinguishable from a builtin. The decisions worth testing are the ones that
are not obvious — namespacing, refusing to trust a server's own safety hint,
and turning every remote failure into an observation rather than an exception.
"""

from __future__ import annotations

from types import SimpleNamespace

from mcpclient.adapter import (
    flatten_result,
    infer_kind,
    mcp_tool_to_tool,
    namespaced,
    normalize_schema,
    select_tools,
    split_namespaced,
)
from tools.base import ToolKind


def _tool(name="search", description="Search things", schema=None, annotations=None):
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema=schema
        if schema is not None
        else {"type": "object", "properties": {"q": {"type": "string"}}},
        annotations=annotations,
    )


def _result(content=(), structured=None, is_error=False):
    return SimpleNamespace(
        content=list(content), structured_content=structured, is_error=is_error
    )


def _text(text):
    return SimpleNamespace(type="text", text=text)


async def _ok(_name, _arguments):
    return _result([_text("done")])


# --------------------------------------------------------------------------
# Namespacing
# --------------------------------------------------------------------------


def test_a_server_tool_is_namespaced():
    tool = mcp_tool_to_tool(server="github", mcp_tool=_tool(), call=_ok)
    assert tool.name == "github__search"


def test_namespacing_round_trips():
    assert split_namespaced(namespaced("github", "search")) == ("github", "search")


def test_an_unnamespaced_name_is_recognised_as_such():
    assert split_namespaced("read_file") is None


def test_namespacing_prevents_collision_with_a_builtin():
    """A server shipping its own `read_file` would otherwise be dropped by the
    registry's keep-the-incumbent rule — silently, with a log nobody reads."""
    tool = mcp_tool_to_tool(server="github", mcp_tool=_tool(name="read_file"), call=_ok)
    assert tool.name != "read_file"


def test_the_description_says_where_the_tool_came_from():
    """The model reads this, and "which of these is GitHub's?" is a real
    question mid-task."""
    tool = mcp_tool_to_tool(server="github", mcp_tool=_tool(), call=_ok)
    assert "mcp:github" in tool.description
    assert "Search things" in tool.description


def test_a_tool_with_no_description_still_names_its_server():
    tool = mcp_tool_to_tool(
        server="github", mcp_tool=_tool(description=None), call=_ok
    )
    assert "github" in tool.description


# --------------------------------------------------------------------------
# Trust: a server does not get to classify itself
# --------------------------------------------------------------------------


def test_a_remote_tool_defaults_to_execute():
    assert infer_kind("search", None, frozenset()) is ToolKind.EXECUTE


def test_a_servers_read_only_hint_is_not_believed_on_its_own():
    """`readOnlyHint` comes from the server author. Trusting it would let any
    server declare itself harmless and skip the approval prompt — the exact
    trust boundary Phase 2 drew."""
    annotations = SimpleNamespace(read_only_hint=True)
    assert infer_kind("search", annotations, frozenset()) is ToolKind.EXECUTE


def test_local_config_can_mark_a_tool_read_only():
    assert infer_kind("search", None, frozenset({"search"})) is ToolKind.READ


def test_the_local_decision_uses_the_unnamespaced_name():
    """`readOnlyTools` in .mcp.json sits inside the server's own block, so it
    names tools the way that server does."""
    tool = mcp_tool_to_tool(
        server="github",
        mcp_tool=_tool(name="search"),
        call=_ok,
        read_only_tools=frozenset({"search"}),
    )
    assert tool.kind is ToolKind.READ


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


def test_a_real_schema_is_kept():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    assert normalize_schema(schema)["properties"] == {"q": {"type": "string"}}


def test_a_missing_schema_becomes_an_empty_object_schema():
    assert normalize_schema(None) == {"type": "object", "properties": {}}


def test_a_non_object_schema_is_replaced():
    """`jsonschema.validate` accepts anything against a non-object schema, so a
    tool with one could be called with nonsense."""
    assert normalize_schema({"type": "string"})["type"] == "object"


def test_an_object_schema_without_properties_gains_them():
    assert normalize_schema({"type": "object"})["properties"] == {}


def test_required_fields_survive_normalization():
    schema = {"type": "object", "properties": {"q": {}}, "required": ["q"]}
    assert normalize_schema(schema)["required"] == ["q"]


# --------------------------------------------------------------------------
# Flattening a result into an observation
# --------------------------------------------------------------------------


def test_text_content_is_returned():
    assert flatten_result(_result([_text("hello")])) == "hello"


def test_several_blocks_are_joined():
    assert flatten_result(_result([_text("one"), _text("two")])) == "one\ntwo"


def test_structured_content_wins_over_prose():
    """It is the machine-readable form the server chose to offer, and a model
    acts on JSON more reliably than on prose describing the same thing."""
    result = _result([_text("ignored")], structured={"count": 3})
    assert '"count": 3' in flatten_result(result)


def test_an_image_is_described_not_dumped():
    """Base64 in the context window is thousands of tokens the model cannot
    see anything in."""
    block = SimpleNamespace(type="image", mimeType="image/png", data="AAAA" * 500)
    flattened = flatten_result(_result([block]))
    assert "image/png" in flattened
    assert "AAAA" not in flattened


def test_an_embedded_resource_yields_its_text():
    block = SimpleNamespace(
        type="resource", resource=SimpleNamespace(text="file body", uri="file:///a")
    )
    assert flatten_result(_result([block])) == "file body"


def test_an_empty_result_says_so_rather_than_returning_nothing():
    """An empty observation reads to the model as a broken tool."""
    assert "no content" in flatten_result(_result([]))


def test_an_error_result_is_marked():
    assert flatten_result(_result([_text("bad id")], is_error=True)).startswith("ERROR")


def test_a_huge_result_is_truncated():
    """Tool output goes straight into the context window; a server returning a
    whole database must not eat the budget."""
    flattened = flatten_result(_result([_text("x" * 50_000)]))
    assert len(flattened) < 20_000
    assert "truncated" in flattened


# --------------------------------------------------------------------------
# Running: failures are data
# --------------------------------------------------------------------------


async def test_running_returns_the_flattened_result():
    tool = mcp_tool_to_tool(server="github", mcp_tool=_tool(), call=_ok)
    assert await tool.run({"q": "x"}) == "done"


async def test_the_arguments_reach_the_server():
    seen = {}

    async def call(name, arguments):
        seen["name"], seen["arguments"] = name, arguments
        return _result([_text("ok")])

    tool = mcp_tool_to_tool(server="github", mcp_tool=_tool(), call=call)
    await tool.run({"q": "forge"})
    # The *unnamespaced* name goes over the wire; the prefix is ours, not the
    # server's, and sending it would make every call fail.
    assert seen == {"name": "search", "arguments": {"q": "forge"}}


async def test_a_dead_server_becomes_an_observation_not_a_crash():
    """Invariant 5: a tool failure is data the model can react to."""

    async def call(_name, _arguments):
        raise ConnectionError("server went away")

    tool = mcp_tool_to_tool(server="github", mcp_tool=_tool(), call=call)
    result = await tool.run({"q": "x"})
    assert result.startswith("ERROR")
    assert "github" in result


async def test_a_timeout_is_also_an_observation():
    async def call(_name, _arguments):
        raise TimeoutError()

    tool = mcp_tool_to_tool(server="github", mcp_tool=_tool(), call=call)
    assert (await tool.run({})).startswith("ERROR")


# --------------------------------------------------------------------------
# Per-server tool allowlist
# --------------------------------------------------------------------------


def test_no_allowlist_takes_every_tool():
    tools = [_tool(name="a"), _tool(name="b")]
    assert len(select_tools(tools, None)) == 2


def test_an_allowlist_narrows_the_set():
    """Every exposed tool costs schema tokens in every single request."""
    tools = [_tool(name="a"), _tool(name="b"), _tool(name="c")]
    assert [t.name for t in select_tools(tools, ("a", "c"))] == ["a", "c"]


def test_an_allowlist_naming_a_missing_tool_is_not_an_error():
    """Servers change; a stale name in .mcp.json should not break startup."""
    assert select_tools([_tool(name="a")], ("a", "gone")) == [_tool(name="a")][:1]
