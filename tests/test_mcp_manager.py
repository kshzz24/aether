"""Connecting to MCP servers, and surviving ones that will not connect.

The load-bearing property is **fail-load isolation**, the same one Phase 2 gave
user tools: an optional GitHub connector being down must not stop FORGE from
starting. An agent that refuses to run because a side feature is broken is worse
than one that runs without it and says so.

No real server is spawned. `build_transport` and `Client` are the seams, and
substituting them is what makes the failure paths — which are the interesting
ones — reachable at all.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mcpclient.config import ServerConfig
from mcpclient.manager import MCPManager
from tools.registry import ToolRegistry


def _config(name="github", transport="stdio", **over) -> ServerConfig:
    base = dict(
        name=name,
        scope="project",
        transport=transport,
        command="npx",
        args=("-y", "server"),
    )
    base.update(over)
    return ServerConfig(**base)


def _tool(name):
    return SimpleNamespace(
        name=name,
        description=f"{name} does things",
        input_schema={"type": "object", "properties": {}},
        annotations=None,
    )


class _FakeClient:
    """Stands in for `mcp.Client`: an async context manager with list/call."""

    def __init__(self, tools, *, on_call=None, fail_at_enter=None) -> None:
        self._tools = tools
        self._on_call = on_call
        self._fail_at_enter = fail_at_enter
        self.closed = False

    async def __aenter__(self):
        if self._fail_at_enter is not None:
            raise self._fail_at_enter
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments):
        if self._on_call is not None:
            return await self._on_call(name, arguments)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"{name} ran")],
            structured_content=None,
            is_error=False,
        )


@pytest.fixture
def patched(monkeypatch):
    """Swap the two seams; return a dict the test fills with fake clients."""
    clients: dict[str, _FakeClient] = {}
    monkeypatch.setattr("mcpclient.manager.build_transport", lambda config: config)
    monkeypatch.setattr(
        "mcpclient.manager.Client", lambda config: clients[config.name]
    )
    return clients


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


async def test_no_servers_means_no_tools():
    assert await MCPManager(configs={}).connect_all() == []


async def test_discovered_tools_are_adapted(patched):
    patched["github"] = _FakeClient([_tool("search"), _tool("create_issue")])
    manager = MCPManager(configs={"github": _config()})

    tools = await manager.connect_all()
    assert [t.name for t in tools] == ["github__search", "github__create_issue"]


async def test_a_connected_server_reports_its_tool_count(patched):
    patched["github"] = _FakeClient([_tool("search")])
    manager = MCPManager(configs={"github": _config()})
    await manager.connect_all()

    status = manager.statuses()[0]
    assert status.connected is True
    assert status.tool_count == 1


async def test_a_per_server_allowlist_is_applied(patched):
    patched["github"] = _FakeClient([_tool("search"), _tool("delete_repo")])
    manager = MCPManager(configs={"github": _config(tools=("search",))})

    tools = await manager.connect_all()
    assert [t.name for t in tools] == ["github__search"]


async def test_tools_land_in_the_ordinary_registry(patched):
    """The payoff for Phase 2: an MCP tool is schema-validated, allowlisted and
    approval-gated exactly like a builtin, with no special case anywhere."""
    patched["github"] = _FakeClient([_tool("search")])
    manager = MCPManager(configs={"github": _config()})
    await manager.connect_all()

    registry = ToolRegistry()
    assert manager.register_into(registry) == 1
    assert registry.get("github__search").name == "github__search"


async def test_a_registered_mcp_tool_validates_like_any_other(patched):
    patched["github"] = _FakeClient([_tool("search")])
    manager = MCPManager(configs={"github": _config()})
    await manager.connect_all()

    registry = ToolRegistry()
    manager.register_into(registry)
    registry.validate_call("github__search", {})  # must not raise


# --------------------------------------------------------------------------
# Fail-load isolation
# --------------------------------------------------------------------------


async def test_a_server_that_will_not_start_is_skipped_not_fatal(patched):
    patched["broken"] = _FakeClient([], fail_at_enter=FileNotFoundError("no npx"))
    manager = MCPManager(configs={"broken": _config(name="broken")})

    assert await manager.connect_all() == []
    assert manager.statuses()[0].connected is False


async def test_the_failure_reason_is_kept_for_the_user(patched):
    patched["broken"] = _FakeClient([], fail_at_enter=FileNotFoundError("no npx"))
    manager = MCPManager(configs={"broken": _config(name="broken")})
    await manager.connect_all()

    assert "no npx" in manager.statuses()[0].error


async def test_one_broken_server_does_not_stop_a_working_one(patched):
    """The whole point: an optional connector being down must not take the
    working ones with it."""
    patched["broken"] = _FakeClient([], fail_at_enter=ConnectionError("refused"))
    patched["github"] = _FakeClient([_tool("search")])
    manager = MCPManager(
        configs={"broken": _config(name="broken"), "github": _config()}
    )

    tools = await manager.connect_all()
    assert [t.name for t in tools] == ["github__search"]


async def test_a_server_that_hangs_is_abandoned(patched, monkeypatch):
    """A hung `npx` must not hold the prompt hostage forever."""
    monkeypatch.setattr("mcpclient.manager.CONNECT_TIMEOUT_SECONDS", 0.05)

    class _Hanging(_FakeClient):
        async def list_tools(self):
            await asyncio.sleep(10)

    patched["slow"] = _Hanging([])
    manager = MCPManager(configs={"slow": _config(name="slow")})

    assert await manager.connect_all() == []
    assert "did not respond" in manager.statuses()[0].error


# --------------------------------------------------------------------------
# Calling through
# --------------------------------------------------------------------------


async def test_calling_a_tool_reaches_the_server(patched):
    seen = {}

    async def on_call(name, arguments):
        seen["name"], seen["arguments"] = name, arguments
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            structured_content=None,
            is_error=False,
        )

    patched["github"] = _FakeClient([_tool("search")], on_call=on_call)
    manager = MCPManager(configs={"github": _config()})
    tools = await manager.connect_all()

    assert await tools[0].run({"q": "forge"}) == "ok"
    assert seen == {"name": "search", "arguments": {"q": "forge"}}


async def test_a_slow_call_times_out_as_an_observation(patched):
    async def on_call(_name, _arguments):
        await asyncio.sleep(10)

    patched["github"] = _FakeClient([_tool("search")], on_call=on_call)
    manager = MCPManager(configs={"github": _config(timeout_ms=50)})
    tools = await manager.connect_all()

    assert (await tools[0].run({})).startswith("ERROR")


async def test_calling_after_close_is_an_observation_not_a_crash(patched):
    patched["github"] = _FakeClient([_tool("search")])
    manager = MCPManager(configs={"github": _config()})
    tools = await manager.connect_all()
    await manager.aclose()

    assert "not connected" in await tools[0].run({})


# --------------------------------------------------------------------------
# Teardown
# --------------------------------------------------------------------------


async def test_closing_releases_the_connections(patched):
    """Without this every session leaves its stdio subprocesses running."""
    client = _FakeClient([_tool("search")])
    patched["github"] = client
    manager = MCPManager(configs={"github": _config()})
    await manager.connect_all()

    await manager.aclose()
    assert client.closed is True


async def test_closing_twice_is_harmless(patched):
    patched["github"] = _FakeClient([_tool("search")])
    manager = MCPManager(configs={"github": _config()})
    await manager.connect_all()
    await manager.aclose()
    await manager.aclose()


# --------------------------------------------------------------------------
# What /mcp shows
# --------------------------------------------------------------------------


def test_no_configuration_says_where_to_put_it():
    rendered = MCPManager(configs={}).render()
    assert ".mcp.json" in rendered


async def test_the_status_names_connected_servers_and_counts(patched):
    patched["github"] = _FakeClient([_tool("search")])
    manager = MCPManager(configs={"github": _config()})
    await manager.connect_all()

    rendered = manager.render()
    assert "github" in rendered
    assert "connected" in rendered
    assert "1 of 1" in rendered


async def test_the_status_surfaces_a_failure(patched):
    patched["broken"] = _FakeClient([], fail_at_enter=ConnectionError("refused"))
    manager = MCPManager(configs={"broken": _config(name="broken")})
    await manager.connect_all()

    rendered = manager.render()
    assert "failed" in rendered
    assert "refused" in rendered
