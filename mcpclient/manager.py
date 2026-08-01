"""Connecting to MCP servers and federating their tools into the registry.

The manager owns connection *lifetime*. Every server stays connected for the
whole session, because a stdio server is a subprocess and reconnecting per call
would pay process startup — often a `npx` download — on every tool use.

Two properties do most of the work here:

**Fail-load isolation.** One server that will not start, times out, or speaks a
broken protocol must not stop FORGE. It is reported and skipped, exactly as
Phase 2 does for a broken user tool. An agent that refuses to start because an
optional GitHub connector is down is worse than one that starts without it.

**Lifetime via a task.** The SDK's client is an async context manager, and
several must stay open at once for an unknown duration. `AsyncExitStack` holds
them, and `aclose()` unwinds them in reverse — so a session teardown does not
leak subprocesses.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import Client

from tools.base import Tool
from tools.registry import ToolRegistry

from .adapter import mcp_tool_to_tool, select_tools
from .config import ServerConfig
from .transport import build_transport

logger = logging.getLogger(__name__)

# A server that cannot connect in this long is not going to. Startup must stay
# interactive; a hung `npx` should not hold the prompt.
CONNECT_TIMEOUT_SECONDS = 30.0


@dataclass
class ServerStatus:
    """What `/mcp` shows for one configured server."""

    name: str
    scope: str
    transport: str
    connected: bool = False
    tool_count: int = 0
    error: str | None = None

    def line(self) -> str:
        if self.connected:
            plural = "" if self.tool_count == 1 else "s"
            return (
                f"  {self.name:<18} {self.transport:<6} connected  "
                f"{self.tool_count} tool{plural}  ({self.scope})"
            )
        return f"  {self.name:<18} {self.transport:<6} failed     {self.error}"


@dataclass
class MCPManager:
    """Owns the connections and the tools discovered over them."""

    configs: dict[str, ServerConfig]
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    _clients: dict[str, Client] = field(default_factory=dict)
    _status: dict[str, ServerStatus] = field(default_factory=dict)
    _tools: list[Tool] = field(default_factory=list)

    # ---------------------------------------------------------------- connect

    async def connect_all(self) -> list[Tool]:
        """Connect every configured server; return the tools that came back.

        Servers are connected concurrently: they are independent, and a
        sequential loop would make startup the sum of every server's spawn
        time rather than the slowest one's.
        """
        if not self.configs:
            return []
        await asyncio.gather(
            *(self._connect_one(config) for config in self.configs.values())
        )
        return list(self._tools)

    async def _connect_one(self, config: ServerConfig) -> None:
        status = ServerStatus(
            name=config.name, scope=config.scope, transport=config.transport
        )
        self._status[config.name] = status
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                client = await self._stack.enter_async_context(
                    Client(build_transport(config))
                )
                discovered = await client.list_tools()
        except TimeoutError:
            status.error = f"did not respond within {CONNECT_TIMEOUT_SECONDS:.0f}s"
            logger.warning("MCP server %r timed out", config.name)
            return
        except Exception as exc:  # noqa: BLE001
            # A missing binary, a refused connection, a protocol mismatch. None
            # of it is a reason for FORGE not to start.
            status.error = f"{type(exc).__name__}: {exc}"
            logger.warning("MCP server %r failed to connect: %s", config.name, exc)
            return

        self._clients[config.name] = client
        wanted = select_tools(list(discovered.tools), config.tools)
        for mcp_tool in wanted:
            self._tools.append(
                mcp_tool_to_tool(
                    server=config.name,
                    mcp_tool=mcp_tool,
                    call=self._caller(config),
                    read_only_tools=config.read_only_tools,
                )
            )
        status.connected = True
        status.tool_count = len(wanted)
        logger.info("MCP server %r: %d tools", config.name, len(wanted))

    def _caller(self, config: ServerConfig):
        """A bound `call(name, arguments)` for one server's adapted tools.

        Late-bound through `self._clients` rather than closing over the client
        object, so the timeout comes from config and the lookup fails loudly if
        the connection was dropped.
        """
        timeout = config.timeout_ms / 1000

        async def call(name: str, arguments: dict):
            client = self._clients.get(config.name)
            if client is None:
                raise RuntimeError(f"server {config.name!r} is not connected")
            async with asyncio.timeout(timeout):
                return await client.call_tool(name, arguments)

        return call

    # ------------------------------------------------------------- inspection

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    def statuses(self) -> list[ServerStatus]:
        return list(self._status.values())

    def render(self) -> str:
        """The body of `/mcp`."""
        if not self.configs:
            return (
                "no MCP servers configured\n"
                "  add them to .mcp.json in this repo, or ~/.forge/.mcp.json"
            )
        lines = [status.line() for status in self.statuses()]
        connected = sum(1 for s in self.statuses() if s.connected)
        header = f"{connected} of {len(self.configs)} servers connected"
        return header + "\n" + "\n".join(lines)

    def register_into(self, registry: ToolRegistry) -> int:
        """Add the discovered tools to the same registry as everything else.

        This one line is the payoff for Phase 2's registry: an MCP tool is
        schema-validated, allowlist-filtered and approval-gated identically to
        a builtin, with no special case anywhere.
        """
        for tool in self._tools:
            registry.register(tool)
        return len(self._tools)

    # ---------------------------------------------------------------- cleanup

    async def aclose(self) -> None:
        """Unwind every connection, closing subprocesses with them."""
        self._clients.clear()
        try:
            await self._stack.aclose()
        except Exception as exc:  # noqa: BLE001
            # Teardown noise (a server already dead, a broken pipe) must not
            # mask whatever the session was actually doing.
            logger.debug("error while closing MCP connections: %s", exc)
        self._stack = AsyncExitStack()
