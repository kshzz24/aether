"""Building an MCP transport from a `ServerConfig`.

The whole point of the transport abstraction is that nothing above this module
knows whether a server is a subprocess or an HTTP endpoint. `mcpclient/manager.py`
opens whatever this returns and talks the same protocol either way — which is
the Phase 6 lesson: the transport is a swappable detail, not a fork in the
calling code.

Auth enters here, and only here. `EnvAuth` reaches a stdio server as process
environment; `HeaderAuth` reaches an HTTP server as request headers. Neither
concept exists in the other transport, which is exactly why `AuthProvider`
exposes both and each transport takes only the half that applies.
"""

from __future__ import annotations

import os

from mcp import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from .auth import AuthProvider, EnvAuth, HeaderAuth
from .config import ServerConfig
from .errors import MCPConfigError


def auth_for(config: ServerConfig) -> AuthProvider:
    """The auth provider a server's config implies.

    Config carries `env` for stdio and `headers` for http; `${VAR}` expansion
    already happened in `config.expand_block`, so by here these are literal
    secrets and must not be logged.
    """
    if config.transport == "stdio":
        return EnvAuth(config.env)
    return HeaderAuth(config.headers)


def _stdio_environment(config: ServerConfig) -> dict[str, str]:
    """The child's environment: ours, plus the server's own entries.

    Inheriting matters more than it looks — a server launched by `npx` needs
    PATH, HOME and the npm cache location to start at all. Starting from an
    empty environment is the classic "works in my shell, hangs under the
    agent" failure.
    """
    return {**os.environ, **config.env}


def build_transport(config: ServerConfig):
    """An async context manager yielding this server's read/write streams.

    Returned unopened: connection lifetime belongs to the manager, which has to
    hold several of these open at once for the length of a session.
    """
    auth = auth_for(config)

    if config.transport == "stdio":
        if not config.command:
            raise MCPConfigError(f'MCP server "{config.name}" has no command')
        return stdio_client(
            StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=_stdio_environment(config),
            )
        )

    if config.transport == "http":
        if not config.url:
            raise MCPConfigError(f'MCP server "{config.name}" has no url')
        headers = auth.headers()
        if headers:
            # httpx is the SDK's own dependency; importing it lazily keeps the
            # stdio path free of it.
            import httpx2

            return streamable_http_client(
                config.url, http_client=httpx2.AsyncClient(headers=headers)
            )
        return streamable_http_client(config.url)

    raise MCPConfigError(
        f'MCP server "{config.name}" has unsupported transport '
        f"{config.transport!r}"
    )
