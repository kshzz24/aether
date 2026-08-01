"""MCP client: consume external tools without hand-writing API clients.

Phase 6. The public surface is deliberately small — load a config, connect, get
`Tool`s that are indistinguishable from builtins once they reach the registry.
"""

from .adapter import mcp_tool_to_tool, namespaced, split_namespaced
from .config import ServerConfig, load_mcp_config
from .errors import MCPConfigError
from .manager import MCPManager, ServerStatus

__all__ = [
    "MCPConfigError",
    "MCPManager",
    "ServerConfig",
    "ServerStatus",
    "load_mcp_config",
    "mcp_tool_to_tool",
    "namespaced",
    "split_namespaced",
]

# Where `.mcp.json` is looked for. Project scope overrides user scope, matching
# the config precedence the rest of FORGE uses.
PROJECT_CONFIG_NAME = ".mcp.json"
USER_CONFIG_RELATIVE = ".forge/.mcp.json"
