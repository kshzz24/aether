import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import MCPConfigError

_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")
DEFAULT_TIMEOUT_MS = 30000


@dataclass(frozen=True)
class ServerConfig:
    name: str
    scope: str
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    tools: tuple[str, ...] | None = None
    read_only_tools: frozenset[str] = frozenset()


def read_from_path(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MCPConfigError(f"Failed to parse {path}: {e}") from e

    return data.get("mcpServers", {})


def tag_scope(blocks: dict, scope: str) -> dict[str, tuple[dict, str]]:
    return {name: (block, scope) for name, block in blocks.items()}


def merge_configs(user: dict, project: dict):

    tagged_user = tag_scope(user, "user")
    tagged_project = tag_scope(project, "project")

    merged = {}

    merged.update(tagged_user)
    merged.update(tagged_project)

    return merged


def expand_string(value: str) -> str:

    def replace(match):

        name = match.group(1)

        default = match.group(3)

        env_value = os.environ.get(name)

        if env_value is not None:
            return env_value

        if match.group(2) is not None:
            return default

        raise MCPConfigError(f"Environment variable {name} is not set")

    return _PATTERN.sub(replace, value)


def expand_block(raw: dict) -> dict:

    shallow_raw = raw.copy()

    if "command" in shallow_raw:
        shallow_raw["command"] = expand_string(shallow_raw["command"])

    if "args" in shallow_raw:
        shallow_raw["args"] = [expand_string(ele) for ele in shallow_raw["args"]]

    if "env" in shallow_raw:
        shallow_raw["env"] = {
            k: expand_string(v) for k, v in shallow_raw["env"].items()
        }

    if "url" in shallow_raw:
        shallow_raw["url"] = expand_string(shallow_raw["url"])

    if "headers" in shallow_raw:
        shallow_raw["headers"] = {
            k: expand_string(v) for k, v in shallow_raw["headers"].items()
        }

    return shallow_raw


def build_server(name: str, raw: dict, scope: str) -> ServerConfig:

    raw = expand_block(raw)
    transport_type = raw.get("type")
    if transport_type is None:
        if "url" in raw:
            raise MCPConfigError(
                f'MCP server "{name}" has a "url" but no "type"; add "type": "http"'
            )
        transport = "stdio"
    elif transport_type == "stdio":
        transport = "stdio"
    elif transport_type in ("http", "streamable-http"):
        transport = "http"
    else:
        raise MCPConfigError(f'MCP server "{name}" has unknown type {transport_type!r}')

    command = raw.get("command")
    url = raw.get("url")
    if transport == "stdio" and not command:
        raise MCPConfigError(f'MCP server "{name}" is stdio but has no "command"')
    if transport == "http" and not url:
        raise MCPConfigError(f'MCP server "{name}" is http but has no "url"')

    # ④ everything else, with defaults
    tools = raw.get("tools")
    return ServerConfig(  # ⑤
        name=name,
        scope=scope,
        transport=transport,
        command=command,
        args=tuple(raw.get("args", ())),
        env=dict(raw.get("env", {})),
        url=url,
        headers=dict(raw.get("headers", {})),
        timeout_ms=raw.get("timeout", DEFAULT_TIMEOUT_MS),
        tools=None if tools is None else tuple(tools),
        read_only_tools=frozenset(raw.get("readOnlyTools", ())),
    )


def load_mcp_config(*, project_path: Path, user_path: Path) -> dict[str, ServerConfig]:
    user_raw = read_from_path(user_path)
    project_raw = read_from_path(project_path)

    merged = merge_configs(user_raw, project_raw)

    return {
        name: build_server(name, raw, scope) for name, (raw, scope) in merged.items()
    }
