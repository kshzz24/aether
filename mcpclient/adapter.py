"""Turning an MCP tool into a FORGE `Tool`.

This is the anti-corruption layer for Phase 6, and it is the same idea as the
one in `client.py`: the MCP SDK's shapes stop here. Past this module a server's
tool is indistinguishable from a builtin — same `Tool` dataclass, same registry,
same schema validation, same approval policy. That is what "federation" means
in practice, and it is why the registry from Phase 2 did not have to change.

Four decisions live here, none of them obvious:

* **Namespacing.** A server tool is registered as `server__tool`. Without it,
  a server shipping a `read_file` would collide with the builtin and the
  registry would keep the incumbent and log — silently giving the model a tool
  that does something else than it thinks.
* **EXECUTE by default.** MCP's `readOnlyHint` is a *hint* from the server
  author, not a guarantee. Trusting it to downgrade a tool to READ would let a
  server opt itself out of the approval prompt, so the hint can only ever be
  corroborated by local config (`readOnlyTools`), never taken on its own.
* **Flattening.** The model needs a string. MCP returns typed content blocks,
  optional structured JSON, and an error flag.
* **Errors as data.** A dead server or a failed call comes back as an
  observation string, exactly like every other tool failure (Invariant 5).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from tools.base import Tool, ToolKind

logger = logging.getLogger(__name__)

# `server__tool`. A double underscore because single underscores are common
# inside tool names and would make the split ambiguous.
NAMESPACE_SEPARATOR = "__"

# Tool output goes straight into the context window. A server returning a whole
# file, or a 5,000-row query, must not silently eat the budget.
MAX_RESULT_CHARS = 8000


def namespaced(server: str, tool: str) -> str:
    return f"{server}{NAMESPACE_SEPARATOR}{tool}"


def split_namespaced(name: str) -> tuple[str, str] | None:
    """`server__tool` -> (server, tool), or None if it is not namespaced."""
    server, separator, tool = name.partition(NAMESPACE_SEPARATOR)
    if not separator or not server or not tool:
        return None
    return server, tool


def infer_kind(tool_name: str, annotations: Any, read_only_tools) -> ToolKind:
    """Classify a remote tool for the approval policy.

    EXECUTE unless the *local* config says otherwise. A server's own
    `readOnlyHint` is advisory — believing it would let any server declare
    itself harmless and skip the confirmation prompt, which is precisely the
    trust boundary Phase 2 drew. The hint is still consulted, but only to
    *agree* with a local decision, never to make one.
    """
    if tool_name in read_only_tools:
        return ToolKind.READ
    hint = getattr(annotations, "read_only_hint", None)
    if hint:
        logger.debug(
            "%r advertises readOnlyHint but is not in readOnlyTools; "
            "treating as EXECUTE",
            tool_name,
        )
    return ToolKind.EXECUTE


def normalize_schema(schema: dict | None) -> dict:
    """A JSON Schema the registry can validate against.

    Servers omit `input_schema`, or send a bare `{}`, or a non-object schema.
    `jsonschema.validate` would accept anything against those, so the shape is
    forced to an object here — a tool whose schema does not constrain arguments
    is a tool the model can call with nonsense.
    """
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    normalized.setdefault("properties", {})
    return normalized


def _block_to_text(block: Any) -> str:
    """One MCP content block as text."""
    kind = getattr(block, "type", None)
    if kind == "text":
        return str(getattr(block, "text", ""))
    if kind == "image":
        # The model cannot see it through a string channel; say so rather than
        # dumping base64 into the context.
        return f"[image: {getattr(block, 'mimeType', 'unknown type')}]"
    if kind == "resource":
        resource = getattr(block, "resource", None)
        text = getattr(resource, "text", None)
        if text is not None:
            return str(text)
        return f"[resource: {getattr(resource, 'uri', 'unknown')}]"
    return str(block)


def flatten_result(result: Any) -> str:
    """A `CallToolResult` as the observation string the model reads.

    Structured content wins when present: it is the machine-readable form the
    server chose to offer, and JSON is easier for a model to act on than prose
    describing the same thing.
    """
    structured = getattr(result, "structured_content", None)
    if structured:
        try:
            body = json.dumps(structured, indent=2, default=str)
        except (TypeError, ValueError):
            body = str(structured)
    else:
        blocks = getattr(result, "content", None) or []
        body = "\n".join(_block_to_text(block) for block in blocks).strip()

    if not body:
        body = "(the server returned no content)"

    if len(body) > MAX_RESULT_CHARS:
        dropped = len(body) - MAX_RESULT_CHARS
        body = body[:MAX_RESULT_CHARS] + f"\n... [truncated {dropped} chars]"

    # `is_error` is the protocol's way of saying "this failed but is not a
    # transport problem" — a tool failure, which Invariant 5 says is data.
    if getattr(result, "is_error", False):
        return f"ERROR: {body}"
    return body


def mcp_tool_to_tool(
    *,
    server: str,
    mcp_tool: Any,
    call: Callable[[str, dict], Awaitable[Any]],
    read_only_tools: frozenset[str] = frozenset(),
) -> Tool:
    """Adapt one discovered MCP tool into a registrable FORGE `Tool`.

    `call` is injected rather than a client being passed in, so this function
    stays testable without a running server — and so the manager keeps sole
    ownership of connection lifetime.
    """
    remote_name = mcp_tool.name

    async def run(arguments: dict) -> str:
        try:
            result = await call(remote_name, arguments)
        except Exception as exc:  # noqa: BLE001
            # A dead server, a timeout, a protocol error: all of it is an
            # observation the model can react to, never a crashed run.
            logger.warning("MCP call %s/%s failed: %s", server, remote_name, exc)
            return f"ERROR: MCP server {server!r} failed: {exc}"
        return flatten_result(result)

    description = (mcp_tool.description or "").strip()
    # The origin is in the description because the model reads it, and "which
    # of these came from GitHub?" is a real question mid-task.
    labelled = f"[mcp:{server}] {description}".strip()
    return Tool(
        name=namespaced(server, remote_name),
        description=labelled,
        parameters=normalize_schema(getattr(mcp_tool, "input_schema", None)),
        kind=infer_kind(
            remote_name, getattr(mcp_tool, "annotations", None), read_only_tools
        ),
        run=run,
    )


def select_tools(tools: list, allowed: tuple[str, ...] | None) -> list:
    """Apply a server's `tools` allowlist from `.mcp.json`.

    A server can expose dozens of tools; every one costs schema tokens in every
    request. `None` means "take them all", which is the default.
    """
    if allowed is None:
        return list(tools)
    wanted = set(allowed)
    return [tool for tool in tools if tool.name in wanted]
