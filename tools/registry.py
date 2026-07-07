from __future__ import annotations

import logging

import jsonschema

from tools.base import Tool


class ToolRegistry:
    _allowlist: set[str]
    _tools: dict[str, Tool]

    def __init__(self, allowlist: set[str] | None = None) -> None:
        self._allowlist = allowlist
        self._tools = {}

    def register(self, tool: Tool) -> None:

        if tool.name in self._tools:
            logging.warning(
                "tool %r already registered; keeping incumbent, ignoring duplicate",
                tool.name,
            )
            return None

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def wire_schemas(self) -> list[dict]:
        return [
            t.to_wire()
            for t in self._tools.values()
            if self._allowlist is None or t.name in self._allowlist
        ]

    def validate_call(self, name: str, args: dict) -> None:
        if self._allowlist is not None and name not in self._allowlist:
            raise ValueError(f"tool {name!r} is not allowed")
        tool = self.get(name)
        jsonschema.validate(instance=args, schema=tool.parameters)
