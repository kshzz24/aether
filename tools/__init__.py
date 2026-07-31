from __future__ import annotations

from typing import TYPE_CHECKING

from tools import (
    edit_file,
    glob,
    grep,
    list_dir,
    read_file,
    repo_map,
    run_shell,
    write_file,
)
from tools.base import Tool
from tools.loader import load_user_tools
from tools.registry import ToolRegistry
from tools.todo import TodoStore, build_todo_tool

if TYPE_CHECKING:
    from config import ForgeConfig

_BUILTIN_MODULES = [
    read_file,
    write_file,
    run_shell,
    edit_file,
    grep,
    glob,
    list_dir,
    repo_map,
]


def build_registry(
    config: ForgeConfig, *, todo_store: TodoStore | None = None
) -> ToolRegistry:
    """Assemble the tool registry: builtins first, then user tools.

    Builtins register first so a user tool that reuses a builtin name is the one
    the registry rejects (builtin wins, loudly). Broken user tools are already
    quarantined by the loader and never reach here.

    `todo_store` is injectable so the caller can hold the same store the tool
    writes to and observe it (the TUI's task panel). Left unset, each registry
    gets its own — runs never share a task list.
    """
    registry = ToolRegistry(allowlist=config.allowlist)
    for module in _BUILTIN_MODULES:
        registry.register(Tool.from_module(module))
    registry.register(build_todo_tool(store=todo_store or TodoStore()))

    user_tools, _load_errors = load_user_tools(config.user_tools_dir)
    for tool in user_tools:
        registry.register(tool)

    return registry
