from config import ForgeConfig
from tools import edit_file, read_file, run_shell, write_file, grep, glob, list_dir
from tools.base import Tool
from tools.loader import load_user_tools
from tools.registry import ToolRegistry

_BUILTIN_MODULES = [read_file, write_file, run_shell, edit_file, grep, glob, list_dir]


def build_registry(config: ForgeConfig) -> ToolRegistry:
    """Assemble the tool registry: builtins first, then user tools.

    Builtins register first so a user tool that reuses a builtin name is the one
    the registry rejects (builtin wins, loudly). Broken user tools are already
    quarantined by the loader and never reach here.
    """
    registry = ToolRegistry(allowlist=config.allowlist)
    for module in _BUILTIN_MODULES:
        registry.register(Tool.from_module(module))

    user_tools, _load_errors = load_user_tools(config.user_tools_dir)
    for tool in user_tools:
        registry.register(tool)

    return registry
