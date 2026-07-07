import importlib.util
import logging
from pathlib import Path

from tools.base import Tool


def load_user_tools(user_tools_dir: Path) -> tuple[list[Tool], list[str]]:

    if not user_tools_dir.exists():
        return ([], [])

    res = []
    err = []
    for path in sorted(user_tools_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue

        try:
            path_name = f"forge_user_tool_{path.stem}"
            spec = importlib.util.spec_from_file_location(name=path_name, location=path)
            module = importlib.util.module_from_spec(spec=spec)
            spec.loader.exec_module(module)
            tool = Tool.from_module(module=module)
            res.append(tool)

        except Exception as e:
            logging.warning("failed to load user tool %s: %s", path.name, e)
            err.append(f"failed to load user tool from {path.name} due to {e}")

    return (res, err)
