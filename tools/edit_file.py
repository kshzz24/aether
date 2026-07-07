import difflib
from pathlib import Path

from tools.base import ToolKind
from tools.fs import atomic_write

KIND = ToolKind.WRITE
SCHEMA = {
    "name": "edit_file",
    "description": "Replace an exact, unique string in a file. Returns a unified diff.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["path", "old_string", "new_string"],
    },
}


async def run(args: dict) -> str:
    path = args["path"]
    old_text = args["old_string"]
    new_text = args["new_string"]

    text = Path(path).read_text(encoding="utf-8")

    count = text.count(old_text)

    if count == 0:
        raise ValueError("old_string not found")

    if count > 1:
        raise ValueError(f"old_string found {count} times, must be unique")

    content = text.replace(old_text, new_text)

    atomic_write(path=path, content=content)

    return "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )
