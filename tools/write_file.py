import difflib
from pathlib import Path

from tools.base import PreviewResult, ToolKind
from tools.fs import atomic_write

KIND = ToolKind.WRITE
SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, creating parent directories. Overwrites.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
}


async def run(args: dict) -> str:
    path_str = args["path"]
    content_str = args["content"]

    atomic_write(path_str, content_str)

    return f"Successfully wrote to {path_str}"


async def PREVIEW(args: dict) -> PreviewResult:
    path = args["path"]
    current = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        args["content"].splitlines(keepends=True),
        fromfile=path,
        tofile=path,
    )

    return PreviewResult(diff="".join(diff))
