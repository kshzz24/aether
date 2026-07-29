import asyncio
from pathlib import Path

from repomap import build_repo_map
from repomap.cache import TagCache
from tools.base import ToolKind
from traversal import find_repo_root

KIND = ToolKind.READ

SCHEMA = {
    "name": "repo_map",
    "description": (
        "Return a ranked, token-budgeted map of the repository's most important "
        "symbols (classes/functions) — the shape of the codebase without reading "
        "every file. Files are ranked by how widely their definitions are "
        "referenced (PageRank). Pass focus_files to bias the map toward files you "
        "care about."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "focus_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "repo-relative paths to bias the ranking toward",
            },
            "max_tokens": {
                "type": "integer",
                "description": "approximate token budget for the map (default 1024)",
            },
        },
        "required": [],
    },
}


def _build(root: Path, focus_files: list[str] | None, max_tokens: int) -> str:
    # The cache is opened inside the worker thread: a sqlite connection belongs to
    # the thread that created it.
    cache = TagCache(root / ".forge" / "cache" / "repo_map.sqlite")
    return build_repo_map(
        root,
        focus_files=focus_files,
        max_tokens=max_tokens,
        cache=cache,
    )


async def run(args: dict) -> str:
    root = find_repo_root(Path.cwd()) or Path.cwd()
    result = await asyncio.to_thread(
        _build, root, args.get("focus_files"), args.get("max_tokens", 1024)
    )
    return result or "(no indexable symbols found)"
