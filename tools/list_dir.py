import asyncio
from pathlib import Path

from tools.base import ToolKind
from tools.traversal import find_repo_root, load_ignore_spec

KIND = ToolKind.READ

SCHEMA = {
    "name": "list_dir",
    "description": (
        "List the immediate contents of a directory (non-recursive). Directories "
        "are suffixed with '/'. Respects .gitignore and skips .git."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "directory to list (default cwd)",
            },
        },
        "required": [],
    },
}

LIMIT = 500


def _list(root: Path) -> str:
    repo_root = find_repo_root(root)
    spec = load_ignore_spec(repo_root)
    base = repo_root or root

    dirs: list[str] = []
    files: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.name == ".git":
            continue
        rel = entry.relative_to(base).as_posix()
        if entry.is_dir():
            if spec.match_file(rel) or spec.match_file(rel + "/"):
                continue
            dirs.append(entry.name + "/")
        else:
            if spec.match_file(rel):
                continue
            files.append(entry.name)

    # Directories first, then files, each already alphabetical from sorted().
    combined = dirs + files
    if not combined:
        return "(empty)"
    if len(combined) > LIMIT:
        out = combined[:LIMIT]
        out.append(f"... truncated ({len(combined) - LIMIT} more)")
        return "\n".join(out)
    return "\n".join(combined)


async def run(args: dict) -> str:
    root = Path(args.get("path") or ".").resolve()
    if not root.exists():
        raise ValueError(f"{root} does not exist")
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    return await asyncio.to_thread(_list, root)
