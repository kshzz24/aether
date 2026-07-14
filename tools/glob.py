import asyncio
from pathlib import Path

import pathspec

from tools.base import ToolKind
from tools.traversal import iter_files

KIND = ToolKind.READ

SCHEMA = {
    "name": "glob",
    "description": (
        "Find files by path glob (e.g. '**/*.py', 'src/**/*.ts'). Returns matching "
        "paths sorted, one per line. Respects .gitignore and skips .git."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "path glob, e.g. **/*.py"},
            "path": {"type": "string", "description": "root directory (default cwd)"},
        },
        "required": ["pattern"],
    },
}

LIMIT = 1000


def _glob(pattern: str, root: Path) -> str:
    # 1. Build PathSpec using gitwildmatch (handles **/*.py, etc.)
    spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])

    # 2. Iterate and filter. iter_files already skips .gitignore and .git.
    matches = []
    for f in iter_files(root):
        rel = f.relative_to(root).as_posix()  # match relative to root
        if spec.match_file(rel):
            matches.append(rel)

    # 3. Deterministic output is non-negotiable
    matches.sort()

    # 4. Apply cap
    if len(matches) > LIMIT:
        truncated = matches[:LIMIT]
        truncated.append(f"... truncated ({len(matches) - LIMIT} more files match)")
        return "\n".join(truncated)

    # 5. Return signal
    return "\n".join(matches) if matches else "no files match"


async def run(args: dict) -> str:
    root = Path(args.get("path") or ".").resolve()
    if not root.exists():
        raise ValueError(f"{root} does not exist")
    return await asyncio.to_thread(_glob, args["pattern"], root)
