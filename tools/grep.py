import asyncio
import re
from pathlib import Path

import pathspec

from tools.base import ToolKind
from tools.traversal import find_repo_root, iter_files

KIND = ToolKind.READ

SCHEMA = {
    "name": "grep",
    "description": (
        "Search file contents with a Python regular expression. Returns matching "
        "lines grouped by file as 'path:line: text'. Respects .gitignore, skips "
        ".git, binary files, and files over 5 MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"},
            "path": {
                "type": "string",
                "description": "file or directory (default cwd)",
            },
            "ignore_case": {"type": "boolean"},
            "include": {
                "type": "string",
                "description": "glob to scope files searched, e.g. *.py",
            },
        },
        "required": ["pattern"],
    },
}

FILE_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MB
PER_FILE_CAP = 50
TOTAL_CAP = 1000


def _is_binary(path: Path) -> bool:
    """Read the first 1 KB, look for a null byte (Git's heuristic)."""
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return True  # Safe default: skip if unreadable


def _search(regex: re.Pattern, root: Path, include: str | None) -> str:
    out: list[str] = []
    total_matches = 0

    # Step 1: Build candidate files
    if root.is_file():
        candidates = [root]
    else:
        candidates = list(iter_files(root=root))

    if include:
        include_spec = pathspec.PathSpec.from_lines("gitwildmatch", [include])
        repo_root = find_repo_root(root)
        base = repo_root or root

        filtered_candidates = []
        for f in candidates:
            rel = f.relative_to(base).as_posix()
            if include_spec.match_file(rel):
                filtered_candidates.append(f)
        candidates = filtered_candidates

    for file_path in candidates:
        if total_matches >= TOTAL_CAP:
            break

        try:
            if file_path.stat().st_size > FILE_SIZE_LIMIT:
                continue

            if _is_binary(file_path):
                continue

            rel_display = (
                file_path.relative_to(root).as_posix()
                if file_path != root
                else file_path.name
            )

            with open(file_path, encoding="utf-8", errors="replace") as f:
                text = f.read()

        except OSError:
            # Skip files with permission issues or that were deleted mid-run
            continue

        file_matches = 0
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                out.append(f"{rel_display}:{i}: {line}")
                file_matches += 1
                total_matches += 1

                # The two caps
                if file_matches >= PER_FILE_CAP:
                    out.append(
                        f"... truncated (maximum {PER_FILE_CAP} matches reached "
                        f"for {rel_display})"
                    )
                    break

                if total_matches >= TOTAL_CAP:
                    out.append(
                        f"... truncated (global limit of {TOTAL_CAP} matches reached)"
                    )
                    break

    # Clean, unambiguous signal
    return "\n".join(out) if out else "no matches"


async def run(args: dict) -> str:
    pattern = args["pattern"]
    ignore_case = args.get("ignore_case", False)
    include = args.get("include")

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"invalid regex {pattern!r}: {e}") from None

    root = Path(args.get("path") or ".").resolve()
    if not root.exists():
        raise ValueError(f"{root} does not exist")

    return await asyncio.to_thread(_search, regex, root, include)
