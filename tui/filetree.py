"""Turning a flat list of paths into a tree, without importing Textual.

Two consumers need this: the `ctrl+b` sidebar mounts it as a `Tree` widget, and
`/files` (plus the summary posted when a run finishes) renders it as text. The
sidebar can import widgets; `tui/commands.py` deliberately cannot, and that
constraint is what puts these functions in their own module rather than
alongside the widget that first needed them.
"""

from __future__ import annotations

from pathlib import Path

# Deep enough to be useful, shallow enough that the pane never becomes the thing
# you scroll instead of the transcript.
MAX_TREE_ENTRIES = 200


def relative(path: Path, root: Path) -> str:
    """`path` under `root`, posix-style. Falls back to the absolute path.

    A tool can legitimately write outside the repo — a scratch file in /tmp, a
    config in the home directory. Rendering that as `../../../tmp/x` would be
    less honest than just showing where it is.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def group_paths(paths: list[Path], root: Path) -> dict:
    """Nest a flat path list into `{dir: {...}, file: None}`."""
    tree: dict = {}
    for path in paths[:MAX_TREE_ENTRIES]:
        parts = [part for part in relative(path, root).split("/") if part]
        if not parts:
            continue
        node = tree
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            # A name used as both a file and a directory: the directory wins,
            # since it is the one with children to show.
            if child is None:
                child = node[part] = {}
            node = child
        node.setdefault(parts[-1], None)
    return tree


def sorted_items(mapping: dict) -> list[tuple[str, dict | None]]:
    """Directories first, then files, each alphabetically — like `tree`."""
    return sorted(mapping.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))


def tree_lines(paths: list[Path], root: Path) -> list[str]:
    """Box-drawing render of `paths`, for the transcript and `/files`."""
    if not paths:
        return []
    lines: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        items = sorted_items(node)
        for index, (name, child) in enumerate(items):
            last = index == len(items) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}")
            if child:
                walk(child, prefix + ("    " if last else "│   "))

    walk(group_paths(paths, root), "")
    if len(paths) > MAX_TREE_ENTRIES:
        lines.append(f"... and {len(paths) - MAX_TREE_ENTRIES} more")
    return lines
