from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pathspec

_ALWAYS_SKIP = {".git"}


def find_repo_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` containing a ``.git`` entry, else None."""
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def load_ignore_spec(repo_root: Path | None) -> pathspec.PathSpec:
    """Build a PathSpec from ``<repo_root>/.gitignore`` (empty spec if absent)."""
    if repo_root is None:
        return pathspec.PathSpec.from_lines("gitwildmatch", [])
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return pathspec.PathSpec.from_lines("gitwildmatch", [])
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def iter_files(root: Path) -> Iterator[Path]:
    """Yield files under ``root``, skipping ``.git/`` and gitignored paths.

    Ignored directories are pruned during the walk so their subtrees are never
    descended into. Paths are matched relative to the repo root (git semantics),
    falling back to ``root`` when ``root`` is not inside a git repo.
    """
    root = root.resolve()
    repo_root = find_repo_root(root)
    spec = load_ignore_spec(repo_root)
    base = repo_root or root

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)
        kept_dirs = []
        for d in dirnames:
            if d in _ALWAYS_SKIP:
                continue
            rel = (current / d).relative_to(base).as_posix()
            if spec.match_file(rel) or spec.match_file(rel + "/"):
                continue
            kept_dirs.append(d)
        dirnames[:] = kept_dirs  # in-place prune: os.walk won't descend removed dirs

        for f in filenames:
            rel = (current / f).relative_to(base).as_posix()
            if spec.match_file(rel):
                continue
            yield current / f
