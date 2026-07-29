from __future__ import annotations

from pathlib import Path

from repomap.cache import TagCache
from repomap.rank import rank_files
from repomap.render import render_map
from repomap.tags import Tag, extract_tags, language_for
from traversal import iter_files

__all__ = ["build_repo_map"]


def build_repo_map(
    root: Path,
    focus_files: list[str] | None = None,
    max_tokens: int = 1024,
    cache: TagCache | None = None,
) -> str:
    """Extract (cached) -> rank -> render a token-budgeted map of `root`."""
    root = root.resolve()
    all_tags: list[Tag] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        if language_for(rel) is None:
            continue
        mtime = path.stat().st_mtime
        tags = cache.get(rel, mtime) if cache else None
        if tags is None:
            try:
                tags = extract_tags(rel, path.read_bytes())
            except Exception:  # Invariant 5: a bad file contributes no tags
                tags = []
            if cache:
                cache.put(rel, mtime, tags)
        all_tags.extend(tags)

    scores = rank_files(all_tags, focus_files)
    ranked = sorted(scores, key=lambda f: scores[f], reverse=True)
    return render_map(ranked, all_tags, max_tokens)
