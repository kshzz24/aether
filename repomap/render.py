from __future__ import annotations

from repomap.tags import Tag


def approx_tokens(text: str) -> int:
    """Cheap token estimate — no tokenizer dependency."""
    return len(text) // 4


def render_map(ranked_files: list[str], tags: list[Tag], max_tokens: int) -> str:
    """Emit ranked files with their definitions, stopping at the token budget.

    The top-ranked file is always included; later files are dropped once adding
    their block would exceed the budget (greedy-by-rank fill). Stopping rather
    than skipping keeps the map a stable prefix of the full ranking.
    """
    defs_by_file: dict[str, list[Tag]] = {}
    for t in tags:
        if t.kind == "def":
            defs_by_file.setdefault(t.path, []).append(t)

    lines: list[str] = []
    for path in ranked_files:
        defs = sorted(defs_by_file.get(path, []), key=lambda t: t.line)
        if not defs:
            continue
        block = [f"{path}:"] + [f"  {d.name} (L{d.line})" for d in defs]
        if lines and approx_tokens("\n".join(lines + block)) > max_tokens:
            break
        lines += block
    return "\n".join(lines)
