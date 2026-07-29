from __future__ import annotations

import numpy as np

from repomap.tags import Tag


def rank_files(
    tags: list[Tag],
    focus_files: list[str] | None = None,
    *,
    damping: float = 0.85,
    iterations: int = 50,
) -> dict[str, float]:
    files = sorted({t.path for t in tags})

    if not files:
        return {}

    index = {f: i for i, f in enumerate(files)}
    n = len(files)
    definitions: dict[str, set[str]] = {}

    for t in tags:
        if t.kind == "def":
            definitions.setdefault(t.name, set()).add(t.path)

    matrix = np.zeros((n, n))

    for t in tags:
        if t.kind == "ref":
            for target in definitions.get(t.name, ()):
                if target != t.path:
                    matrix[index[target], index[t.path]] += 1.0

    col_sums = matrix.sum(axis=0)
    nonzero = col_sums > 0
    matrix[:, nonzero] /= col_sums[nonzero]

    teleport = np.zeros(n)
    focus = [f for f in (focus_files or []) if f in index]
    if focus:
        for f in focus:
            teleport[index[f]] = 1.0 / len(focus)
    else:
        teleport[:] = 1.0 / n

    dangling = ~nonzero  # sources with no out-links redistribute via teleport
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        dangling_mass = rank[dangling].sum()
        rank = (1 - damping) * teleport + damping * (
            matrix @ rank + dangling_mass * teleport
        )
    rank /= rank.sum()
    return {f: float(rank[index[f]]) for f in files}
