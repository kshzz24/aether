from __future__ import annotations

import numpy as np

from repomap.tags import Tag

# A symbol defined in more files than this is treated as a name collision rather
# than a dependency: `get`/`run`/`put` are defined all over a codebase, so a call
# to one carries no information about which file the caller actually depends on.
_MAX_DEFINING_FILES = 3


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def rank_files(
    tags: list[Tag],
    focus_files: list[str] | None = None,
    *,
    damping: float = 0.85,
    iterations: int = 50,
) -> dict[str, float]:
    """Personalized PageRank over the file dependency graph.

    Edge A->B (rank flows into B) when file A references a symbol defined in B,
    so widely-depended-on files score high. `focus_files` biases the teleport
    vector toward files of interest.
    """
    files = sorted({t.path for t in tags})

    if not files:
        return {}

    index = {f: i for i, f in enumerate(files)}
    n = len(files)
    definitions: dict[str, set[str]] = {}

    for t in tags:
        # A dunder is defined everywhere and depended on nowhere.
        if t.kind == "def" and not _is_dunder(t.name):
            definitions.setdefault(t.name, set()).add(t.path)

    # Drop ambiguous names before any edge is built, so the edge construction
    # below stays a plain ref -> definer mapping.
    definitions = {
        name: paths
        for name, paths in definitions.items()
        if len(paths) <= _MAX_DEFINING_FILES
    }

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
