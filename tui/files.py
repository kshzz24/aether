"""`@file` completion: turn `@con` into `@config.py` while typing.

The file index is built once at startup from `traversal.iter_files`, the same
gitignore-aware walk the repo-map uses — so `@` never offers a path from
`node_modules/` or `.venv/` that the agent would then fail to read.

Rebuilt only on demand (`/reindex`). A filesystem watcher is the "correct"
answer and the wrong trade for a process that lives for one task: it costs a
dependency and a thread to catch files created during a session, which the
agent can already read by typing the path out.
"""

from __future__ import annotations

from pathlib import Path

from traversal import iter_files

# Enough to cover any repo a CLI agent works in, bounded so a stray checkout of
# something enormous cannot stall startup.
_MAX_INDEXED = 20_000
_MAX_SUGGESTIONS = 50


def build_file_index(root: Path, *, limit: int = _MAX_INDEXED) -> list[str]:
    """Repo-relative POSIX paths, shortest first.

    Shortest-first means `@config` offers `config.py` before
    `gateway/config.py` — top-level files are what people usually mean.
    """
    paths: list[str] = []
    for path in iter_files(root):
        try:
            paths.append(path.relative_to(root).as_posix())
        except ValueError:
            continue
        if len(paths) >= limit:
            break
    paths.sort(key=lambda p: (len(p), p))
    return paths


def split_mention(text: str) -> tuple[str, str] | None:
    """Split `text` into (prefix, partial) at a trailing `@mention`.

    Returns None when the caret is not in a mention, so callers can tell "no
    completion applies" from "completes to nothing".
    """
    at = text.rfind("@")
    if at == -1:
        return None
    partial = text[at + 1 :]
    # A mention ends at whitespace: "@a.py and @b" is completing "b", and
    # "read @a.py now" is not completing at all.
    if any(ch.isspace() for ch in partial):
        return None
    return text[:at], partial


def complete_mention(text: str, index: list[str]) -> str | None:
    """Full replacement line for the best `@` match, or None."""
    matches = match_paths(text, index)
    if not matches:
        return None
    split = split_mention(text)
    assert split is not None  # match_paths returns [] when there is no mention
    prefix, _ = split
    return f"{prefix}@{matches[0]}"


def match_paths(text: str, index: list[str]) -> list[str]:
    """Ranked matches for the trailing `@mention` in `text`."""
    split = split_mention(text)
    if split is None:
        return []
    _, partial = split
    if not partial:
        return index[:_MAX_SUGGESTIONS]

    needle = partial.lower()
    starts: list[str] = []
    basename: list[str] = []
    contains: list[str] = []
    for path in index:
        lowered = path.lower()
        if lowered.startswith(needle):
            starts.append(path)
        elif lowered.rsplit("/", 1)[-1].startswith(needle):
            basename.append(path)
        elif needle in lowered:
            contains.append(path)

    # Prefix beats basename beats substring: typing "@gate" should reach
    # "gateway/..." before "tests/test_gateway_cache.py".
    return (starts + basename + contains)[:_MAX_SUGGESTIONS]
