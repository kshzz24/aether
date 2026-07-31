"""Subsequence fuzzy matching, shared by every picker in the TUI.

One scorer, three consumers: `ctrl+r` history search, the `/prompt` template
picker, and (eventually) a file picker. Written once here rather than three
times inside three screens, so the ranking is consistent everywhere and testable
without a running app.

The model is the one every fuzzy finder uses: a candidate matches if the needle
is a *subsequence* of it, and the score rewards matches that a human would call
obvious — consecutive characters, and characters that start a word. `cfg` should
rank `config.toml` above `.claude/forge/g.py`, even though both technically
contain c-f-g in order.
"""

from __future__ import annotations

from collections.abc import Iterable

# Characters after which the next character counts as starting a word. Path
# separators are in here because these are usually filenames.
_BOUNDARIES = " \t_-./\\:@"

_MATCH = 10        # every matched character is worth something
_CONTIGUOUS = 12   # ...but a run of them is worth much more
_BOUNDARY = 9      # matching the start of a word reads as intentional
_LEADING_GAP = 2   # small penalty per character skipped before the first match


def score(needle: str, haystack: str) -> int | None:
    """Rank `haystack` against `needle`, or None when it doesn't match.

    None rather than 0 because "no match" and "a bad match" must be
    distinguishable: the caller filters on the first and sorts on the second.
    """
    if not needle:
        return 0

    lowered_needle = needle.lower()
    lowered_hay = haystack.lower()

    total = 0
    hay_index = 0
    previous_match = -1

    for char in lowered_needle:
        found = lowered_hay.find(char, hay_index)
        if found == -1:
            return None

        total += _MATCH
        if found == previous_match + 1:
            total += _CONTIGUOUS
        elif found == 0 or lowered_hay[found - 1] in _BOUNDARIES:
            total += _BOUNDARY

        if previous_match == -1:
            total -= min(found, 20) * _LEADING_GAP

        previous_match = found
        hay_index = found + 1

    # A needle that covers most of a short candidate beats the same needle
    # scattered through a long one.
    return total + max(0, 20 - len(haystack) // 4)


def rank(needle: str, items: Iterable[str]) -> list[str]:
    """Matching items, best first. Ties keep their original order."""
    scored = []
    for position, item in enumerate(items):
        value = score(needle, item)
        if value is not None:
            scored.append((value, position, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored]
