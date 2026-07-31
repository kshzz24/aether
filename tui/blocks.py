"""Splitting model prose into copyable pieces.

Rendering a whole model response as one `Markdown` renderable looks right but is
useless the moment you want the code out of it — you get the prose too, with the
fences. So the transcript mounts prose and fenced code as *separate* widgets, and
the code ones carry a copy button.

Both functions here are pure, which is the point: fence parsing has more edge
cases than it looks (unclosed fences, `~~~`, indented fences, a fence inside a
fence) and none of them should require booting a Textual app to test.
"""

from __future__ import annotations

from dataclasses import dataclass

# Long enough that folding is a relief rather than an annoyance.
LONG_PROSE_LINES = 25


@dataclass(frozen=True)
class Prose:
    """A run of ordinary markdown between code fences."""

    text: str


@dataclass(frozen=True)
class Code:
    """A fenced block. `language` is "" when the fence carried no info string."""

    text: str
    language: str = ""


Block = Prose | Code


def _fence(line: str) -> tuple[str, str] | None:
    """(marker, info) if `line` opens or closes a fence, else None."""
    stripped = line.lstrip()
    # Markdown allows up to three spaces of indent before a fence; more than
    # that is an indented code block, which is not our business.
    if len(line) - len(stripped) > 3:
        return None
    for marker in ("```", "~~~"):
        if stripped.startswith(marker):
            return marker, stripped[len(marker) :].strip()
    return None


def split_blocks(text: str) -> list[Block]:
    """Split markdown into alternating prose and fenced-code blocks.

    An unclosed fence takes the rest of the text as code rather than being
    dropped: a streamed response is routinely cut mid-block, and losing the
    half-written code is worse than rendering it early.
    """
    blocks: list[Block] = []
    buffer: list[str] = []
    open_marker: str | None = None
    language = ""

    def flush_prose() -> None:
        body = "\n".join(buffer).strip("\n")
        if body.strip():
            blocks.append(Prose(body))
        buffer.clear()

    for line in text.splitlines():
        found = _fence(line)

        if open_marker is None:
            if found is not None:
                flush_prose()
                open_marker, language = found
            else:
                buffer.append(line)
            continue

        # Inside a fence: only the same marker closes it, so a ``` inside a
        # ~~~ block stays literal.
        if found is not None and found[0] == open_marker and not found[1]:
            blocks.append(Code("\n".join(buffer), language))
            buffer.clear()
            open_marker = None
            language = ""
        else:
            buffer.append(line)

    if open_marker is not None:
        blocks.append(Code("\n".join(buffer), language))
    else:
        flush_prose()

    return blocks


def looks_like_diff(text: str) -> bool:
    """True for unified-diff text, so the transcript can colour it.

    Deliberately strict. Tool output is full of prose containing dashes and
    plus signs, and syntax-highlighting an ordinary sentence as a diff paints
    half of it red for no reason. Require a real structural marker: a hunk
    header, or a `---`/`+++` file-header pair.
    """
    lines = text.splitlines()
    if any(line.startswith("@@ ") and line.rstrip().endswith("@@") for line in lines):
        return True
    if any(line.startswith("@@ ") and " @@" in line for line in lines):
        return True
    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)
    return has_old and has_new


def is_long(text: str) -> bool:
    """True when a prose block is long enough to be worth folding."""
    return text.count("\n") + 1 > LONG_PROSE_LINES
