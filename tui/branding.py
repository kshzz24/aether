"""The wordmark and the opening info block.

Kept as pure renderable-builders so the app just mounts what it is given, and so
the strings can be asserted on in a test without a screen.

The mark is half-block glyphs rather than a font-dependent logo: it renders the
same in Windows Terminal, WSL, iTerm, and a bare console, which is more than can
be said for most ASCII art.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Group, RenderableType
from rich.text import Text

WORDMARK = r"""
 ███████  ██████  ██████   ██████  ███████
 ██      ██    ██ ██   ██ ██       ██
 █████   ██    ██ ██████  ██   ███ █████
 ██      ██    ██ ██   ██ ██    ██ ██
 ██       ██████  ██   ██  ██████  ███████
""".strip("\n")

TAGLINE = "agentic coding assistant"

# Shown under the mark on every launch. Two columns of the things a new user
# needs within the first minute — anything longer than this is /help's job.
TIPS: tuple[tuple[str, str], ...] = (
    ("/help", "commands"),
    ("/keys", "shortcuts"),
    ("@", "reference a file"),
    ("esc", "interrupt a run"),
)


def _short_path(path: Path) -> str:
    """Render a path relative to home, so the mark line stays short."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return path.as_posix()


def banner(*, version: str, cwd: Path) -> RenderableType:
    """The full opening block: mark, tagline, location, tips.

    Colour comes from the mounting widget's CSS class, not from styles baked in
    here — that is what keeps the theme switch able to recolour the mark.
    """
    mark = Text(WORDMARK, style="bold")

    meta = Text()
    meta.append(TAGLINE)
    meta.append("  ·  ", style="dim")
    meta.append(f"v{version}")
    meta.append("\n")
    meta.append(_short_path(cwd), style="dim")

    width = max(len(key) for key, _ in TIPS)
    tips = Text()
    for index, (key, description) in enumerate(TIPS):
        if index and index % 2 == 0:
            tips.append("\n")
        tips.append(f"  {key:<{width}}  ", style="bold")
        tips.append(f"{description:<20}", style="dim")

    return Group(mark, Text(""), meta, Text(""), tips)
