"""Getting text out of the TUI and into the system clipboard.

Textual's `App.copy_to_clipboard` writes an OSC 52 escape sequence. That is the
right primitive — it is the only one that works over SSH — but terminals that
don't implement it discard the sequence *silently*. A copy button you cannot
tell has failed is worse than no copy button, so this module does both: it sends
OSC 52 and also shells out to a native helper when one exists, then reports
which path actually carried the text.

`detect_backend` takes the platform string and a `which` callable rather than
reading `sys.platform` and `shutil.which` itself, so the whole matrix —
Windows, macOS, Wayland, X11, WSL, and a bare box with none of them — is
testable from one machine.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Tried in order. wl-copy first: on a Wayland session xclip may exist but talk
# to an XWayland clipboard the user's other apps don't read.
_LINUX_CANDIDATES: tuple[tuple[str, list[str]], ...] = (
    ("wl-copy", ["wl-copy"]),
    ("xclip", ["xclip", "-selection", "clipboard"]),
    ("xsel", ["xsel", "--clipboard", "--input"]),
    # WSL: platform reads as linux, but the Windows clipboard is the real one.
    ("clip.exe", ["clip.exe"]),
)

_TIMEOUT_SECONDS = 5


def detect_backend(
    platform: str, which: Callable[[str], str | None] = shutil.which
) -> list[str] | None:
    """The argv of a clipboard helper for `platform`, or None if there is none."""
    if platform == "win32":
        return ["clip.exe"] if which("clip.exe") else None
    if platform == "darwin":
        return ["pbcopy"] if which("pbcopy") else None
    for binary, argv in _LINUX_CANDIDATES:
        if which(binary):
            return argv
    return None


def copy_via_backend(text: str, argv: list[str]) -> bool:
    """Pipe `text` to a clipboard helper. False on any failure."""
    try:
        subprocess.run(
            argv,
            input=text.encode("utf-8"),
            check=True,
            timeout=_TIMEOUT_SECONDS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # A clipboard helper failing must never take the session with it.
        logger.debug("clipboard backend %s failed: %s", argv[0], exc)
        return False
    return True


def copy(send_osc52: Callable[[str], None], text: str) -> str:
    """Copy `text` and describe where it went, for the toast.

    Both paths are attempted: OSC 52 is free and works over SSH, the native
    helper survives terminals that ignore OSC 52. Naming the winner is what
    makes a failed copy visible instead of mysterious.

    The OSC-52 sender is passed in rather than pulled off an app object, so this
    can sit *underneath* `App.copy_to_clipboard` — which is what routes
    Textual's own drag-select `ctrl+c` (screen.py:991) through the native
    fallback as well.
    """
    if not text:
        return "nothing to copy"

    terminal_ok = True
    try:
        send_osc52(text)
    except Exception as exc:  # noqa: BLE001 -- never lose the session over a copy
        logger.debug("OSC 52 copy failed: %s", exc)
        terminal_ok = False

    argv = detect_backend(sys.platform)
    native_ok = copy_via_backend(text, argv) if argv else False

    lines = text.count("\n") + 1
    if native_ok:
        return f"copied {lines} lines via {argv[0]}"
    if terminal_ok:
        return f"copied {lines} lines via the terminal"
    return "copy failed — no clipboard backend available"
