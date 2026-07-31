"""Saved prompts, loaded from `~/.forge/prompts/*.md`.

The same reusable-prompt idea as a shell alias: the fifth time you type "review
this for the invariants in CLAUDE.md and flag anything that leaks provider shape
into the loop", it should be `/prompt review`.

Separate from both `tui/pickers.py` (which imports Textual) and
`tui/commands.py` (which must not), because both need it.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path.home() / ".forge" / "prompts"


def load_templates(directory: Path) -> dict[str, str]:
    """Read `*.md` templates, keyed by filename stem.

    A missing directory is the normal case, not an error — most people never
    create one, and the picker should say "no saved prompts" rather than fail.
    """
    if not directory.is_dir():
        return {}
    templates: dict[str, str] = {}
    for path in sorted(directory.glob("*.md")):
        try:
            body = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            # One unreadable template must not hide the rest.
            logger.debug("skipping template %s: %s", path, exc)
            continue
        if body:
            templates[path.stem] = body
    return templates
