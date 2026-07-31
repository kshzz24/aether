"""What the status bar needs to know about the conversation.

Three small pure functions behind the context meter and the project-rules chip.

The token estimate is deliberately a division, not a tokenizer. A real
tokenizer means a per-provider dependency and a model-specific vocabulary, for a
number whose only job is to drive a ten-character bar. `len // 4` is the
standard rule of thumb, it is never off by enough to change what the bar looks
like, and it costs nothing. The agent's own compaction trigger uses the
provider's real `input_tokens` (`agent.py:266`) — this is the display, not the
governor, and it says so.
"""

from __future__ import annotations

from pathlib import Path

from client import Message, TextBlock, ToolCallBlock, ToolResultBlock

# Files a coding agent conventionally reads as standing instructions. Shown as a
# chip so it is obvious *which* rules are in play — a stale CLAUDE.md silently
# steering the agent is a confusing afternoon.
RULE_FILES = (
    "CLAUDE.md",
    "FORGE.md",
    "AGENTS.md",
    "GEMINI.md",
    ".airules",
    ".cursorrules",
)

_CHARS_PER_TOKEN = 4

_FILLED = "█"
_EMPTY = "░"


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token count for a message list. See the module docstring."""
    characters = 0
    for message in messages:
        for block in message.blocks:
            match block:
                case TextBlock(text=text):
                    characters += len(text)
                case ToolResultBlock(content=content):
                    characters += len(content)
                case ToolCallBlock(name=name, arguments=arguments):
                    characters += len(name) + sum(
                        len(str(key)) + len(str(value))
                        for key, value in arguments.items()
                    )
    return characters // _CHARS_PER_TOKEN


def meter(used: int, budget: int, width: int = 10) -> str:
    """A fixed-width bar, e.g. `████░░░░░░ 45%`.

    Clamps rather than overflowing: past the budget the agent compacts, and a
    bar that ran off the end of the status line would be a rendering bug
    reported as a cost bug.
    """
    if budget <= 0:
        return "—"
    fraction = min(1.0, max(0.0, used / budget))
    filled = round(fraction * width)
    return f"{_FILLED * filled}{_EMPTY * (width - filled)} {fraction * 100:.0f}%"


def find_project_rules(root: Path) -> list[Path]:
    """Standing-instruction files present at `root`, in RULE_FILES order."""
    return [root / name for name in RULE_FILES if (root / name).is_file()]


def rules_label(root: Path) -> str:
    """The status-bar chip, or "" when there are no rules to mention."""
    found = find_project_rules(root)
    if not found:
        return ""
    return "rules: " + ", ".join(path.name for path in found)
