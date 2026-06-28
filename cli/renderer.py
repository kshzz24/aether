"""The renderer is the ONLY component permitted to write to stdout.

The agent core and tools yield `Event` objects; this is where they become text.
Keep all `print()` calls in this file (Phase 0 print-discipline invariant).
"""

from events import (
    CostEvent,
    Event,
    StatusEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)

_MAX_RESULT_CHARS = 2000


def _truncate(text: str, limit: int = _MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _format_args(arguments: dict[str, object]) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = str(value).replace("\n", "\\n")
        if len(rendered) > 60:
            rendered = rendered[:60] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class Renderer:
    """Turns the agent's Event stream into terminal output."""

    def render(self, event: Event) -> None:
        if isinstance(event, StatusEvent):
            print(f"\n[ {event.message} ]")
        elif isinstance(event, TextEvent):
            print(event.text)
        elif isinstance(event, ToolCallEvent):
            print(f"  -> {event.name}({_format_args(event.arguments)})")
        elif isinstance(event, ToolResultEvent):
            print(_indent(_truncate(event.result)))
        elif isinstance(event, CostEvent):
            print(
                f"  [${event.cost_usd:.4f} this turn"
                f" | ${event.total_cost_usd:.4f} total]"
            )
