"""The renderer is the ONLY component permitted to write to stdout.

The agent core and tools yield `Event` objects; this is where they become text.
Keep all `print()` calls in this file (Phase 0 print-discipline invariant).
"""

from typing import assert_never

from events import (
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    TerminalEvent,
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
        match event:
            case StatusEvent(message=message):
                print(f"\n[ {message} ]")
            case TextEvent(text=text):
                print(text)
            case ToolCallEvent(name=name, arguments=arguments):
                print(f"  -> {name}({_format_args(arguments)})")
            case ToolResultEvent(result=result):
                print(_indent(_truncate(result)))
            case CostEvent(cost_usd=cost_usd, total_cost_usd=total):
                print(f"  [${cost_usd:.4f} this turn | ${total:.4f} total]")
            case ConfirmRequestEvent(
                tool_name=name, arguments=arguments, reason=reason
            ):
                print(f"\n[ confirm? {name}({_format_args(arguments)}) — {reason} ]")
            case TerminalEvent(reason=reason, detail=detail):
                label = reason.name.lower().replace("_", " ")
                print(f"\n[ {label}{f': {detail}' if detail else ''} ]")
            case _ as unreachable:
                assert_never(unreachable)
