"""The renderer is the ONLY component permitted to write to stdout.

The agent core and tools yield `Event` objects; this is where they become text.
Keep all `print()` calls in this file (Phase 0 print-discipline invariant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from events import (
    ApprovalDecisionEvent,
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    SubagentEvent,
    TerminalEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from persistence import SessionMeta

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

            case SubagentEvent(task=task, phase=phase, detail=detail):
                if phase == "started":
                    print(f"\n[ subagent -> {task} ]")
                else:
                    suffix = f": {detail}" if detail else ""
                    print(f"[ subagent done{suffix} ]")

            case CostEvent(cost_usd=cost_usd, total_cost_usd=total):
                print(f"  [${cost_usd:.4f} this turn | ${total:.4f} total]")

            case ApprovalDecisionEvent(
                tool_name=name, verdict=verdict, source=source, danger_reasons=reasons
            ):
                suffix = f" — {'; '.join(reasons)}" if reasons else ""
                print(f"[ decision {name}: {verdict.name} ({source}){suffix} ]")

            case ConfirmRequestEvent(
                tool_name=name, arguments=arguments, reason=reason
            ):
                print(f"\n[ confirm? {name}({_format_args(arguments)}) — {reason} ]")

            case TerminalEvent(reason=reason, detail=detail):
                label = reason.name.lower().replace("_", " ")
                print(f"\n[ {label}{f': {detail}' if detail else ''} ]")

            case _ as unreachable:
                assert_never(unreachable)

    def notice(self, text: str) -> None:
        """A composition-root notice (e.g. the session id) — not an Event."""
        print(f"[ {text} ]")

    def sessions(self, metas: list[SessionMeta]) -> None:
        """Render the saved-session list for --list-sessions."""
        if not metas:
            print("[ no saved sessions ]")
            return
        for m in metas:
            goal = m.goal if len(m.goal) <= 60 else m.goal[:57] + "..."
            print(
                f"{m.id}  ${m.total_cost:.4f}  {m.turns:>3} turns  "
                f"{m.updated_at}  {goal}"
            )
