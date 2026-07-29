from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tools.base import Tool, ToolKind

if TYPE_CHECKING:
    from agent import Agent

_PARAMS = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "A self-contained task for the subagent to carry out. It runs "
                "its own loop with the full toolset and returns only a distilled "
                "result."
            ),
        }
    },
    "required": ["prompt"],
}

_DESCRIPTION = (
    "Delegate a self-contained sub-task to a child agent that runs its own loop "
    "with the full toolset and returns only a distilled result. Use it to keep "
    "noisy exploration or a well-scoped chunk of work out of your own context."
)


def build_subagent_tool(*, make_child: Callable[[], Agent]) -> Tool:

    from events import (
        CostEvent,
        StatusEvent,
        TerminalEvent,
        TerminalReason,
        TextEvent,
    )

    async def run(args: dict) -> str:
        prompt = args["prompt"]
        child = make_child()

        turn_texts: list[str] = []
        cost = 0.0
        reason: TerminalReason | None = None
        detail = ""

        async for event in child.run(prompt):
            match event:
                case StatusEvent(message="thinking"):
                    turn_texts = []
                case TextEvent(text=text):
                    turn_texts.append(text)
                case CostEvent(total_cost_usd=total):
                    cost = total
                case TerminalEvent(reason=terminal_reason, detail=terminal_detail):
                    reason = terminal_reason
                    detail = terminal_detail

        answer = "\n".join(turn_texts).strip()
        reason_label = reason.name.lower() if reason is not None else "unknown"
        footer = f"[subagent {reason_label}, ${cost:.4f}]"

        if reason is not TerminalReason.COMPLETED:
            note = f"subagent did not complete ({reason_label})"
            if detail:
                note += f":{detail}"

            body = answer or "(no output)"
            return f"{note}\n{body}\n\n{footer}"

        return f"{answer or '(subagent produced no summary text)'}\n\n{footer}"

    return Tool(
        name="task",
        description=_DESCRIPTION,
        parameters=_PARAMS,
        kind=ToolKind.AGENT,
        run=run,
    )
