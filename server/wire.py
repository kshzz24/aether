from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

from events import (
    ApprovalDecisionEvent,
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    SubagentEvent,
    TerminalEvent,
    TextDeltaEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)

_MAX_RESULT_CHARS = 2000


@dataclass(frozen=True)
class RunParams:
    """Everything `build_composition` needs, with nothing implicit left over.

    Frozen because a surface that wants a different model builds a *new*
    invocation rather than mutating the live one — `dataclasses.replace` in the
    TUI's `_rebuild` rejects a typo'd key outright, where the old `setattr` onto
    a Namespace accepted it and silently did nothing.

    Every field defaults to None so a caller can name only what it cares about.
    The server will build `RunParams(model=..., project_root=...)` and nothing
    else; `replace` can only ever produce a complete instance if the original
    was constructible bare.
    """

    goal: str | None = None
    resume: str | None = None
    gateway_url: str | None = None
    provider: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    max_iterations: int | None = None
    max_cost_usd: float | None = None
    project_root: Path | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> RunParams:
        """Translate argv once, at the CLI boundary.

        `getattr` with a default rather than attribute access: `--setup`
        short-circuits parsing in ways that leave attributes unset, and the TUI
        tests build partial namespaces. A missing flag must read as None, not
        raise.

        `project_root` is deliberately absent. argv cannot express it — the CLI
        is always rooted at wherever the user is standing — so it stays None and
        `build_composition` is the single place that resolves None -> Path.cwd().
        """
        return cls(
            goal=getattr(args, "goal", None),
            resume=getattr(args, "resume", None),
            gateway_url=getattr(args, "gateway_url", None),
            provider=getattr(args, "provider", None),
            model=getattr(args, "model", None),
            approval_mode=getattr(args, "approval_mode", None),
            max_iterations=getattr(args, "max_iterations", None),
            max_cost_usd=getattr(args, "max_cost_usd", None),
            project_root=None,
        )


def encode(event: Event) -> dict[str, Any]:
    match event:
        case StatusEvent(message=message):
            print(f"\n[ {message} ]")

        case TextDeltaEvent():
            # Ignored on purpose. This renderer writes a one-shot log, and
            # the authoritative TextEvent follows with the same words — so
            # rendering deltas as well would print every answer twice. The
            # TUI, which can replace what it drew, uses them.
            pass

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

        case ConfirmRequestEvent(tool_name=name, arguments=arguments, reason=reason):
            print(f"\n[ confirm? {name}({_format_args(arguments)}) — {reason} ]")

        case TerminalEvent(reason=reason, detail=detail):
            label = reason.name.lower().replace("_", " ")
            print(f"\n[ {label}{f': {detail}' if detail else ''} ]")

        case _ as unreachable:
            assert_never(unreachable)


def frame(event: Event, seq: int) -> dict[str, Any]:
    return {**encode(event), "seq": seq}
