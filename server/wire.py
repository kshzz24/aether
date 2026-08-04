from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class RunParams:
    goal: str | None
    resume: str | None
    gateway_url: str | None
    provider: str | None
    model: str | None
    approval_mode: str | None
    max_iterations: int | None
    max_cost_usd: float | None
    project_root: Path | None

    @classmethod
    def from_namespace(cls, args):
        return cls(
            prompt=getattr(args, "prompt", None),
            model=getattr(args, "model", None),
            system=getattr(args, "system", None),
            temperature=getattr(args, "temperature", None),
            max_tokens=getattr(args, "max_tokens", None),
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
