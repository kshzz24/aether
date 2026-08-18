from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

from approval import ApprovalRequest
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
    """Translate one `Event` into a JSON-serializable frame.

    Pure and stateless: `seq` is session state and is applied by `frame`, which
    is what lets the replay buffer re-encode a stored event and reproduce the
    frame a live subscriber already saw.

    The `type` discriminator is spelled out here rather than read off
    `event.type` because two members of the union (`TerminalEvent`,
    `ConfirmRequestEvent`) do not carry one, and because this encoder — not
    `events.py` — owns the wire contract deployed clients are written against.

    Enums encode as `.name.lower()`, never `.value`: `TerminalReason`,
    `Verdict` and `ToolKind` are all `Enum(auto())`, so their values are
    *positional* integers and reordering the enum would silently change the
    wire format for every client and mis-decode every replayed transcript.
    """
    match event:
        case StatusEvent(message=message):
            return {"type": "status", "message": message}

        case TextDeltaEvent(text=text):
            # Encoded, unlike in `cli/renderer.py`. That renderer drops deltas
            # because it also prints the authoritative TextEvent and would
            # double every answer; a browser streams, so dropping them here
            # would leave the UI silent until a turn completed.
            return {"type": "text_delta", "text": text}

        case TextEvent(text=text):
            return {"type": "text", "text": text}

        case ToolCallEvent(name=name, arguments=arguments):
            # `arguments` stays a nested object: the browser renders it as a
            # table and the confirm modal needs it editable.
            return {"type": "tool_call", "name": name, "arguments": arguments}

        case ToolResultEvent(name=name, result=result, flags=flags):
            # Deliberately untruncated. The renderer caps at 2000 chars for a
            # terminal scrollback; here a replayed frame must match the live
            # one byte-for-byte, so trimming for display is the client's job.
            return {
                "type": "tool_result",
                "name": name,
                "result": result,
                "flags": flags,
            }

        case SubagentEvent(task=task, phase=phase, detail=detail):
            return {
                "type": "subagent",
                "task": task,
                "phase": phase,
                "detail": detail,
            }

        case CostEvent(
            cost_usd=cost_usd,
            total_cost_usd=total,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ):
            return {
                "type": "cost",
                "cost_usd": cost_usd,
                "total_cost_usd": total,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        case ApprovalDecisionEvent(
            tool_name=tool_name,
            kind=kind,
            danger_reasons=danger_reasons,
            verdict=verdict,
            approved=approved,
            source=source,
        ):
            return {
                "type": "approval_decision",
                "tool_name": tool_name,
                "kind": kind.name.lower(),
                "danger_reasons": danger_reasons,
                "verdict": verdict.name.lower(),
                "approved": approved,
                "source": source,
            }

        case ConfirmRequestEvent(
            tool_name=tool_name, arguments=arguments, reason=reason
        ):
            # No `request_id`: a ConfirmRequestEvent off the agent's stream is a
            # notification. The correlation id belongs to `ServerApprover`'s
            # parked future and arrives on its own frame — minting one here
            # would produce ids that resolve nothing, and every replayed
            # transcript would re-offer confirms answered long ago.
            return {
                "type": "confirm_request",
                "tool_name": tool_name,
                "arguments": arguments,
                "reason": reason,
            }

        case TerminalEvent(reason=reason, detail=detail):
            # Underscores kept. The CLI prints `loop detected` because a human
            # reads it; the client switches on an identifier.
            return {
                "type": "terminal",
                "reason": reason.name.lower(),
                "detail": detail,
            }

        case _ as unreachable:
            assert_never(unreachable)


def frame(event: Event, seq: int) -> dict[str, Any]:
    return {**encode(event), "seq": seq}


def confirm_frame(request: ApprovalRequest, request_id: str) -> dict[str, Any]:
    """The answerable question, minted by `ServerApprover` rather than the agent.

    A second frame is needed because `ConfirmRequestEvent` cannot carry any of
    this: no `kind`, no `danger_reasons`, no `diff`, and no correlation id. Those
    live on `ApprovalRequest`, which only the approver sees, and a browser modal
    needs all four — the id to answer with, `danger_reasons` to decide whether to
    offer "always", the diff to show what a write would do.

    No `seq` here, for the same reason `encode` has none: numbering is session
    state, applied by `AgentSession.publish_frame`. Unlike the overflow notice
    this frame *does* get one and *does* enter the transcript, so a browser that
    reconnects mid-confirm is re-shown the pending question.

    `offers_always` is computed here, server-side, and not left to the client:
    withholding "always" from a flagged call is a security rule
    (`tui/approver.py:76-84`), and a security rule enforced only in browser
    JavaScript is not enforced.
    """
    return {
        "type": "confirm",
        "request_id": request_id,
        "tool_name": request.tool_name,
        "arguments": request.arguments,
        "kind": request.kind.name.lower(),
        "danger_reasons": list(request.danger_reasons),
        "diff": request.diff,
        "offers_always": not request.danger_reasons,
    }
