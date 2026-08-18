"""A trace is a transparent event-stream wrapper, not an agent change.

`traced()` wraps the same iterator the CLI/TUI/server already drive
(`agent.run(...)`), re-yields every event unchanged, and records spans on the
side. Zero changes to `agent.py`'s control flow: this stays in the
composition/driver layer, the same plane as `persistence.py`.

Span boundaries, read off the event sequence (no new hooks needed):
  - `StatusEvent("thinking")` -> open a turn span.
  - `CostEvent` -> close the current turn span with cost_usd/tokens. Its id is
    kept as the parent for any tool spans opened later in the same turn.
  - `ToolCallEvent` -> open a tool span, nested under the current turn.
  - `ToolResultEvent` -> close it.
  - `TerminalEvent` -> close the run span (`reason` in `detail`), flush.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from events import (
    CostEvent,
    Event,
    StatusEvent,
    TerminalEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tracing.store import append_span


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str  # "run" | "turn" | "tool"
    name: str
    started_at: float
    ended_at: float
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    detail: str = ""


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _write(span: Span, traces_dir) -> None:
    append_span(
        {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "kind": span.kind,
            "name": span.name,
            "started_at": span.started_at,
            "ended_at": span.ended_at,
            "cost_usd": span.cost_usd,
            "input_tokens": span.input_tokens,
            "output_tokens": span.output_tokens,
            "detail": span.detail,
        },
        traces_dir,
    )


async def traced(
    events: AsyncIterator[Event],
    *,
    trace_id: str,
    prompt_hash: str,
    traces_dir=None,
) -> AsyncIterator[Event]:
    """Pass every event through unchanged; record spans on the side."""
    run_span_id = _new_id()
    run_started = time.time()

    turn_span_id: str | None = None
    turn_started: float | None = None

    tool_span_id: str | None = None
    tool_name: str = ""
    tool_started: float | None = None

    async for event in events:
        yield event

        if isinstance(event, StatusEvent) and event.message == "thinking":
            turn_span_id = _new_id()
            turn_started = time.time()

        elif isinstance(event, CostEvent) and turn_span_id is not None:
            _write(
                Span(
                    trace_id=trace_id,
                    span_id=turn_span_id,
                    parent_id=run_span_id,
                    kind="turn",
                    name="turn",
                    started_at=turn_started,
                    ended_at=time.time(),
                    cost_usd=event.cost_usd,
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                ),
                traces_dir,
            )
            # id kept (not cleared): tool spans issued later this same turn
            # still nest under it.

        elif isinstance(event, ToolCallEvent):
            tool_span_id = _new_id()
            tool_name = event.name
            tool_started = time.time()

        elif isinstance(event, ToolResultEvent) and tool_span_id is not None:
            _write(
                Span(
                    trace_id=trace_id,
                    span_id=tool_span_id,
                    parent_id=turn_span_id or run_span_id,
                    kind="tool",
                    name=tool_name,
                    started_at=tool_started,
                    ended_at=time.time(),
                    detail="; ".join(event.flags),
                ),
                traces_dir,
            )
            tool_span_id = None

        elif isinstance(event, TerminalEvent):
            _write(
                Span(
                    trace_id=trace_id,
                    span_id=run_span_id,
                    parent_id=None,
                    kind="run",
                    name=prompt_hash,
                    started_at=run_started,
                    ended_at=time.time(),
                    detail=event.reason.name,
                ),
                traces_dir,
            )
