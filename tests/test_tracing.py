import asyncio
import json

from events import (
    CostEvent,
    StatusEvent,
    TerminalEvent,
    TerminalReason,
    ToolCallEvent,
    ToolResultEvent,
)
from tracing.recorder import traced
from tracing.store import read_spans


async def _fake_events():
    yield StatusEvent(type="status", message="thinking")
    yield CostEvent(
        type="cost",
        cost_usd=0.01,
        total_cost_usd=0.01,
        input_tokens=10,
        output_tokens=5,
    )
    yield ToolCallEvent(type="tool_call", name="read_file", arguments={"path": "a.py"})
    yield ToolResultEvent(type="tool_result", name="read_file", result="ok", flags=[])
    yield TerminalEvent(reason=TerminalReason.COMPLETED)


def _drive(traces_dir):
    async def _run():
        out = []
        async for event in traced(
            _fake_events(),
            trace_id="t1",
            prompt_hash="deadbeef1234",
            traces_dir=traces_dir,
        ):
            out.append(event)
        return out

    return asyncio.run(_run())


def test_every_event_is_reyielded_unchanged(tmp_path):
    events = _drive(tmp_path)
    assert len(events) == 5
    assert isinstance(events[0], StatusEvent)
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.COMPLETED


def test_spans_are_written_with_correct_kinds_and_nesting(tmp_path):
    _drive(tmp_path)
    spans = read_spans("t1", tmp_path)

    kinds = {s["kind"] for s in spans}
    assert kinds == {"run", "turn", "tool"}

    run_span = next(s for s in spans if s["kind"] == "run")
    turn_span = next(s for s in spans if s["kind"] == "turn")
    tool_span = next(s for s in spans if s["kind"] == "tool")

    assert run_span["parent_id"] is None
    assert turn_span["parent_id"] == run_span["span_id"]
    assert tool_span["parent_id"] == turn_span["span_id"]

    assert turn_span["cost_usd"] == 0.01
    assert turn_span["input_tokens"] == 10
    assert turn_span["output_tokens"] == 5

    assert run_span["detail"] == "COMPLETED"
    assert tool_span["name"] == "read_file"


def test_jsonl_round_trips_through_the_file(tmp_path):
    _drive(tmp_path)
    path = tmp_path / "t1.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # turn, tool, run
    for line in lines:
        json.loads(line)  # every line stands alone as valid JSON


def test_read_spans_skips_a_truncated_final_line(tmp_path):
    _drive(tmp_path)
    path = tmp_path / "t1.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"trace_id": "t1", "span_id": "broken"')  # no closing brace/newline

    spans = read_spans("t1", tmp_path)
    assert len(spans) == 3  # the truncated tail is skipped, not raised
