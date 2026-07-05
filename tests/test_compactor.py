import asyncio
import logging

from client import (
    Message,
    NormalizedResponse,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)


def drive(coro):
    """Run a single coroutine to completion and return its result."""
    return asyncio.run(coro)


class StubClient:
    """A fake LLMClient that records calls and returns a canned summary.

    The compactor must summarize THROUGH the client (never a provider SDK), so
    the test injects this and asserts it was called exactly once and that the
    cost it reports flows back out.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.last_system = None

    async def create(self, messages, tools, system) -> NormalizedResponse:
        self.calls += 1
        self.last_system = system
        return NormalizedResponse(
            blocks=[TextBlock(text="SUMMARY: earlier work happened")],
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            stop_reason="end_turn",
        )


def _transcript() -> list[Message]:
    """Goal + turns with a tool_call/tool_result pair near the tail.

    Laid out so a naive cut (keep_recent=2) would land on the tool_result and
    orphan it from its tool_call one turn earlier. The compactor must walk the
    cut back to the assistant boundary and keep the pair together.
    """
    return [
        Message(role="user", blocks=[TextBlock(text="do the thing")]),        # 0 goal
        Message(role="assistant", blocks=[TextBlock(text="step 1")]),          # 1
        Message(role="user", blocks=[TextBlock(text="obs 1")]),                # 2
        Message(role="assistant", blocks=[TextBlock(text="step 2")]),          # 3
        Message(role="user", blocks=[TextBlock(text="obs 2")]),                # 4
        Message(
            role="assistant",
            blocks=[ToolCallBlock(id="abc", name="read_file", arguments={})],
        ),                                                                      # 5
        Message(
            role="user",
            blocks=[ToolResultBlock(tool_call_id="abc", content="file body")],
        ),                                                                      # 6
        Message(role="assistant", blocks=[TextBlock(text="final")]),           # 7
    ]


def test_needs_compaction_fires_at_eighty_percent():
    from context.compactor import needs_compaction

    # int(0.8 * 100) == 80: just under stays False, at/over the line is True.
    assert needs_compaction(79, 100) is False
    assert needs_compaction(80, 100) is True
    assert needs_compaction(200, 100) is True


def test_compact_folds_summary_into_goal_and_keeps_recent_pair(caplog):
    from context.compactor import compact

    client = StubClient()
    original = _transcript()

    with caplog.at_level(logging.INFO):
        new, summary_cost = drive(compact(client, original, keep_recent=2))

    # Summarized exactly once, THROUGH the injected client.
    assert client.calls == 1

    # The synthetic summary is folded into the goal message, not emitted as a
    # standalone user turn (that would double up the user role at the seam).
    assert new[0].role == "user"
    goal_text = " ".join(b.text for b in new[0].blocks if isinstance(b, TextBlock))
    assert "do the thing" in goal_text        # original goal survives
    assert "earlier work happened" in goal_text  # summary was folded in

    # The transcript shrank.
    assert len(new) < len(original)

    # The kept tail starts on an assistant turn -> no orphaned tool_result and
    # no two same-role messages in a row (Anthropic rejects that).
    assert new[1].role == "assistant"

    # The tool_call/tool_result pair stayed together in the kept tail.
    tail_blocks = [b for m in new[1:] for b in m.blocks]
    call_ids = {b.id for b in tail_blocks if isinstance(b, ToolCallBlock)}
    result_ids = {b.tool_call_id for b in tail_blocks if isinstance(b, ToolResultBlock)}
    assert result_ids <= call_ids  # every kept result has its call in the tail

    # The most recent turns are preserved verbatim.
    assert new[-2:] == original[-2:]

    # The summarization cost flows back out so the agent can meter it.
    assert summary_cost == 0.001

    # Compaction is visible in the logs.
    assert any("compact" in r.message.lower() for r in caplog.records)


def test_compact_noop_when_keep_recent_exceeds_transcript():
    from context.compactor import compact

    client = StubClient()
    # 5 messages, keep_recent=6 -> a naive cut goes negative and would land
    # `recent` on an orphaned tool_result (its tool_call swept into the middle),
    # which Anthropic/Groq reject. When nothing can be dropped while keeping
    # whole turns, compaction must no-op instead of corrupting the transcript.
    messages = [
        Message(role="user", blocks=[TextBlock(text="do the thing")]),
        Message(
            role="assistant",
            blocks=[ToolCallBlock(id="t1", name="run_shell", arguments={})],
        ),
        Message(role="user", blocks=[ToolResultBlock(tool_call_id="t1", content="ok")]),
        Message(
            role="assistant",
            blocks=[ToolCallBlock(id="t2", name="write_file", arguments={})],
        ),
        Message(role="user", blocks=[ToolResultBlock(tool_call_id="t2", content="ok")]),
    ]

    new, summary_cost = drive(compact(client, messages, keep_recent=6))

    assert new == messages     # untouched, no orphaned tool_result
    assert summary_cost == 0.0
    assert client.calls == 0


def test_compact_noop_when_nothing_to_summarize():
    from context.compactor import compact

    client = StubClient()
    # goal + one recent turn, keep_recent=2 -> nothing in the middle to drop.
    messages = [
        Message(role="user", blocks=[TextBlock(text="do the thing")]),
        Message(role="assistant", blocks=[TextBlock(text="done")]),
    ]

    new, summary_cost = drive(compact(client, messages, keep_recent=2))

    assert new == messages       # unchanged
    assert summary_cost == 0.0    # no model call, no cost
    assert client.calls == 0      # the client was never touched
