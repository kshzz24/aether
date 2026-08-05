"""The third exhaustiveness gate on the `Event` union (Phase 8, stage 1).

`tests/test_renderer.py` already pins two: `sample_events()` must name every
member of the union, and `cli/renderer.py` must produce output for each. The
TUI's transcript is a third consumer under the same discipline.

`server/wire.py` now joins them, and it is the strictest of the four, because
its output leaves the process. The other surfaces render for a human who can
tolerate a missing line; the encoder feeds a browser that will throw on an
unknown `type` and a replay buffer that has to reproduce a transcript
byte-for-byte after a reconnect. So the gate is not "it emitted something" but
"it emitted a JSON-serializable frame with a stable discriminator".

The two tests worth reading twice are `test_enums_encode_by_name_not_value`
(the wire-format landmine) and `test_tool_result_is_not_truncated` (where the
encoder must *not* copy the renderer it was cloned from).
"""

from __future__ import annotations

import json
from typing import get_args

import pytest
from conftest import sample_events

from events import (
    Event,
    TerminalEvent,
    TerminalReason,
    ToolCallEvent,
    ToolResultEvent,
)
from server.wire import encode, frame

# Spelled out rather than derived from the encoder. A computed expectation
# would happily agree with a typo; this list is the actual contract the browser
# client and the replay buffer are written against.
EXPECTED_TYPES = {
    "status",
    "text_delta",
    "text",
    "tool_call",
    "tool_result",
    "cost",
    "subagent",
    "approval_decision",
    "confirm_request",
    "terminal",
}


@pytest.mark.parametrize("event", sample_events(), ids=lambda e: type(e).__name__)
def test_every_sample_event_encodes_to_json(event: Event):
    """The gate itself.

    Add a variant to the union -> `test_sample_covers_every_event_in_union`
    fails -> you add it to `sample_events()` -> it falls through this encoder's
    `match` and hits `assert_never` -> red here too. One list, four gates.
    """
    encoded = encode(event)
    assert isinstance(encoded, dict)
    assert "type" in encoded, f"{type(event).__name__} encoded without a discriminator"
    json.dumps(encoded)  # raises TypeError on anything not wire-representable


def test_encoded_types_are_unique_and_complete():
    """One frame type per event type, no collisions.

    `TerminalEvent` and `ConfirmRequestEvent` have no `type` field in
    `events.py` at all, so the encoder is the only thing assigning them one —
    which is exactly where two variants can quietly end up sharing a name.
    """
    types = [encode(event)["type"] for event in sample_events()]
    assert len(types) == len(set(types)), "two event types share a wire discriminator"
    assert set(types) == EXPECTED_TYPES
    assert len(types) == len(get_args(Event))


def test_enums_encode_by_name_not_value():
    """`TerminalReason`, `Verdict` and `ToolKind` are all `Enum(auto())`.

    `auto()` assigns values *positionally*, so `.value` is 1, 2, 3... in source
    order. Encoding those integers means reordering the enum — a refactor with
    no semantic content — silently changes the wire format for every deployed
    client, and a replayed transcript decodes as the wrong reason. `.name` is
    the stable identifier; this test is what stops someone "simplifying" the
    encoder to `reason.value`.
    """
    assert encode(TerminalEvent(reason=TerminalReason.LOOP_DETECTED))["reason"] == (
        "loop_detected"
    )
    assert encode(TerminalEvent(reason=TerminalReason.MAX_COST))["reason"] == "max_cost"

    decision = next(
        e for e in sample_events() if type(e).__name__ == "ApprovalDecisionEvent"
    )
    encoded = encode(decision)
    assert encoded["verdict"] == "auto_approve"
    assert encoded["kind"] == "write"
    assert encoded["approved"] is True
    assert encoded["source"] == "policy"


def test_terminal_reason_keeps_its_underscores():
    """The CLI renderer prints `reason.name.lower().replace("_", " ")` because
    a human reads it. The wire must not: `loop detected` is prose, and the
    client switches on an identifier."""
    assert "_" in encode(TerminalEvent(reason=TerminalReason.MAX_ITERATIONS))["reason"]


def test_tool_result_is_not_truncated():
    """The encoder was cloned from `cli/renderer.py`, which caps results at
    2000 chars — correct for a terminal scrollback, wrong here.

    The replay buffer re-sends this frame after a `Last-Event-ID` reconnect. If
    the live stream carried 5000 chars and the replay carries a truncated 2000,
    the same session shows two different transcripts depending on whether the
    browser dropped a connection. Trimming for display is the client's job.
    """
    body = "x" * 5000
    encoded = encode(ToolResultEvent(type="tool_result", name="run_shell", result=body))
    assert encoded["result"] == body


def test_tool_call_arguments_survive_as_a_nested_object():
    """Arguments must stay structured, not be flattened to the renderer's
    `key=value` display string — the browser renders them as a table and the
    confirm modal needs them editable."""
    encoded = encode(
        ToolCallEvent(
            type="tool_call",
            name="write_file",
            arguments={"path": "a.py", "content": "x", "count": 3},
        )
    )
    assert encoded["arguments"] == {"path": "a.py", "content": "x", "count": 3}


def test_text_delta_is_encoded_not_dropped():
    """`cli/renderer.py` deliberately ignores deltas because it also prints the
    authoritative `TextEvent` and would double every answer. The browser
    streams, like the TUI does, so dropping deltas here would leave the UI
    silent until a turn completed."""
    delta = next(e for e in sample_events() if type(e).__name__ == "TextDeltaEvent")
    assert encode(delta) == {"type": "text_delta", "text": "hel"}


def test_confirm_request_carries_no_request_id():
    """A `ConfirmRequestEvent` off the agent's stream is a *notification*.

    The correlation id belongs to `ServerApprover`'s parked future (stage 3)
    and arrives on its own frame. Minting one here would produce ids that
    resolve nothing, and every replayed transcript would re-offer confirms that
    were answered long ago.
    """
    confirm = next(
        e for e in sample_events() if type(e).__name__ == "ConfirmRequestEvent"
    )
    encoded = encode(confirm)
    assert "request_id" not in encoded
    assert encoded["arguments"] == {"cmd": "rm -rf /"}
    assert encoded["reason"] == "destructive shell command"


def test_encode_is_pure_and_stable():
    """Called twice, identical output. `encode` holds no counter and no clock —
    that is what lets the replay buffer re-encode a stored event and get the
    frame the live subscriber already saw."""
    for event in sample_events():
        assert encode(event) == encode(event)


def test_frame_adds_seq_and_changes_nothing_else():
    """`seq` is *session* state, not event state, which is why it is applied
    here rather than inside `encode`. Splitting them is what keeps `encode`
    testable across the whole union without inventing a counter."""
    event = TerminalEvent(reason=TerminalReason.COMPLETED, detail="")
    framed = frame(event, 7)
    assert framed["seq"] == 7
    assert {k: v for k, v in framed.items() if k != "seq"} == encode(event)


def test_frames_are_json_encodable_end_to_end():
    """What a subscriber actually does: enumerate, frame, serialize. Fails if
    any field is a dataclass, a Path, an Enum, or a set."""
    for seq, event in enumerate(sample_events()):
        line = json.dumps(frame(event, seq))
        assert json.loads(line)["seq"] == seq
