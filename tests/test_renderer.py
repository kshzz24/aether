"""Exhaustiveness guard for the event taxonomy + renderer (piece #5, half A).

The point of "the renderer handles every event type" is that it must be
*impossible* to add an Event variant and silently forget to render it. Two
tests enforce that from the runtime side (the pytest gate you already own):

- test_sample_covers_every_event_in_union: the sample list must name every
  member of the Event union, or it fails. Add a variant to the union without
  adding a sample here -> red.
- test_renderer_produces_output_for_every_event_type: each sampled event must
  make render() emit *something*. An unhandled variant (isinstance chain falls
  through, or match hits assert_never) prints nothing / raises -> red.

Together they mean: a new event forces a sample, and the sample forces a render
branch. Static enforcement (pyright + match/assert_never) is the edit-time
upgrade on top of this; this is the floor.
"""

from typing import get_args

from conftest import sample_events as _one_of_each

from approval import Verdict
from cli.renderer import Renderer
from events import (
    ApprovalDecisionEvent,
    Event,
    SubagentEvent,
)
from tools.base import ToolKind


def test_sample_covers_every_event_in_union():
    # If a variant is added to the Event union but not to _one_of_each(), the
    # counts diverge and this fails -- forcing the sample (and the render test
    # below) to stay complete.
    assert len(_one_of_each()) == len(get_args(Event))


# Variants this renderer deliberately drops, with the reason. Anything not
# listed here must produce output, so a variant can never be forgotten by
# accident -- only ignored on purpose, in writing.
_DELIBERATELY_SILENT = {
    # The authoritative TextEvent follows with the same words. This renderer
    # writes a one-shot log and cannot replace what it already printed, so
    # rendering deltas too would print every answer twice.
    "TextDeltaEvent",
}


def test_renderer_produces_output_for_every_event_type(capsys):
    # Every event must make the renderer emit something. A variant the renderer
    # doesn't handle prints nothing (silent isinstance fall-through) or raises
    # (match + assert_never) -- either way this catches it.
    renderer = Renderer()
    for event in _one_of_each():
        name = type(event).__name__
        renderer.render(event)
        out = capsys.readouterr().out
        if name in _DELIBERATELY_SILENT:
            assert not out.strip(), f"{name} is listed as silent but printed"
            continue
        assert out.strip(), f"{name} rendered nothing"


def test_renderer_handles_approval_decision_event(capsys):
    Renderer().render(ApprovalDecisionEvent(
        type="approval_decision", tool_name="write_file", kind=ToolKind.WRITE,
        danger_reasons=[], verdict=Verdict.AUTO_APPROVE, approved=True,
        source="policy",
    ))
    out = capsys.readouterr().out
    assert "write_file" in out


def test_renderer_handles_subagent_event(capsys):
    Renderer().render(SubagentEvent(
        type="subagent", task="explore the repo", phase="started",
    ))
    out = capsys.readouterr().out
    assert "explore the repo" in out
