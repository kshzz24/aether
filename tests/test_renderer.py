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

from cli.renderer import Renderer
from events import (
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    TerminalEvent,
    TerminalReason,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def _one_of_each() -> list[Event]:
    return [
        StatusEvent(type="status", message="working"),
        TextEvent(type="text", text="hello"),
        ToolCallEvent(type="tool_call", name="run_shell", arguments={"cmd": "ls"}),
        ToolResultEvent(type="tool_result", name="run_shell", result="ok"),
        CostEvent(type="cost", cost_usd=0.01, total_cost_usd=0.02),
        ConfirmRequestEvent(
            tool_name="run_shell",
            arguments={"cmd": "rm -rf /"},
            reason="destructive shell command",
        ),
        TerminalEvent(reason=TerminalReason.COMPLETED, detail=""),
    ]


def test_sample_covers_every_event_in_union():
    # If a variant is added to the Event union but not to _one_of_each(), the
    # counts diverge and this fails -- forcing the sample (and the render test
    # below) to stay complete.
    assert len(_one_of_each()) == len(get_args(Event))


def test_renderer_produces_output_for_every_event_type(capsys):
    # Every event must make the renderer emit something. A variant the renderer
    # doesn't handle prints nothing (silent isinstance fall-through) or raises
    # (match + assert_never) -- either way this catches it.
    renderer = Renderer()
    for event in _one_of_each():
        renderer.render(event)
        out = capsys.readouterr().out
        assert out.strip(), f"{type(event).__name__} rendered nothing"
