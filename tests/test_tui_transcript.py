"""The TUI's Event mapping — pure, no app.

`tui/transcript.py` is the *second* consumer of the `Event` union. The union's
exhaustiveness gate lives in `tests/test_renderer.py`; these tests are the TUI's
half of it: every event `conftest.sample_events()` produces must render to
something visible here too.
"""

from __future__ import annotations

import io

from conftest import sample_events
from rich.console import Console

from events import (
    StatusEvent,
    TerminalEvent,
    TerminalReason,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tui.transcript import event_to_renderable


def _render(event) -> str:
    """Draw a renderable to plain text so we can assert on what a user sees."""
    buffer = io.StringIO()
    Console(file=buffer, width=80, no_color=True, legacy_windows=False).print(
        event_to_renderable(event)
    )
    return buffer.getvalue()


def test_every_event_type_renders_something():
    """The TUI half of the exhaustiveness gate.

    A variant added to the union forces an entry in `sample_events()`, which
    forces a `match` branch here — an unhandled one hits `assert_never`.
    """
    for event in sample_events():
        assert _render(event).strip(), f"{type(event).__name__} rendered nothing"


def test_every_terminal_reason_has_a_style():
    """`_TERMINAL_STYLES` is a dict lookup, so a missing reason is a KeyError at
    the worst possible moment — the end of a run."""
    for reason in TerminalReason:
        assert _render(TerminalEvent(reason=reason, detail="")).strip()


def test_terminal_detail_is_shown_when_present():
    out = _render(TerminalEvent(reason=TerminalReason.ERROR, detail="boom happened"))
    assert "boom happened" in out


def test_tool_call_renders_name_and_arguments():
    out = _render(
        ToolCallEvent(type="tool_call", name="read_file", arguments={"path": "a.py"})
    )
    assert "read_file" in out
    assert "path=a.py" in out


def test_tool_call_with_no_arguments_still_renders():
    out = _render(ToolCallEvent(type="tool_call", name="list_dir", arguments={}))
    assert "list_dir" in out


def test_long_tool_result_is_truncated_with_a_marker():
    out = _render(
        ToolResultEvent(type="tool_result", name="run_shell", result="x" * 5000)
    )
    assert "truncated" in out
    assert len(out) < 5000


def test_short_tool_result_is_untouched():
    out = _render(ToolResultEvent(type="tool_result", name="run_shell", result="fine"))
    assert "fine" in out
    assert "truncated" not in out


def test_long_arguments_are_clipped():
    out = _render(
        ToolCallEvent(
            type="tool_call", name="write_file", arguments={"body": "y" * 500}
        )
    )
    assert "..." in out
    assert len(out) < 500


def test_newlines_in_arguments_do_not_break_the_panel():
    out = _render(
        ToolCallEvent(
            type="tool_call", name="write_file", arguments={"body": "a\nb\nc"}
        )
    )
    assert "\\n" in out


def test_model_text_containing_markup_is_not_interpreted():
    """Model and tool output is data, not styling. Rendering it as Rich markup
    would let a tool result restyle the transcript — or crash on bad syntax."""
    out = _render(TextEvent(type="text", text="use [bold]this[/bold] and [not-a-tag]"))
    assert "[bold]" in out
    assert "[not-a-tag]" in out


def test_tool_result_containing_markup_is_not_interpreted():
    out = _render(
        ToolResultEvent(type="tool_result", name="grep", result="match: [/unclosed")
    )
    assert "[/unclosed" in out


def test_status_message_is_visible():
    assert "compacted" in _render(StatusEvent(type="status", message="compacted"))
