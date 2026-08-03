"""Selecting, yanking, and copying from the transcript.

Selection is entry-wise rather than character-wise: entries are mounted widgets,
so there is no character grid to select across and a collapsed `Collapsible`'s
body has no position at all. The terminal's own shift+drag still works over the
top of this, which is why it is documented in `MOUSE_TIPS`.

The load-bearing test here is the last one in the select-mode section: `escape`
must leave select mode *without* also interrupting the agent, because `escape`
is the panic stop for a run that is spending money.
"""

from __future__ import annotations

import argparse

import pytest

import persistence
from events import TerminalEvent, TerminalReason, TextDeltaEvent, TextEvent
from tui.app import ForgeApp
from tui.transcript import CodeBlock, TranscriptView


def _args(**over) -> argparse.Namespace:
    base = dict(
        goal=None,
        gateway_url=None,
        resume=None,
        list_sessions=False,
        tui=True,
        setup=False,
        provider="groq",
        model="stub-model",
        max_iterations=None,
        max_cost_usd=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")
    return tmp_path


@pytest.fixture
def clipboard(monkeypatch):
    """Capture what would have gone to the system clipboard."""
    captured: list[str] = []
    monkeypatch.setattr(
        "tui.app.clipboard_copy", lambda _app, text: captured.append(text) or "copied"
    )
    return captured


async def _fill(app, pilot, count: int = 4) -> TranscriptView:
    transcript = app.query_one(TranscriptView)
    transcript.clear()
    for n in range(count):
        transcript.notice(f"entry {n}")
    transcript.focus()
    await pilot.pause()
    return transcript


# --------------------------------------------------------------------------
# Select mode
# --------------------------------------------------------------------------


async def test_v_enters_select_mode(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("v")
        await pilot.pause()
        assert transcript.selecting is True


async def test_select_mode_starts_at_the_newest_entry(sessions_dir):
    """What you just read is what you want to copy."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("v")
        await pilot.pause()
        assert "entry 3" in transcript.selected_text()


async def test_k_moves_the_cursor_without_growing_the_selection(sessions_dir):
    """`v` selects one entry and `j`/`k` move it. Extending on every move (vim's
    visual mode) would make a single-entry copy unreachable from the bottom of
    the transcript, which is where the cursor always starts."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("v", "k")
        await pilot.pause()
        assert "entry 2" in transcript.selected_text()
        assert "entry 3" not in transcript.selected_text()


async def test_shift_v_makes_j_and_k_grow_the_selection(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("v", "k", "V", "k")
        await pilot.pause()
        text = transcript.selected_text()
        assert "entry 1" in text and "entry 2" in text
        assert "entry 3" not in text


async def test_v_collapses_an_extended_selection_back_to_one(sessions_dir):
    """The obvious way out of an over-wide selection."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("v", "V", "k", "k")
        await pilot.pause()
        assert len(transcript.selected_text().strip().splitlines()) > 1

        await pilot.press("v")
        await pilot.pause()
        assert "entry 1" in transcript.selected_text()
        assert "entry 3" not in transcript.selected_text()


async def test_shift_v_alone_enters_select_mode(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("V")
        await pilot.pause()
        assert transcript.selecting is True


async def test_the_cursor_stops_at_the_top(sessions_dir):
    """Walking off the end must not throw or wrap around."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("v", "k", "k", "k", "k", "k", "k")
        await pilot.pause()
        assert "entry 0" in transcript.selected_text()


async def test_y_yanks_the_selection(sessions_dir, clipboard):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await _fill(app, pilot)
        await pilot.press("v", "k", "y")
        await pilot.pause()
        assert clipboard
        assert "entry 2" in clipboard[0]


async def test_yanking_leaves_select_mode(sessions_dir, clipboard):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        await pilot.press("v", "y")
        await pilot.pause()
        assert transcript.selecting is False


async def test_escape_leaves_select_mode_without_interrupting(sessions_dir):
    """`escape` is the panic stop for a paying run. Swallowing it whenever the
    transcript happened to have focus would take that away, so it only consumes
    the key while a selection is actually active."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        interrupted = []
        app.action_interrupt = lambda: interrupted.append(True)

        await pilot.press("v")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert transcript.selecting is False
        assert interrupted == [], "escape interrupted the run instead of deselecting"


async def test_escape_outside_select_mode_still_interrupts(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await _fill(app, pilot)
        interrupted = []
        app.action_interrupt = lambda: interrupted.append(True)

        await pilot.press("escape")
        await pilot.pause()
        assert interrupted == [True]


async def test_select_mode_on_an_empty_transcript_is_harmless(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = app.query_one(TranscriptView)
        transcript.clear()
        transcript.focus()
        await pilot.pause()
        await pilot.press("v", "y")
        await pilot.pause()
        assert transcript.selecting is False


# --------------------------------------------------------------------------
# Direct copy shortcuts
# --------------------------------------------------------------------------


async def test_c_copies_the_last_entry(sessions_dir, clipboard):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await _fill(app, pilot)
        await pilot.press("c")
        await pilot.pause()
        assert "entry 3" in clipboard[0]


async def test_shift_c_copies_the_last_code_block(sessions_dir, clipboard):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        transcript._add_model_text("here:\n```python\nx = 1\n```")
        await pilot.pause()
        await pilot.press("C")
        await pilot.pause()
        assert clipboard == ["x = 1"]


async def test_copying_code_copies_the_source_not_the_rendering(
    sessions_dir, clipboard
):
    """Selecting a rendered block on screen gives you wrapped lines and syntax
    colouring; the button must hand back something you can paste into a file."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot)
        source = "def f():\n    return 1"
        transcript._add_model_text(f"```python\n{source}\n```")
        await pilot.pause()
        await pilot.press("C")
        await pilot.pause()
        assert clipboard == [source]


async def test_shift_c_with_no_code_block_says_so(sessions_dir, clipboard):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await _fill(app, pilot)
        await pilot.press("C")
        await pilot.pause()
        assert clipboard == []


# --------------------------------------------------------------------------
# Code blocks as widgets
# --------------------------------------------------------------------------


async def test_fenced_code_becomes_its_own_widget(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript._add_model_text("prose\n```python\nx = 1\n```\nmore prose")
        await pilot.pause()
        assert len(transcript.query(CodeBlock)) == 1


async def test_prose_around_code_is_kept(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript._add_model_text("before\n```\nx\n```\nafter")
        await pilot.pause()
        assert "before" in transcript.text
        assert "after" in transcript.text


async def test_the_copy_button_copies_that_block(sessions_dir, clipboard):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript._add_model_text("```\nfirst\n```\ntext\n```\nsecond\n```")
        await pilot.pause()
        blocks = list(transcript.query(CodeBlock))
        blocks[0].post_message(CodeBlock.CopyRequested(blocks[0].code))
        await pilot.pause()
        assert clipboard == ["first"]


async def test_an_unknown_fence_language_does_not_crash(sessions_dir):
    """Models write fences like ```mermaid and ```text all the time."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript._add_model_text("```not-a-real-lexer\nbody\n```")
        await pilot.pause()
        assert len(transcript.query(CodeBlock)) == 1


# --------------------------------------------------------------------------
# Find still works over the new widget mix
# --------------------------------------------------------------------------


async def test_find_matches_text_inside_a_code_block(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript._add_model_text("```python\nneedle = 1\n```")
        await pilot.pause()
        assert transcript.highlight("needle") == 1


# --------------------------------------------------------------------------
# The streaming preview
# --------------------------------------------------------------------------


def _delta(text: str) -> TextDeltaEvent:
    return TextDeltaEvent(type="text_delta", text=text)


async def test_deltas_appear_as_they_arrive(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript.append(_delta("Hel"))
        await pilot.pause()
        assert "Hel" in transcript.text


async def test_deltas_accumulate_into_one_entry(sessions_dir):
    """Mounting a widget per fragment would produce hundreds of them and a
    layout that reflows on every token."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        before = len(transcript.entries)
        for part in ("Hel", "lo ", "there"):
            transcript.append(_delta(part))
        await pilot.pause()

        assert len(transcript.entries) == before + 1
        assert transcript.streaming_text == "Hello there"


async def test_the_text_event_replaces_the_preview(sessions_dir):
    """The preview is raw text; the TextEvent is the same answer rendered as
    markdown. Appending both would show every answer twice."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript.append(_delta("Hello there"))
        await pilot.pause()
        transcript.append(TextEvent(type="text", text="Hello there"))
        await pilot.pause()

        assert transcript.text.count("Hello there") == 1
        assert transcript.streaming_text == ""


async def test_a_streamed_code_block_ends_up_as_a_widget(sessions_dir):
    """The point of replacing rather than keeping the preview: the finished
    answer still gets its copy button."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        for part in ("```python\n", "x = 1\n", "```"):
            transcript.append(_delta(part))
        await pilot.pause()
        assert not transcript.query(CodeBlock)

        transcript.append(TextEvent(type="text", text="```python\nx = 1\n```"))
        await pilot.pause()
        assert len(transcript.query(CodeBlock)) == 1


async def test_a_terminal_event_also_clears_the_preview(sessions_dir):
    """A run that streams and then fails must not leave a preview stranded."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript.append(_delta("partial"))
        await pilot.pause()
        transcript.append(
            TerminalEvent(reason=TerminalReason.ERROR, detail="went wrong")
        )
        await pilot.pause()
        assert transcript.streaming_text == ""


async def test_clearing_resets_the_preview(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        transcript = await _fill(app, pilot, count=0)
        transcript.append(_delta("partial"))
        await pilot.pause()
        transcript.clear()
        await pilot.pause()

        transcript.append(_delta("fresh"))
        await pilot.pause()
        assert transcript.streaming_text == "fresh"
