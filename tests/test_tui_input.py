"""The prompt box and mouse selection — the two things that felt missing.

Both turned out to be discoverability rather than mechanics. The box *did* grow
with content, but at one line tall it is indistinguishable from a single-line
input, so nobody finds shift+enter. And drag-selection *did* work — Textual
binds `ctrl+c` to `screen.copy_text` — but that path copies via OSC 52 only
(screen.py:991), which terminals discard silently, so it looked like nothing
happened.
"""

from __future__ import annotations

import argparse

import pytest

import persistence
from tui.app import ForgeApp
from tui.prompt import PromptArea, is_incomplete
from tui.transcript import TranscriptView


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
    captured: list[str] = []
    monkeypatch.setattr(
        "tui.app.clipboard_copy",
        lambda _send, text: captured.append(text) or "copied",
    )
    return captured


# --------------------------------------------------------------------------
# Continuation — when enter should not send
# --------------------------------------------------------------------------


def test_ordinary_text_is_complete():
    assert is_incomplete("fix the config loader") is False


def test_an_open_fence_is_incomplete():
    assert is_incomplete("look at this:\n```python\nx = 1") is True


def test_a_closed_fence_is_complete():
    assert is_incomplete("```python\nx = 1\n```") is False


def test_a_balanced_pair_of_fences_is_complete():
    assert is_incomplete("here:\n```\nx = 1\n```\nthoughts?") is False


def test_a_trailing_open_bracket_is_incomplete():
    assert is_incomplete("def handler(") is True


def test_prose_with_an_unclosed_bracket_is_still_sendable():
    """"(see below" is ordinary prose. Counting every unbalanced bracket was
    the obvious implementation and would make the prompt refuse to send, which
    is far more annoying than one extra shift+enter."""
    assert is_incomplete("rewrite this (see below for why") is False


def test_an_apostrophe_does_not_block_sending():
    """Quotes are ignored entirely — apostrophes make them useless as a
    signal."""
    assert is_incomplete("don't touch the loop") is False


def test_a_smiley_does_not_block_sending():
    assert is_incomplete("thanks :)") is False


def test_empty_text_is_complete():
    assert is_incomplete("") is False
    assert is_incomplete("   \n  ") is False


async def test_enter_inside_an_open_fence_adds_a_line(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.text = "```python\nx = 1"
        prompt.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert prompt.line_count == 3, "enter sent a half-written code block"


async def test_enter_on_a_complete_prompt_still_sends(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.text = "/help"
        prompt.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert prompt.text == ""


# --------------------------------------------------------------------------
# The box is visibly multi-line
# --------------------------------------------------------------------------


async def test_the_box_grows_with_the_text(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        one = prompt.size.height
        prompt.text = "\n".join(f"line {n}" for n in range(6))
        await pilot.pause()
        await pilot.pause()
        assert prompt.size.height > one


async def test_the_box_stops_growing_before_it_eats_the_transcript(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.text = "\n".join(f"line {n}" for n in range(200))
        await pilot.pause()
        await pilot.pause()
        assert prompt.size.height <= 12


async def test_the_hint_line_advertises_a_newline_that_actually_works(sessions_dir):
    """Nobody discovers a keystroke that is never written down, and an empty
    one-line box looks exactly like a single-line input.

    It names `ctrl+j` and `\\⏎` rather than `shift+enter` on purpose: the hint
    line has one line of room, and spending it on the gesture most terminals
    cannot deliver is how you get "multi-line doesn't work". The full list,
    caveats included, is in `f1`.
    """
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        hint = str(app.query_one("#hints").content)
        assert "ctrl+j" in hint
        assert "\\" in hint


async def test_the_hint_line_counts_lines_once_multi_line(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.text = "one\ntwo\nthree"
        await pilot.pause()
        assert "3 lines" in str(app.query_one("#hints").content)


async def test_the_hint_line_explains_why_enter_is_not_sending(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.text = "```python\nx = 1"
        await pilot.pause()
        assert "unclosed" in str(app.query_one("#hints").content)


async def test_the_hint_line_explains_the_menu_while_it_is_open(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.focus()
        await pilot.press(*"/co")
        await pilot.pause()
        assert "accept" in str(app.query_one("#hints").content)


async def test_the_box_is_labelled(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "forge" in str(app.query_one(PromptArea).border_title)


# --------------------------------------------------------------------------
# Mouse selection
# --------------------------------------------------------------------------


async def test_the_transcript_allows_selection(sessions_dir):
    """Textual will not let the mouse select inside a widget that opts out."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(TranscriptView).ALLOW_SELECT is True


async def test_ctrl_c_copies_the_selection_rather_than_quitting(
    sessions_dir, clipboard, monkeypatch
):
    """The regression this whole change is about: ctrl+c on a selection used to
    fall through to interrupt/quit."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(type(app), "_selection", lambda _self: "dragged text")
        exited = []
        monkeypatch.setattr(type(app), "exit", lambda _self, *a, **k: exited.append(1))

        app.action_interrupt_or_quit()
        await pilot.pause()

        assert clipboard == ["dragged text"]
        assert exited == [], "ctrl+c quit instead of copying the selection"


async def test_ctrl_c_without_a_selection_still_quits(sessions_dir, monkeypatch):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        exited = []
        monkeypatch.setattr(type(app), "exit", lambda _self, *a, **k: exited.append(1))

        app.action_interrupt_or_quit()
        await pilot.pause()
        assert exited == [1]


async def test_the_copy_goes_through_the_native_backend(sessions_dir, monkeypatch):
    """Textual's own drag-select copy calls straight into App.copy_to_clipboard,
    so overriding that is what gives *its* path the native fallback too."""
    seen: list[str] = []
    monkeypatch.setattr(
        "tui.app.clipboard_copy", lambda _send, text: seen.append(text) or "ok"
    )
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.copy_to_clipboard("from textual")
        await pilot.pause()
        assert seen == ["from textual"]


async def test_y_prefers_a_mouse_selection_over_the_entry_cursor(
    sessions_dir, clipboard, monkeypatch
):
    """If you just highlighted something with the mouse, that is unambiguously
    what you meant to copy."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        transcript.clear()
        transcript.notice("an entry")
        transcript.focus()
        await pilot.pause()
        monkeypatch.setattr(
            type(transcript), "_mouse_selection", lambda _self: "highlighted"
        )

        await pilot.press("y")
        await pilot.pause()
        assert clipboard == ["highlighted"]
