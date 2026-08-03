"""The `@file` / `/command` completion popup, and copy-on-select.

The matching was never broken — `tui/files.py` indexes the repo and ranks
correctly, and `tab` always worked. What was missing was any *visible* menu, so
the feature read as absent. These tests pin the popup's behaviour: it opens by
itself, the arrows walk it, enter and tab take a row, escape dismisses it
without interrupting the agent.
"""

from __future__ import annotations

import argparse

import pytest

import persistence
from tui.app import ForgeApp
from tui.completions import CompletionMenu
from tui.prompt import PromptArea


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


async def _type(pilot, text: str) -> None:
    pilot.app.query_one(PromptArea).focus()
    await pilot.pause()
    await pilot.press(*text)
    await pilot.pause()


def _menu(app) -> CompletionMenu:
    return app.query_one(CompletionMenu)


# --------------------------------------------------------------------------
# The menu as a widget — no app needed
# --------------------------------------------------------------------------


def test_a_new_menu_is_closed():
    assert CompletionMenu().is_open is False


def test_showing_nothing_keeps_it_closed():
    menu = CompletionMenu()
    menu.show([])
    assert menu.is_open is False


def test_current_is_none_when_closed():
    assert CompletionMenu().current is None


async def test_move_wraps_around(sessions_dir):
    """A short menu is faster to cycle than to walk to the end of."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        menu = _menu(app)
        menu.show(["a", "b", "c"])
        await pilot.pause()
        menu.highlighted = 2
        menu.move(1)
        assert menu.highlighted == 0
        menu.move(-1)
        assert menu.highlighted == 2


async def test_retyping_the_same_prefix_keeps_your_highlight(sessions_dir):
    """Rebuilding the list on every keystroke would yank the cursor back to the
    top mid-selection."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        menu = _menu(app)
        menu.show(["a", "b", "c"])
        await pilot.pause()
        menu.highlighted = 2
        menu.show(["a", "b", "c"])
        assert menu.highlighted == 2


async def test_a_fully_typed_path_offers_nothing(sessions_dir):
    """Otherwise accepting a completion re-opens the menu offering you what you
    just chose, and you have to press escape to dismiss it."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.focus()
        await pilot.press(*"@tui/app.py")
        await pilot.pause()
        assert prompt.suggestions() == []


async def test_a_fully_typed_command_offers_nothing(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.focus()
        await pilot.press(*"/help")
        await pilot.pause()
        assert prompt.suggestions() == []


# --------------------------------------------------------------------------
# It opens by itself
# --------------------------------------------------------------------------


async def test_the_menu_is_hidden_with_an_empty_prompt(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert _menu(app).is_open is False


async def test_typing_a_slash_opens_the_menu(sessions_dir):
    """Waiting for tab was the bug: nobody presses a key to reveal a list they
    have no reason to think exists."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "/co")
        assert _menu(app).is_open is True


async def test_typing_an_at_sign_opens_the_menu(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@tui/ap")
        assert _menu(app).is_open is True
        assert "tui/app.py" in _menu(app)._values


async def test_ordinary_text_leaves_the_menu_closed(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "fix the loader")
        assert _menu(app).is_open is False


async def test_the_menu_closes_when_the_mention_no_longer_matches(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@zzzzzznotafile")
        assert _menu(app).is_open is False


async def test_the_menu_is_bounded(sessions_dir):
    """Taller than a handful of rows and it covers the conversation you are
    writing about."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@")
        assert _menu(app).option_count <= 8


# --------------------------------------------------------------------------
# Driving it
# --------------------------------------------------------------------------


async def test_down_walks_the_menu_instead_of_the_history(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@tui/")
        await pilot.press("down")
        await pilot.pause()
        assert _menu(app).highlighted == 1


async def test_enter_accepts_the_highlighted_row(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@tui/ap")
        chosen = _menu(app).current
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(PromptArea).text == f"@{chosen}"


async def test_enter_accepts_rather_than_sending(sessions_dir):
    """With the menu open, enter must complete — sending a half-typed path is
    never what was meant."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@tui/ap")
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(PromptArea).text != ""


async def test_tab_also_accepts(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@tui/ap")
        await pilot.press("tab")
        await pilot.pause()
        assert app.query_one(PromptArea).text.startswith("@tui/app")


async def test_accepting_closes_the_menu(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "@tui/ap")
        await pilot.press("enter")
        await pilot.pause()
        assert _menu(app).is_open is False


async def test_a_mention_keeps_the_text_before_it(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "explain @tui/ap")
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(PromptArea).text.startswith("explain @tui/app")


async def test_escape_dismisses_the_menu_without_interrupting(sessions_dir):
    """`escape` is the panic stop for a paying run; a dropdown must not eat it
    permanently, only while it is open."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        interrupted = []
        app.action_interrupt = lambda: interrupted.append(True)

        await _type(pilot, "@tui/ap")
        await pilot.press("escape")
        await pilot.pause()

        assert _menu(app).is_open is False
        assert interrupted == []


async def test_escape_with_no_menu_still_interrupts(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        interrupted = []
        app.action_interrupt = lambda: interrupted.append(True)

        await _type(pilot, "hello")
        await pilot.press("escape")
        await pilot.pause()
        assert interrupted == [True]


async def test_history_still_works_when_the_menu_is_closed(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.remember("an earlier prompt")
        prompt.focus()
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert prompt.text == "an earlier prompt"


# --------------------------------------------------------------------------
# Copy on select — no ctrl+c
# --------------------------------------------------------------------------


async def test_finishing_a_selection_copies_immediately(sessions_dir, clipboard):
    """What a terminal does, and what "select to copy" means."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._selection = lambda: "dragged text"
        app.on_text_selected(None)
        await pilot.pause()
        assert clipboard == ["dragged text"]


async def test_a_plain_click_copies_nothing(sessions_dir, clipboard):
    """TextSelected fires on every mouse-up; a click clears the selection
    first, so an empty selection is the click filter."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.on_text_selected(None)
        await pilot.pause()
        assert clipboard == []


async def test_autocopy_can_be_turned_off(sessions_dir, clipboard):
    """One toast per drag is right when copying and wrong when highlighting to
    read."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.text = "/autocopy"
        prompt.post_message(PromptArea.Submitted("/autocopy"))
        await pilot.pause()
        await pilot.pause()

        assert app._autocopy is False
        app._selection = lambda: "dragged text"
        app.on_text_selected(None)
        await pilot.pause()
        assert clipboard == []


# --------------------------------------------------------------------------
# Newline keys
# --------------------------------------------------------------------------


async def test_ctrl_j_inserts_a_newline(sessions_dir):
    """The reliable one: most terminals send an identical byte for enter and
    shift+enter, so only ctrl+j is guaranteed to reach us."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "one")
        await pilot.press("ctrl+j")
        await _type(pilot, "two")
        assert app.query_one(PromptArea).line_count == 2


async def test_ctrl_j_does_not_send(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "hello")
        await pilot.press("ctrl+j")
        await pilot.pause()
        assert app.query_one(PromptArea).text != ""


async def test_alt_enter_also_inserts_a_newline(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "one")
        await pilot.press("alt+enter")
        await pilot.pause()
        assert app.query_one(PromptArea).line_count == 2


def test_the_keys_list_admits_shift_enter_is_unreliable():
    """Documenting a keystroke that silently does nothing in the user's
    terminal is worse than not documenting it."""
    from tui.commands import KEYS

    assert "kitty" in KEYS["shift+enter"]
    assert "every terminal" in KEYS["ctrl+j"]
    assert "everywhere" in KEYS["\\ then enter"]


# --------------------------------------------------------------------------
# Backslash continuation — the newline that needs no terminal support
# --------------------------------------------------------------------------


def test_a_trailing_backslash_wants_continuation():
    from tui.prompt import wants_continuation

    assert wants_continuation("write a test \\") is True


def test_trailing_spaces_after_the_backslash_are_tolerated():
    """People type a space before hitting enter constantly."""
    from tui.prompt import wants_continuation

    assert wants_continuation("write a test \\   ") is True


def test_ordinary_text_does_not_want_continuation():
    from tui.prompt import wants_continuation

    assert wants_continuation("write a test") is False


def test_a_backslash_mid_line_does_not_want_continuation():
    """`C:\\Users` in the middle of a sentence must still send."""
    from tui.prompt import wants_continuation

    assert wants_continuation("look at C:\\Users\\hp and report") is False


async def test_continuation_leaves_the_caret_ready_to_type(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "first \\")
        await pilot.press("enter")
        await _type(pilot, "second")
        assert app.query_one(PromptArea).text == "first\nsecond"
