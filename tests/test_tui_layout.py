"""Nothing overlaps, at any prompt size.

This exists because of a bug you could only see by looking: the completion menu,
the prompt, the hint line and the footer each carried `dock: bottom`, so all
four claimed the *same* final row. The prompt's bottom border was painted over
by the hint line, and the hint line by the footer — the app looked like it was
being cut off at the bottom of the terminal.

Geometry is checked numerically rather than by eye, because a one-row overlap is
invisible in a screenshot until someone types enough to notice.
"""

from __future__ import annotations

import argparse

import pytest
from textual.widgets import Footer

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


def _rows(app, selector) -> range:
    region = app.query_one(selector).region
    return range(region.y, region.y + region.height)


def _assert_stacked(app) -> None:
    """Every piece of bottom furniture owns its own rows, in order."""
    body = _rows(app, "#body")
    prompt = _rows(app, "#prompt")
    hints = _rows(app, "#hints")
    footer = app.query_one(Footer).region

    assert body.stop <= prompt.start, "the transcript overlaps the prompt"
    assert prompt.stop <= hints.start, (
        f"the prompt's last row {prompt.stop - 1} is overdrawn by the hint line"
    )
    assert hints.stop <= footer.y, (
        f"the hint line at row {hints.start} is overdrawn by the footer"
    )
    assert footer.y + footer.height <= app.screen.size.height


async def test_nothing_overlaps_with_an_empty_prompt(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        _assert_stacked(app)


async def test_nothing_overlaps_when_the_prompt_wraps(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.query_one(PromptArea).text = "x" * 400
        await pilot.pause()
        await pilot.pause()
        _assert_stacked(app)


async def test_nothing_overlaps_at_the_prompts_maximum_height(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.query_one(PromptArea).text = "\n".join(str(n) for n in range(200))
        await pilot.pause()
        await pilot.pause()
        _assert_stacked(app)


async def test_nothing_overlaps_with_the_completion_menu_open(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        prompt.focus()
        await pilot.press(*"@tui/")
        await pilot.pause()

        assert app.query_one(CompletionMenu).is_open, "the menu did not open"
        menu = _rows(app, "#completions")
        assert menu.stop <= _rows(app, "#prompt").start, "the menu covers the prompt"
        _assert_stacked(app)


async def test_nothing_overlaps_in_a_short_terminal(sessions_dir):
    """The furniture has to fit before the transcript does."""
    app = ForgeApp(_args())
    async with app.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        _assert_stacked(app)


async def test_nothing_overlaps_with_the_sidebar_open(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        _assert_stacked(app)


async def test_the_hint_line_is_actually_on_screen(sessions_dir):
    """It was being drawn on the footer's row, so it existed and was invisible
    — the failure that makes multi-line feel undiscoverable."""
    app = ForgeApp(_args())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        hints = app.query_one("#hints")
        assert hints.region.height == 1
        assert hints.region.y < app.query_one(Footer).region.y


async def test_the_whole_prompt_border_is_drawn(sessions_dir):
    """A three-row prompt is border, text, border. Losing the last row to an
    overlap is what made the app look cut off."""
    app = ForgeApp(_args())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea).region
        assert prompt.height >= 3
        assert prompt.y + prompt.height <= app.query_one("#hints").region.y
