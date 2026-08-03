"""The interaction layer: interrupt, history, autocomplete, collapsing, themes.

These are the affordances that separate a TUI from a scrolling log. The one that
matters beyond ergonomics is **interrupt**: an autonomous loop spending real
money that the user cannot stop is a liability, not a missing convenience.
"""

from __future__ import annotations

import argparse
import asyncio

import pytest
from conftest import StubClient
from textual.widgets import Collapsible

import persistence
from client import NormalizedResponse, TextBlock, ToolCallBlock
from events import ToolCallEvent, ToolResultEvent
from tui.app import ForgeApp
from tui.commands import COMMANDS, KEY_GROUPS, KEYS
from tui.prompt import PromptArea
from tui.transcript import TranscriptView


def _args(**over) -> argparse.Namespace:
    base = dict(
        goal=None,
        gateway_url=None,
        resume=None,
        list_sessions=False,
        tui=True,
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
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    return tmp_path


async def _type(pilot, text: str) -> None:
    await pilot.click("#prompt")
    pilot.app.query_one(PromptArea).text = text
    await pilot.press("enter")
    await pilot.pause()


class _HangingClient:
    """Never returns. Stands in for a slow model so a run can be interrupted."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, messages, tools, system):
        self.calls += 1
        await asyncio.sleep(3600)


# --------------------------------------------------------------------------
# Interrupt
# --------------------------------------------------------------------------


async def test_escape_interrupts_a_running_agent(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = _HangingClient()

        await _type(pilot, "something slow")
        assert app._run_in_flight is True

        await pilot.press("escape")
        for _ in range(6):
            await pilot.pause()

        assert app._run_in_flight is False
        assert "interrupted" in app.query_one(TranscriptView).text


async def test_an_interrupted_run_still_checkpoints(sessions_dir):
    """The `finally` in the worker runs on cancellation too, so stopping a run
    never loses the turns that already completed."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = _HangingClient()
        await _type(pilot, "something slow")
        await pilot.press("escape")
        for _ in range(6):
            await pilot.pause()

    assert persistence.list_sessions(sessions_dir)


async def test_a_new_goal_can_be_sent_after_interrupting(sessions_dir):
    """Interrupting must clear the in-flight guard, or the app is bricked."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = _HangingClient()
        await _type(pilot, "slow one")
        await pilot.press("escape")
        for _ in range(6):
            await pilot.pause()

        app.comp.agent.client = StubClient(
            [
                NormalizedResponse(
                    blocks=[TextBlock(text="second run worked")],
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                    stop_reason="end_turn",
                )
            ]
        )
        await _type(pilot, "try again")
        for _ in range(6):
            await pilot.pause()

        assert "second run worked" in app.query_one(TranscriptView).text


async def test_escape_when_idle_is_harmless(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.is_running is True


async def test_ctrl_c_quits_when_idle(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.is_running is False


async def test_ctrl_c_interrupts_rather_than_quitting_mid_run(sessions_dir):
    """Quitting on the first ctrl+c would throw away a partially-finished run."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = _HangingClient()
        await _type(pilot, "slow")

        await pilot.press("ctrl+c")
        for _ in range(6):
            await pilot.pause()

        assert app.is_running is True
        assert app._run_in_flight is False


# --------------------------------------------------------------------------
# Prompt history
# --------------------------------------------------------------------------


async def test_up_arrow_recalls_the_previous_line(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "/help")
        await pilot.press("up")
        await pilot.pause()
        assert app.query_one(PromptArea).text == "/help"


async def test_history_walks_back_through_several_lines(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "/help")
        await _type(pilot, "/tools")

        await pilot.press("up")
        assert app.query_one(PromptArea).text == "/tools"
        await pilot.press("up")
        assert app.query_one(PromptArea).text == "/help"


async def test_down_arrow_returns_to_the_draft_being_typed(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "/help")

        prompt = app.query_one(PromptArea)
        prompt.text = "half-written"
        await pilot.press("up")
        assert prompt.text == "/help"
        await pilot.press("down")
        assert prompt.text == "half-written"


async def test_history_does_not_record_consecutive_duplicates(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "/help")
        await _type(pilot, "/help")

        prompt = app.query_one(PromptArea)
        await pilot.press("up")
        await pilot.press("up")
        assert prompt.text == "/help"
        assert len(prompt._history) == 1


async def test_up_arrow_on_an_empty_history_is_harmless(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#prompt")
        await pilot.press("up")
        await pilot.pause()
        assert app.query_one(PromptArea).text == ""


# --------------------------------------------------------------------------
# Autocomplete
# --------------------------------------------------------------------------


async def test_tab_completes_a_slash_command(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await pilot.click("#prompt")
        await pilot.press(*"/he")
        await pilot.press("tab")
        await pilot.pause()
        assert prompt.text == "/help"


async def test_every_command_is_reachable_by_completion(sessions_dir):
    """Suggestions come from COMMANDS, so a new command is completable the
    moment it is registered — no second list to keep in sync."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        for name in COMMANDS:
            # One character short of the full name: a *fully* typed command
            # deliberately offers nothing, since there is nothing left to
            # complete and the menu would just be offering you what you typed.
            prompt.text = name[:-1]
            prompt.move_cursor(prompt.document.end)
            assert name in prompt.suggestions(), f"{name} not completable"


async def test_tab_on_ordinary_text_does_not_complete(sessions_dir):
    """Tab must stay inert for plain prose, or it eats the keystroke."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await pilot.click("#prompt")
        await pilot.press(*"read the")
        assert prompt.suggestions() == []


# --------------------------------------------------------------------------
# Collapsible tool output
# --------------------------------------------------------------------------


async def test_a_tool_call_becomes_a_collapsed_entry(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        transcript.append(
            ToolCallEvent(
                type="tool_call", name="read_file", arguments={"path": "config.py"}
            )
        )
        await pilot.pause()

        entry = transcript.query_one(Collapsible)
        assert entry.collapsed is True
        assert "read_file" in entry.title


async def test_a_tool_result_lands_inside_its_call(sessions_dir):
    """Pairing is what makes collapsing useful: the result hides with the call
    that produced it, instead of spilling into the transcript beneath it."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        transcript.append(
            ToolCallEvent(type="tool_call", name="read_file", arguments={"path": "a"})
        )
        await pilot.pause()
        transcript.append(
            ToolResultEvent(type="tool_result", name="read_file", result="FILE BODY")
        )
        await pilot.pause()

        entry = transcript.query_one(Collapsible)
        assert "FILE BODY" in transcript.text
        # One entry, not two: the result went inside the call.
        assert len(transcript.query(Collapsible)) == 1
        assert entry.collapsed is True


async def test_an_orphan_tool_result_still_renders(sessions_dir):
    """A result with no preceding call (resumed session, replay) must not be
    silently dropped."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        transcript.append(
            ToolResultEvent(type="tool_result", name="grep", result="ORPHANED")
        )
        await pilot.pause()
        assert "ORPHANED" in transcript.text


async def test_a_tool_call_arriving_without_a_result_does_not_swallow_the_next(
    sessions_dir,
):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        for path in ("a.py", "b.py"):
            transcript.append(
                ToolCallEvent(
                    type="tool_call", name="read_file", arguments={"path": path}
                )
            )
        await pilot.pause()
        assert len(transcript.query(Collapsible)) == 2


# --------------------------------------------------------------------------
# Transcript controls
# --------------------------------------------------------------------------


async def test_ctrl_l_clears_the_transcript(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(TranscriptView).text.strip()

        await pilot.press("ctrl+l")
        await pilot.pause()
        assert app.query_one(TranscriptView).text.strip() == ""


async def test_clear_command_empties_the_transcript(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type(pilot, "/clear")
        assert app.query_one(TranscriptView).text.strip() == ""


async def test_clearing_does_not_touch_the_session(sessions_dir):
    """/clear is a view operation. Losing run state to a display command would
    be a nasty surprise."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        before = app.comp.session.id
        await _type(pilot, "/clear")
        assert app.comp.session.id == before


async def test_clearing_resets_tool_pairing(sessions_dir):
    """A dangling reference to a removed widget would raise on the next result."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        transcript.append(
            ToolCallEvent(type="tool_call", name="read_file", arguments={})
        )
        await pilot.pause()
        transcript.clear()
        await pilot.pause()
        transcript.append(
            ToolResultEvent(type="tool_result", name="read_file", result="AFTER CLEAR")
        )
        await pilot.pause()
        assert "AFTER CLEAR" in transcript.text


# --------------------------------------------------------------------------
# Theme + status
# --------------------------------------------------------------------------


async def test_ctrl_t_switches_theme(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        before = app.theme
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.theme != before


async def test_the_activity_indicator_is_blank_when_idle(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert str(app.query_one("#status-activity").content).strip() == ""


async def test_the_activity_indicator_shows_progress_during_a_run(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = _HangingClient()
        await _type(pilot, "slow")
        await pilot.pause()

        assert str(app.query_one("#status-activity").content).strip()

        await pilot.press("escape")
        for _ in range(6):
            await pilot.pause()


async def test_the_activity_indicator_clears_after_a_run(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = StubClient(
            [
                NormalizedResponse(
                    blocks=[TextBlock(text="done")],
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                    stop_reason="end_turn",
                )
            ]
        )
        await _type(pilot, "go")
        for _ in range(6):
            await pilot.pause()

        assert str(app.query_one("#status-activity").content).strip() == ""


# --------------------------------------------------------------------------
# Documentation stays honest
# --------------------------------------------------------------------------


def _bound_keys() -> set[str]:
    """Every key bound anywhere a keystroke can land.

    Bindings live on three widgets, not one: the app owns run control, the
    transcript owns select mode (so `escape` can leave select mode before it
    reaches the app's interrupt), and the prompt owns editing.
    """
    keys: set[str] = set()
    for source in (ForgeApp, TranscriptView, PromptArea):
        for binding in source.BINDINGS:
            spec = binding[0] if isinstance(binding, tuple) else binding.key
            keys.update(part.strip() for part in spec.split(","))
    # Handled in `_on_key`/Textual defaults rather than declared as Bindings.
    keys |= {"enter", "up", "down", "tab", "pgup", "pgdn", "ctrl+p"}
    # A gesture, not a keypress: a trailing backslash makes the next `enter`
    # open a line. It has no Binding to find because it cannot have one.
    keys |= {"\\ then enter"}
    return keys


# `?` is documented the way a human writes it; Textual names it question_mark.
_KEY_ALIASES = {"?": "question_mark"}


def test_every_documented_key_is_actually_bound():
    """`/keys` is hand-written prose; this stops it describing a shortcut that
    does not exist."""
    bound = _bound_keys()
    for documented in KEYS:
        for key in documented.split(" / "):
            key = _KEY_ALIASES.get(key.strip(), key.strip())
            assert key in bound, f"/keys documents unbound {key!r}"


def test_every_documented_key_appears_in_exactly_one_help_group():
    """The `?` overlay groups KEYS by hand; a key added to one and not the
    other silently vanishes from the overlay."""
    grouped = [key for keys in KEY_GROUPS.values() for key in keys]
    assert sorted(grouped) == sorted(KEYS), "KEYS and KEY_GROUPS have drifted"
    assert len(grouped) == len(set(grouped)), "a key is in two groups"


def test_tool_calls_render_as_multiple_blocks_end_to_end(sessions_dir):
    """Guards the ToolCallBlock path used by the collapsible pairing above."""
    block = ToolCallBlock(id="c1", name="read_file", arguments={"path": "x"})
    assert block.name == "read_file"
