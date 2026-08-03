"""Session-level TUI features: multi-line input, the task panel, runtime
switching of model/provider/approval, cost, plan mode, compaction, find, copy.

The theme running through these: a long task should not need a restart. Changing
model, tightening approval, or reclaiming context mid-run all preserve the
conversation rather than starting over.
"""

from __future__ import annotations

import argparse

import pytest
from conftest import StubClient

import persistence
from approval import ApprovalMode
from client import Message, NormalizedResponse, TextBlock
from tui.app import ForgeApp
from tui.prompt import PromptArea
from tui.todos import TodoPanel
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
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")
    return tmp_path


async def _run_command(pilot, text: str) -> None:
    pilot.app.query_one(PromptArea).text = text
    pilot.app.query_one(PromptArea).post_message(PromptArea.Submitted(text))
    await pilot.pause()
    await pilot.pause()


def _say(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="end_turn",
    )


# --------------------------------------------------------------------------
# Multi-line input
# --------------------------------------------------------------------------


async def test_shift_enter_inserts_a_newline_when_the_terminal_can_send_it(
    sessions_dir,
):
    """CAUTION: this passes in more terminals than it works in.

    `pilot.press("shift+enter")` synthesizes a key Textual can only produce from
    the Kitty keyboard protocol (`_xterm_parser.py:392`). Terminals without it
    deliver a bare `enter` that is indistinguishable from a send, and no amount
    of application code recovers the difference. The gesture people can actually
    rely on is the backslash continuation below, and `ctrl+j`.
    """
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await pilot.click("#prompt")
        await pilot.press(*"line one")
        await pilot.press("shift+enter")
        await pilot.press(*"line two")
        await pilot.pause()

        assert prompt.text == "line one\nline two"
        assert app._run_in_flight is False  # nothing was submitted


async def test_a_trailing_backslash_opens_a_new_line(sessions_dir):
    """The newline gesture that needs no terminal support at all."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await pilot.click("#prompt")
        await pilot.press(*"line one \\")
        await pilot.press("enter")
        await pilot.press(*"line two")
        await pilot.pause()

        assert prompt.text == "line one\nline two"
        assert app._run_in_flight is False, "the backslash line was submitted"


async def test_the_backslash_itself_is_not_sent(sessions_dir):
    """It was punctuation asking for a newline, not part of the message —
    exactly as a shell treats it."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await pilot.click("#prompt")
        await pilot.press(*"ask \\")
        await pilot.press("enter")
        await pilot.pause()
        assert "\\" not in prompt.text


async def test_enter_submits_a_multi_line_prompt_whole(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = StubClient([_say("got it")])
        prompt = app.query_one(PromptArea)

        await pilot.click("#prompt")
        await pilot.press(*"first")
        await pilot.press("shift+enter")
        await pilot.press(*"second")
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

        assert prompt.text == ""
        sent = app.comp.agent.messages[0].blocks[0].text
        assert "first\nsecond" in sent


async def test_up_arrow_moves_the_caret_before_it_walks_history(sessions_dir):
    """On a multi-line prompt, `up` must move within the text first — stealing
    it for history would make multi-line editing impossible."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        await pilot.click("#prompt")
        await pilot.press(*"one")
        await pilot.press("shift+enter")
        await pilot.press(*"two")

        await pilot.press("up")
        await pilot.pause()
        assert prompt.text == "one\ntwo"       # unchanged
        assert prompt.cursor_location[0] == 0  # caret moved up a line


# --------------------------------------------------------------------------
# The task panel
# --------------------------------------------------------------------------


async def test_the_task_panel_is_hidden_until_there_are_tasks(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(TodoPanel).display is False


async def test_adding_a_todo_reveals_and_fills_the_panel(sessions_dir):
    """The panel observes the same store the tool writes to — no polling, and
    no backchannel into the agent core."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.comp.registry.get("todo").run(
            {"action": "add", "text": "ship the panel"}
        )
        await pilot.pause()

        panel = app.query_one(TodoPanel)
        assert panel.display is True
        assert "ship the panel" in str(panel.content)


async def test_completing_a_todo_updates_the_panel(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        todo = app.comp.registry.get("todo")
        await todo.run({"action": "add", "text": "a task"})
        await todo.run({"action": "complete", "id": 1})
        await pilot.pause()

        assert "1/1" in str(app.query_one(TodoPanel).content)


async def test_the_todo_command_shows_the_same_list(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.comp.registry.get("todo").run({"action": "add", "text": "visible"})
        await _run_command(pilot, "/todo")
        assert "visible" in app.query_one(TranscriptView).text


# --------------------------------------------------------------------------
# Runtime switching
# --------------------------------------------------------------------------


async def test_model_switches_and_shows_in_the_status_bar(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _run_command(pilot, "/model llama-3.3-70b")
        assert app.comp.config.model == "llama-3.3-70b"
        assert "llama-3.3-70b" in str(app.query_one("#status-model").content)


async def test_switching_model_keeps_the_conversation(sessions_dir):
    """Changing model mid-task should continue the task, not restart it."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.messages = [
            Message(role="user", blocks=[TextBlock(text="earlier turn")])
        ]
        await _run_command(pilot, "/model other-model")

        assert app.comp.agent.messages[0].blocks[0].text == "earlier turn"


async def test_switching_model_keeps_the_task_list(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.comp.registry.get("todo").run({"action": "add", "text": "survives"})
        await _run_command(pilot, "/model other-model")

        assert [t.text for t in app.comp.todos.list()] == ["survives"]


async def test_provider_switches(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _run_command(pilot, "/provider openai")
        assert app.comp.config.provider == "openai"


async def test_an_unknown_provider_is_rejected_without_rebuilding(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _run_command(pilot, "/provider notreal")
        assert app.comp.config.provider == "groq"
        assert "unknown provider" in app.query_one(TranscriptView).text


async def test_approval_mode_switches_and_takes_effect(sessions_dir):
    """The config value is cosmetic; the policy the agent consults is the point."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _run_command(pilot, "/approval never")
        assert app.comp.agent.policy.mode is ApprovalMode.NEVER


async def test_bare_switch_commands_report_the_current_value(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _run_command(pilot, "/model")
        assert "stub-model" in app.query_one(TranscriptView).text


# --------------------------------------------------------------------------
# Plan mode
# --------------------------------------------------------------------------


async def test_plan_mode_forces_approval_on(sessions_dir):
    """The prompt preamble is advice; the policy is the guarantee."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.policy.mode = ApprovalMode.AUTO

        await _run_command(pilot, "/plan")
        assert app._plan_mode is True
        assert app.comp.agent.policy.mode is ApprovalMode.ON_REQUEST


async def test_plan_mode_adds_a_preamble_to_the_system_prompt(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _run_command(pilot, "/plan")
        assert "PLAN MODE IS ON" in app.comp.agent.system


async def test_leaving_plan_mode_restores_the_previous_policy(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.policy.mode = ApprovalMode.AUTO

        await _run_command(pilot, "/plan")
        await _run_command(pilot, "/plan")

        assert app._plan_mode is False
        assert app.comp.agent.policy.mode is ApprovalMode.AUTO
        assert "PLAN MODE IS ON" not in app.comp.agent.system


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


async def test_cost_reports_no_turns_before_a_run(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _run_command(pilot, "/cost")
        assert "no turns yet" in app.query_one(TranscriptView).text


async def test_cost_breaks_down_by_turn_after_a_run(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = StubClient([_say("done")])
        await _run_command(pilot, "run something")
        for _ in range(6):
            await pilot.pause()

        await _run_command(pilot, "/cost")
        text = app.query_one(TranscriptView).text
        assert "total" in text
        assert "mean" in text


# --------------------------------------------------------------------------
# Compaction, find, copy
# --------------------------------------------------------------------------


async def test_compact_with_nothing_to_drop_says_so(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = StubClient([_say("summary")])
        await _run_command(pilot, "/compact")
        for _ in range(6):
            await pilot.pause()

        assert "compact" in app.query_one(TranscriptView).text


async def test_compact_shortens_a_long_transcript(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.messages = [
            Message(role="user" if i % 2 == 0 else "assistant",
                    blocks=[TextBlock(text=f"turn {i}")])
            for i in range(20)
        ]
        app.comp.agent.client = StubClient([_say("a summary of the middle")])

        await _run_command(pilot, "/compact")
        for _ in range(8):
            await pilot.pause()

        assert len(app.comp.agent.messages) < 20


async def test_find_dims_non_matching_entries(sessions_dir):
    """Dimming, not filtering: hiding the steps between matches would make the
    transcript lie about what happened."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        transcript.notice("needle in here")
        transcript.notice("nothing relevant")
        await pilot.pause()

        await _run_command(pilot, "/find needle")
        faded = [c for c in transcript.children if c.has_class("faded")]
        assert faded, "nothing was dimmed"
        assert not any(
            "needle in here" in str(getattr(c, "content", "")) for c in faded
        )


async def test_an_empty_find_clears_the_highlight(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        transcript.notice("something")
        await pilot.pause()

        await _run_command(pilot, "/find zzz")
        await _run_command(pilot, "/find")
        assert not any(c.has_class("faded") for c in transcript.children)


async def test_copy_puts_the_transcript_on_the_clipboard(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(TranscriptView).notice("copy me please")
        await pilot.pause()

        await _run_command(pilot, "/copy")
        assert "copy me please" in app.clipboard


async def test_reindex_rebuilds_the_file_index(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(PromptArea).file_index = []
        await _run_command(pilot, "/reindex")
        assert app.query_one(PromptArea).file_index
