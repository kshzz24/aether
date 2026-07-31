"""ForgeApp integration tests, driven by Textual's built-in `run_test` pilot.

No snapshot testing: `pytest-textual-snapshot` pins rendered output and breaks on
every styling tweak, which makes it a poor gate for behaviour. These assert what
the app *does* — runs the agent, routes commands, returns Decisions — not how it
looks.
"""

from __future__ import annotations

import argparse

import pytest
from conftest import StubClient
from textual.app import App

import persistence
from approval import ApprovalRequest, Decision
from client import NormalizedResponse, TextBlock
from tools.base import ToolKind
from tui.app import ForgeApp
from tui.approver import ConfirmScreen, TuiApprover
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
    """Isolate the app: temp sessions dir, and a dummy key so client
    construction succeeds. Every test swaps in a StubClient before any call is
    made, so the key is never used — the OpenAI-compatible SDK just refuses to
    be constructed without one."""
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    return tmp_path


def _transcript_text(app: App) -> str:
    return app.query_one(TranscriptView).text


def _say(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.0,
        stop_reason="end_turn",
    )


# --------------------------------------------------------------------------
# Boot
# --------------------------------------------------------------------------


async def test_app_boots_and_announces_the_session(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.comp is not None
        assert "session" in _transcript_text(app)


async def test_status_bar_shows_provider_and_model(sessions_dir):
    app = ForgeApp(_args(provider="groq", model="llama-3.3"))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "llama-3.3" in str(app.query_one("#status-model").content)


async def test_a_bad_resume_id_disables_input_instead_of_crashing(sessions_dir):
    app = ForgeApp(_args(resume="no-such-session"))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.comp is None
        assert "cannot resume" in _transcript_text(app)


# --------------------------------------------------------------------------
# Driving the agent
# --------------------------------------------------------------------------


async def test_submitting_a_goal_runs_the_agent_and_renders_its_text(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = StubClient([_say("the answer is 42")])

        await pilot.click("#prompt")
        await pilot.press(*"hello")
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

        assert "the answer is 42" in _transcript_text(app)


async def test_a_completed_run_checkpoints_the_session(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = StubClient([_say("done")])

        await pilot.click("#prompt")
        await pilot.press(*"build it")
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

    assert persistence.list_sessions(sessions_dir), "run did not checkpoint"


async def test_the_first_goal_becomes_the_session_goal(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.comp.agent.client = StubClient([_say("ok")])

        await pilot.click("#prompt")
        await pilot.press(*"ship it")
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

        assert app.comp.session.goal == "ship it"


# --------------------------------------------------------------------------
# Commands vs goals
# --------------------------------------------------------------------------


async def test_a_slash_command_renders_without_starting_a_run(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        # No StubClient installed: if /tools started a run, the real client
        # would be called and the test would error rather than pass.
        await pilot.click("#prompt")
        await pilot.press(*"/tools")
        await pilot.press("enter")
        await pilot.pause()

        assert "tools" in _transcript_text(app)
        assert app._run_in_flight is False


async def test_the_input_is_cleared_after_submitting(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#prompt")
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one("#prompt").text == ""


async def test_quit_command_exits_the_app(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#prompt")
        await pilot.press(*"/quit")
        await pilot.press("enter")
        await pilot.pause()

    assert app.is_running is False


# --------------------------------------------------------------------------
# The approver round-trip — the one bidirectional flow
# --------------------------------------------------------------------------


class _ApproverHarness(App):
    """Minimal app that asks once, exactly as the agent loop would."""

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self._request = request
        self.decision: Decision | None = None

    def on_mount(self) -> None:
        self._ask()

    @staticmethod
    def _request_for(**over) -> ApprovalRequest:
        base = dict(
            tool_name="write_file",
            arguments={"path": "main.py"},
            kind=ToolKind.WRITE,
            danger_reasons=[],
            diff=None,
        )
        base.update(over)
        return ApprovalRequest(**base)

    def _ask(self) -> None:
        self.run_worker(self._ask_worker(), exclusive=True)

    async def _ask_worker(self) -> None:
        self.decision = await TuiApprover(self).decide(self._request)


async def _decide_with(request: ApprovalRequest, key: str) -> Decision:
    app = _ApproverHarness(request)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen), "modal never appeared"
        await pilot.press(key)
        for _ in range(4):
            await pilot.pause()
    return app.decision


async def test_pressing_y_returns_an_approval():
    """The deadlock test. `push_screen_wait` awaited outside worker context
    never returns, and the app hangs with no error — so this is the first
    thing to check when touching the approver."""
    decision = await _decide_with(_ApproverHarness._request_for(), "y")
    assert decision is not None, "decide() never returned — deadlock"
    assert decision.approved is True


async def test_pressing_n_returns_a_denial_with_a_reason():
    decision = await _decide_with(_ApproverHarness._request_for(), "n")
    assert decision.approved is False
    assert decision.reason


async def test_escape_denies():
    decision = await _decide_with(_ApproverHarness._request_for(), "escape")
    assert decision.approved is False


async def test_the_modal_shows_the_diff_when_one_is_present():
    request = _ApproverHarness._request_for(
        diff="--- a/main.py\n+++ b/main.py\n+import mcpclient\n"
    )
    app = _ApproverHarness(request)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query("#confirm-diff")
        await pilot.press("n")
        await pilot.pause()


async def test_the_modal_shows_danger_reasons():
    request = _ApproverHarness._request_for(
        tool_name="run_shell",
        kind=ToolKind.EXECUTE,
        danger_reasons=["destructive shell command"],
    )
    app = _ApproverHarness(request)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query("#confirm-danger")
        await pilot.press("n")
        await pilot.pause()
