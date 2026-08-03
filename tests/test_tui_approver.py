"""The approval modal, and `a` — "stop asking me about this tool".

`a` lives entirely in the surface. `Decision` (`approval.py:34`) carries only
`approved` and `reason`, so "remember this" cannot be expressed in what the
agent receives — and shouldn't be. Whether a human wants to stop being asked is
a property of this session at this surface; the agent keeps asking exactly as it
always did, and the approver answers from memory.

The security-relevant test is `test_always_is_withheld_from_a_flagged_call`.
"""

from __future__ import annotations

import json

import pytest
from textual.app import App
from textual.widgets import Input

from approval import ApprovalRequest, Decision
from tools.base import ToolKind
from tui.approver import ConfirmScreen, TuiApprover


def _request(name: str = "write_file", *, danger: list[str] | None = None):
    return ApprovalRequest(
        name,
        {"path": "a.py", "content": "x"},
        ToolKind.WRITE,
        danger or [],
        diff="--- a/a.py\n+++ b/a.py\n-old\n+new",
    )


class _Harness(App):
    """Hosts a ConfirmScreen alone, the way the agent's worker would."""

    def __init__(self, request) -> None:
        super().__init__()
        self._request = request
        self.outcome = "unset"

    def on_mount(self) -> None:
        self.push_screen(ConfirmScreen(self._request), self._done)

    def _done(self, outcome) -> None:
        self.outcome = outcome


# --------------------------------------------------------------------------
# The modal
# --------------------------------------------------------------------------


async def test_y_approves():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
    decision, remember = app.outcome
    assert decision.approved is True
    assert remember is False


async def test_n_denies():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
    assert app.outcome[0].approved is False


async def test_escape_denies():
    """The safe default: walking away must not authorise a write."""
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.outcome[0].approved is False


async def test_a_approves_and_asks_to_be_remembered():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
    decision, remember = app.outcome
    assert decision.approved is True
    assert remember is True


async def test_the_always_key_is_advertised():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "always" in str(app.screen.query_one("#confirm-keys").content)


async def test_a_diff_is_shown_when_there_is_one():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query("#confirm-diff")


# --------------------------------------------------------------------------
# `e` — correct the call instead of denying it
# --------------------------------------------------------------------------


async def test_the_editor_is_hidden_until_asked_for():
    """Most confirmations are one keypress; a text box would be in the way."""
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query_one("#confirm-edit", Input).display is False


async def test_e_opens_the_editor_prefilled():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()

        editor = app.screen.query_one("#confirm-edit", Input)
        assert editor.display is True
        assert json.loads(editor.value) == {"path": "a.py", "content": "x"}


async def test_submitting_an_edit_returns_the_new_arguments():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#confirm-edit", Input).value = '{"path": "b.py"}'
        await pilot.press("enter")
        await pilot.pause()

    decision, _remember = app.outcome
    assert decision.approved is True
    assert decision.arguments == {"path": "b.py"}


async def test_malformed_json_is_refused_in_the_modal():
    """Rejected here rather than passed to the agent as a broken call."""
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#confirm-edit", Input).value = "{not json"
        await pilot.press("enter")
        await pilot.pause()

        assert app.outcome == "unset", "a malformed edit was sent onward"
        assert "JSON" in str(app.screen.query_one("#confirm-error").content)


async def test_a_json_value_that_is_not_an_object_is_refused():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#confirm-edit", Input).value = '["a", "list"]'
        await pilot.press("enter")
        await pilot.pause()
        assert app.outcome == "unset"


async def test_the_edit_key_is_advertised():
    app = _Harness(_request())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "edit" in str(app.screen.query_one("#confirm-keys").content)


async def test_editing_is_offered_even_on_a_flagged_call():
    """Unlike `a`, editing is how you *fix* a dangerous call — and whatever
    comes back is re-checked by the agent anyway."""
    app = _Harness(_request("run_shell", danger=["recursive delete"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        assert app.screen.query_one("#confirm-edit", Input).display is True


# --------------------------------------------------------------------------
# The danger carve-out
# --------------------------------------------------------------------------


async def test_always_is_withheld_from_a_flagged_call():
    """Silencing the prompt for a call that tripped a danger check would disarm
    the check for the rest of the session — the one case where being asked
    every time is the entire point."""
    app = _Harness(_request("run_shell", danger=["recursive delete"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.offers_always is False
        await pilot.press("a")
        await pilot.pause()
        assert app.outcome == "unset", "a dangerous call was always-approved"


async def test_a_flagged_call_does_not_advertise_always():
    app = _Harness(_request("run_shell", danger=["recursive delete"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "always" not in str(app.screen.query_one("#confirm-keys").content)


async def test_the_danger_reason_is_shown():
    app = _Harness(_request("run_shell", danger=["recursive delete"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        shown = str(app.screen.query_one("#confirm-danger").content)
        assert "recursive delete" in shown


# --------------------------------------------------------------------------
# TuiApprover — the remembering
# --------------------------------------------------------------------------


class _StubApp:
    """Records what was pushed and answers with a scripted outcome."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.asked = 0

    async def push_screen_wait(self, _screen):
        self.asked += 1
        return self._outcomes.pop(0)


async def test_a_plain_approval_is_not_remembered():
    app = _StubApp([(Decision(approved=True), False)] * 2)
    approver = TuiApprover(app)

    await approver.decide(_request())
    await approver.decide(_request())
    assert app.asked == 2


async def test_always_suppresses_the_second_prompt():
    app = _StubApp([(Decision(approved=True), True)])
    approver = TuiApprover(app)

    first = await approver.decide(_request())
    second = await approver.decide(_request())

    assert app.asked == 1, "the second call still prompted"
    assert first.approved and second.approved


async def test_always_covers_only_the_tool_it_was_granted_for():
    app = _StubApp([(Decision(approved=True), True), (Decision(approved=True), False)])
    approver = TuiApprover(app)

    await approver.decide(_request("write_file"))
    await approver.decide(_request("run_shell"))
    assert app.asked == 2


async def test_a_remembered_tool_is_still_asked_about_when_flagged():
    """`a` was granted on an ordinary call; a later call that trips a danger
    check is a different question and must be asked again."""
    app = _StubApp([(Decision(approved=True), True), (Decision(approved=False), False)])
    approver = TuiApprover(app)

    await approver.decide(_request("run_shell"))
    await approver.decide(_request("run_shell", danger=["recursive delete"]))
    assert app.asked == 2


async def test_the_remembered_set_is_visible():
    app = _StubApp([(Decision(approved=True), True)])
    approver = TuiApprover(app)
    await approver.decide(_request("write_file"))
    assert approver.always_allowed == frozenset({"write_file"})


async def test_reset_clears_the_remembered_set():
    app = _StubApp([(Decision(approved=True), True), (Decision(approved=True), False)])
    approver = TuiApprover(app)

    await approver.decide(_request())
    approver.reset()
    await approver.decide(_request())
    assert app.asked == 2


@pytest.mark.parametrize("approved", [True, False])
async def test_the_decision_reaches_the_caller_unchanged(approved):
    app = _StubApp([(Decision(approved=approved, reason="because"), False)])
    decision = await TuiApprover(app).decide(_request())
    assert decision.approved is approved
    assert decision.reason == "because"
