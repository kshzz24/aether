"""The TUI's Approver: a modal that returns a Decision into the agent loop.

This is the phase's only *bidirectional* flow. Everything else is events flowing
out of `agent.run`; here a `Decision` flows back in, through the `Approver`
Protocol that Phase 5 already injected into the agent (`approval.py:40`). No
core change was needed to make this work — that seam was built for exactly this.

`push_screen_wait` must be awaited from a worker context. It is: `decide()` is
called from inside `agent.run`, which the app drives as a Textual worker. Await
it from the main event loop instead and the app deadlocks.

**`a` (always) lives here, not in the core.** `Decision` carries only
`approved` and `reason`, so "remember this" cannot be expressed in the value the
agent receives — and it shouldn't be. Whether a human wants to stop being asked
is a property of this session at this surface, so the approver holds it and the
agent keeps asking exactly as it always did.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from approval import ApprovalRequest, Decision

if TYPE_CHECKING:
    from textual.app import App

_MAX_DIFF_LINES = 40

# (decision, remember-this-tool-for-the-session)
ConfirmOutcome = tuple[Decision, bool]


def _format_args(arguments: dict) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = str(value).replace("\n", "\\n")
        if len(rendered) > 70:
            rendered = rendered[:70] + "..."
        parts.append(f"{key} = {rendered}")
    return "\n".join(parts) or "(no arguments)"


def _clip_diff(diff: str) -> str:
    lines = diff.splitlines()
    if len(lines) <= _MAX_DIFF_LINES:
        return diff
    kept = lines[:_MAX_DIFF_LINES]
    kept.append(f"... [{len(lines) - _MAX_DIFF_LINES} more lines]")
    return "\n".join(kept)


class ConfirmScreen(ModalScreen[ConfirmOutcome]):
    """Asks the human; dismisses with the Decision the agent loop is awaiting."""

    BINDINGS = [
        ("y", "approve", "allow"),
        ("a", "always", "always"),
        ("e", "edit", "edit"),
        ("n", "deny", "deny"),
        ("escape", "deny", "deny"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self._request = request

    @property
    def offers_always(self) -> bool:
        """`a` is withheld from flagged calls.

        Letting someone silence the prompt for a tool call that tripped a danger
        check would disarm the check for the rest of the session — the one case
        where being asked every time is the entire point.
        """
        return not self._request.danger_reasons

    def compose(self) -> ComposeResult:
        req = self._request
        with VerticalScroll(id="confirm-body"):
            yield Static(
                Text(f"{req.tool_name}  ({req.kind.name.lower()})", style="bold"),
                id="confirm-title",
            )
            yield Static(Text(_format_args(req.arguments)), id="confirm-args")
            if req.danger_reasons:
                yield Static(
                    Text("! " + "; ".join(req.danger_reasons), style="bold yellow"),
                    id="confirm-danger",
                )
            if req.diff is not None:
                yield Static(
                    Syntax(_clip_diff(req.diff), "diff", theme="ansi_dark"),
                    id="confirm-diff",
                )

            yield Input(id="confirm-edit")
            yield Static(Text(""), id="confirm-error")

            keys = Text.assemble(("  y  ", "bold green"), "allow      ")
            if self.offers_always:
                keys.append_text(
                    Text.assemble(
                        ("  a  ", "bold green"),
                        f"always {req.tool_name}      ",
                    )
                )
            keys.append_text(Text.assemble(("  e  ", "bold yellow"), "edit      "))
            keys.append_text(Text.assemble(("  n  ", "bold red"), "deny"))
            yield Static(keys, id="confirm-keys")

    def on_mount(self) -> None:
        # The editor is a second step, not the default view: most confirmations
        # are a single keypress and a text box would be in the way.
        self.query_one("#confirm-edit", Input).display = False
        self.query_one("#confirm-error", Static).display = False

    def action_approve(self) -> None:
        self.dismiss((Decision(approved=True), False))

    def action_always(self) -> None:
        if not self.offers_always:
            self.app.bell()
            return
        self.dismiss((Decision(approved=True), True))

    def action_edit(self) -> None:
        """Correct the call instead of denying it.

        Denying makes the model guess again, which costs a turn and often
        produces the same call. Editing is the shorter path — and the agent
        re-validates and re-checks whatever comes back (`agent.py`), so this is
        no more powerful than making the call yourself.
        """
        editor = self.query_one("#confirm-edit", Input)
        editor.display = True
        editor.value = json.dumps(self._request.arguments)
        editor.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            arguments = json.loads(event.value)
        except json.JSONDecodeError as exc:
            self._complain(f"not valid JSON: {exc.msg}")
            return
        if not isinstance(arguments, dict):
            self._complain("arguments must be a JSON object, e.g. {\"path\": \"a.py\"}")
            return
        self.dismiss((Decision(approved=True, arguments=arguments), False))

    def _complain(self, message: str) -> None:
        """Reject malformed input here rather than passing it to the agent."""
        error = self.query_one("#confirm-error", Static)
        error.display = True
        error.update(Text(message, style="bold red"))

    def action_deny(self) -> None:
        self.dismiss((Decision(approved=False, reason="user declined"), False))


class TuiApprover:
    """Implements the `Approver` Protocol against a running Textual app."""

    def __init__(self, app: App) -> None:
        self._app = app
        # Tools the human said "always" to, for this session only. Never
        # persisted: an approval that outlives the session you granted it in is
        # a policy change disguised as a keystroke.
        self._always: set[str] = set()

    @property
    def always_allowed(self) -> frozenset[str]:
        return frozenset(self._always)

    def reset(self) -> None:
        self._always.clear()

    async def decide(self, request: ApprovalRequest) -> Decision:
        # A remembered tool is still re-checked for danger: `a` was granted on
        # an ordinary call and must not cover a later flagged one.
        if request.tool_name in self._always and not request.danger_reasons:
            return Decision(approved=True, reason="always allowed this session")

        decision, remember = await self._app.push_screen_wait(ConfirmScreen(request))
        if remember:
            self._always.add(request.tool_name)
        return decision
