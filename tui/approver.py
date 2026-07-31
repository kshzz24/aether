"""The TUI's Approver: a modal that returns a Decision into the agent loop.

This is the phase's only *bidirectional* flow. Everything else is events flowing
out of `agent.run`; here a `Decision` flows back in, through the `Approver`
Protocol that Phase 5 already injected into the agent (`approval.py:40`). No
core change was needed to make this work — that seam was built for exactly this.

`push_screen_wait` must be awaited from a worker context. It is: `decide()` is
called from inside `agent.run`, which the app drives as a Textual worker. Await
it from the main event loop instead and the app deadlocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from approval import ApprovalRequest, Decision

if TYPE_CHECKING:
    from textual.app import App

_MAX_DIFF_LINES = 40


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


class ConfirmScreen(ModalScreen[Decision]):
    """Asks the human; dismisses with the Decision the agent loop is awaiting."""

    BINDINGS = [
        ("y", "approve", "allow"),
        ("n", "deny", "deny"),
        ("escape", "deny", "deny"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self._request = request

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
            yield Static(
                Text.assemble(
                    ("  y  ", "bold green"), "allow      ",
                    ("  n  ", "bold red"), "deny",
                ),
                id="confirm-keys",
            )

    def action_approve(self) -> None:
        self.dismiss(Decision(approved=True))

    def action_deny(self) -> None:
        self.dismiss(Decision(approved=False, reason="user declined"))


class TuiApprover:
    """Implements the `Approver` Protocol against a running Textual app."""

    def __init__(self, app: App) -> None:
        self._app = app

    async def decide(self, request: ApprovalRequest) -> Decision:
        return await self._app.push_screen_wait(ConfirmScreen(request))
