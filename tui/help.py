"""The `?` / `f1` shortcuts overlay.

`/keys` prints the same information into the transcript, which is fine when you
already know the command exists. This is for when you don't: a centred modal you
can reach by pressing the key people press when they are lost.

`?` is a printable character, so it is bound on the **transcript**, not the app —
typing a question mark into the prompt must produce a question mark. `f1` is
bound app-wide because it is not printable anywhere.

The content comes from `tui.commands.KEYS`, the same dict `/keys` reads, so the
two can never describe different shortcuts.
"""

from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from tui.commands import KEY_GROUPS, KEYS, MOUSE_TIPS


def _table(keys: tuple[str, ...]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold", min_width=12)
    table.add_column()
    for key in keys:
        table.add_row(key, KEYS.get(key, ""))
    return table


class HelpScreen(ModalScreen[None]):
    """Every shortcut, grouped, over a dimmed transcript."""

    BINDINGS = [
        ("escape", "close", "close"),
        ("question_mark", "close", "close"),
        ("f1", "close", "close"),
        ("q", "close", "close"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-body"):
            yield Label("Keyboard shortcuts", id="help-title")
            for group, keys in KEY_GROUPS.items():
                yield Label(group, classes="help-group")
                yield Static(_table(keys), classes="help-table")

            yield Label("Mouse", classes="help-group")
            mouse = Table.grid(padding=(0, 2))
            mouse.add_column(justify="right", style="bold", min_width=12)
            mouse.add_column()
            for gesture, meaning in MOUSE_TIPS:
                mouse.add_row(gesture, meaning)
            yield Static(mouse, classes="help-table")

            yield Static(
                Text("esc or ? to close · /help for commands", style="dim"),
                id="help-footer",
            )

    def action_close(self) -> None:
        self.dismiss(None)
