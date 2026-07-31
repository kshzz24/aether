"""Fuzzy pickers: ctrl+r over your history, /prompt over your templates.

One screen, two uses. Both are "filter a list of strings as you type, then pick
one", which is the same shape as the setup wizard's `OptionList`
(`tui/setup.py`) — including the trap it found: `clear_options()` leaves nothing
highlighted, so `enter` becomes a silent no-op and the list renders with no
cursor. `highlighted = 0` after every repopulate is not optional.

Ranking lives in `tui/fuzzy.py` so the ordering is identical everywhere and
testable without a running app.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from tui.fuzzy import rank

# One row per candidate; more than this and you should be typing, not scrolling.
_MAX_ROWS = 200

# History entries are whole prompts, which can be paragraphs.
_MAX_LABEL_CHARS = 110


def _label(value: str) -> str:
    """One line, bounded. Multi-line prompts would break the row height."""
    single = " ".join(value.split())
    if len(single) > _MAX_LABEL_CHARS:
        single = single[: _MAX_LABEL_CHARS - 1] + "…"
    return single


class FuzzyPicker(ModalScreen[str | None]):
    """Type to filter, enter to choose, escape to cancel."""

    BINDINGS = [
        ("escape", "cancel", "cancel"),
        ("up", "previous", "up"),
        ("down", "next", "down"),
    ]

    def __init__(self, title: str, items: list[str], *, placeholder: str = "") -> None:
        super().__init__()
        self._title = title
        self._items = items
        self._placeholder = placeholder or "type to filter"

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-body"):
            yield Label(self._title, id="picker-title")
            yield Input(placeholder=self._placeholder, id="picker-input")
            yield OptionList(id="picker-options")

    def on_mount(self) -> None:
        self._populate(self._items)
        self.query_one(Input).focus()

    def _populate(self, values: list[str]) -> None:
        options = self.query_one(OptionList)
        options.clear_options()
        shown = values[:_MAX_ROWS]
        # `id` carries the full value; the prompt is the shortened label, so
        # picking a truncated row still returns the whole prompt.
        options.add_options(
            [Option(_label(value), id=str(index)) for index, value in enumerate(shown)]
        )
        self._shown = shown
        if shown:
            options.highlighted = 0
        self.query_one("#picker-title", Label).update(
            f"{self._title}  ({len(shown)} of {len(self._items)})"
            if len(shown) != len(self._items)
            else f"{self._title}  ({len(self._items)})"
        )

    # --------------------------------------------------------------- handlers

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate(rank(event.value.strip(), self._items))

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._choose()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        index = int(event.option.id or 0)
        if 0 <= index < len(self._shown):
            self.dismiss(self._shown[index])

    def _choose(self) -> None:
        options = self.query_one(OptionList)
        index = options.highlighted
        if index is not None and 0 <= index < len(self._shown):
            self.dismiss(self._shown[index])
        else:
            self.dismiss(None)

    # ---------------------------------------------------------------- actions

    def action_previous(self) -> None:
        """Move the list while the caret stays in the filter box."""
        self.query_one(OptionList).action_cursor_up()

    def action_next(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cancel(self) -> None:
        self.dismiss(None)
