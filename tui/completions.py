"""The `@file` and `/command` completion menu.

The matching logic was always right — `tui/files.py` indexes the repo and ranks
correctly — but the only way to *see* a match was a line of grey text under the
box, and the only way to take one was `tab`. That is not a completion menu, it
is a hint, and it reads as "completion doesn't work".

So: a real popup. It opens by itself as you type, `up`/`down` walk it, `enter`
or `tab` takes the highlighted row, `escape` dismisses it. It is a sibling of
the prompt rather than a child so it can overlay the transcript without the
prompt having to grow.
"""

from __future__ import annotations

from textual.widgets import OptionList
from textual.widgets.option_list import Option

# Taller than this and the menu covers the conversation you are writing about.
MAX_ROWS = 8


class CompletionMenu(OptionList):
    """A popup over the prompt. Hidden whenever there is nothing to offer."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.display = False
        self._values: list[str] = []

    @property
    def is_open(self) -> bool:
        return bool(self.display and self._values)

    @property
    def current(self) -> str | None:
        """The highlighted candidate, if any."""
        index = self.highlighted
        if index is None or not (0 <= index < len(self._values)):
            return None
        return self._values[index]

    def show(self, values: list[str]) -> None:
        """Offer `values`, keeping the highlight on the best one."""
        if not values:
            self.hide()
            return
        shown = values[:MAX_ROWS]
        if shown == self._values:
            # Same candidates: leave the user's highlight where they put it,
            # or every keystroke would yank the cursor back to the top.
            self.display = True
            return
        self._values = shown
        self.clear_options()
        self.add_options([Option(value, id=value) for value in shown])
        # `clear_options` leaves nothing highlighted, which makes enter a
        # silent no-op and renders the list with no visible cursor.
        self.highlighted = 0
        self.display = True

    def hide(self) -> None:
        self.display = False
        self._values = []

    def move(self, delta: int) -> None:
        """Walk the list, wrapping — a short menu is faster to cycle."""
        if not self._values:
            return
        current = self.highlighted or 0
        self.highlighted = (current + delta) % len(self._values)
