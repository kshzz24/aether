"""The prompt: multi-line editing, history, and `/` + `@` completion.

A `TextArea` rather than an `Input`, because a coding agent gets pasted
stack traces and multi-paragraph instructions, and a single-line box silently
flattens them.

`enter` submits and `shift+enter` inserts a newline — the convention every chat
surface uses. TextArea binds neither by default, so both are ours to define, and
`up`/`down` fall through to normal cursor movement unless the caret is already
on the first/last line (only then do they walk history).

Completion is `tab`-driven rather than ghost-text: `Input`'s `Suggester` has no
`TextArea` equivalent, and inline ghost text in a multi-line editor fights with
the caret. `tab` completes the token under the caret — `/co` to `/config`,
`@gate` to `@gateway/config.py` — and the app shows the runner-up matches.
"""

from __future__ import annotations

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

from tui.commands import COMMANDS
from tui.files import complete_mention, match_paths, split_mention


class PromptArea(TextArea):
    """Multi-line prompt that submits on enter."""

    BINDINGS = [
        Binding("shift+enter", "newline", "newline", show=False),
        Binding("ctrl+j", "newline", "newline", show=False),
    ]

    class Submitted(Message):
        """Posted when the user presses enter on a non-empty prompt."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, **kwargs) -> None:
        super().__init__(soft_wrap=True, tab_behavior="focus", **kwargs)
        self._history: list[str] = []
        # Sits one past the end when not browsing, like a shell.
        self._index = 0
        # What was typed before browsing started, restored on the way back down.
        self._draft = ""
        self.file_index: list[str] = []

    # ------------------------------------------------------------------ keys

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self._submit()
            return
        if event.key == "tab":
            if self._complete():
                event.prevent_default()
                event.stop()
                return
        if event.key == "up" and self._at_first_line():
            event.prevent_default()
            event.stop()
            self.action_history_prev()
            return
        if event.key == "down" and self._at_last_line():
            event.prevent_default()
            event.stop()
            self.action_history_next()
            return
        await super()._on_key(event)

    def action_newline(self) -> None:
        self.insert("\n")

    def _at_first_line(self) -> bool:
        return self.cursor_location[0] == 0

    def _at_last_line(self) -> bool:
        return self.cursor_location[0] == self.document.line_count - 1

    # ------------------------------------------------------------------ submit

    def _submit(self) -> None:
        value = self.text.strip()
        if not value:
            return
        self.post_message(self.Submitted(value))

    def clear(self) -> None:
        self.text = ""

    # ----------------------------------------------------------------- history

    def remember(self, line: str) -> None:
        """Record a submitted line. Consecutive duplicates collapse."""
        if line and (not self._history or self._history[-1] != line):
            self._history.append(line)
        self._index = len(self._history)
        self._draft = ""

    def action_history_prev(self) -> None:
        if not self._history:
            return
        if self._index == len(self._history):
            self._draft = self.text  # stash the in-progress line
        self._index = max(0, self._index - 1)
        self._set(self._history[self._index])

    def action_history_next(self) -> None:
        if self._index >= len(self._history):
            return
        self._index += 1
        if self._index == len(self._history):
            self._set(self._draft)  # back to what was being typed
        else:
            self._set(self._history[self._index])

    def _set(self, value: str) -> None:
        self.text = value
        self.move_cursor(self.document.end)

    # -------------------------------------------------------------- completion

    @property
    def _current_line(self) -> str:
        """Text from the start of the caret's line up to the caret."""
        row, column = self.cursor_location
        return self.document.get_line(row)[:column]

    def suggestions(self) -> list[str]:
        """Candidates for the token under the caret, best first."""
        line = self._current_line
        if split_mention(line) is not None:
            return match_paths(line, self.file_index)
        token = line.strip()
        if token.startswith("/") and " " not in token:
            return [c for c in sorted(COMMANDS) if c.startswith(token.lower())]
        return []

    def _complete(self) -> bool:
        """Replace the token under the caret with its best match."""
        line = self._current_line
        row, column = self.cursor_location

        completed = complete_mention(line, self.file_index)
        if completed is None:
            matches = self.suggestions()
            if not matches or split_mention(line) is not None:
                return False
            # A slash command occupies the whole line by construction.
            completed = matches[0]

        self.replace(completed, (row, 0), (row, column))
        self.move_cursor((row, len(completed)))
        return True
