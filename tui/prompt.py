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

_PAIRS = {"(": ")", "[": "]", "{": "}"}


def wants_continuation(text: str) -> bool:
    """True when the line ends in a backslash, shell-style.

    This exists because `shift+enter` is not deliverable in most terminals. The
    keystroke reaches the app as a plain `enter` — Textual can only name it
    `shift+enter` when the terminal implements the Kitty keyboard protocol
    (`_xterm_parser.py:392`), which Windows Terminal only does in recent
    versions. Nothing in the application layer can recover the difference.

    A trailing backslash needs no terminal support at all, and it is already the
    line-continuation convention everyone using a shell knows.
    """
    return text.rstrip(" \t").endswith("\\")


def is_incomplete(text: str) -> bool:
    """True when `enter` should insert a newline instead of sending.

    Deliberately narrow. Two signals only:

    * an odd number of ``` fences — you are mid code block;
    * the last non-space character is an opening bracket — you are mid
      structure.

    Counting *all* unbalanced brackets was the obvious implementation and is
    wrong: "(see below" is ordinary prose, and a prompt that silently refuses to
    send is far more annoying than one extra shift+enter. Quotes are ignored
    entirely, because apostrophes make them useless as a signal.
    """
    if not text.strip():
        return False
    if text.count("```") % 2 == 1:
        return True
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _PAIRS


class PromptArea(TextArea):
    """Multi-line prompt that submits on enter."""

    # shift+enter is listed first because it is what people try, but it is the
    # least reliable: most terminals send an identical byte for enter and
    # shift+enter, and only those implementing the Kitty keyboard protocol can
    # tell them apart. ctrl+j sends U+000A directly and works everywhere, so it
    # is the one documented in `/keys` as the fallback.
    BINDINGS = [
        Binding("shift+enter", "newline", "newline", show=False),
        Binding("ctrl+j", "newline", "newline", show=False),
        Binding("alt+enter", "newline", "newline", show=False),
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

    @property
    def menu(self):
        """The completion popup, if the app has one mounted."""
        found = self.app.query("#completions") if self.is_mounted else None
        return found.first() if found else None

    async def _on_key(self, event: events.Key) -> None:
        menu = self.menu

        # While the menu is open it owns the arrows and the accept keys — that
        # is what makes it a menu rather than a list of hints you have to
        # retype. Everything else still falls through to normal editing.
        if menu is not None and menu.is_open:
            if event.key in ("down", "up"):
                event.prevent_default()
                event.stop()
                menu.move(1 if event.key == "down" else -1)
                return
            if event.key in ("enter", "tab"):
                event.prevent_default()
                event.stop()
                chosen = menu.current
                menu.hide()
                if chosen is not None:
                    self._apply_completion(chosen)
                return
            if event.key == "escape":
                event.prevent_default()
                event.stop()
                menu.hide()
                return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            # A trailing backslash continues the line. The one newline gesture
            # that works in every terminal, because it needs no key beyond the
            # ones already arriving.
            if wants_continuation(self.text):
                self.action_continue_line()
                return
            # Pasting a code block and hitting enter halfway through it should
            # keep typing, not send half a fence to the model.
            if is_incomplete(self.text):
                self.action_newline()
                return
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

    def action_continue_line(self) -> None:
        """Drop the trailing backslash and open a new line.

        The backslash is consumed rather than kept, exactly as a shell does —
        it was punctuation asking for a newline, not part of the message.
        """
        kept = self.text.rstrip(" \t")[:-1].rstrip(" \t")
        self.text = kept + "\n"
        self.move_cursor(self.document.end)

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

    @property
    def line_count(self) -> int:
        return self.document.line_count

    @property
    def status(self) -> str:
        """What the hint line says when there is nothing to complete.

        The box does grow with content, but at one line tall it is
        indistinguishable from a single-line input, so nobody discovers
        shift+enter. Saying it is the whole fix.
        """
        if wants_continuation(self.text):
            return " ⏎ opens a new line (trailing \\)"
        if is_incomplete(self.text):
            return " ⏎ continues — unclosed block"
        if self.line_count > 1:
            return f" {self.line_count} lines · enter sends · ctrl+j newline"
        return " enter sends · ctrl+j or \\⏎ newline · @ file · / command · f1 keys"

    @property
    def past_prompts(self) -> list[str]:
        """Everything submitted this session, oldest first. Read by ctrl+r.

        NB: not `history` — `TextArea` already owns that name for its undo
        stack, and shadowing it breaks the widget at construction. Check any new
        public name here against `dir(TextArea)` first.
        """
        return list(self._history)

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
        """Candidates for the token under the caret, best first.

        A token that already *is* a candidate returns nothing. Without that,
        accepting a completion immediately re-opens the menu — the completed
        text still matches itself — and you have to press escape to get rid of
        a list offering you what you just chose.
        """
        line = self._current_line
        split = split_mention(line)
        if split is not None:
            _, partial = split
            matches = match_paths(line, self.file_index)
            return [] if partial in matches else matches
        token = line.strip()
        if token.startswith("/") and " " not in token:
            lowered = token.lower()
            matches = [c for c in sorted(COMMANDS) if c.startswith(lowered)]
            return [] if lowered in matches else matches
        return []

    def _apply_completion(self, value: str) -> None:
        """Swap the token under the caret for `value`.

        A mention keeps whatever came before the `@`; a slash command occupies
        the whole line by construction.
        """
        line = self._current_line
        row, column = self.cursor_location
        split = split_mention(line)
        completed = f"{split[0]}@{value}" if split is not None else value
        self.replace(completed, (row, 0), (row, column))
        self.move_cursor((row, len(completed)))

    def _complete(self) -> bool:
        """Replace the token under the caret with its best match."""
        line = self._current_line

        completed = complete_mention(line, self.file_index)
        if completed is not None:
            row, column = self.cursor_location
            self.replace(completed, (row, 0), (row, column))
            self.move_cursor((row, len(completed)))
            return True

        matches = self.suggestions()
        if not matches or split_mention(line) is not None:
            return False
        self._apply_completion(matches[0])
        return True
