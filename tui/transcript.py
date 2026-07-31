"""The TUI's view of the agent's Event stream.

This is the second consumer of the `Event` union (`cli/renderer.py` is the
first), and it carries the same obligation: handle every variant, and fail
loudly when a new one appears. Hence `match` + `assert_never`, exactly as the
renderer does.

Three layers, deliberately separated:

* `event_to_renderable` — a pure function, `Event` -> Rich renderable. Total over
  the union, so exhaustiveness is testable without booting an app.
* `CodeBlock` — a fenced block mounted as its own widget so it can carry a copy
  button. Whole-response markdown looks right but is useless the moment you want
  the code out of it.
* `TranscriptView` — decides how each renderable is *mounted*, and owns the
  entry cursor that `v`/`y` select and yank.

Model and tool output is wrapped in `rich.text.Text` (or parsed as Markdown, for
model prose) rather than interpolated into a Rich markup string: a tool result
containing "[bold]" is data, not styling, and letting Rich parse it as markup
would be an injection.
"""

from __future__ import annotations

from typing import assert_never

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Collapsible, Static

from events import (
    ApprovalDecisionEvent,
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    SubagentEvent,
    TerminalEvent,
    TerminalReason,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tui.blocks import Code, Prose, is_long, looks_like_diff, split_blocks

_MAX_RESULT_CHARS = 4000
_MAX_ARG_CHARS = 80
_MAX_TITLE_CHARS = 90

# Mounted-widget transcripts are more capable than a flat log (collapsible tool
# output, per-entry styling) but every entry is a live widget, so an unbounded
# run would degrade layout. Prune the oldest beyond this.
_MAX_ENTRIES = 400

_TERMINAL_STYLES: dict[TerminalReason, str] = {
    TerminalReason.COMPLETED: "bold green",
    TerminalReason.MAX_ITERATIONS: "bold yellow",
    TerminalReason.MAX_COST: "bold yellow",
    TerminalReason.LOOP_DETECTED: "bold yellow",
    TerminalReason.ERROR: "bold red",
}


def _plain(renderable: RenderableType) -> str:
    """Render to plain text.

    Segments rather than `Console.print` so the print-discipline grep over
    `tui/` stays a clean, unexplained pass — and `str(Markdown)` is a repr, not
    the prose, so `str()` is not an option either.
    """
    console = Console(width=100, no_color=True, legacy_windows=False)
    return "".join(
        segment.text for segment in console.render(renderable, console.options)
    )


def _truncate(text: str, limit: int = _MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _format_args(arguments: dict[str, object]) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = str(value).replace("\n", "\\n")
        if len(rendered) > _MAX_ARG_CHARS:
            rendered = rendered[:_MAX_ARG_CHARS] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def tool_call_title(name: str, arguments: dict[str, object]) -> str:
    """The one-line summary shown on a collapsed tool entry."""
    title = f"{name}({_format_args(arguments)})"
    if len(title) > _MAX_TITLE_CHARS:
        title = title[:_MAX_TITLE_CHARS] + "...)"
    return title


def result_renderable(result: str) -> RenderableType:
    """Tool output, coloured as a diff when it is one.

    `write_file` and `edit_file` return diffs, and reading a diff without the
    red and green is materially harder — it is the one place colour carries
    information rather than decoration.
    """
    clipped = _truncate(result)
    if looks_like_diff(clipped):
        return Syntax(clipped, "diff", theme="ansi_dark", word_wrap=True)
    return Text(clipped, style="dim")


def event_to_renderable(event: Event) -> RenderableType:
    """Map one Event to something Rich can draw. Total over the Event union."""
    match event:
        case StatusEvent(message=message):
            return Text(f"— {message}", style="dim italic")

        case TextEvent(text=text):
            # Models emit Markdown; rendering it raw is the single biggest
            # visual gap in a plain log. Markdown does not interpret Rich
            # markup, so "[bold]" still shows as literal text.
            return Markdown(text)

        case ToolCallEvent(name=name, arguments=arguments):
            return Text(tool_call_title(name, arguments), style="cyan")

        case ToolResultEvent(result=result):
            return result_renderable(result)

        case SubagentEvent(task=task, phase=phase, detail=detail):
            if phase == "started":
                return Text(f"▸ subagent: {task}", style="bold magenta")
            suffix = f": {detail}" if detail else ""
            return Text(f"▪ subagent done{suffix}", style="magenta")

        case CostEvent(cost_usd=cost_usd, total_cost_usd=total):
            return Text(
                f"  ${cost_usd:.4f} this turn · ${total:.4f} total", style="dim"
            )

        case ApprovalDecisionEvent(
            tool_name=name, verdict=verdict, source=source, danger_reasons=reasons
        ):
            line = Text(f"· {name}: {verdict.name.lower()} ({source})", style="dim")
            if reasons:
                line.append(f"\n  ! {'; '.join(reasons)}", style="yellow")
            return line

        case ConfirmRequestEvent(tool_name=name, arguments=arguments):
            # The modal does the asking; this is the transcript's record of it.
            return Text(
                f"? confirm {name}({_format_args(arguments)})", style="bold yellow"
            )

        case TerminalEvent(reason=reason, detail=detail):
            label = reason.name.lower().replace("_", " ")
            body = Text(f"■ {label}", style=_TERMINAL_STYLES[reason])
            if detail:
                body = Group(body, Text(f"  {detail}", style="dim"))
            return body

        case _ as unreachable:
            assert_never(unreachable)


class CodeBlock(Static):
    """A fenced code block with its own copy button.

    Its own widget rather than part of the surrounding Markdown because the
    button needs somewhere to live and the raw source needs somewhere to be
    kept — copying a *rendered* code block gives you syntax-highlighting escape
    codes and wrapped lines, not something you can paste into an editor.
    """

    class CopyRequested(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, code: str, language: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.code = code
        # Fences carry things like "python title=foo.py"; only the first word is
        # the lexer name, and an unknown one makes Syntax fall back gracefully.
        self.language = (language.split() or [""])[0] or "text"

    def compose(self) -> ComposeResult:
        with Horizontal(classes="code-bar"):
            yield Static(Text(self.language, style="dim"), classes="code-lang")
            yield Button("copy", classes="code-copy", variant="default")
        yield Static(
            Syntax(
                self.code,
                self.language,
                theme="ansi_dark",
                word_wrap=True,
                background_color="default",
            ),
            classes="code-body",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.CopyRequested(self.code))

    @property
    def content_text(self) -> str:
        return self.code


class TranscriptView(VerticalScroll):
    """Scrollable run transcript built from mounted widgets.

    Auto-follows the tail, but stops following the moment the user scrolls up —
    reading back through output while an agent is still writing is the whole
    point of having scrollback.

    Also owns *select mode*: `v` puts a cursor on an entry, `j`/`k` move it,
    `V` extends the range, `y` yanks. Selection is entry-wise rather than
    character-wise because entries are mounted widgets — there is no character
    grid to select across, and a `Collapsible`'s hidden body has no position at
    all. The terminal's own shift+drag still works over the top of this.
    """

    can_focus = True

    BINDINGS = [
        Binding("v", "select_mode", "select", show=False),
        Binding("V", "extend", "extend", show=False),
        Binding("j", "cursor_down", "down", show=False),
        Binding("k", "cursor_up", "up", show=False),
        Binding("y", "yank", "yank", show=False),
        Binding("c", "copy_entry", "copy entry", show=False),
        Binding("C", "copy_code", "copy code", show=False),
        Binding("escape", "leave_select", "leave select", show=False),
        Binding("question_mark", "help", "help", show=False),
        # `u` is printable, so it lives here rather than on the app — typing a
        # `u` into the prompt must not revert the agent's last edit.
        Binding("u", "undo", "undo", show=False),
    ]

    class CopyRequested(Message):
        """Bubbles to the app, which owns the clipboard."""

        def __init__(self, text: str, label: str) -> None:
            super().__init__()
            self.text = text
            self.label = label

    class HelpRequested(Message):
        pass

    class UndoRequested(Message):
        pass

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # The body of the tool entry currently awaiting its result, so a
        # ToolResultEvent lands inside the call that produced it.
        self._open_body: Static | None = None
        self._selecting = False
        # `v` moves a one-entry cursor; `V` pins the anchor so `j`/`k` grow the
        # range instead. Two distinct verbs, rather than vim's charwise/linewise
        # split, which is meaningless when every entry is already a unit.
        self._extending = False
        self._cursor = 0
        self._anchor = 0

    # ------------------------------------------------------------------ state

    @property
    def following(self) -> bool:
        """True when pinned to the tail (within a line of the bottom)."""
        return self.scroll_offset.y >= self.max_scroll_y - 1

    @property
    def entries(self) -> list:
        return list(self.children)

    @property
    def selecting(self) -> bool:
        return self._selecting

    def entry_text(self, widget) -> str:
        """Plain text of one entry, whatever kind of widget it is."""
        if isinstance(widget, CodeBlock):
            return widget.code
        content = getattr(widget, "content", None)
        if content is not None:
            return _plain(content)
        # A Collapsible: the title plus whatever its body holds.
        title = str(getattr(widget, "title", ""))
        body = "".join(
            _plain(child.content)
            for child in widget.query(Static)
            if getattr(child, "content", None) is not None
        )
        return f"{title}\n{body}" if body else title

    @property
    def text(self) -> str:
        """Everything on screen as plain text."""
        return "".join(self.entry_text(child) for child in self.children)

    def last_code_block(self) -> str | None:
        """The most recent fenced block, for `C`."""
        for child in reversed(self.children):
            if isinstance(child, CodeBlock):
                return child.code
        return None

    # ----------------------------------------------------------------- mutate

    def _add(self, widget) -> None:
        was_following = self.following
        self.mount(widget)
        self._prune()
        if was_following:
            self.scroll_end(animate=False)

    def _prune(self) -> None:
        entries = list(self.children)
        if len(entries) > _MAX_ENTRIES:
            for stale in entries[: len(entries) - _MAX_ENTRIES]:
                stale.remove()

    def banner(self, renderable: RenderableType) -> None:
        """The opening wordmark block. Styled by CSS, not by baked-in colour."""
        self._add(Static(renderable, classes="banner"))

    def user_turn(self, text: str) -> None:
        """Echo what the user asked, on the accent rail.

        Without this a transcript reads as one long monologue — you cannot tell
        which output belongs to which request.
        """
        self._add(Static(Text(f"❯ {text}"), classes="turn-user"))

    def error(self, text: str) -> None:
        self._add(Static(Text(text), classes="turn-error"))

    def _add_model_text(self, text: str) -> None:
        """Mount prose and fenced code as separate widgets."""
        blocks = split_blocks(text)
        if not blocks:
            return
        for block in blocks:
            match block:
                case Code(text=code, language=language):
                    self._add(CodeBlock(code, language, classes="turn-assistant"))
                case Prose(text=prose):
                    body = Static(Markdown(prose), classes="turn-assistant")
                    if is_long(prose):
                        # Long answers stay readable but become foldable, so
                        # scrollback doesn't turn into one wall of text.
                        self._add(
                            Collapsible(
                                body,
                                title=prose.strip().splitlines()[0][:80],
                                collapsed=False,
                                classes="long-entry",
                            )
                        )
                    else:
                        self._add(body)

    def append(self, event: Event) -> None:
        match event:
            case ToolCallEvent(name=name, arguments=arguments):
                # Collapsed by default: verbose tool output is available on
                # demand instead of burying the model's reasoning.
                body = Static(Text("(running...)", style="dim"))
                self._open_body = body
                self._add(
                    Collapsible(
                        body,
                        title=tool_call_title(name, arguments),
                        collapsed=True,
                        classes="tool-entry",
                    )
                )

            case ToolResultEvent():
                if self._open_body is not None:
                    self._open_body.update(event_to_renderable(event))
                    self._open_body = None
                else:
                    self._add(
                        Static(event_to_renderable(event), classes="turn-assistant")
                    )

            case TextEvent(text=text):
                self._add_model_text(text)

            case _:
                self._add(Static(event_to_renderable(event), classes="turn-meta"))

    def notice(self, text: str) -> None:
        """A composition-root message (session id, command output) — not an Event."""
        self._add(Static(Text(text), classes="notice"))

    def preview(self, path, body: str, language: str) -> None:
        """Show a file picked in the sidebar."""
        self._add(Static(Text(str(path), style="bold"), classes="turn-meta"))
        self._add(CodeBlock(body, language, classes="turn-assistant"))

    def clear(self) -> None:
        self._open_body = None
        self._selecting = False
        self.remove_children()

    def highlight(self, needle: str) -> int:
        """Dim entries that don't contain `needle`; return the number that do.

        Dimming rather than filtering: a transcript is a causal record, and
        hiding the steps between two matches makes it lie about what happened.
        An empty needle clears the highlight.
        """
        needle = needle.lower()
        matches = 0
        for child in self.children:
            hit = bool(needle) and needle in self.entry_text(child).lower()
            matches += hit
            child.set_class(bool(needle) and not hit, "faded")
        return matches

    # ----------------------------------------------------------- select mode

    def _paint_selection(self) -> None:
        low, high = sorted((self._anchor, self._cursor))
        for index, child in enumerate(self.children):
            child.set_class(self._selecting and low <= index <= high, "selected")
            child.set_class(self._selecting and index == self._cursor, "cursor")

    def _selected_indices(self) -> range:
        low, high = sorted((self._anchor, self._cursor))
        return range(low, high + 1)

    def action_select_mode(self) -> None:
        """`v` — select one entry, and move it with `j`/`k`."""
        entries = self.entries
        if not entries:
            return
        if not self._selecting:
            self._selecting = True
            # Start at the bottom: what you just read is what you want to copy.
            self._cursor = len(entries) - 1
        # Pressing `v` again collapses an extended range back to one entry,
        # which is the obvious way out of an over-wide selection.
        self._extending = False
        self._anchor = self._cursor
        self._paint_selection()

    def action_extend(self) -> None:
        """`V` — pin the anchor here, so `j`/`k` grow the selection."""
        if not self._selecting:
            self.action_select_mode()
        self._extending = True
        self._anchor = self._cursor
        self._paint_selection()

    def action_leave_select(self) -> None:
        """Leave select mode, or let `escape` fall through to interrupt.

        The fall-through matters: `escape` is the panic stop for a running
        agent, and swallowing it here whenever the transcript happened to have
        focus would take that away.
        """
        if not self._selecting:
            self.app.action_interrupt()
            return
        self._selecting = False
        self._extending = False
        for child in self.children:
            child.set_class(False, "selected")
            child.set_class(False, "cursor")

    def _move(self, delta: int) -> None:
        entries = self.entries
        if not entries:
            return
        if not self._selecting:
            self.scroll_relative(y=delta * 2, animate=False)
            return
        self._cursor = max(0, min(len(entries) - 1, self._cursor + delta))
        if not self._extending:
            self._anchor = self._cursor
        entries[self._cursor].scroll_visible(animate=False)
        self._paint_selection()

    def action_cursor_down(self) -> None:
        self._move(1)

    def action_cursor_up(self) -> None:
        self._move(-1)

    def selected_text(self) -> str:
        entries = self.entries
        if not (self._selecting and entries):
            return ""
        return "\n".join(
            self.entry_text(entries[index])
            for index in self._selected_indices()
            if index < len(entries)
        )

    def _mouse_selection(self) -> str:
        """Text the mouse has dragged over, if any.

        Checked first by `y` and `c`: if you have just highlighted something
        with the mouse, that is unambiguously what you meant to copy, and
        handing back a whole entry instead would be a surprise.
        """
        try:
            return self.screen.get_selected_text() or ""
        except Exception:  # noqa: BLE001 -- a selection probe must never raise
            return ""

    def action_yank(self) -> None:
        dragged = self._mouse_selection()
        if dragged:
            self.app.clear_selection()
            self.post_message(self.CopyRequested(dragged, "selection"))
            return
        text = self.selected_text()
        if not text:
            return
        count = len(self._selected_indices())
        self.action_leave_select()
        self.post_message(
            self.CopyRequested(text, f"{count} {'entry' if count == 1 else 'entries'}")
        )

    def action_copy_entry(self) -> None:
        """`c` — copy the mouse selection, else the entry under the cursor."""
        dragged = self._mouse_selection()
        if dragged:
            self.app.clear_selection()
            self.post_message(self.CopyRequested(dragged, "selection"))
            return
        entries = self.entries
        if not entries:
            return
        index = self._cursor if self._selecting else len(entries) - 1
        self.post_message(self.CopyRequested(self.entry_text(entries[index]), "entry"))

    def action_copy_code(self) -> None:
        code = self.last_code_block()
        if code is None:
            self.app.notify("no code block to copy", severity="warning")
            return
        self.post_message(self.CopyRequested(code, "code block"))

    def action_help(self) -> None:
        self.post_message(self.HelpRequested())

    def action_undo(self) -> None:
        self.post_message(self.UndoRequested())
