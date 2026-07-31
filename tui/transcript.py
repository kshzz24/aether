"""The TUI's view of the agent's Event stream.

This is the second consumer of the `Event` union (`cli/renderer.py` is the
first), and it carries the same obligation: handle every variant, and fail
loudly when a new one appears. Hence `match` + `assert_never`, exactly as the
renderer does.

Two layers, deliberately separated:

* `event_to_renderable` — a pure function, `Event` -> Rich renderable. Total over
  the union, so exhaustiveness is testable without booting an app.
* `TranscriptView` — decides how each renderable is *mounted*. Tool calls become
  collapsed disclosure widgets whose body is filled in when the result arrives;
  everything else is a plain line.

Model and tool output is wrapped in `rich.text.Text` (or parsed as Markdown, for
model prose) rather than interpolated into a Rich markup string: a tool result
containing "[bold]" is data, not styling, and letting Rich parse it as markup
would be an injection.
"""

from __future__ import annotations

from typing import assert_never

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static

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
            return Text(_truncate(result), style="dim")

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


class TranscriptView(VerticalScroll):
    """Scrollable run transcript built from mounted widgets.

    Auto-follows the tail, but stops following the moment the user scrolls up —
    reading back through output while an agent is still writing is the whole
    point of having scrollback.
    """

    can_focus = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # The body of the tool entry currently awaiting its result, so a
        # ToolResultEvent lands inside the call that produced it.
        self._open_body: Static | None = None

    # ------------------------------------------------------------------ state

    @property
    def following(self) -> bool:
        """True when pinned to the tail (within a line of the bottom)."""
        return self.scroll_offset.y >= self.max_scroll_y - 1

    @property
    def text(self) -> str:
        """Everything on screen as plain text.

        Entries hold Rich renderables (Markdown, Text, Group), so this renders
        each one rather than `str()`-ing it — `str(Markdown)` is a repr, not the
        prose. Rendered to segments rather than via `Console.print` so the
        print-discipline grep over `tui/` stays a clean, unexplained pass.
        """
        console = Console(width=100, no_color=True, legacy_windows=False)
        parts: list[str] = []
        for child in self.query(Static):
            content = getattr(child, "content", None)
            if content is None:
                continue
            parts.append(
                "".join(seg.text for seg in console.render(content, console.options))
            )
        return "".join(parts)

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
                    self._add(Static(event_to_renderable(event)))

            case _:
                self._add(Static(event_to_renderable(event)))

    def notice(self, text: str) -> None:
        """A composition-root message (session id, command output) — not an Event."""
        self._add(Static(Text(text, style="bold blue"), classes="notice"))

    def clear(self) -> None:
        self._open_body = None
        self.remove_children()

    def highlight(self, needle: str) -> int:
        """Dim entries that don't contain `needle`; return the number that do.

        Dimming rather than filtering: a transcript is a causal record, and
        hiding the steps between two matches makes it lie about what happened.
        An empty needle clears the highlight.
        """
        console = Console(width=100, no_color=True, legacy_windows=False)
        needle = needle.lower()
        matches = 0
        for child in self.children:
            content = getattr(child, "content", None)
            rendered = (
                "".join(seg.text for seg in console.render(content, console.options))
                if content is not None
                else getattr(child, "title", "")
            )
            hit = bool(needle) and needle in rendered.lower()
            matches += hit
            child.set_class(bool(needle) and not hit, "faded")
        return matches
