"""ForgeApp — the full-screen terminal surface.

The structural point of this file: it is *a* subscriber to `agent.run`, not its
owner. `cli/renderer.py` remains a second, independent subscriber, and neither
knows about the other. The agent core was not modified to make this work.

Who owns the event loop is what changed. In the plain CLI, `main._run` drives
`async for event in agent.run(...)`. Here Textual owns the loop and that same
iteration happens inside a worker, pushing into widgets instead of stdout. The
agent cannot tell the difference — it yields Events to whoever iterates.
"""

from __future__ import annotations

import argparse

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Label
from textual.worker import Worker

import main as composition_root
from approval import ApprovalMode
from context.compactor import compact
from events import CostEvent, StatusEvent, TerminalEvent
from tui.approver import TuiApprover
from tui.commands import CommandContext, dispatch
from tui.files import build_file_index
from tui.prompt import PromptArea
from tui.todos import TodoPanel
from tui.transcript import TranscriptView

# Frames for the "working" indicator. Braille spinners are near-universal in
# modern terminals and degrade to boxes rather than breaking layout.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Appended to the system prompt while plan mode is on. Plan mode is two things
# at once: this instruction, and forcing approval on every non-read tool — the
# prompt alone is a suggestion the model can ignore, the policy is the guarantee.
PLAN_PREAMBLE = (
    "\n\nPLAN MODE IS ON. Before changing anything, state a short numbered plan "
    "of what you intend to do and why. Use the `todo` tool to record the steps. "
    "Prefer reading and searching over editing. Expect every write or command to "
    "be confirmed by the user."
)

# How many recent messages /compact preserves verbatim.
_COMPACT_KEEP_RECENT = 6


class ForgeApp(App):
    """Drives an Agent from an input box and renders its events."""

    CSS_PATH = "forge.tcss"
    TITLE = "FORGE"

    BINDINGS = [
        ("escape", "interrupt", "interrupt"),
        ("ctrl+c", "interrupt_or_quit", "interrupt / quit"),
        ("ctrl+l", "clear", "clear"),
        ("ctrl+t", "cycle_theme", "theme"),
        ("ctrl+d", "quit", "quit"),
    ]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args
        self._error: str | None = None
        self._turns = 0
        # NB: not `_running` — Textual's App already owns that name and flips it
        # to True once the app starts, so a guard reading it would always trip.
        # Check any new attribute here against `dir(App)`; these collisions fail
        # silently (a shadowed `_context` hangs startup with no traceback).
        self._run_in_flight = False
        self._worker: Worker | None = None
        self._spinner_frame = 0
        self._activity = ""
        self._turn_costs: list[float] = []
        self._plan_mode = False
        self._saved_mode = ApprovalMode.ON_REQUEST
        self._base_system: str | None = None
        try:
            # TuiApprover(self) is safe here: it only stores the reference.
            self.comp = composition_root.build_composition(
                args, approver=TuiApprover(self)
            )
        except composition_root.CompositionError as exc:
            self.comp = None
            self._error = str(exc)

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="statusbar"):
            yield Label("", id="status-model")
            yield Label("", id="status-activity")
            yield Label("", id="status-cost")
        yield TranscriptView(id="transcript")
        if self.comp is not None:
            yield TodoPanel(self.comp.todos, id="todos")
        yield PromptArea(id="prompt")
        yield Label("", id="hints")
        yield Footer()

    def on_mount(self) -> None:
        transcript = self.query_one(TranscriptView)
        if self.comp is None:
            transcript.notice(self._error or "failed to start")
            self.query_one(PromptArea).disabled = True
            return

        self._base_system = self.comp.agent.system
        self._refresh_status()
        transcript.notice(
            f"session {self.comp.session.id}  ·  /help for commands, "
            f"@ to reference a file, shift+enter for a new line"
        )
        prompt = self.query_one(PromptArea)
        prompt.focus()
        self._reindex_files()
        self.set_interval(0.1, self._tick)

        # A goal on argv (or --resume) starts immediately, matching the CLI.
        if self.comp.goal:
            self._start(self.comp.goal, history=self.comp.history)

    def _reindex_files(self) -> None:
        self.query_one(PromptArea).file_index = build_file_index(
            self.comp.agent.repo_root
        )

    def _tick(self) -> None:
        if not self._run_in_flight:
            return
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER)
        self._paint_activity()

    def _paint_activity(self) -> None:
        label = self.query_one("#status-activity", Label)
        if self._run_in_flight:
            frame = _SPINNER[self._spinner_frame]
            label.update(f" {frame} {self._activity or 'working'} ")
        else:
            label.update(" plan mode " if self._plan_mode else "")

    def _refresh_status(self) -> None:
        if self.comp is None:
            return
        cfg = self.comp.config
        self.query_one("#status-model", Label).update(f" {cfg.provider}/{cfg.model} ")
        self.query_one("#status-cost", Label).update(
            f" turn {self._turns}/{cfg.max_iterations} "
            f" ${self.comp.agent.total_cost:.4f}/${cfg.max_cost_usd:.2f} "
        )
        self._paint_activity()

    # ------------------------------------------------------------ completion

    def on_text_area_changed(self, _event) -> None:
        """Show what `tab` would complete to, and the runners-up.

        Tolerates a partial widget tree: TextArea posts Changed during mount and
        teardown, when #hints may not exist yet (or any more).
        """
        found = self.query("#hints")
        if not found:
            return
        hints = found.first(Label)
        matches = self.query_one(PromptArea).suggestions()
        if not matches:
            hints.update("")
            return
        shown = "  ".join(matches[:6])
        more = f"  (+{len(matches) - 6})" if len(matches) > 6 else ""
        hints.update(f" tab → {shown}{more}")

    # -------------------------------------------------------------- commands

    def _command_context(self) -> CommandContext:
        # NB: not `_context` — that name is taken by Textual's App, which uses
        # it as a context manager during startup. Shadowing it hangs the app.
        return CommandContext(
            config=self.comp.config,
            registry=self.comp.registry,
            session=self.comp.session,
            sessions_dir=self.comp.sessions_dir,
            total_cost=self.comp.agent.total_cost,
            turns=self._turns,
            todos=self.comp.todos,
            turn_costs=tuple(self._turn_costs),
            plan_mode=self._plan_mode,
        )

    def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        line = event.value.strip()
        prompt = self.query_one(PromptArea)
        prompt.clear()
        self.query_one("#hints", Label).update("")
        if not line or self.comp is None:
            return
        prompt.remember(line)

        transcript = self.query_one(TranscriptView)

        result = dispatch(line, self._command_context())
        if result is not None:
            self._apply(result)
            return

        if self._run_in_flight:
            self.notify("a run is already in progress", severity="warning")
            return

        transcript.notice(f"> {line}")
        self._start(line, history=None)

    def _apply(self, result) -> None:
        """Carry out a CommandResult. Dispatch stays pure; the effects land here."""
        transcript = self.query_one(TranscriptView)

        if result.clear:
            transcript.clear()
        elif result.toast:
            self.notify(result.text)
        elif result.text:
            transcript.notice(result.text)

        if result.find is not None:
            count = transcript.highlight(result.find)
            if result.find:
                self.notify(f"{count} matching entries")
        if result.toggle_plan:
            self._toggle_plan()
        if result.rebuild:
            self._rebuild(result.rebuild)
        if result.reindex:
            self._reindex_files()
            self.notify(f"{len(self.query_one(PromptArea).file_index)} files indexed")
        if result.copy:
            self.copy_to_clipboard(transcript.text)
        if result.compact:
            self._compact()
        if result.quit:
            self.exit()
        elif result.resume_id:
            self._resume(result.resume_id)

    def _toggle_plan(self) -> None:
        """Plan mode = a prompt preamble plus a stricter approval policy.

        The policy half is what makes it real: a preamble is advice the model
        may ignore, but ON_REQUEST means every write is gated regardless.
        """
        self._plan_mode = not self._plan_mode
        agent = self.comp.agent
        agent.system = (self._base_system or "") + (
            PLAN_PREAMBLE if self._plan_mode else ""
        )
        if self._plan_mode:
            # Save the mode actually in effect, not the config default: a user
            # who ran `/approval never` first must get `never` back on exit.
            self._saved_mode = agent.policy.mode
            agent.policy.mode = ApprovalMode.ON_REQUEST
        else:
            agent.policy.mode = self._saved_mode
        self._refresh_status()

    def _rebuild(self, overrides: dict) -> None:
        """Rebuild the agent under new config (model / provider / approval)."""
        for key, value in overrides.items():
            setattr(self._args, key, value)
        history = self.comp.agent.messages
        try:
            self.comp = composition_root.build_composition(
                self._args, approver=TuiApprover(self), todos=self.comp.todos
            )
        except composition_root.CompositionError as exc:
            self.query_one(TranscriptView).notice(str(exc))
            return
        # Carry the conversation across the swap: switching model mid-task
        # should continue the task, not restart it.
        self.comp.agent.messages = history
        self._base_system = self.comp.agent.system
        if self._plan_mode:
            self._plan_mode = False
            self._toggle_plan()
        self._refresh_status()

    def _compact(self) -> None:
        self.run_worker(self._compact_worker(), group="compact")

    async def _compact_worker(self) -> None:
        agent = self.comp.agent
        before = len(agent.messages)
        messages, cost = await compact(
            agent.client, agent.messages, _COMPACT_KEEP_RECENT
        )
        agent.messages = messages
        agent.total_cost += cost
        dropped = before - len(messages)
        self.query_one(TranscriptView).notice(
            f"compacted {dropped} messages (${cost:.4f})"
            if dropped > 0
            else "nothing to compact yet"
        )
        self._refresh_status()

    def _resume(self, session_id: str) -> None:
        """Reload the composition against a saved session and replay it."""
        self._args.resume = session_id
        try:
            self.comp = composition_root.build_composition(
                self._args, approver=TuiApprover(self)
            )
        except composition_root.CompositionError as exc:
            self.query_one(TranscriptView).notice(str(exc))
            return
        self._turns = 0
        self._turn_costs.clear()
        self._base_system = self.comp.agent.system
        self._refresh_status()
        if self.comp.goal:
            self._start(self.comp.goal, history=self.comp.history)

    # --------------------------------------------------------------- actions

    def action_clear(self) -> None:
        self.query_one(TranscriptView).clear()

    def action_cycle_theme(self) -> None:
        self.theme = (
            "textual-light" if self.theme == "textual-dark" else "textual-dark"
        )

    def action_interrupt(self) -> None:
        """Stop the agent mid-run.

        An autonomous loop spending real money that the user cannot stop is a
        liability, not a UI gap — this is the counterpart to the iteration and
        cost governors, with a human at the switch.
        """
        if not self._run_in_flight or self._worker is None:
            return
        self._worker.cancel()
        self.query_one(TranscriptView).notice("interrupted")
        self.notify("run interrupted", severity="warning")

    def action_interrupt_or_quit(self) -> None:
        """Ctrl+C stops the run if one is going, otherwise exits."""
        if self._run_in_flight:
            self.action_interrupt()
        else:
            self.exit()

    # ------------------------------------------------------------------- run

    def _start(self, goal: str, *, history) -> None:
        if not self.comp.session.goal:
            self.comp.session.goal = goal
        self._run_in_flight = True
        self._turns = 0
        self._turn_costs.clear()
        self._activity = "thinking"
        self._worker = self._drive(goal, history)
        self._refresh_status()

    def _drive(self, goal: str, history) -> Worker:
        return self.run_worker(
            self._drive_worker(goal, history), exclusive=True, group="agent"
        )

    async def _drive_worker(self, goal: str, history) -> None:
        """The agent loop, as a worker. Mirrors `main._run`'s body exactly."""
        transcript = self.query_one(TranscriptView)
        try:
            async for event in self.comp.agent.run(goal, history=history):
                transcript.append(event)
                if isinstance(event, CostEvent):
                    self._turn_costs.append(event.cost_usd)
                # StatusEvent fires at each turn's start (state complete through
                # the prior turn); TerminalEvent at the end. Snapshot on both — a
                # crash loses at most the in-flight turn.
                if isinstance(event, (StatusEvent, TerminalEvent)):
                    if isinstance(event, StatusEvent):
                        self._turns += 1
                        self._activity = event.message
                    composition_root.checkpoint(self.comp)
                self._refresh_status()
        finally:
            # Also runs on cancellation, so an interrupted run still checkpoints
            # and the UI never gets stuck showing a spinner.
            self._run_in_flight = False
            self._worker = None
            self._activity = ""
            if self.comp is not None:
                composition_root.checkpoint(self.comp)
            self._refresh_status()
