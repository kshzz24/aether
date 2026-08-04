"""ForgeApp — the full-screen terminal surface.

The structural point of this file: it is *a* subscriber to `agent.run`, not its
owner. `cli/renderer.py` remains a second, independent subscriber, and neither
knows about the other. The agent core was not modified to make this work.

Who owns the event loop is what changed. In the plain CLI, `main._run` drives
`async for event in agent.run(...)`. Here Textual owns the loop and that same
iteration happens inside a worker, pushing into widgets instead of stdout. The
agent cannot tell the difference — it yields Events to whoever iterates.

The one thing that reaches *into* the composition is `Hooks`: the app passes an
`UndoStack`'s `before_tool`/`after_tool` so a file can be snapshotted before the
agent overwrites it. That is the Phase-2 seam being used for the first time, not
a new hole in the core.
"""

from __future__ import annotations

import argparse
import os
import time
import tomllib
from asyncio import CancelledError
from dataclasses import replace
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Label
from textual.worker import Worker

import main as composition_root
import persistence
from approval import ApprovalMode
from context.compactor import compact
from events import CostEvent, StatusEvent, TerminalEvent
from server.wire import RunParams
from tui.approver import TuiApprover
from tui.branding import banner
from tui.catalog import find, load_catalog
from tui.clipboard import copy as clipboard_copy
from tui.commands import CommandContext, dispatch
from tui.completions import CompletionMenu
from tui.context import estimate_tokens, meter, rules_label
from tui.files import build_file_index
from tui.filetree import tree_lines
from tui.help import HelpScreen
from tui.pickers import FuzzyPicker
from tui.prompt import PromptArea
from tui.setup import SetupScreen
from tui.sidebar import Sidebar
from tui.templates import TEMPLATES_DIR, load_templates
from tui.theme import DEFAULT_THEME, THEMES, next_theme
from tui.todos import TodoPanel
from tui.transcript import CodeBlock, TranscriptView
from tui.undo import UndoStack

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

# Below this, a finished run is something you were watching; above it, something
# you walked away from and want to be told about.
_BELL_AFTER_SECONDS = 10.0

# A sidebar preview is a glance, not a file viewer.
_PREVIEW_MAX_BYTES = 200_000

_USER_CONFIG = Path.home() / ".forge" / "config.toml"
_PROJECT_CONFIG = Path.cwd() / ".forge" / "config.toml"

# Suffix -> Pygments lexer, for sidebar previews. Unknown suffixes fall back to
# "text", which renders fine — this is a nicety, not a parser.
_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".css": "css",
    ".tcss": "css",
    ".html": "html",
    ".sql": "sql",
}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("forge")
    except Exception:  # noqa: BLE001 -- a missing dist must not block startup
        return "0.1.0"


def _load_prices() -> dict:
    """prices.toml drives the model picker. Absent is survivable: the wizard
    falls back to free-text model entry for every provider."""
    try:
        with open("prices.toml", "rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _explain(exc: BaseException, provider: str | None = None) -> str:
    """Turn a provider SDK error into a sentence naming the fix.

    The auth case is worth special-casing: it is by far the most common first-run
    failure, and the SDK's own message ("Missing credentials. Please pass an
    `api_key`...", "Could not resolve authentication method...") names the SDK's
    parameters rather than the environment variable the user has to export.

    `provider` is passed in where the caller knows it — a `--provider` flag beats
    guessing from config files, which would otherwise name the wrong variable in
    exactly the situation the message exists to fix.
    """
    text = str(exc)
    lowered = text.lower()
    auth = (
        "api_key" in lowered
        or "authentication" in lowered
        or "credentials" in lowered
        or "401" in lowered
    )
    if auth:
        name = provider or _current_provider()
        env_var = composition_root.ENV_KEYS.get(name, "the API key")
        return (
            f"no API key for {name} — {env_var} is not set in this shell.\n"
            f"  export {env_var}='...'   then restart\n"
            f"  or press ctrl+s to pick a provider whose key you already have.\n"
            f"(WSL does not inherit Windows environment variables; set it in the "
            f"Linux shell you launched from, or add it to ~/.bashrc.)"
        )
    return f"{type(exc).__name__}: {text}"


def _current_provider() -> str:
    """Best-effort provider name for error messages, from the config layers."""
    for path in (_PROJECT_CONFIG, _USER_CONFIG):
        if path.exists():
            try:
                with open(path, "rb") as handle:
                    provider = tomllib.load(handle).get("provider")
            except (OSError, tomllib.TOMLDecodeError):
                continue
            if provider:
                return str(provider)
    return "anthropic"


def _needs_setup(args: argparse.Namespace) -> bool:
    """True when the user has never said which provider they want.

    Explicit flags or either config file count as having chosen — the wizard
    must never interrupt someone who already configured FORGE.
    """
    if getattr(args, "setup", False):
        return True
    if args.provider is not None or args.model is not None:
        return False
    return not (_USER_CONFIG.exists() or _PROJECT_CONFIG.exists())


def _write_user_config(provider: str, model: str) -> None:
    """Persist the choice at *user* scope.

    "Which provider do I use" is a property of the person, not of the repo, so
    it belongs in ~/.forge rather than in a .forge committed alongside code.
    Merges into an existing file rather than replacing it, so a hand-set
    max_cost_usd is not silently discarded.
    """
    existing: dict = {}
    if _USER_CONFIG.exists():
        try:
            with open(_USER_CONFIG, "rb") as handle:
                existing = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            existing = {}
    existing["provider"] = provider
    existing["model"] = model

    _USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in existing.items():
        rendered = f'"{value}"' if isinstance(value, str) else repr(value).lower()
        lines.append(f"{key} = {rendered}")
    _USER_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ForgeApp(App):
    """Drives an Agent from an input box and renders its events."""

    CSS_PATH = "forge.tcss"
    TITLE = "FORGE"

    BINDINGS = [
        ("escape", "interrupt", "interrupt"),
        ("ctrl+c", "interrupt_or_quit", "interrupt / quit"),
        ("ctrl+l", "clear", "clear"),
        ("ctrl+t", "cycle_theme", "theme"),
        ("ctrl+b", "toggle_sidebar", "files"),
        ("ctrl+r", "search_history", "history"),
        # Bound as well as being a command, because when startup fails there is
        # no prompt to type `/setup` into — and that is exactly when you need it.
        ("ctrl+s", "setup", "setup"),
        ("f1", "help", "help"),
        ("ctrl+d", "quit", "quit"),
    ]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # Themes must be registered before the stylesheet is parsed: forge.tcss
        # references $rail-user and friends, and Textual resolves CSS variables
        # at startup, well before on_mount.
        for theme in THEMES:
            self.register_theme(theme)
        self.theme = DEFAULT_THEME

        # Two representations of the same invocation, on purpose. `_params` is
        # what `build_composition` consumes and what `/model`, `/provider` and
        # `/approval` mutate. `_args` survives only to feed `_needs_setup`,
        # which asks argv-shaped questions (`--setup`) that have no business in
        # a composition parameter object.
        self._args = args
        self._params = RunParams.from_namespace(args)
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
        self._bell_enabled = True
        self._autocopy = True
        self._run_started = 0.0
        # Set just before a copy so the toast can say what was copied. Consumed
        # (and cleared) by `copy_to_clipboard`, which Textual also calls.
        self._copy_label = ""
        # Tracks the message count across turns so an internal compaction
        # (agent.py:266, which yields no event) can still be reported.
        self._message_count = 0
        self._undo = UndoStack()
        try:
            # TuiApprover(self) is safe here: it only stores the reference.
            self.comp = composition_root.build_composition(
                self._params, approver=TuiApprover(self), hooks=self._undo.hooks()
            )
        except composition_root.CompositionError as exc:
            self.comp = None
            self._error = str(exc)
        except Exception as exc:  # noqa: BLE001
            # Provider SDKs raise from their *constructor* when the key is
            # absent, so this fires before a single event is produced. Catching
            # only CompositionError here let an OpenAIError escape into a
            # traceback — the exact failure `_explain` exists to replace.
            self.comp = None
            self._error = _explain(exc, provider=getattr(args, "provider", None))

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="statusbar"):
            yield Label("", id="status-model")
            yield Label("", id="status-rules")
            yield Label("", id="status-activity")
            yield Label("", id="status-context")
            yield Label("", id="status-cost")
        with Horizontal(id="body"):
            with Vertical(id="main-column"):
                yield TranscriptView(id="transcript")
                if self.comp is not None:
                    yield TodoPanel(self.comp.todos, id="todos")
            if self.comp is not None:
                yield Sidebar(self.comp.agent.repo_root, id="sidebar")
        # One docked container, not three docked siblings. Docking the menu,
        # the prompt and the hint line separately made all three claim the same
        # bottom row: the prompt's own bottom border got overdrawn by the hints,
        # and the hints by the footer. Grouping them means the stack has a
        # single `auto` height that grows as one.
        with Vertical(id="composer"):
            yield CompletionMenu(id="completions")
            yield PromptArea(id="prompt")
            yield Label("", id="hints")
            # Inside the composer, not beside it: Footer carries its own
            # `dock: bottom`, so as a sibling it docked to the *screen* and
            # landed on the hint line rather than below it.
            yield Footer()

    def on_mount(self) -> None:
        transcript = self.query_one(TranscriptView)
        transcript.banner(banner(version=_version(), cwd=Path.cwd()))

        if self.comp is None:
            # No agent, so nothing can be typed at it — but the wizard is a real
            # fix (pick a provider whose key you have), so offer it rather than
            # leaving ctrl+c as the only working key.
            transcript.error(self._error or "failed to start")
            self.query_one(PromptArea).disabled = True
            self._open_setup()
            return

        # Hidden until ctrl+b: an empty pane on startup is just lost width.
        self.query_one(Sidebar).display = False

        self._base_system = self.comp.agent.system
        self._refresh_status()
        transcript.notice(f"session {self.comp.session.id}")
        prompt = self.query_one(PromptArea)
        prompt.border_title = "❯ ask forge"
        self.query_one("#hints", Label).update(prompt.status)
        prompt.focus()
        self._reindex_files()
        self.set_interval(0.1, self._tick)
        # In a worker, not inline: a stdio server may `npx`-download a package
        # on first run, and the prompt must not be dead while that happens.
        self.run_worker(self._connect_mcp(), group="mcp")

        if _needs_setup(self._args):
            self._open_setup()
            return

        # A goal on argv (or --resume) starts immediately, matching the CLI.
        if self.comp.goal:
            self._start(self.comp.goal, history=self.comp.history)

    # ----------------------------------------------------------------- setup

    def _open_setup(self) -> None:
        catalog = load_catalog(
            prices=_load_prices(),
            env=os.environ,
            env_keys=composition_root.ENV_KEYS,
        )
        self.push_screen(SetupScreen(catalog), self._setup_done)

    def _setup_done(self, result) -> None:
        if not result:
            self.query_one(TranscriptView).notice(
                "setup skipped — nothing is configured, so fix the key above and "
                "restart, or press ctrl+s to choose again"
                if self.comp is None
                else "setup skipped — using defaults; run /setup to choose again"
            )
            return
        provider, model = result
        _write_user_config(provider, model)
        self._rebuild({"provider": provider, "model": model})
        self.query_one(TranscriptView).notice(
            f"using {provider}/{model} · saved to ~/.forge/config.toml"
        )
        info = find(
            load_catalog(
                prices=_load_prices(),
                env=os.environ,
                env_keys=composition_root.ENV_KEYS,
            ),
            provider,
        )
        # Say it now, plainly, rather than letting the first turn fail with an
        # SDK traceback — the whole reason this screen exists.
        if info is not None and not info.has_key:
            self.query_one(TranscriptView).error(
                f"{info.env_var} is not set — export it before sending a goal"
            )

    async def _connect_mcp(self) -> None:
        """Federate MCP tools into the registry, in the background.

        Failures are already contained by the manager — a dead server shows up
        in `/mcp` rather than stopping the session — so this only reports.
        """
        if self.comp is None or self.comp.mcp is None:
            return
        try:
            summary = await composition_root.connect_mcp(self.comp)
        except Exception as exc:  # noqa: BLE001
            self.query_one(TranscriptView).notice(f"mcp: {_explain(exc)}")
            return
        if summary:
            self.query_one(TranscriptView).notice(summary)

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

    def _context_usage(self) -> tuple[int, int]:
        if self.comp is None:
            return 0, 0
        agent = self.comp.agent
        return estimate_tokens(agent.messages), getattr(agent, "max_context_token", 0)

    def _refresh_status(self) -> None:
        if self.comp is None:
            return
        cfg = self.comp.config
        self.query_one("#status-model", Label).update(f" {cfg.provider}/{cfg.model} ")

        rules = rules_label(self.comp.agent.repo_root)
        self.query_one("#status-rules", Label).update(f" {rules} " if rules else "")

        used, budget = self._context_usage()
        self.query_one("#status-context", Label).update(
            f" {meter(used, budget, width=8)} " if budget else ""
        )
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
        prompt = self.query_one(PromptArea)
        matches = prompt.suggestions()

        menu = self.query("#completions")
        if menu:
            # The popup opens by itself. Waiting for `tab` was the bug: nobody
            # presses a key to reveal a list they have no reason to think
            # exists.
            menu.first(CompletionMenu).show(matches)

        if not matches:
            # Never blank: this line is the only place shift+enter is
            # advertised, and an empty one-line box looks single-line.
            hints.update(prompt.status)
            return
        extra = f"  (+{len(matches) - 8})" if len(matches) > 8 else ""
        hints.update(f" ↑↓ choose · enter or tab accept · esc dismiss{extra}")

    # -------------------------------------------------------------- commands

    def _command_context(self) -> CommandContext:
        # NB: not `_context` — that name is taken by Textual's App, which uses
        # it as a context manager during startup. Shadowing it hangs the app.
        used, budget = self._context_usage()
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
            undo=self._undo,
            repo_root=self.comp.agent.repo_root,
            mcp=self.comp.mcp,
            bell=self._bell_enabled,
            autocopy=self._autocopy,
            context_tokens=used,
            context_budget=budget,
        )

    def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        line = event.value.strip()
        prompt = self.query_one(PromptArea)
        prompt.clear()
        self.query_one("#hints", Label).update(prompt.status)
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

        transcript.user_turn(line)
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
            self._copy(transcript.text, "transcript")
        if result.compact:
            self._compact()
        if result.undo:
            self._refresh_changed()
        if result.toggle_bell:
            self._bell_enabled = not self._bell_enabled
        if result.toggle_autocopy:
            self._autocopy = not self._autocopy
        if result.fill_prompt:
            self._fill_prompt(result.fill_prompt)
        if result.pick_template:
            self._pick_template()
        if result.pick_session:
            self._pick_session()
        if result.setup:
            self._open_setup()
        if result.cycle_theme:
            self.action_cycle_theme()
        if result.quit:
            self.exit()
        elif result.resume_id:
            self._resume(result.resume_id)

    def _fill_prompt(self, text: str) -> None:
        """Put a template in the box rather than sending it.

        A saved prompt is a starting point; submitting it verbatim is rarely
        what you want, and there is no undo on a sent goal.
        """
        prompt = self.query_one(PromptArea)
        prompt.text = text
        prompt.move_cursor(prompt.document.end)
        prompt.focus()

    def _pick_template(self) -> None:
        templates = load_templates(TEMPLATES_DIR)
        if not templates:
            return

        def chosen(name: str | None) -> None:
            if name:
                self._fill_prompt(templates[name])

        self.push_screen(FuzzyPicker("Saved prompts", sorted(templates)), chosen)

    def _pick_session(self) -> None:
        """`/resume` with no id. Beats reading an id off `/sessions` and
        typing it back in."""
        metas = persistence.list_sessions(self.comp.sessions_dir)
        if not metas:
            return
        # The id leads so it survives the label truncation, and so the fuzzy
        # filter matches on it as well as on the goal.
        rows = [
            f"{m.id}  ${m.total_cost:.4f}  {m.turns:>3} turns  {m.goal}" for m in metas
        ]

        def chosen(row: str | None) -> None:
            if row:
                self._resume(row.split()[0])

        self.push_screen(FuzzyPicker("Resume a session", rows), chosen)

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
        """Rebuild the agent under new config (model / provider / approval).

        Also the recovery path when startup failed outright: `self.comp` is None
        there, so nothing may be read off it until the rebuild succeeds.
        """
        # `replace` on a frozen dataclass rejects an unknown key outright, where
        # the old `setattr` onto a Namespace accepted a typo'd override and
        # silently did nothing.
        self._params = replace(self._params, **overrides)
        previous = self.comp
        history = previous.agent.messages if previous is not None else None
        try:
            self.comp = composition_root.build_composition(
                self._params,
                approver=TuiApprover(self),
                todos=previous.todos if previous is not None else None,
                hooks=self._undo.hooks(),
            )
        except Exception as exc:  # noqa: BLE001
            self.comp = previous
            self.query_one(TranscriptView).error(
                _explain(exc, provider=self._params.provider)
            )
            return
        # Carry the conversation across the swap: switching model mid-task
        # should continue the task, not restart it.
        if history is not None:
            self.comp.agent.messages = history
        self._base_system = self.comp.agent.system
        if self._plan_mode:
            self._plan_mode = False
            self._toggle_plan()
        # Startup may have failed before there was an agent to talk to; a
        # successful rebuild is what makes the prompt usable again.
        prompt = self.query_one(PromptArea)
        if prompt.disabled:
            prompt.disabled = False
            prompt.focus()
            self._reindex_files()
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
        self._message_count = len(messages)
        self.query_one(TranscriptView).notice(
            f"compacted {dropped} messages (${cost:.4f})"
            if dropped > 0
            else "nothing to compact yet"
        )
        self._refresh_status()

    def _resume(self, session_id: str) -> None:
        """Reload the composition against a saved session and replay it."""
        # `resume` stays set on `_params` afterwards, so a later `/model` rebuild
        # re-resumes this session. That is the pre-existing behaviour of the
        # mutated Namespace, preserved deliberately — changing it is a behaviour
        # fix, not part of this refactor.
        self._params = replace(self._params, resume=session_id)
        try:
            self.comp = composition_root.build_composition(
                self._params, approver=TuiApprover(self), hooks=self._undo.hooks()
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

    # ------------------------------------------------------------- clipboard

    def copy_to_clipboard(self, text: str) -> None:
        """Every copy in the app funnels through here — including Textual's own.

        Textual binds `ctrl+c` to `screen.copy_text`, which drag-selection uses,
        and that calls straight into `App.copy_to_clipboard` — i.e. OSC 52 only
        (screen.py:991). Terminals that don't implement OSC 52 discard it
        silently, so selecting text and pressing ctrl+c looked like it did
        nothing. Overriding here gives the *native* fallback and a toast to
        every copy path at once, ours and Textual's.
        """
        message = clipboard_copy(super().copy_to_clipboard, text)
        label, self._copy_label = self._copy_label, ""
        self.notify(f"{label}: {message}" if label else message)

    def _copy(self, text: str, label: str) -> None:
        """Copy with a label saying what was copied."""
        self._copy_label = label
        self.copy_to_clipboard(text)

    def _selection(self) -> str:
        """Text the mouse has selected, if any."""
        try:
            return self.screen.get_selected_text() or ""
        except Exception:  # noqa: BLE001 -- a selection probe must never raise
            return ""

    def on_text_selected(self, _event) -> None:
        """Copy the moment a drag ends, without waiting for ctrl+c.

        This is what a terminal does (X11's primary selection), and it is what
        people mean by "select to copy". Textual posts `TextSelected` on every
        mouse-up, including plain clicks — but a click clears the selection
        first, so the emptiness check is also the click filter.
        """
        if not self._autocopy:
            return
        text = self._selection()
        if text:
            # The highlight is deliberately left on screen: it is the receipt
            # for what just went to the clipboard.
            self._copy(text, "selection")

    def on_transcript_view_copy_requested(
        self, event: TranscriptView.CopyRequested
    ) -> None:
        event.stop()
        self._copy(event.text, event.label)

    def on_code_block_copy_requested(self, event: CodeBlock.CopyRequested) -> None:
        event.stop()
        self._copy(event.text, "code block")

    def on_transcript_view_help_requested(self, _event) -> None:
        self.action_help()

    def on_transcript_view_undo_requested(self, _event) -> None:
        """`u` on the transcript — the same path as `/undo`."""
        if self.comp is not None:
            self._apply(dispatch("/undo", self._command_context()))

    # --------------------------------------------------------------- sidebar

    def _refresh_changed(self) -> None:
        found = self.query("#sidebar")
        if found:
            found.first(Sidebar).refresh_changed(self._undo.touched())

    def on_sidebar_file_chosen(self, event: Sidebar.FileChosen) -> None:
        event.stop()
        transcript = self.query_one(TranscriptView)
        path = event.path
        try:
            if path.stat().st_size > _PREVIEW_MAX_BYTES:
                transcript.notice(f"{path.name} is too large to preview")
                return
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            transcript.notice(f"cannot read {path}: {exc}")
            return
        transcript.preview(path, body, _LANGUAGES.get(path.suffix.lower(), "text"))

    # --------------------------------------------------------------- actions

    async def on_unmount(self) -> None:
        """Close MCP connections, and the subprocesses behind them.

        Without this, every `forge` session leaves its stdio servers running.
        """
        if self.comp is not None and self.comp.mcp is not None:
            await self.comp.mcp.aclose()

    def action_clear(self) -> None:
        self.query_one(TranscriptView).clear()

    def action_cycle_theme(self) -> None:
        self.theme = next_theme(self.theme)
        self.notify(f"theme: {self.theme}")

    def action_toggle_sidebar(self) -> None:
        found = self.query("#sidebar")
        if not found:
            return
        sidebar = found.first(Sidebar)
        sidebar.display = not sidebar.display
        if sidebar.display:
            self._refresh_changed()

    def action_help(self) -> None:
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())

    def action_setup(self) -> None:
        if not isinstance(self.screen, SetupScreen):
            self._open_setup()

    def action_search_history(self) -> None:
        """ctrl+r over everything typed this session."""
        prompt = self.query_one(PromptArea)
        # Newest first: an empty filter should offer what you just ran.
        entries = list(reversed(prompt.past_prompts))
        if not entries:
            self.notify("no history yet")
            return

        def chosen(line: str | None) -> None:
            if line:
                self._fill_prompt(line)

        self.push_screen(FuzzyPicker("History", entries), chosen)

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
        """Ctrl+C copies a selection, else stops the run, else exits.

        Selection wins because that is what ctrl+c means everywhere else, and
        because Textual's own binding already claims it — `screen.copy_text`
        runs first and raises `SkipAction` when nothing is selected, which is
        how control reaches this method at all. The explicit check here is
        belt-and-braces for the paths that call this action directly.
        """
        selected = self._selection()
        if selected:
            self._copy(selected, "selection")
            self.clear_selection()
            return
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
        self._run_started = time.monotonic()
        self._message_count = len(self.comp.agent.messages)
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
            await self._iterate(goal, history, transcript)
        except Exception as exc:  # noqa: BLE001
            # A provider auth failure, a dropped connection, a bad model id --
            # none of these should dump a Textual crash screen over the user's
            # session. Surface it where they are already looking, with the fix.
            transcript.error(_explain(exc, provider=self.comp.config.provider))
        finally:
            # Also runs on cancellation, so an interrupted run still checkpoints
            # and the UI never gets stuck showing a spinner.
            self._run_in_flight = False
            self._worker = None
            self._activity = ""
            if self.comp is not None:
                composition_root.checkpoint(self.comp)
            self._finish_run(transcript)
            self._refresh_status()

    def _finish_run(self, transcript: TranscriptView) -> None:
        """Summarise what changed, refresh the pane, and ring if it was long."""
        touched = self._undo.touched()
        if touched:
            plural = "" if len(touched) == 1 else "s"
            lines = tree_lines(touched, self.comp.agent.repo_root)
            transcript.notice(
                f"{len(touched)} file{plural} changed  (/undo reverts the last turn)\n"
                + "\n".join(lines)
            )
            self._refresh_changed()
        if self._bell_enabled and time.monotonic() - self._run_started > (
            _BELL_AFTER_SECONDS
        ):
            self.bell()

    def _note_compaction(self, transcript: TranscriptView) -> None:
        """Report the agent's own compaction, which yields no event.

        Rather than add an Event variant (a core change), watch the message
        count: it only ever shrinks when `agent.py:266` compacts.
        """
        count = len(self.comp.agent.messages)
        if self._message_count and count < self._message_count:
            transcript.notice(
                f"context was getting full — summarised "
                f"{self._message_count - count} older messages"
            )
        self._message_count = count

    async def _iterate(self, goal: str, history, transcript) -> None:
        """Drive the event stream. Errors propagate to `_drive_worker`."""
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
                        self._note_compaction(transcript)
                        # One undo batch per turn, so /undo reverts a step
                        # rather than a single write or the whole session.
                        self._undo.start_batch()
                    composition_root.checkpoint(self.comp)
                self._refresh_status()
        except CancelledError:
            # An interrupt is a normal outcome, not an error to report.
            self.query_one(TranscriptView).notice("interrupted")
