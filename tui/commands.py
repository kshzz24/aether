"""Slash-command dispatch for the TUI.

`dispatch` is deliberately free of widgets, screens, and the app object: it takes
a context struct and returns a result struct. That keeps every command testable
without booting a Textual app, which is most of why the command surface can grow
without the test suite getting slower or flakier.

"Pure" here means *free of UI*, not free of I/O — `/save`, `/sessions` and
`/resume` touch the sessions directory, because that is what they are for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import persistence
from approval import ApprovalMode
from config import ForgeConfig
from tools.registry import ToolRegistry
from tools.todo import TodoStore
from tui.context import meter
from tui.filetree import tree_lines
from tui.templates import TEMPLATES_DIR, load_templates
from tui.undo import UndoStack


@dataclass(frozen=True)
class CommandContext:
    """Everything the commands are allowed to see."""

    config: ForgeConfig
    registry: ToolRegistry
    session: persistence.Session
    sessions_dir: Path
    total_cost: float
    turns: int
    todos: TodoStore | None = None
    # Per-turn costs, oldest first. The agent tracks only a running total, so
    # the breakdown is accumulated by the surface from CostEvents.
    turn_costs: tuple[float, ...] = ()
    plan_mode: bool = False
    # The write-snapshot stack behind `/undo` and `/files`.
    undo: UndoStack | None = None
    repo_root: Path | None = None
    bell: bool = True
    autocopy: bool = True
    context_tokens: int = 0
    context_budget: int = 0


@dataclass(frozen=True)
class CommandResult:
    """Text to show, plus any action only the app can carry out.

    `toast` marks output short and transient enough for a notification popup
    instead of a permanent transcript entry — "saved" is noise in a scrollback
    you are reading for the agent's reasoning.
    """

    text: str
    quit: bool = False
    resume_id: str | None = None
    clear: bool = False
    toast: bool = False
    setup: bool = False
    cycle_theme: bool = False
    # Actions only the app can perform. Kept as data so dispatch stays pure and
    # every command is testable without a running app.
    rebuild: dict | None = None      # config overrides -> rebuild the agent
    toggle_plan: bool = False
    compact: bool = False
    copy: bool = False
    reindex: bool = False
    find: str | None = None
    undo: bool = False
    toggle_bell: bool = False
    toggle_autocopy: bool = False
    # Text to drop into the prompt box instead of sending. `/prompt <name>`
    # fills the box so the template can be edited before it is submitted.
    fill_prompt: str | None = None
    pick_template: bool = False
    pick_session: bool = False


# name -> one-line help. The single source of truth for both `/help` and the
# "unknown command" message, so the two can never drift apart.
COMMANDS: dict[str, str] = {
    "/help": "show this list",
    "/config": "show the resolved configuration",
    "/tools": "list registered tools and their kinds",
    "/mcp": "show connected MCP servers",
    "/stats": "show cost and turn counts for this run",
    "/save": "checkpoint the session now",
    "/sessions": "list saved sessions",
    "/resume": "resume a saved session: /resume [id], or pick from a list",
    "/clear": "clear the transcript (the session is untouched)",
    "/keys": "show keyboard shortcuts",
    "/cost": "per-turn spend breakdown for this run",
    "/todo": "show the agent's task list",
    "/model": "switch model: /model <id>",
    "/provider": "switch provider: /provider <name>",
    "/approval": "switch approval mode: /approval auto|on-request|never",
    "/plan": "toggle plan mode — propose before editing",
    "/compact": "summarize the transcript to reclaim context",
    "/copy": "copy the transcript to the clipboard",
    "/find": "highlight matching transcript entries: /find <text>",
    "/reindex": "rebuild the @file completion index",
    "/undo": "revert the files the agent changed in its last turn",
    "/redo": "put back what /undo reverted",
    "/yolo": "approve every tool call for the rest of the session",
    "/files": "show the files changed this session",
    "/prompt": "insert a saved prompt: /prompt [name]",
    "/context": "show how full the context window is",
    "/bell": "toggle the notification bell on long runs",
    "/autocopy": "toggle copying the moment you finish selecting",
    "/setup": "choose a provider and model again",
    "/theme": "switch between the forge themes",
    "/quit": "exit FORGE",
}

# Provider -> the environment variable holding its key. Mirrors main.ENV_KEYS;
# duplicated rather than imported because importing `main` from a command module
# would make the composition root a dependency of the thing it composes.
PROVIDERS = ("anthropic", "openai", "groq", "openrouter", "gemini", "together")

KEYS: dict[str, str] = {
    # Prompt
    "enter": "send the prompt, or take the highlighted completion",
    "ctrl+j": "newline without sending — works in every terminal",
    "shift+enter": "newline, if your terminal can send it (many cannot)",
    "alt+enter": "newline, another terminal-dependent alias",
    "up / down": "walk the completion menu, or your prompt history",
    "tab": "accept the suggested command or @file",
    "ctrl+r": "fuzzy-search everything you have typed",
    # Transcript
    "pgup / pgdn": "scroll the transcript",
    "v": "select the entry under the cursor",
    "V": "pin the anchor, so j/k grow the selection",
    "j / k": "move the selection",
    "y": "yank the selected entries",
    "c": "copy the mouse selection, or the entry under the cursor",
    "C": "copy the last code block",
    "u": "undo the agent's last batch of file changes",
    # Panels
    "ctrl+b": "show what the agent changed",
    "ctrl+s": "choose a provider and model",
    "f1": "this list",
    "?": "this list (from the transcript)",
    # Run control
    "escape": "interrupt the agent, or leave select mode",
    "ctrl+c": "copy the selection · interrupt · quit when idle",
    "ctrl+l": "clear the transcript",
    "ctrl+t": "switch theme",
    "ctrl+p": "command palette",
    "ctrl+d": "quit",
}

# Which section of the `?` overlay each key belongs to. `/keys` prints KEYS flat;
# the overlay groups it, and a test asserts the two never drift apart.
KEY_GROUPS: dict[str, tuple[str, ...]] = {
    "Prompt": (
        "enter",
        "ctrl+j",
        "shift+enter",
        "alt+enter",
        "up / down",
        "tab",
        "ctrl+r",
    ),
    "Transcript": ("pgup / pgdn", "v", "V", "j / k", "y", "c", "C", "u"),
    "Panels": ("ctrl+b", "ctrl+s", "f1", "?"),
    "Run": ("escape", "ctrl+c", "ctrl+l", "ctrl+t", "ctrl+p", "ctrl+d"),
}

# Not keys, so they stay out of the "is this actually bound?" test — but the
# first one is the most useful thing in the app and nothing else advertises it.
MOUSE_TIPS: tuple[tuple[str, str], ...] = (
    ("drag", "select text — it copies as soon as you let go (/autocopy)"),
    ("shift+drag", "hand selection to the terminal itself, bypassing FORGE"),
    ("click", "fold or unfold a tool call"),
    ("scroll", "move through the transcript"),
)


def _help() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"  {name:<{width}}  {desc}" for name, desc in COMMANDS.items()]
    return "commands:\n" + "\n".join(lines)


def _config(cfg: ForgeConfig) -> str:
    rows = [
        ("provider", cfg.provider),
        ("model", cfg.model),
        ("max_iterations", cfg.max_iterations),
        ("max_cost_usd", f"${cfg.max_cost_usd:.2f}"),
        ("approval_mode", cfg.approval_mode.value),
        ("allowlist", "all tools" if cfg.allowlist is None else sorted(cfg.allowlist)),
        ("skills_dir", cfg.skills_dir),
        ("user_tools_dir", cfg.user_tools_dir),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)


def _tools(registry: ToolRegistry) -> str:
    tools = sorted(registry.list(), key=lambda t: t.name)
    if not tools:
        return "no tools registered"
    width = max(len(t.name) for t in tools)
    lines = [f"  {t.name:<{width}}  {t.kind.name.lower()}" for t in tools]
    return f"{len(tools)} tools:\n" + "\n".join(lines)


def _mcp() -> str:
    # TODO(phase-6): replace with real per-server status once mcpclient lands.
    return "no MCP servers connected (Phase 6 not built yet)"


def _stats(ctx: CommandContext) -> str:
    return (
        f"  session       {ctx.session.id}\n"
        f"  turns         {ctx.turns}\n"
        f"  cost          ${ctx.total_cost:.4f} of ${ctx.config.max_cost_usd:.2f}\n"
        f"  iteration cap {ctx.config.max_iterations}"
    )


def _keys() -> str:
    width = max(len(k) for k in KEYS)
    return "keys:\n" + "\n".join(f"  {k:<{width}}  {v}" for k, v in KEYS.items())


def _save(ctx: CommandContext) -> str:
    ctx.session.updated_at = datetime.now().isoformat(timespec="seconds")
    persistence.save(ctx.session, ctx.sessions_dir)
    return f"saved {ctx.session.id}"


def _sessions(sessions_dir: Path) -> str:
    metas = persistence.list_sessions(sessions_dir)
    if not metas:
        return "no saved sessions"
    lines = []
    for m in metas:
        goal = m.goal if len(m.goal) <= 50 else m.goal[:47] + "..."
        lines.append(f"  {m.id}  ${m.total_cost:.4f}  {m.turns:>3} turns  {goal}")
    return "\n".join(lines)


def _cost(ctx: CommandContext) -> str:
    if not ctx.turn_costs:
        return f"no turns yet · budget ${ctx.config.max_cost_usd:.2f}"
    total = ctx.total_cost
    budget = ctx.config.max_cost_usd
    lines = [
        f"  {'turn':<6} {'cost':>10}",
        *(
            f"  {i:<6} {c:>10.4f}"
            for i, c in enumerate(ctx.turn_costs, start=1)
        ),
        f"  {'total':<6} {total:>10.4f}   of ${budget:.2f} "
        f"({total / budget * 100:.0f}% used)" if budget else "",
        f"  {'mean':<6} {total / len(ctx.turn_costs):>10.4f}",
    ]
    return "\n".join(line for line in lines if line)


def _todos(store: TodoStore | None) -> str:
    if store is None:
        return "no task list available"
    return store.render()


def _model(arg: str, ctx: CommandContext) -> CommandResult:
    if not arg:
        return CommandResult(f"current model: {ctx.config.model}\nusage: /model <id>")
    return CommandResult(f"model -> {arg}", rebuild={"model": arg}, toast=True)


def _provider(arg: str, ctx: CommandContext) -> CommandResult:
    if not arg:
        return CommandResult(
            f"current provider: {ctx.config.provider}\n"
            f"available: {', '.join(PROVIDERS)}"
        )
    if arg not in PROVIDERS:
        return CommandResult(
            f"unknown provider {arg!r} — one of: {', '.join(PROVIDERS)}"
        )
    return CommandResult(f"provider -> {arg}", rebuild={"provider": arg}, toast=True)


def _approval(arg: str, ctx: CommandContext) -> CommandResult:
    modes = [m.value for m in ApprovalMode]
    if not arg:
        return CommandResult(
            f"current approval mode: {ctx.config.approval_mode.value}\n"
            f"available: {', '.join(modes)}"
        )
    if arg not in modes:
        return CommandResult(f"unknown mode {arg!r} — one of: {', '.join(modes)}")
    return CommandResult(
        f"approval -> {arg}", rebuild={"approval_mode": arg}, toast=True
    )


def _plan(ctx: CommandContext) -> CommandResult:
    now_on = not ctx.plan_mode
    detail = (
        "plan mode ON — the agent will propose a plan and ask before every "
        "write or command"
        if now_on
        else "plan mode OFF — back to the configured approval mode"
    )
    return CommandResult(detail, toggle_plan=True)


def _find(arg: str) -> CommandResult:
    if not arg:
        return CommandResult("usage: /find <text>   (empty /find clears)", find="")
    return CommandResult(f"highlighting {arg!r}", find=arg)


def _undo(ctx: CommandContext) -> CommandResult:
    """Revert the last turn's writes.

    The work happens here rather than in the app for the same reason `/save`
    does: dispatch is free of *UI*, not of I/O, and a command that touches the
    filesystem is far easier to test as a function than as a keypress.
    """
    if ctx.undo is None:
        return CommandResult("undo is not available")
    result = ctx.undo.undo_last()
    return CommandResult(result.summary(), undo=True, toast=result.changed > 0)


def _redo(ctx: CommandContext) -> CommandResult:
    if ctx.undo is None:
        return CommandResult("redo is not available")
    result = ctx.undo.redo_last()
    return CommandResult(
        result.summary("nothing to redo"), undo=True, toast=result.changed > 0
    )


def _files(ctx: CommandContext) -> str:
    if ctx.undo is None or ctx.repo_root is None:
        return "no file history available"
    touched = ctx.undo.touched()
    if not touched:
        return "the agent has not changed any files this session"
    lines = tree_lines(touched, ctx.repo_root)
    plural = "" if len(touched) == 1 else "s"
    return f"{len(touched)} file{plural} changed:\n" + "\n".join(lines)


def _context(ctx: CommandContext) -> str:
    used, budget = ctx.context_tokens, ctx.context_budget
    if budget <= 0:
        return f"~{used:,} tokens in context (no budget configured)"
    return (
        f"  {meter(used, budget, width=24)}\n"
        f"  ~{used:,} of {budget:,} tokens\n"
        f"  the agent compacts automatically as this fills; /compact forces it"
    )


def _prompt(arg: str, ctx: CommandContext) -> CommandResult:
    templates = load_templates(TEMPLATES_DIR)
    if not templates:
        return CommandResult(
            f"no saved prompts — put markdown files in {TEMPLATES_DIR}"
        )
    if not arg:
        return CommandResult("", pick_template=True)
    if arg not in templates:
        return CommandResult(
            f"unknown prompt {arg!r} — have: {', '.join(sorted(templates))}"
        )
    # Filled into the box rather than sent, so it can be edited first. A
    # template is a starting point; sending it verbatim is rarely what you want.
    return CommandResult("", fill_prompt=templates[arg])


def _bell(ctx: CommandContext) -> CommandResult:
    state = "off" if ctx.bell else "on"
    return CommandResult(f"notification bell {state}", toggle_bell=True, toast=True)


def _autocopy(ctx: CommandContext) -> CommandResult:
    """Selecting with the mouse copies immediately, as a terminal does.

    Toggleable because it is one toast per drag, which is the right trade when
    you are copying and the wrong one when you are just highlighting to read.
    """
    state = "off" if ctx.autocopy else "on"
    return CommandResult(
        f"copy-on-select {state}"
        + ("" if ctx.autocopy else " — drag to copy, no ctrl+c needed"),
        toggle_autocopy=True,
        toast=True,
    )


def _yolo(ctx: CommandContext) -> CommandResult:
    """Approve everything for the rest of the session.

    Named for what people actually call it. It is a real mode, not a joke: an
    agent working through a long refactor in a scratch checkout should not stop
    forty times. The warning is part of the output because the danger checks
    stop applying too.
    """
    if ctx.config.approval_mode is ApprovalMode.AUTO:
        return CommandResult("already in auto-approve mode")
    return CommandResult(
        "auto-approve ON — every tool call runs without asking, including "
        "flagged ones. /approval on-request puts the guard rails back.",
        rebuild={"approval_mode": ApprovalMode.AUTO.value},
    )


def _resume(arg: str, sessions_dir: Path) -> CommandResult:
    if not arg:
        # No id: offer a picker rather than making the user run /sessions,
        # read an id off the screen, and type it back in.
        if not persistence.list_sessions(sessions_dir):
            return CommandResult("no saved sessions")
        return CommandResult("", pick_session=True)
    try:
        session = persistence.load(arg, sessions_dir)
    except (OSError, ValueError) as exc:
        return CommandResult(f"cannot resume {arg!r}: {exc}")
    return CommandResult(f"resumed {session.id}: {session.goal}", resume_id=arg)


def dispatch(line: str, ctx: CommandContext) -> CommandResult | None:
    """Handle `line` if it is a slash command.

    Returns None when the line is an ordinary goal, so the caller knows to send
    it to the agent instead. An unrecognised slash command returns an error
    result rather than None — otherwise a typo like `/toolz` would be silently
    forwarded to the model as a goal, and cost real money to find out.
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None

    name, _, arg = stripped.partition(" ")
    name = name.lower()
    arg = arg.strip()

    match name:
        case "/help":
            return CommandResult(_help())
        case "/config":
            return CommandResult(_config(ctx.config))
        case "/tools":
            return CommandResult(_tools(ctx.registry))
        case "/mcp":
            return CommandResult(_mcp())
        case "/stats":
            return CommandResult(_stats(ctx))
        case "/keys":
            return CommandResult(_keys())
        case "/save":
            return CommandResult(_save(ctx), toast=True)
        case "/sessions":
            return CommandResult(_sessions(ctx.sessions_dir))
        case "/resume":
            return _resume(arg, ctx.sessions_dir)
        case "/clear":
            return CommandResult("", clear=True)
        case "/cost":
            return CommandResult(_cost(ctx))
        case "/todo":
            return CommandResult(_todos(ctx.todos))
        case "/model":
            return _model(arg, ctx)
        case "/provider":
            return _provider(arg, ctx)
        case "/approval":
            return _approval(arg, ctx)
        case "/plan":
            return _plan(ctx)
        case "/compact":
            return CommandResult("compacting transcript...", compact=True)
        case "/copy":
            return CommandResult("transcript copied", copy=True, toast=True)
        case "/find":
            return _find(arg)
        case "/reindex":
            return CommandResult("rebuilding @file index...", reindex=True)
        case "/undo":
            return _undo(ctx)
        case "/redo":
            return _redo(ctx)
        case "/yolo":
            return _yolo(ctx)
        case "/files":
            return CommandResult(_files(ctx))
        case "/prompt":
            return _prompt(arg, ctx)
        case "/context":
            return CommandResult(_context(ctx))
        case "/bell":
            return _bell(ctx)
        case "/autocopy":
            return _autocopy(ctx)
        case "/setup":
            return CommandResult("", setup=True)
        case "/theme":
            return CommandResult("", cycle_theme=True)
        case "/quit":
            return CommandResult("bye", quit=True)
        case _:
            return CommandResult(f"unknown command {name!r} — try /help")
