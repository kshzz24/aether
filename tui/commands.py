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
    # Actions only the app can perform. Kept as data so dispatch stays pure and
    # every command is testable without a running app.
    rebuild: dict | None = None      # config overrides -> rebuild the agent
    toggle_plan: bool = False
    compact: bool = False
    copy: bool = False
    reindex: bool = False
    find: str | None = None


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
    "/resume": "resume a saved session: /resume <id>",
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
    "/quit": "exit FORGE",
}

# Provider -> the environment variable holding its key. Mirrors main.ENV_KEYS;
# duplicated rather than imported because importing `main` from a command module
# would make the composition root a dependency of the thing it composes.
PROVIDERS = ("anthropic", "openai", "groq", "openrouter", "gemini", "together")

KEYS: dict[str, str] = {
    "enter": "send the prompt",
    "up / down": "walk back through what you have typed",
    "tab": "accept the suggested command",
    "escape": "interrupt the running agent",
    "ctrl+l": "clear the transcript",
    "ctrl+t": "switch theme",
    "ctrl+p": "command palette",
    "ctrl+c": "interrupt, or quit when idle",
    "pgup / pgdn": "scroll the transcript",
}


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


def _resume(arg: str, sessions_dir: Path) -> CommandResult:
    if not arg:
        return CommandResult("usage: /resume <id>  (see /sessions)")
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
        case "/quit":
            return CommandResult("bye", quit=True)
        case _:
            return CommandResult(f"unknown command {name!r} — try /help")
