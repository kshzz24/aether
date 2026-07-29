"""FORGE entry point: wire a client + tools + the agent loop, and render events.

This is the composition root. It is allowed to read argv and the environment and
to drive the renderer, but it does NOT contain agent logic and does NOT print
directly (the renderer owns stdout).
"""

import argparse
import asyncio
import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import persistence
from agent import Agent
from cli.approver import CliApprover
from cli.renderer import Renderer
from client import make_client
from config import load_config
from events import StatusEvent, TerminalEvent
from gateway.client import GatewayClient
from policy import PolicyEngine
from skills import build_skill_registry, build_skill_tool, render_skill_menu
from tools import build_registry
from tools.hooks import Hooks
from tools.subagent import build_subagent_tool
from tools.traversal import find_repo_root

# Per-token (USD) pricing as (input_rate, output_rate). $/token = $/Mtok / 1e6.

SUBAGENT_SYSTEM = (
    "You are a FORGE subagent: a focused worker spawned to carry out one "
    "delegated task. You have the full toolset. Work in small steps, then end "
    "with a concise summary of what you did or found — that summary is the only "
    "thing returned to the parent agent."
)
# Environment variable holding the API key, per provider.
ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "together": "TOGETHER_API_KEY",
}

SYSTEM = (
    "You are FORGE, a coding assistant operating in a terminal workspace. "
    "You can read, write, and edit files and run shell commands via tools. "
    "Inspect before you change, work in small steps, and stop once the task is "
    "complete."
)

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "skills" / "builtin"


async def _run(args: argparse.Namespace) -> None:
    sessions_dir = persistence.default_sessions_dir()
    renderer = Renderer()

    # Resume: load the saved session. Its provider/model/goal drive the run so it
    # continues faithfully; the saved messages seed the loop as history.
    resume_session = None
    if args.resume:
        try:
            resume_session = persistence.load(args.resume, sessions_dir)
        except (OSError, ValueError) as exc:
            renderer.notice(f"cannot resume {args.resume!r}: {exc}")
            return

    # gateway_url is a composition-root concern, not validated config: ForgeConfig
    # uses extra="forbid", so pull it out before the merge. (For .forge/config.toml
    # support, add a gateway_url field to ForgeConfig; left CLI-only here by scope.)
    gateway_url = args.gateway_url

    # Only flags the user *explicitly* set reach the config merge; unset flags
    # are None sentinels and must not clobber file/default config.
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ("goal", "gateway_url", "resume", "list_sessions") and v is not None
    }
    config = load_config(cli_overrides)

    # On resume the session's provider/model/goal win; otherwise use config + argv.
    provider = resume_session.provider if resume_session else config.provider
    model = resume_session.model if resume_session else config.model
    goal = resume_session.goal if resume_session else args.goal
    history = resume_session.messages if resume_session else None
    skill_registry = build_skill_registry(BUILTIN_SKILLS_DIR, config.skills_dir)
    skill_tool = build_skill_tool(registry=skill_registry)
    menu = render_skill_menu(skill_registry.list())
    system = SYSTEM + (f"\n\n{menu}" if menu else "")
    subagent_system = SUBAGENT_SYSTEM + (f"\n\n{menu}" if menu else "")

    with open("prices.toml", "rb") as f:
        prices = tomllib.load(f)
    api_key = os.environ.get(ENV_KEYS.get(provider, ""), "")
    rates = prices.get(provider, {})

    # The direct provider client. With a gateway configured it becomes the
    # degrade-to-passthrough fallback; otherwise the agent talks to it directly.
    client = make_client(provider=provider, model=model, api_key=api_key, rates=rates)
    if gateway_url:
        client = GatewayClient(
            gateway_url=gateway_url,
            model=model,
            fallback=client,
            rates=rates,
        )
    approver = CliApprover()
    repo_root = find_repo_root(Path.cwd()) or Path.cwd()
    child_registry = build_registry(config)
    child_registry.register(skill_tool)

    def make_child() -> Agent:
        return Agent(
            client=client,
            model=model,
            registry=child_registry,
            system=subagent_system,
            max_iterations=config.subagent_max_iterations,
            max_cost_usd=config.subagent_max_cost_usd,
            policy=PolicyEngine(config.approval_mode),
            approver=approver,
            repo_root=repo_root,
            hooks=Hooks(),
        )

    registry = build_registry(config)
    registry.register(build_subagent_tool(make_child=make_child))
    registry.register(skill_tool)

    agent = Agent(
        client=client,
        model=model,
        registry=registry,
        system=system,
        max_iterations=config.max_iterations,
        max_cost_usd=config.max_cost_usd,
        policy=PolicyEngine(config.approval_mode),
        approver=approver,
        repo_root=repo_root,
        hooks=Hooks(),
    )

    # The session we auto-checkpoint after every turn (crash-recoverable).
    now = datetime.now().isoformat(timespec="seconds")
    session = resume_session or persistence.Session(
        id=persistence.new_session_id(),
        goal=goal,
        provider=provider,
        model=model,
        created_at=now,
        updated_at=now,
        total_cost=0.0,
        messages=[],
    )
    renderer.notice(f"session {session.id}")

    async for event in agent.run(goal, history=history):
        renderer.render(event)
        # StatusEvent fires at each turn's start (state complete through the prior
        # turn); TerminalEvent at the end. Snapshot on both — a crash loses at
        # most the in-flight turn.
        if isinstance(event, (StatusEvent, TerminalEvent)):
            session.messages = agent.messages
            session.total_cost = agent.total_cost
            session.updated_at = datetime.now().isoformat(timespec="seconds")
            persistence.save(session, sessions_dir)


def main() -> None:
    # Windows consoles default to cp1252; model output routinely contains
    # characters outside it (em-dashes, smart quotes, non-breaking hyphens).
    # Render as UTF-8 so a stray character never crashes a run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="forge", description="FORGE - an agentic CLI coding assistant"
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=None,
        help="the task for the agent (omit with --resume / --list-sessions)",
    )
    # default=None so unset flags fall through to file/default config layers.
    parser.add_argument("--provider", default=None, help="LLM provider")
    parser.add_argument("--model", default=None, help="model id")
    parser.add_argument(
        "--max-iter",
        dest="max_iterations",
        type=int,
        default=None,
        help="maximum agent loop iterations",
    )
    parser.add_argument(
        "--max-cost",
        dest="max_cost_usd",
        type=float,
        default=None,
        help="maximum spend in USD before the run aborts",
    )
    parser.add_argument(
        "--gateway-url",
        dest="gateway_url",
        default=None,
        help=(
            "base URL of the FORGE gateway (e.g. http://localhost:8000/v1); "
            "when set, requests route through it and fall back to a direct "
            "provider call if it is unreachable"
        ),
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        default=None,
        metavar="ID",
        help="resume a saved session by id, continuing its original goal",
    )
    parser.add_argument(
        "--list-sessions",
        dest="list_sessions",
        action="store_true",
        help="list saved sessions and exit",
    )
    args = parser.parse_args()

    if args.list_sessions:
        metas = persistence.list_sessions(persistence.default_sessions_dir())
        Renderer().sessions(metas)
        return

    if args.resume is None and args.goal is None:
        parser.error("a goal is required (or use --resume <id> / --list-sessions)")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
