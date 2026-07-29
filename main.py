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
from pathlib import Path

from agent import Agent
from cli.approver import CliApprover
from cli.renderer import Renderer
from client import make_client
from config import load_config
from gateway.client import GatewayClient
from policy import PolicyEngine
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


async def _run(goal: str, args: argparse.Namespace) -> None:
    # gateway_url is a composition-root concern, not validated config: ForgeConfig
    # uses extra="forbid", so pull it out before the merge. (For .forge/config.toml
    # support, add a gateway_url field to ForgeConfig; left CLI-only here by scope.)

    gateway_url = args.gateway_url

    # Only flags the user *explicitly* set reach the config merge; unset flags
    # are None sentinels and must not clobber file/default config.
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ("goal", "gateway_url") and v is not None
    }
    config = load_config(cli_overrides)

    with open("prices.toml", "rb") as f:
        prices = tomllib.load(f)
    api_key = os.environ.get(ENV_KEYS.get(config.provider, ""), "")
    rates = prices.get(config.provider, {})

    # The direct provider client. With a gateway configured it becomes the
    # degrade-to-passthrough fallback; otherwise the agent talks to it directly.
    client = make_client(
        provider=config.provider, model=config.model, api_key=api_key, rates=rates
    )
    if gateway_url:
        client = GatewayClient(
            gateway_url=gateway_url,
            model=config.model,
            fallback=client,
            rates=rates,
        )
    approver = CliApprover()
    repo_root = find_repo_root(Path.cwd()) or Path.cwd()
    child_registry = build_registry(config)

    def make_child() -> Agent:
        return Agent(
            client=client,
            model=config.model,
            registry=child_registry,
            system=SUBAGENT_SYSTEM,
            max_iterations=config.subagent_max_iterations,
            max_cost_usd=config.subagent_max_cost_usd,
            policy=PolicyEngine(config.approval_mode),
            approver=approver,
            repo_root=repo_root,
            hooks=Hooks(),
        )

    registry = build_registry(config)
    registry.register(build_subagent_tool(make_child=make_child))

    agent = Agent(
        client=client,
        model=config.model,
        registry=registry,
        system=SYSTEM,
        max_iterations=config.max_iterations,
        max_cost_usd=config.max_cost_usd,
        policy=PolicyEngine(config.approval_mode),
        approver=approver,
        repo_root=repo_root,
        hooks=Hooks(),
    )

    renderer = Renderer()
    async for event in agent.run(goal):
        renderer.render(event)


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
    parser.add_argument("goal", help="the task for the agent to accomplish")
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
    args = parser.parse_args()
    asyncio.run(_run(args.goal, args))


if __name__ == "__main__":
    main()
