"""FORGE entry point: wire a client + tools + the agent loop, and render events.

This is the composition root. It is allowed to read argv and the environment and
to drive the renderer, but it does NOT contain agent logic and does NOT print
directly (the renderer owns stdout).
"""

import argparse
import asyncio
import os
import tomllib

from agent import Agent
from cli.renderer import Renderer
from client import make_client
from config import load_config
from tools import build_registry
from tools.hooks import Hooks

# Per-token (USD) pricing as (input_rate, output_rate). $/token = $/Mtok / 1e6.


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
    # Only flags the user *explicitly* set reach the config merge; unset flags
    # are None sentinels and must not clobber file/default config.
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if k != "goal" and v is not None
    }
    config = load_config(cli_overrides)

    with open("prices.toml", "rb") as f:
        prices = tomllib.load(f)
    api_key = os.environ.get(ENV_KEYS.get(config.provider, ""), "")
    rates = prices.get(config.provider, {})
    client = make_client(
        provider=config.provider, model=config.model, api_key=api_key, rates=rates
    )

    agent = Agent(
        client=client,
        model=config.model,
        registry=build_registry(config),
        system=SYSTEM,
        max_iterations=config.max_iterations,
        max_cost_usd=config.max_cost_usd,
        auto_approve=config.auto_approve,
        hooks=Hooks(),
    )

    renderer = Renderer()
    async for event in agent.run(goal):
        renderer.render(event)


def main() -> None:
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
    args = parser.parse_args()
    asyncio.run(_run(args.goal, args))


if __name__ == "__main__":
    main()
