"""FORGE entry point: wire a client + tools + the agent loop, and render events.

This is the composition root. It is allowed to read argv and the environment and
to drive the renderer, but it does NOT contain agent logic and does NOT print
directly (the renderer owns stdout).
"""

import argparse
import asyncio
import os

from agent import Agent
from cli.renderer import Renderer
from client import make_client
from tools import build_tools

# Per-token (USD) pricing as (input_rate, output_rate). $/token = $/Mtok / 1e6.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5e-6, 25e-6),
    "claude-opus-4-7": (5e-6, 25e-6),
    "claude-sonnet-4-6": (3e-6, 15e-6),
    "claude-haiku-4-5": (1e-6, 5e-6),
}

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


async def _run(args: argparse.Namespace) -> None:
    api_key = os.environ.get(ENV_KEYS.get(args.provider, ""), "")
    client = make_client(provider=args.provider, model=args.model, api_key=api_key)

    agent = Agent(
        client=client,
        model=args.model,
        tools=build_tools(),
        system=SYSTEM,
        max_iterations=args.max_iter,
        max_cost_usd=args.max_cost,
        pricing={args.model: PRICING.get(args.model, (0.0, 0.0))},
    )

    renderer = Renderer()
    async for event in agent.run(args.goal):
        renderer.render(event)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="forge", description="FORGE - an agentic CLI coding assistant"
    )
    parser.add_argument("goal", help="the task for the agent to accomplish")
    parser.add_argument("--provider", default="anthropic", help="LLM provider")
    parser.add_argument("--model", default="claude-opus-4-8", help="model id")
    parser.add_argument(
        "--max-iter", dest="max_iter", type=int, default=25,
        help="maximum agent loop iterations",
    )
    parser.add_argument(
        "--max-cost", dest="max_cost", type=float, default=1.0,
        help="maximum spend in USD before the run aborts",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
