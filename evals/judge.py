"""LLM-as-judge for golden tasks whose output can't be checked structurally.

Live suite only: calls a real model, so it never runs in CI (`evals.deterministic`
only drives tasks with `scripted` responses, and a judged task has none).
"""

import asyncio
import os

from client import Message, TextBlock, make_client

_JUDGE_PROVIDER = os.environ.get("FORGE_JUDGE_PROVIDER", "groq")
_JUDGE_MODEL = os.environ.get("FORGE_JUDGE_MODEL", "llama-3.3-70b-versatile")

_INSTRUCTION = (
    "Answer PASS or FAIL on the first line, then one sentence why.\n\n{prompt}"
)


def judge(prompt: str, content: str) -> tuple[bool, str]:
    """Ask a cheap model to grade `content` against `prompt`.

    Returns (passed, explanation). Any malformed answer (no PASS/FAIL on the
    first line) is treated as a failure -- a judge that can't be parsed cannot
    be trusted to have passed the task.
    """
    api_key = os.environ.get(
        {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(_JUDGE_PROVIDER, ""),
        "",
    )
    client = make_client(
        provider=_JUDGE_PROVIDER, model=_JUDGE_MODEL, api_key=api_key, rates={}
    )

    async def _ask() -> str:
        response = await client.create(
            messages=[
                Message(
                    role="user",
                    blocks=[TextBlock(text=_INSTRUCTION.format(prompt=prompt))],
                )
            ],
            tools=[],
            system="You are a strict, terse grader.",
        )
        return "".join(b.text for b in response.blocks if isinstance(b, TextBlock))

    answer = asyncio.run(_ask()).strip()
    first_line = answer.splitlines()[0].strip().upper() if answer else ""
    if not (first_line.startswith("PASS") or first_line.startswith("FAIL")):
        return False, f"unparseable judge answer: {answer!r}"
    return first_line.startswith("PASS"), answer
