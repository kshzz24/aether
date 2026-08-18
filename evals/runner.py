import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from agent import Agent
from client import Message, NormalizedResponse
from events import TerminalEvent, TerminalReason

_PRICES_PATH = Path(__file__).resolve().parent / "prices.toml"


@dataclass(frozen=True)
class GoldenTask:
    name: str
    goal: str
    check: Callable[[Path], bool]
    # Seed the workspace with input fixtures before the agent runs. A read task
    # with nothing to read can never pass; this is where its inputs come from.
    setup: Callable[[Path], None] = lambda ws: None
    # {content} template for an LLM-as-judge check. Live suite only -- a stub
    # can't judge anything, so this never runs in CI.
    judge_prompt: str | None = None
    # The exact response sequence for the deterministic (CI) suite's
    # StubClient. An entry may be an Exception instance instead of a
    # NormalizedResponse -- evals.deterministic's client raises it, which is
    # how `malformed-call-repair` exercises agent.py's repair path.
    # None means this task never runs in evals.deterministic.
    scripted: list[NormalizedResponse | Exception] | None = field(
        default=None, repr=False
    )
    # Reads the judged artifact's text out of the finished workspace, fed into
    # `judge_prompt`'s {content}. Only consulted when judge_prompt is set.
    judge_target: Callable[[Path], str] = lambda ws: ""
    # A second check that inspects the finished conversation, not just the
    # filesystem -- what `redacts-a-leaked-secret` needs to assert a
    # ToolResultBlock's content was redacted, which never touches disk.
    check_transcript: Callable[[list[Message]], bool] | None = None
    # False for a task that proves harness plumbing (repair, guardrails)
    # rather than model capability -- it has no live-model form, so the live
    # gate (`main()`) skips it. Deterministic-only tasks must also set
    # `scripted`, or nothing ever runs them.
    live: bool = True


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    reason: str


async def run_task(agent: Agent, task: GoldenTask, workspace: Path) -> EvalResult:
    task.setup(workspace)
    events = [event async for event in agent.run(task.goal)]
    terminal = events[-1] if events else None

    # A pass requires a *clean* finish. An artifact left by an aborted run
    # (loop detected, cap hit) does not count -- that's the whole point of the
    # third test.
    if not (
        isinstance(terminal, TerminalEvent)
        and terminal.reason is TerminalReason.COMPLETED
    ):
        reason = (
            terminal.reason.name
            if isinstance(terminal, TerminalEvent)
            else "NO_TERMINAL"
        )
        return EvalResult(task.name, passed=False, reason=reason)

    passed = task.check(workspace)
    if passed and task.check_transcript is not None:
        passed = task.check_transcript(agent.messages)
    if not passed:
        return EvalResult(task.name, False, "CHECK_FAILED")

    if task.judge_prompt is None:
        return EvalResult(task.name, True, "COMPLETED")

    # Live suite only: a stub can't judge anything, and a task with a
    # judge_prompt but no scripted responses never reaches here from CI.
    from evals.judge import judge

    content = task.judge_target(workspace)
    ok, reason = judge(task.judge_prompt.format(content=content), content)
    return EvalResult(task.name, ok, "COMPLETED" if ok else f"JUDGE_FAILED: {reason}")


@contextmanager
def _chdir(path: Path):
    # Run the agent with `path` as cwd so its tools' relative writes land in the
    # task's workspace. Tasks run sequentially, so this process-wide chdir is
    # safe (no concurrent runs to race it).
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


async def run_suite(
    tasks: list[GoldenTask],
    make_agent: Callable[[GoldenTask], Agent],
    base_dir: str | Path | None = None,
) -> list[EvalResult]:
    """Run each task in its own fresh workspace and collect the results.

    Pure of stdout: it returns EvalResults; presentation belongs to main().
    """
    results = []
    for task in tasks:
        workspace = Path(
            tempfile.mkdtemp(prefix=f"forge-eval-{task.name}-", dir=base_dir)
        )
        with _chdir(workspace):
            result = await run_task(make_agent(task), task, workspace)
        results.append(result)
    return results


def main() -> None:
    """Live gate step: run the golden suite against a real provider.

    Not a pytest test -- a live model would flake the suite. This is the
    eval's presentation boundary (the one place here allowed to write stdout),
    analogous to the agent's renderer.
    """
    import argparse
    import asyncio
    import tomllib

    from client import make_client
    from config import ForgeConfig
    from evals.tasks import GOLDEN_TASKS
    from main import ENV_KEYS, SYSTEM
    from tools import build_registry

    parser = argparse.ArgumentParser(
        prog="forge-eval", description="Run FORGE's golden-task smoke suite."
    )
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--max-iter", dest="max_iter", type=int, default=25)
    parser.add_argument("--max-cost", dest="max_cost", type=float, default=1.0)
    args = parser.parse_args()

    with open(_PRICES_PATH, "rb") as f:
        prices = tomllib.load(f)
    api_key = os.environ.get(ENV_KEYS.get(args.provider, ""), "")
    rates = prices.get(args.provider, {})

    config = ForgeConfig(
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iter,
        max_cost_usd=args.max_cost,
    )

    def make_agent(task: GoldenTask) -> Agent:
        client = make_client(
            provider=args.provider, model=args.model, api_key=api_key, rates=rates
        )
        return Agent(
            client=client,
            model=args.model,
            registry=build_registry(config),
            system=SYSTEM,
            max_iterations=args.max_iter,
            max_cost_usd=args.max_cost,
        )

    live_tasks = [t for t in GOLDEN_TASKS if t.live]
    results = asyncio.run(run_suite(live_tasks, make_agent))

    passed = sum(1 for r in results if r.passed)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name} ({r.reason})")
    print(f"{passed}/{len(results)} passed")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
