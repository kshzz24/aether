"""Deterministic (CI) eval gate: the same GoldenTask list, driven by a StubClient.

No network, no API key, no cost -- every task run here carries `scripted`, an
exact response sequence, so the suite is fully reproducible.

Named limitation, stated plainly: a scripted response can drift from what a
real model would actually produce. This suite proves the *harness and tool
wiring* didn't regress (a broken tool, a broken loop, a broken repair path) --
it is not a substitute for the live gate (`evals.runner.main`), which is the
only thing that tests real model capability.
"""

from __future__ import annotations

import asyncio

from client import NormalizedResponse
from config import ForgeConfig
from evals.runner import GoldenTask, run_suite
from evals.tasks import GOLDEN_TASKS
from main import SYSTEM
from tools import build_registry

_MAX_ITERATIONS = 10
_MAX_COST_USD = 10.0


class _ScriptedStubClient:
    """Pops scripted entries in order; an Exception entry is raised, not returned.

    This is what lets `malformed-call-repair` script a first call that fails
    with `ToolCallingUnsupportedError` and a second that succeeds -- the plain
    response-popping StubClient used elsewhere has no way to raise.
    """

    def __init__(self, script: list[NormalizedResponse | Exception]) -> None:
        self._script = list(script)

    async def create(self, messages, tools, system) -> NormalizedResponse:
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_stub_agent(task: GoldenTask):
    """Build an Agent identical to the live suite's, except for the client."""
    from agent import Agent

    config = ForgeConfig()
    return Agent(
        client=_ScriptedStubClient(task.scripted),
        model="stub-model",
        registry=build_registry(config),
        system=SYSTEM,
        max_iterations=_MAX_ITERATIONS,
        max_cost_usd=_MAX_COST_USD,
    )


def main() -> None:
    scripted_tasks = [t for t in GOLDEN_TASKS if t.scripted is not None]

    results = asyncio.run(run_suite(scripted_tasks, make_stub_agent))

    passed = sum(1 for r in results if r.passed)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name} ({r.reason})")
    print(f"{passed}/{len(results)} passed")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
