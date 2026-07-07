"""Specs for the eval smoke test (piece #5, half B).

Design decisions pinned here (chosen, not yet ratified -- veto welcome):

- The committed test drives the runner with a *scripted* client, not a live
  model: it guards loop/tool/runner plumbing deterministically and for free.
  Running the real GOLDEN_TASKS through a live provider is a manual gate step,
  like the real-Groq cost check.
- A task passes iff the run ends on a clean TerminalReason.COMPLETED *and* the
  artifact check passes. An artifact left behind by an aborted run must NOT
  count -- that's the difference between "it finished the job" and "it happened
  to touch a file before dying".
- Checks are the cheapest thing that still catches a regression: the artifact
  exists and contains an expected marker. Each golden check must also reject an
  empty workspace, or it's vacuous and guards nothing.

Runner interface spec'd:
    GoldenTask(name, goal, check: Callable[[Path], bool])
    EvalResult(name, passed, reason)
    run_task(agent, task, workspace) -> EvalResult
    evals.tasks.GOLDEN_TASKS -> exactly 3 GoldenTasks
"""

import asyncio

from agent import Agent
from client import NormalizedResponse, TextBlock, ToolCallBlock
from config import ForgeConfig
from tools import build_registry
from tools.base import Tool, ToolKind
from tools.registry import ToolRegistry


class ScriptedClient:
    """Returns canned responses in order; no network, fully deterministic."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)

    async def create(self, messages, tools, system) -> NormalizedResponse:
        return self._responses.pop(0)


def _tool_use(name, arguments):
    return NormalizedResponse(
        blocks=[ToolCallBlock(id="c1", name=name, arguments=arguments)],
        input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
    )


def _finish(text="done"):
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
    )


def _agent(client, max_iterations=5):
    return Agent(
        client=client, model="m", registry=build_registry(ForgeConfig()),
        system="s", max_iterations=max_iterations, max_cost_usd=10.0,
    )


def test_run_task_passes_when_completed_and_check_true(tmp_path):
    from evals.runner import GoldenTask, run_task

    out = tmp_path / "out.txt"
    client = ScriptedClient(
        [_tool_use("write_file", {"path": str(out), "content": "has a TODO here"}),
         _finish()]
    )
    task = GoldenTask(
        name="write-out",
        goal="write out.txt containing a TODO",
        check=lambda ws: (ws / "out.txt").exists()
        and "TODO" in (ws / "out.txt").read_text(encoding="utf-8"),
    )

    result = asyncio.run(run_task(_agent(client), task, tmp_path))

    assert result.name == "write-out"
    assert result.passed is True


def test_run_task_fails_when_completed_but_artifact_missing(tmp_path):
    from evals.runner import GoldenTask, run_task

    # The run ends cleanly but never produced the artifact.
    client = ScriptedClient([_finish("I chose to do nothing")])
    task = GoldenTask(
        name="missing",
        goal="write out.txt",
        check=lambda ws: (ws / "out.txt").exists(),
    )

    result = asyncio.run(run_task(_agent(client), task, tmp_path))

    assert result.passed is False


def test_run_task_fails_when_run_aborts_even_if_artifact_exists(tmp_path):
    from evals.runner import GoldenTask, run_task

    # Writes the artifact, then never stops -> aborts on the iteration cap.
    # A leftover file must NOT be scored as a pass without a clean COMPLETED.
    out = tmp_path / "out.txt"
    call = _tool_use("write_file", {"path": str(out), "content": "TODO"})
    client = ScriptedClient([call, call, call])  # never yields end_turn
    agent = _agent(client, max_iterations=2)
    task = GoldenTask(
        name="aborts",
        goal="write out.txt",
        check=lambda ws: (ws / "out.txt").exists(),
    )

    result = asyncio.run(run_task(agent, task, tmp_path))

    assert out.exists()            # the artifact really is on disk...
    assert result.passed is False  # ...but the run didn't finish cleanly
    assert result.reason == "MAX_ITERATIONS"  # and the reason says why


def test_run_task_seeds_workspace_before_running_agent(tmp_path):
    from evals.runner import GoldenTask, run_task

    # A read task with nothing to read can never pass, so setup must populate
    # the workspace BEFORE the agent runs. Record the order to prove it.
    order = []

    def setup(ws):
        order.append("setup")

    async def spy_tool(args):
        order.append("agent")
        return "ok"

    tool = Tool(
        name="t", description="t",
        parameters={"type": "object", "properties": {}},
        kind=ToolKind.READ, run=spy_tool,
    )
    registry = ToolRegistry()
    registry.register(tool)
    agent = Agent(
        client=ScriptedClient([_tool_use("t", {}), _finish()]),
        model="m", registry=registry, system="s",
        max_iterations=5, max_cost_usd=10.0,
    )
    task = GoldenTask(name="order", goal="g", check=lambda ws: True, setup=setup)

    asyncio.run(run_task(agent, task, tmp_path))

    assert order == ["setup", "agent"]


def test_run_suite_runs_each_task_in_its_own_workspace(tmp_path):
    from evals.runner import GoldenTask, run_suite

    # "good" writes a RELATIVE path -> it must land in the task's own workspace
    # (proving run_suite runs the agent with that workspace as cwd). "bad"
    # finishes cleanly but never produces its artifact.
    good = GoldenTask("good", "g", check=lambda ws: (ws / "a.txt").exists())
    bad = GoldenTask("bad", "b", check=lambda ws: (ws / "missing.txt").exists())

    def make_agent(task):
        if task.name == "good":
            client = ScriptedClient(
                [_tool_use("write_file", {"path": "a.txt", "content": "x"}), _finish()]
            )
        else:
            client = ScriptedClient([_finish()])
        return _agent(client)

    results = asyncio.run(run_suite([good, bad], make_agent, base_dir=tmp_path))

    by_name = {r.name: r for r in results}
    assert by_name["good"].passed is True
    assert by_name["bad"].passed is False
    assert len(results) == 2


def test_golden_suite_has_three_tasks():
    from evals.runner import GoldenTask
    from evals.tasks import GOLDEN_TASKS

    assert len(GOLDEN_TASKS) == 3
    assert all(isinstance(t, GoldenTask) for t in GOLDEN_TASKS)
    assert all(t.name and t.goal for t in GOLDEN_TASKS)


def test_golden_checks_reject_an_empty_workspace(tmp_path):
    from evals.tasks import GOLDEN_TASKS

    # A check that passes against an untouched workspace catches no regression.
    for task in GOLDEN_TASKS:
        assert task.check(tmp_path) is False, f"{task.name} check is vacuous"
