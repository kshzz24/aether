import asyncio

from agent import Agent
from client import (
    Message,
    NormalizedResponse,
    TextBlock,
    ToolCallBlock,
    ToolCallingUnsupportedError,
    ToolResultBlock,
)
from events import (
    ConfirmRequestEvent,
    CostEvent,
    SubagentEvent,
    TerminalEvent,
    TerminalReason,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tools.base import Tool, ToolKind
from tools.hooks import Hooks
from tools.registry import ToolRegistry


class StubClient:
    """A fake LLMClient that returns scripted responses, no network."""

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self.received: list[list] = []  # snapshot of messages per create() call

    async def create(self, messages, tools, system) -> NormalizedResponse:
        self.received.append(list(messages))  # copy: agent mutates the list
        return self._responses.pop(0)


def collect(agent: Agent, goal: str) -> list:
    """Drive the async generator to completion and return all events."""

    async def _drive():
        return [event async for event in agent.run(goal)]

    return asyncio.run(_drive())


def make_tool(name: str, fn, kind: ToolKind = ToolKind.READ) -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        kind=kind,
        run=fn,
    )


def make_registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_happy_path_runs_tool_then_finishes():
    async def echo(args):
        return "ok"

    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="echo", arguments={"n": 1})],
            input_tokens=10, output_tokens=5, cost_usd=0.0, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="all done")],
            input_tokens=4, output_tokens=2, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    client = StubClient(responses)
    agent = Agent(
        client=client, model="m",
        registry=make_registry(make_tool("echo", echo)),
        system="sys", max_iterations=5, max_cost_usd=1.0,
    )

    events = collect(agent, "do it")

    assert any(isinstance(e, ToolCallEvent) and e.name == "echo" for e in events)
    assert any(isinstance(e, ToolResultEvent) and e.result == "ok" for e in events)
    assert any(isinstance(e, TextEvent) and e.text == "all done" for e in events)
    assert any(isinstance(e, CostEvent) for e in events)
    second_call = client.received[1]
    assert any(
        isinstance(b, ToolResultBlock) and b.content == "ok"
        for m in second_call for b in m.blocks
    )
    # A normal end_turn finish is a terminal event, not just a trailing text.
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.COMPLETED


def test_iteration_cap_stops_cleanly():
    async def echo(args):
        return "ok"

    forever = NormalizedResponse(
        blocks=[ToolCallBlock(id="c1", name="echo", arguments={})],
        input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
    )
    client = StubClient([forever])
    agent = Agent(
        client=client, model="m",
        registry=make_registry(make_tool("echo", echo)),
        system="s", max_iterations=1, max_cost_usd=99.0,
    )

    events = collect(agent, "go")

    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.MAX_ITERATIONS


def test_cost_cap_stops_cleanly():
    async def echo(args):
        return "ok"

    resp = NormalizedResponse(
        blocks=[ToolCallBlock(id="c1", name="echo", arguments={})],
        input_tokens=10, output_tokens=10, cost_usd=2.0, stop_reason="tool_use",
    )
    client = StubClient([resp])
    agent = Agent(
        client=client, model="m",
        registry=make_registry(make_tool("echo", echo)),
        system="s", max_iterations=10, max_cost_usd=1.0,
    )

    events = collect(agent, "go")

    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.MAX_COST
    assert not any(isinstance(e, ToolResultEvent) for e in events)


def test_repeating_action_trips_loop_detector():
    # The model keeps issuing the SAME tool call and the tool keeps returning
    # the SAME observation: no progress. The loop detector must abort early --
    # long before the (deliberately huge) iteration cap.
    async def echo(args):
        return "same result"

    class RepeatingClient:
        def __init__(self, response):
            self._response = response
            self.calls = 0

        async def create(self, messages, tools, system):
            self.calls += 1
            return self._response

    resp = NormalizedResponse(
        blocks=[ToolCallBlock(id="c1", name="echo", arguments={})],
        input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
    )
    client = RepeatingClient(resp)
    agent = Agent(
        client=client, model="m",
        registry=make_registry(make_tool("echo", echo)),
        system="s", max_iterations=50, max_cost_usd=99.0,
    )

    events = collect(agent, "go")

    # Aborted cleanly via the detector, not a traceback and not the cap.
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.LOOP_DETECTED
    # Tripped at the third identical step (min_repeats=3), not at iteration 50.
    assert client.calls == 3


def test_unsupported_tool_calling_stops_gracefully():
    class RaisingClient:
        async def create(self, messages, tools, system):
            raise ToolCallingUnsupportedError("llama-3.3-70b-versatile")

    agent = Agent(
        client=RaisingClient(), model="m",
        registry=ToolRegistry(), system="s", max_iterations=5, max_cost_usd=10.0,
    )

    events = collect(agent, "go")

    # ends on a clean terminal event, not a traceback
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.ERROR
    assert "llama-3.3-70b-versatile" in events[-1].detail
    assert "does not support tool calling" in events[-1].detail


def test_tool_failure_becomes_observation():
    async def boom(args):
        raise ValueError("kaboom")

    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="boom", arguments={})],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="recovered")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    client = StubClient(responses)
    agent = Agent(
        client=client, model="m",
        registry=make_registry(make_tool("boom", boom)),
        system="s", max_iterations=5, max_cost_usd=10.0,
    )

    events = collect(agent, "go")

    errs = [e for e in events if isinstance(e, ToolResultEvent)]
    assert errs and errs[0].result.startswith("ERROR:")
    assert "kaboom" in errs[0].result
    # the loop CONTINUED to completion instead of crashing
    assert any(isinstance(e, TextEvent) and e.text == "recovered" for e in events)


def _run_shell_then_finish() -> list[NormalizedResponse]:
    """Model calls run_shell (a dangerous tool), then finishes on the next turn."""
    return [
        NormalizedResponse(
            blocks=[
                ToolCallBlock(id="c1", name="run_shell", arguments={"cmd": "rm -rf /"})
            ],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="understood")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]


def _build_agent(client, registry, *, mode, approver):
    from approval import ApprovalMode  # noqa: F401
    from policy import PolicyEngine
    return Agent(
        client=client, model="m", registry=registry, system="s",
        max_iterations=5, max_cost_usd=10.0,
        policy=PolicyEngine(mode), approver=approver,
    )


def test_on_request_approved_runs_and_surfaces_confirm(ScriptedApprover_):
    from approval import ApprovalMode, Decision
    ran = {"n": 0}
    async def shell(args):
        ran["n"] += 1
        return "executed"
    approver = ScriptedApprover_([Decision(approved=True)])
    agent = _build_agent(
        StubClient(_run_shell_then_finish()),
        make_registry(make_tool("run_shell", shell, kind=ToolKind.EXECUTE)),
        mode=ApprovalMode.ON_REQUEST, approver=approver,
    )
    events = collect(agent, "go")
    assert [e for e in events if isinstance(e, ConfirmRequestEvent)]
    assert ran["n"] == 1
    assert approver.seen and approver.seen[0].tool_name == "run_shell"


def test_on_request_denied_does_not_run_but_feeds_observation(ScriptedApprover_):
    from approval import ApprovalMode, Decision
    ran = {"n": 0}
    async def shell(args):
        ran["n"] += 1
        return "executed"
    agent = _build_agent(
        StubClient(_run_shell_then_finish()),
        make_registry(make_tool("run_shell", shell, kind=ToolKind.EXECUTE)),
        mode=ApprovalMode.ON_REQUEST,
        approver=ScriptedApprover_([Decision(approved=False)]),
    )
    events = collect(agent, "go")
    assert ran["n"] == 0
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results and results[0].result.startswith("DENIED")


def test_never_mode_auto_denies_without_calling_approver(ScriptedApprover_):
    from approval import ApprovalMode
    ran = {"n": 0}
    async def shell(args):
        ran["n"] += 1
        return "executed"
    approver = ScriptedApprover_([])  # empty: any decide() call would IndexError
    agent = _build_agent(
        StubClient(_run_shell_then_finish()),
        make_registry(make_tool("run_shell", shell, kind=ToolKind.EXECUTE)),
        mode=ApprovalMode.NEVER, approver=approver,
    )
    events = collect(agent, "go")
    assert ran["n"] == 0
    assert approver.seen == []  # approver never consulted under NEVER
    from events import ApprovalDecisionEvent
    decisions = [e for e in events if isinstance(e, ApprovalDecisionEvent)]
    assert decisions and decisions[0].source == "policy"


def test_hooks_fire_at_their_points_during_a_run():
    async def echo(args):
        return "ok"

    fired = []
    hooks = Hooks(
        before_run=lambda goal: fired.append(("before_run", goal)),
        after_run=lambda goal: fired.append(("after_run", goal)),
        before_tool=lambda name, args: fired.append(("before_tool", name)),
        after_tool=lambda name, args, result: fired.append(("after_tool", name)),
    )
    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="echo", arguments={})],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="done")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    agent = Agent(
        client=StubClient(responses), model="m",
        registry=make_registry(make_tool("echo", echo)),
        system="s", max_iterations=5, max_cost_usd=10.0, hooks=hooks,
    )

    collect(agent, "the goal")

    names = [f[0] for f in fired]
    # run brackets everything; the tool bracket sits inside it, in order.
    assert names == ["before_run", "before_tool", "after_tool", "after_run"]
    assert fired[0] == ("before_run", "the goal")
    assert fired[-1] == ("after_run", "the goal")


def test_on_error_hook_fires_on_tool_failure():
    async def boom(args):
        raise ValueError("kaboom")

    errors = []
    hooks = Hooks(on_error=lambda exc: errors.append(exc))
    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="boom", arguments={})],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="recovered")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    agent = Agent(
        client=StubClient(responses), model="m",
        registry=make_registry(make_tool("boom", boom)),
        system="s", max_iterations=5, max_cost_usd=10.0, hooks=hooks,
    )

    collect(agent, "go")

    assert errors and isinstance(errors[0], ValueError)
    assert "kaboom" in str(errors[0])


def test_edit_file_base_hash_mismatch_blocks_apply(tmp_path, ScriptedApprover_):
    # A real edit_file call previewed against base B, but the file changes to B'
    # before apply -> the loop must refuse to apply the stale diff.
    from dataclasses import replace

    from approval import ApprovalMode, Decision
    from tools.base import PreviewResult

    f = tmp_path / "c.py"
    f.write_text("a = 1\n", encoding="utf-8")

    async def _run(args):  # would apply if reached
        f.write_text("APPLIED\n", encoding="utf-8")
        return "applied"

    async def _preview(args):
        # base hash of some *other* content -> guaranteed mismatch at apply
        return PreviewResult(diff="d", base_hash="deadbeef", hashed_path=str(f))

    tool = replace(make_tool("edit_file", _run, kind=ToolKind.WRITE),
                   preview=_preview)  # frozen-dataclass-safe way to attach preview

    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="edit_file",
                                  arguments={"path": str(f)})],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use"),
        NormalizedResponse(blocks=[TextBlock(text="ok")], input_tokens=1,
                           output_tokens=1, cost_usd=0.0, stop_reason="end_turn"),
    ]
    # Approve unconditionally: the path is under tmp_path (outside cwd/repo_root),
    # so path_escape may force ASK under AUTO -- approving isolates the hash guard,
    # which runs AFTER approval regardless of the verdict path.
    agent = _build_agent(StubClient(responses), make_registry(tool),
                         mode=ApprovalMode.AUTO,
                         approver=ScriptedApprover_([Decision(approved=True)]))
    events = collect(agent, "go")

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results and "file changed since approval" in results[0].result
    assert f.read_text(encoding="utf-8") == "a = 1\n"  # never applied


def test_agent_brackets_agent_kind_tool_with_subagent_events():
    # A tool tagged ToolKind.AGENT is bracketed by SubagentEvent(started) before
    # the run and SubagentEvent(completed) after, with task = the `prompt` arg.
    async def fake_task(args):
        return "distilled result"

    registry = make_registry(make_tool("task", fake_task, ToolKind.AGENT))
    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="task",
                                  arguments={"prompt": "go explore"})],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="done")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    agent = Agent(
        client=StubClient(responses), model="m", registry=registry,
        system="s", max_iterations=5, max_cost_usd=1.0,
    )
    events = collect(agent, "use task")
    subs = [e for e in events if isinstance(e, SubagentEvent)]
    assert [s.phase for s in subs] == ["started", "completed"]
    assert subs[0].task == "go explore"


def test_run_resumes_from_history_without_reseeding_goal():
    # A saved conversation awaiting the model's next move.
    history = [
        Message(role="user", blocks=[TextBlock(text="original goal")]),
        Message(role="assistant",
                blocks=[ToolCallBlock(id="c1", name="echo", arguments={})]),
        Message(role="user",
                blocks=[ToolResultBlock(tool_call_id="c1", content="ok")]),
    ]
    responses = [
        NormalizedResponse(
            blocks=[TextBlock(text="finished")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    client = StubClient(responses)
    agent = Agent(
        client=client, model="m", registry=make_registry(),
        system="s", max_iterations=5, max_cost_usd=1.0,
    )

    async def _drive():
        return [e async for e in agent.run("original goal", history=history)]

    asyncio.run(_drive())

    # The model saw exactly the resumed history on its first call -- the goal was
    # NOT re-appended as a 4th message.
    first_call = client.received[0]
    assert first_call == history
    # self.messages reflects the final state, including the new assistant turn.
    assert agent.messages[-1].blocks[0].text == "finished"


def test_repair_recovers_from_one_malformed_tool_call():
    class FlakyClient:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = 0

        async def create(self, messages, tools, system):
            self.calls += 1
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    responses = [
        ToolCallingUnsupportedError("m"),
        NormalizedResponse(
            blocks=[TextBlock(text="recovered")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    client = FlakyClient(responses)
    agent = Agent(
        client=client, model="m", registry=make_registry(),
        system="s", max_iterations=5, max_cost_usd=10.0,
    )

    events = collect(agent, "go")

    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.COMPLETED
    assert client.calls == 2


def test_repair_gives_up_after_the_cap():
    class AlwaysRaisingClient:
        def __init__(self):
            self.calls = 0

        async def create(self, messages, tools, system):
            self.calls += 1
            raise ToolCallingUnsupportedError("m")

    client = AlwaysRaisingClient()
    agent = Agent(
        client=client, model="m", registry=make_registry(),
        system="s", max_iterations=10, max_cost_usd=10.0,
    )

    events = collect(agent, "go")

    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].reason is TerminalReason.ERROR
    # REPAIR_ATTEMPTS=2: two retried calls, then the third raise gives up.
    from agent import REPAIR_ATTEMPTS
    assert client.calls == REPAIR_ATTEMPTS + 1


def test_agent_exposes_final_messages_and_cost_for_checkpointing():
    responses = [
        NormalizedResponse(
            blocks=[TextBlock(text="hi")],
            input_tokens=1, output_tokens=1, cost_usd=0.05, stop_reason="end_turn",
        ),
    ]
    agent = Agent(
        client=StubClient(responses), model="m", registry=make_registry(),
        system="s", max_iterations=5, max_cost_usd=1.0,
    )

    collect(agent, "greet")

    assert agent.messages[0].blocks[0].text == "greet"   # goal seeded
    assert agent.messages[-1].blocks[0].text == "hi"      # final answer appended
    assert agent.total_cost == 0.05
