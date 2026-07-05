import asyncio

from agent import Agent, Tool
from client import (
    NormalizedResponse,
    TextBlock,
    ToolCallBlock,
    ToolCallingUnsupportedError,
    ToolResultBlock,
)
from events import (
    ConfirmRequestEvent,
    CostEvent,
    TerminalEvent,
    TerminalReason,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)


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


def make_tool(name: str, fn) -> Tool:
    return Tool(
        schema={
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
        run=fn,
    )


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
        tools={"echo": make_tool("echo", echo)},
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
        tools={"echo": make_tool("echo", echo)},
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
        tools={"echo": make_tool("echo", echo)},
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
        tools={"echo": make_tool("echo", echo)},
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
        tools={}, system="s", max_iterations=5, max_cost_usd=10.0,
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
        tools={"boom": make_tool("boom", boom)},
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


def test_dangerous_tool_surfaces_confirm_and_runs_when_auto_approved():
    # run_shell is in DANGEROUS_TOOLS, so the request is always surfaced. With
    # auto_approve=True the stubbed resolver says yes and the tool runs.
    ran = {"count": 0}

    async def shell(args):
        ran["count"] += 1
        return "executed"

    client = StubClient(_run_shell_then_finish())
    agent = Agent(
        client=client, model="m",
        tools={"run_shell": make_tool("run_shell", shell)},
        system="s", max_iterations=5, max_cost_usd=10.0, auto_approve=True,
    )

    events = collect(agent, "go")

    confirms = [e for e in events if isinstance(e, ConfirmRequestEvent)]
    assert confirms, "a dangerous tool must always surface a ConfirmRequestEvent"
    assert confirms[0].tool_name == "run_shell"
    assert confirms[0].arguments == {"cmd": "rm -rf /"}
    assert confirms[0].reason  # a non-empty reason the renderer can display
    # Approved -> it actually ran and the model saw the real output.
    assert ran["count"] == 1
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results and results[0].result == "executed"


def test_dangerous_tool_surfaces_confirm_and_is_denied_when_not_auto_approved():
    # Same call, but the stubbed resolver says no. The request is still
    # surfaced; the tool must NOT run and the model gets a denial observation
    # to react to (not the real output).
    ran = {"count": 0}

    async def shell(args):
        ran["count"] += 1
        return "executed"

    client = StubClient(_run_shell_then_finish())
    agent = Agent(
        client=client, model="m",
        tools={"run_shell": make_tool("run_shell", shell)},
        system="s", max_iterations=5, max_cost_usd=10.0, auto_approve=False,
    )

    events = collect(agent, "go")

    confirms = [e for e in events if isinstance(e, ConfirmRequestEvent)]
    assert confirms and confirms[0].tool_name == "run_shell"
    # Denied -> the tool body never executed...
    assert ran["count"] == 0
    # ...but the loop still fed an observation back so the model can self-correct.
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results, "a denied tool still owes the model an observation"
    assert results[0].result != "executed"


def test_safe_tool_bypasses_confirmation_even_when_not_auto_approved():
    # read_file is NOT dangerous: no confirmation seam, runs regardless of the
    # auto_approve flag. Proves the gate is scoped to DANGEROUS_TOOLS only.
    async def read_file(args):
        return "file body"

    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="read_file", arguments={"path": "a"})],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="done")],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="end_turn",
        ),
    ]
    client = StubClient(responses)
    agent = Agent(
        client=client, model="m",
        tools={"read_file": make_tool("read_file", read_file)},
        system="s", max_iterations=5, max_cost_usd=10.0, auto_approve=False,
    )

    events = collect(agent, "go")

    assert not any(isinstance(e, ConfirmRequestEvent) for e in events)
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results and results[0].result == "file body"
