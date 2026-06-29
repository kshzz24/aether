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
    CostEvent,
    StatusEvent,
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
            input_tokens=10, output_tokens=5, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="all done")],
            input_tokens=4, output_tokens=2, stop_reason="end_turn",
        ),
    ]
    client = StubClient(responses)
    agent = Agent(
        client=client, model="m",
        tools={"echo": make_tool("echo", echo)},
        system="sys", max_iterations=5, max_cost_usd=1.0,
        pricing={"m": (0.0, 0.0)},
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


def test_iteration_cap_stops_cleanly():
    async def echo(args):
        return "ok"

    forever = NormalizedResponse(
        blocks=[ToolCallBlock(id="c1", name="echo", arguments={})],
        input_tokens=1, output_tokens=1, stop_reason="tool_use",
    )
    client = StubClient([forever])
    agent = Agent(
        client=client, model="m",
        tools={"echo": make_tool("echo", echo)},
        system="s", max_iterations=1, max_cost_usd=99.0,
        pricing={"m": (0.0, 0.0)},
    )

    events = collect(agent, "go")

    assert isinstance(events[-1], StatusEvent)
    assert events[-1].message == "stopped: max iterations reached"


def test_cost_cap_stops_cleanly():
    async def echo(args):
        return "ok"

    resp = NormalizedResponse(
        blocks=[ToolCallBlock(id="c1", name="echo", arguments={})],
        input_tokens=10, output_tokens=10, stop_reason="tool_use",
    )
    client = StubClient([resp])
    agent = Agent(
        client=client, model="m",
        tools={"echo": make_tool("echo", echo)},
        system="s", max_iterations=10, max_cost_usd=1.0,
        pricing={"m": (1.0, 1.0)},
    )

    events = collect(agent, "go")

    assert isinstance(events[-1], StatusEvent)
    assert events[-1].message == "stopped: cost limit reached"
    assert not any(isinstance(e, ToolResultEvent) for e in events)


def test_unsupported_tool_calling_stops_gracefully():
    class RaisingClient:
        async def create(self, messages, tools, system):
            raise ToolCallingUnsupportedError("llama-3.3-70b-versatile")

    agent = Agent(
        client=RaisingClient(), model="m",
        tools={}, system="s", max_iterations=5, max_cost_usd=10.0,
        pricing={"m": (0.0, 0.0)},
    )

    events = collect(agent, "go")

    # ends on a clean status event, not a traceback
    assert isinstance(events[-1], StatusEvent)
    assert events[-1].message.startswith("stopped:")
    assert "llama-3.3-70b-versatile" in events[-1].message
    assert "does not support tool calling" in events[-1].message


def test_tool_failure_becomes_observation():
    async def boom(args):
        raise ValueError("kaboom")

    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="boom", arguments={})],
            input_tokens=1, output_tokens=1, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="recovered")],
            input_tokens=1, output_tokens=1, stop_reason="end_turn",
        ),
    ]
    client = StubClient(responses)
    agent = Agent(
        client=client, model="m",
        tools={"boom": make_tool("boom", boom)},
        system="s", max_iterations=5, max_cost_usd=10.0,
        pricing={"m": (0.0, 0.0)},
    )

    events = collect(agent, "go")

    errs = [e for e in events if isinstance(e, ToolResultEvent)]
    assert errs and errs[0].result.startswith("ERROR:")
    assert "kaboom" in errs[0].result
    # the loop CONTINUED to completion instead of crashing
    assert any(isinstance(e, TextEvent) and e.text == "recovered" for e in events)
