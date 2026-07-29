import asyncio

from agent import Agent
from client import NormalizedResponse, TextBlock, ToolCallBlock
from tools.base import Tool, ToolKind
from tools.registry import ToolRegistry
from tools.subagent import build_subagent_tool


class StubClient:
    """A fake LLMClient that returns scripted responses, no network."""

    def __init__(self, responses):
        self._responses = list(responses)

    async def create(self, messages, tools, system):
        return self._responses.pop(0)


def make_tool(name, fn, kind=ToolKind.READ):
    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        kind=kind,
        run=fn,
    )


def make_registry(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _child_factory(responses, registry):
    def factory() -> Agent:
        return Agent(
            client=StubClient(responses), model="m", registry=registry,
            system="s", max_iterations=3, max_cost_usd=1.0,
        )
    return factory


def test_task_tool_returns_distilled_final_text_and_isolates_chatter():
    async def peek(args):
        return "noisy internal tool output"

    registry = make_registry(make_tool("peek", peek, ToolKind.READ))
    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id="c1", name="peek", arguments={})],
            input_tokens=1, output_tokens=1, cost_usd=0.01, stop_reason="tool_use",
        ),
        NormalizedResponse(
            blocks=[TextBlock(text="Here is the summary.")],
            input_tokens=1, output_tokens=1, cost_usd=0.02, stop_reason="end_turn",
        ),
    ]
    tool = build_subagent_tool(make_child=_child_factory(responses, registry))

    result = asyncio.run(tool.run({"prompt": "summarize the repo"}))

    assert "Here is the summary." in result           # distilled final answer
    assert "noisy internal tool output" not in result  # child chatter isolated
    assert "completed" in result                       # footer reason


def test_task_tool_is_agent_kind_named_task():
    tool = build_subagent_tool(make_child=lambda: None)
    assert tool.name == "task"
    assert tool.kind is ToolKind.AGENT
    assert "prompt" in tool.parameters["properties"]


def test_task_tool_returns_error_string_when_child_does_not_complete():
    # Child keeps calling a tool and never finishes -> a non-COMPLETED terminal
    # reason (MAX_ITERATIONS or LOOP_DETECTED). Must be returned, not raised.
    async def peek(args):
        return "x"

    registry = make_registry(make_tool("peek", peek, ToolKind.READ))
    responses = [
        NormalizedResponse(
            blocks=[ToolCallBlock(id=f"c{i}", name="peek", arguments={})],
            input_tokens=1, output_tokens=1, cost_usd=0.0, stop_reason="tool_use",
        )
        for i in range(3)
    ]
    tool = build_subagent_tool(make_child=_child_factory(responses, registry))

    result = asyncio.run(tool.run({"prompt": "never ends"}))

    assert "did not complete" in result   # failure surfaced as an observation


def test_child_registry_has_no_task_tool_depth_cap():
    # Mirrors main.py's assembly: the parent registry gets `task` registered
    # onto it; a child registry is a fresh build_registry(config) that never
    # contains `task`, so a child cannot spawn a grandchild.
    from config import ForgeConfig
    from tools import build_registry

    config = ForgeConfig()
    parent = build_registry(config)
    parent.register(build_subagent_tool(make_child=lambda: None))
    child = build_registry(config)

    parent_names = {t.name for t in parent.list()}
    child_names = {t.name for t in child.list()}

    assert "task" in parent_names
    assert "task" not in child_names
