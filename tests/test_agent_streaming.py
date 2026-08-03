"""The agent's streaming branch, and edited approvals.

Two claims are load-bearing here.

**Compatibility.** Streaming is an optional capability probed with `getattr`,
not a change to `LLMClient`. A client without `stream` must produce a
byte-identical event sequence to before — that is the whole reason the design
is a second Protocol rather than a new signature on `create`.

**An edited call is a new call.** `e` at the approval prompt would otherwise be
a bypass for the schema, the dangerous-command check and the path guard: approve
a harmless `ls`, edit it into `rm -rf /`.
"""

from __future__ import annotations

from pathlib import Path

from conftest import ScriptedApprover, StubClient

from agent import Agent
from approval import ApprovalMode, Decision
from client import NormalizedResponse, TextBlock, ToolCallBlock
from events import TextDeltaEvent, TextEvent, ToolResultEvent
from policy import PolicyEngine
from tools.base import Tool, ToolKind
from tools.registry import ToolRegistry


def _response(blocks, stop="end_turn") -> NormalizedResponse:
    return NormalizedResponse(
        blocks=blocks,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        stop_reason=stop,
    )


class _Stream:
    def __init__(self, deltas: list[str], response: NormalizedResponse) -> None:
        self._deltas = deltas
        self.response = response

    def __aiter__(self):
        async def gen():
            for delta in self._deltas:
                yield delta

        return gen()


class StreamingStub(StubClient):
    """A StubClient that also satisfies `StreamingClient`."""

    def __init__(self, responses, deltas: list[list[str]]) -> None:
        super().__init__(responses)
        self._deltas = list(deltas)

    def stream(self, messages, tools, system):
        self.received.append(list(messages))
        return _Stream(self._deltas.pop(0), self._responses.pop(0))


def _echo_tool() -> tuple[Tool, list[dict]]:
    """A run_shell stand-in, plus the list of calls it actually received.

    `Tool` is frozen, so the record rides alongside rather than on the tool.
    What it records is the assertion that matters for edited approvals: whether
    a call reached execution at all, and with which arguments.
    """
    seen: list[dict] = []

    async def run(args: dict) -> str:
        seen.append(dict(args))
        return f"ran {args.get('command', '')}"

    tool = Tool(
        name="run_shell",
        description="run a command",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        kind=ToolKind.EXECUTE,
        run=run,
    )
    return tool, seen


def _agent(client, *, approver=None, tools=(), mode=ApprovalMode.AUTO) -> Agent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return Agent(
        client=client,
        model="m",
        registry=registry,
        system="s",
        max_iterations=4,
        max_cost_usd=1.0,
        policy=PolicyEngine(mode),
        approver=approver,
        repo_root=Path.cwd(),
    )


async def _events(agent, goal="go") -> list:
    return [event async for event in agent.run(goal)]


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


async def test_a_streaming_client_yields_deltas():
    client = StreamingStub([_response([TextBlock(text="hello")])], [["hel", "lo"]])
    events = await _events(_agent(client))
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["hel", "lo"]


async def test_the_full_text_event_still_follows():
    """Deltas are a preview; the TextEvent is what gets rendered as markdown."""
    client = StreamingStub([_response([TextBlock(text="hello")])], [["hel", "lo"]])
    events = await _events(_agent(client))
    assert [e.text for e in events if isinstance(e, TextEvent)] == ["hello"]


async def test_deltas_precede_the_text_event():
    client = StreamingStub([_response([TextBlock(text="hello")])], [["hel", "lo"]])
    events = await _events(_agent(client))
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("TextDeltaEvent") < kinds.index("TextEvent")


async def test_the_streamed_response_drives_the_loop():
    """`turn.response` must be used for stop_reason, or the loop never ends."""
    client = StreamingStub([_response([TextBlock(text="done")])], [["done"]])
    events = await _events(_agent(client))
    assert type(events[-1]).__name__ == "TerminalEvent"


async def test_a_streaming_client_still_runs_tools():
    tool, _seen = _echo_tool()
    client = StreamingStub(
        [
            _response(
                [ToolCallBlock(id="c1", name="run_shell", arguments={"command": "ls"})],
                stop="tool_use",
            ),
            _response([TextBlock(text="finished")]),
        ],
        [[], []],
    )
    events = await _events(_agent(client, tools=[tool]))
    assert any(isinstance(e, ToolResultEvent) for e in events)


# --------------------------------------------------------------------------
# Compatibility: a non-streaming client is untouched
# --------------------------------------------------------------------------


async def test_a_client_without_stream_emits_no_deltas():
    client = StubClient([_response([TextBlock(text="hello")])])
    events = await _events(_agent(client))
    assert not any(isinstance(e, TextDeltaEvent) for e in events)


async def test_a_client_without_stream_produces_the_same_events_as_before():
    """The compatibility guarantee the whole design exists for: probing must
    not perturb the buffered path at all."""
    client = StubClient([_response([TextBlock(text="hello")])])
    kinds = [type(e).__name__ for e in await _events(_agent(client))]
    assert kinds == ["StatusEvent", "CostEvent", "TextEvent", "TerminalEvent"]


async def test_the_two_paths_agree_on_everything_but_deltas():
    buffered = await _events(_agent(StubClient([_response([TextBlock(text="hi")])])))
    streamed = await _events(
        _agent(StreamingStub([_response([TextBlock(text="hi")])], [["h", "i"]]))
    )
    strip = lambda evts: [  # noqa: E731
        type(e).__name__ for e in evts if not isinstance(e, TextDeltaEvent)
    ]
    assert strip(buffered) == strip(streamed)


# --------------------------------------------------------------------------
# Edited approvals — an edited call is a NEW call
# --------------------------------------------------------------------------


async def test_an_edited_call_runs_with_the_edited_arguments():
    tool, seen = _echo_tool()
    approver = ScriptedApprover(
        [Decision(approved=True, arguments={"command": "echo edited"})]
    )
    client = StubClient(
        [
            _response(
                [ToolCallBlock(id="c1", name="run_shell", arguments={"command": "ls"})],
                stop="tool_use",
            ),
            _response([TextBlock(text="done")]),
        ]
    )
    await _events(
        _agent(client, approver=approver, tools=[tool], mode=ApprovalMode.ON_REQUEST)
    )
    assert seen == [{"command": "echo edited"}]


async def test_an_edit_that_introduces_danger_is_refused():
    """The security content of the feature. Without the re-check, `e` is a hole
    straight through the dangerous-command guard."""
    tool, seen = _echo_tool()
    approver = ScriptedApprover(
        [Decision(approved=True, arguments={"command": "rm -rf /"})]
    )
    client = StubClient(
        [
            _response(
                [ToolCallBlock(id="c1", name="run_shell", arguments={"command": "ls"})],
                stop="tool_use",
            ),
            _response([TextBlock(text="done")]),
        ]
    )
    events = await _events(
        _agent(client, approver=approver, tools=[tool], mode=ApprovalMode.ON_REQUEST)
    )

    assert seen == [], "a dangerous edited command was executed"
    denied = [e for e in events if isinstance(e, ToolResultEvent)]
    assert denied and denied[0].result.startswith("DENIED")


async def test_an_edit_that_breaks_the_schema_is_refused():
    tool, seen = _echo_tool()
    approver = ScriptedApprover([Decision(approved=True, arguments={"wrong": "field"})])
    client = StubClient(
        [
            _response(
                [ToolCallBlock(id="c1", name="run_shell", arguments={"command": "ls"})],
                stop="tool_use",
            ),
            _response([TextBlock(text="done")]),
        ]
    )
    events = await _events(
        _agent(client, approver=approver, tools=[tool], mode=ApprovalMode.ON_REQUEST)
    )

    assert seen == []
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results and "invalid" in results[0].result


async def test_the_refusal_reason_reaches_the_model():
    """It is an observation, not a crash — the model gets to try again."""
    approver = ScriptedApprover(
        [Decision(approved=True, arguments={"command": "rm -rf /"})]
    )
    client = StubClient(
        [
            _response(
                [ToolCallBlock(id="c1", name="run_shell", arguments={"command": "ls"})],
                stop="tool_use",
            ),
            _response([TextBlock(text="ok")]),
        ]
    )
    tool, _seen = _echo_tool()
    await _events(
        _agent(client, approver=approver, tools=[tool], mode=ApprovalMode.ON_REQUEST)
    )
    # The second create() call carries the denial as a tool result.
    second_turn = client.received[1]
    assert any("DENIED" in str(block) for m in second_turn for block in m.blocks)


async def test_a_decision_without_arguments_is_unchanged():
    """The default path: every existing approver keeps working untouched."""
    tool, seen = _echo_tool()
    approver = ScriptedApprover([Decision(approved=True)])
    client = StubClient(
        [
            _response(
                [ToolCallBlock(id="c1", name="run_shell", arguments={"command": "ls"})],
                stop="tool_use",
            ),
            _response([TextBlock(text="done")]),
        ]
    )
    await _events(
        _agent(client, approver=approver, tools=[tool], mode=ApprovalMode.ON_REQUEST)
    )
    assert seen == [{"command": "ls"}]


async def test_arguments_on_a_denial_are_ignored():
    tool, seen = _echo_tool()
    approver = ScriptedApprover(
        [Decision(approved=False, arguments={"command": "echo sneaky"})]
    )
    client = StubClient(
        [
            _response(
                [ToolCallBlock(id="c1", name="run_shell", arguments={"command": "ls"})],
                stop="tool_use",
            ),
            _response([TextBlock(text="done")]),
        ]
    )
    await _events(
        _agent(client, approver=approver, tools=[tool], mode=ApprovalMode.ON_REQUEST)
    )
    assert seen == []
