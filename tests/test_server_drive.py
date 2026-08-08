"""Phase 8, stage 2b — the drive task.

Stage 2a pinned the publish side with a hand-driven `publish`. This file connects
it to the real agent: `start(goal)` fires a background task that pulls from
`agent.run(...)` and pushes into `publish`, and returns immediately so an HTTP
handler can answer 202 while the run continues.

Three properties here are subtle enough that they are the reason the file exists:

- `running` must flip inside `start`, not inside the task body. `create_task`
  schedules a coroutine; it runs no line of it until the caller next awaits, so a
  flag set in `_drive` lets two `start()` calls in one tick both pass the busy
  check. See `test_running_is_true_the_moment_start_returns`.
- multi-turn needs the *new goal appended to* the prior history, not the prior
  history alone. `agent.py:92-95` assigns `history` straight onto
  `self.messages` and never appends the goal, so passing bare history drops the
  turn. See `test_second_turn_history_ends_with_the_new_goal`.
- session cost is not `agent.total_cost`. `agent.py:98` resets it per run because
  `max_cost_usd` is a per-run bound (invariant 6), so the session has to
  accumulate its own. See `test_session_cost_accumulates_across_turns`.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from conftest import ScriptedApprover, StubClient

import main
import persistence
from client import NormalizedResponse, TextBlock
from server.session import AgentSession, SessionBusy
from server.wire import RunParams

# --- doubles ------------------------------------------------------------------


def say(text: str) -> NormalizedResponse:
    """A one-turn answer: no tool calls, so the run ends after it."""
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="end_turn",
    )


class BlockingClient:
    """Parks inside `create` until released, so a run can be caught mid-flight.

    `StubClient` returns instantly, which makes an interrupt a race: the run is
    usually over before `interrupt()` is called. This gives the test a point
    where the agent is provably suspended.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.received: list[list] = []

    async def create(self, messages, tools, system):
        self.received.append(list(messages))
        self.entered.set()
        await self.release.wait()
        return say("done")


class ExplodingClient:
    """Raises where the drive task's `except Exception` has to catch it."""

    def __init__(self) -> None:
        self.received: list[list] = []

    async def create(self, messages, tools, system):
        self.received.append(list(messages))
        raise RuntimeError("provider exploded")


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def make_comp(tmp_path, monkeypatch):
    """Same neutralisation as `test_server_sessions.py` — see its docstring.

    The `load_mcp_manager` patch is the load-bearing one: it reads
    `Path.home() / ".mcp.json"`, an absolute path `project_root` does not
    contain, so unpatched every test here spawns real stdio subprocesses.
    """
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: None)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")

    def _make(client):
        comp = main.build_composition(
            RunParams(
                provider="groq",
                model="stub-model",
                project_root=tmp_path,
            ),
            approver=ScriptedApprover([]),
        )
        comp.agent.client = client
        return comp

    return _make


@pytest.fixture
def session(make_comp):
    """A session whose client answers one turn and stops."""
    return AgentSession(make_comp(StubClient([say("hello")])))


async def run_once(session: AgentSession, goal: str = "do it") -> None:
    """Start a run and wait for the drive task to finish."""
    session.start(goal)
    await asyncio.wait_for(session.wait(), 2.0)


def types_of(session: AgentSession) -> list[str]:
    return [f["type"] for f in session.transcript]


# --- one turn, end to end -----------------------------------------------------


async def test_start_drives_the_agent_and_publishes_the_run(session):
    """A `_say` turn is exactly four events, in this order.

    Spelled out rather than derived, so a change to the agent's event order is
    caught here rather than silently reshaping every client.
    """
    await run_once(session)
    assert types_of(session) == ["status", "cost", "text", "terminal"]


async def test_the_terminal_frame_reports_completion(session):
    await run_once(session)
    assert session.transcript[-1] == {
        "type": "terminal",
        "reason": "completed",
        "detail": "",
        "seq": 3,
    }


async def test_a_subscriber_sees_the_run_live(session):
    """The point of the whole stage: frames reach a consumer that is not driving.

    Subscribing *before* `start` also proves the drive task pushes rather than
    waiting to be pulled — nothing here ever touches `agent.run`.
    """
    sub = session.subscribe()
    await run_once(session)
    seen = [await asyncio.wait_for(anext(sub), 1.0) for _ in range(4)]
    assert [f["type"] for f in seen] == ["status", "cost", "text", "terminal"]


async def test_publishing_continues_with_no_subscribers(session):
    """A run started by POST /goal produces frames before the browser connects."""
    await run_once(session)
    assert len(session.transcript) == 4


# --- the in-flight flag -------------------------------------------------------


async def test_start_is_synchronous(session):
    """`start` must not be a coroutine.

    The busy check and `_running = True` have to happen in one uninterrupted
    block. An `async def start` puts an await between the caller and the flag,
    and two concurrent starts would both pass the check — two agents writing the
    same files, interleaved into one transcript.
    """
    assert not inspect.iscoroutinefunction(AgentSession.start)


async def test_running_is_true_the_moment_start_returns(session):
    """No await between `start()` and this assertion, deliberately.

    `asyncio.create_task` only *schedules* `_drive`; not one line of it has run
    yet. So this passes only if `start` itself set the flag.
    """
    session.start("do it")
    assert session.running
    await asyncio.wait_for(session.wait(), 2.0)


async def test_running_clears_when_the_run_finishes(session):
    await run_once(session)
    assert not session.running


async def test_second_start_while_running_raises_session_busy(make_comp):
    client = BlockingClient()
    session = AgentSession(make_comp(client))

    session.start("first")
    await asyncio.wait_for(client.entered.wait(), 2.0)

    with pytest.raises(SessionBusy):
        session.start("second")

    client.release.set()
    await asyncio.wait_for(session.wait(), 2.0)


async def test_a_new_run_is_allowed_once_the_previous_one_ends(make_comp):
    session = AgentSession(make_comp(StubClient([say("one"), say("two")])))
    await run_once(session, "first")
    await run_once(session, "second")
    assert len(session.transcript) == 8


# --- multi-turn ---------------------------------------------------------------


async def test_seq_continues_across_two_turns(make_comp):
    """`seq` is session-scoped. A reset would make `Last-Event-ID` address the
    wrong offset and splice turn 2 over turn 1 in the browser."""
    session = AgentSession(make_comp(StubClient([say("one"), say("two")])))
    await run_once(session, "first")
    await run_once(session, "second")
    assert [f["seq"] for f in session.transcript] == list(range(8))


async def test_first_turn_sends_only_the_goal(make_comp):
    client = StubClient([say("one")])
    session = AgentSession(make_comp(client))
    await run_once(session, "first goal")

    (messages,) = client.received
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].blocks[0].text == "first goal"


async def test_second_turn_carries_the_first_turns_exchange(make_comp):
    """Turn 2 must see turn 1. `agent.run` reassigns `self.messages` on every
    call (`agent.py:92-95`), so continuity exists only if the session hands the
    prior messages back in."""
    client = StubClient([say("one"), say("two")])
    session = AgentSession(make_comp(client))
    await run_once(session, "first goal")
    await run_once(session, "second goal")

    first, second = client.received
    assert len(first) == 1
    assert [m.role for m in second] == ["user", "assistant", "user"]
    assert second[1].blocks[0].text == "one"


async def test_second_turn_history_ends_with_the_new_goal(make_comp):
    """The trap the plan's §4.6 does not cover.

    `agent.py:92-95` assigns `history` verbatim onto `self.messages` and never
    appends the goal — the resume path relies on the goal already being message
    zero. So passing the prior history *alone* silently drops turn 2's text and
    the model is simply re-shown turn 1. The session must append it:

        history = [*prior, Message(role="user", blocks=[TextBlock(text=goal)])]
    """
    client = StubClient([say("one"), say("two")])
    session = AgentSession(make_comp(client))
    await run_once(session, "first goal")
    await run_once(session, "second goal")

    _, second = client.received
    assert second[-1].role == "user"
    assert second[-1].blocks[0].text == "second goal"


# --- cost ---------------------------------------------------------------------


async def test_session_cost_accumulates_across_turns(make_comp):
    """`agent.total_cost` is per-run by design; the session total is not.

    `agent.py:98` resets it each run because `max_cost_usd` is a per-run bound
    (invariant 6) that must not inherit the previous turn's spend — so the reset
    stays. But `main.checkpoint` then *assigns* that per-run figure onto
    `session.total_cost`, which means turn 2 overwrites turn 1. The session
    accumulates its own by summing `CostEvent.cost_usd`, the per-call delta.
    Summing `total_cost_usd` instead would double-count: that field is already
    cumulative within a run.
    """
    session = AgentSession(make_comp(StubClient([say("one"), say("two")])))
    await run_once(session, "first")
    await run_once(session, "second")

    assert session.total_cost == pytest.approx(0.02)
    assert session._comp.agent.total_cost == pytest.approx(0.01)


async def test_checkpoint_records_the_session_total_not_the_run_total(make_comp):
    comp = make_comp(StubClient([say("one"), say("two")]))
    session = AgentSession(comp)
    await run_once(session, "first")
    await run_once(session, "second")

    saved = persistence.load(comp.session.id, comp.sessions_dir)
    assert saved.total_cost == pytest.approx(0.02)


# --- interrupt ----------------------------------------------------------------


async def test_interrupt_ends_the_run_and_clears_running(make_comp):
    client = BlockingClient()
    session = AgentSession(make_comp(client))

    session.start("do it")
    await asyncio.wait_for(client.entered.wait(), 2.0)
    await asyncio.wait_for(session.interrupt(), 2.0)

    assert not session.running


async def test_interrupt_publishes_a_notice(make_comp):
    """An interrupt is a normal outcome, not an error — the same call the TUI
    makes at `tui/app.py:984`."""
    client = BlockingClient()
    session = AgentSession(make_comp(client))

    session.start("do it")
    await asyncio.wait_for(client.entered.wait(), 2.0)
    await asyncio.wait_for(session.interrupt(), 2.0)

    assert session.transcript[-1]["type"] == "status"
    assert "interrupt" in session.transcript[-1]["message"]


async def test_interrupt_still_checkpoints(make_comp):
    """The `finally` runs on the cancel path too, or an interrupted session is
    unresumable. This is why `aclose` must *await* the cancelled task rather
    than only requesting cancellation."""
    client = BlockingClient()
    comp = make_comp(client)
    session = AgentSession(comp)

    session.start("do it")
    await asyncio.wait_for(client.entered.wait(), 2.0)
    await asyncio.wait_for(session.interrupt(), 2.0)

    saved = persistence.load(comp.session.id, comp.sessions_dir)
    assert saved.id == comp.session.id


async def test_interrupt_with_no_run_in_flight_is_a_no_op(session):
    await asyncio.wait_for(session.interrupt(), 2.0)
    assert not session.running
    assert session.transcript == []


async def test_a_run_can_start_again_after_an_interrupt(make_comp):
    client = BlockingClient()
    session = AgentSession(make_comp(client))

    session.start("first")
    await asyncio.wait_for(client.entered.wait(), 2.0)
    await asyncio.wait_for(session.interrupt(), 2.0)

    client.release.set()
    session.start("second")  # must not raise SessionBusy
    await asyncio.wait_for(session.wait(), 2.0)


# --- failure ------------------------------------------------------------------


async def test_a_crashed_run_publishes_a_terminal_error(make_comp):
    """The failure encodes as a `TerminalEvent`, not a bespoke error frame.

    `wire.encode`'s match ends in `assert_never` so the wire vocabulary stays
    closed over the `Event` union, and `TerminalReason.ERROR` already exists.
    Reusing it means the browser's terminal handling covers a crashed run with
    no new client code.
    """
    session = AgentSession(make_comp(ExplodingClient()))
    await run_once(session)

    last = session.transcript[-1]
    assert last["type"] == "terminal"
    assert last["reason"] == "error"
    assert "exploded" in last["detail"]


async def test_a_crashed_run_does_not_leave_running_stuck(make_comp):
    """Without the `finally`, one provider error bricks the session: `running`
    stays True and every later goal is refused with 409 forever."""
    session = AgentSession(make_comp(ExplodingClient()))
    await run_once(session)
    assert not session.running


async def test_a_crashed_run_is_still_checkpointed(make_comp):
    comp = make_comp(ExplodingClient())
    session = AgentSession(comp)
    await run_once(session)

    saved = persistence.load(comp.session.id, comp.sessions_dir)
    assert saved.id == comp.session.id


# --- isolation ----------------------------------------------------------------


async def test_two_sessions_do_not_bleed(make_comp):
    """The gate for stage 2: two sessions, zero shared state.

    Class-level mutable attributes are the failure this catches — `_subs = set()`
    on the class passes every single-session test and fails this one instantly.
    """
    a = AgentSession(make_comp(StubClient([say("from a")])))
    b = AgentSession(make_comp(StubClient([say("from b")])))

    sub_a = a.subscribe()
    await asyncio.gather(run_once(a, "goal a"), run_once(b, "goal b"))

    assert len(a.transcript) == 4
    assert len(b.transcript) == 4
    assert [f["seq"] for f in a.transcript] == [0, 1, 2, 3]

    texts_a = [f["text"] for f in a.transcript if f["type"] == "text"]
    assert texts_a == ["from a"]

    seen = [await asyncio.wait_for(anext(sub_a), 1.0) for _ in range(4)]
    assert all("from b" not in str(f.get("text", "")) for f in seen)


async def test_two_sessions_keep_separate_histories(make_comp):
    client_a = StubClient([say("from a")])
    client_b = StubClient([say("from b")])
    a = AgentSession(make_comp(client_a))
    b = AgentSession(make_comp(client_b))

    await run_once(a, "goal a")
    await run_once(b, "goal b")

    assert client_a.received[0][0].blocks[0].text == "goal a"
    assert client_b.received[0][0].blocks[0].text == "goal b"


# --- teardown -----------------------------------------------------------------


async def test_aclose_ends_every_subscriber(session):
    a, b = session.subscribe(), session.subscribe()
    await asyncio.wait_for(session.aclose(), 2.0)

    for sub in (a, b):
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(sub), 1.0)


async def test_aclose_cancels_an_in_flight_run(make_comp):
    """Shutdown must not wait on a run that may never finish."""
    client = BlockingClient()
    session = AgentSession(make_comp(client))

    session.start("do it")
    await asyncio.wait_for(client.entered.wait(), 2.0)
    await asyncio.wait_for(session.aclose(), 2.0)

    assert not session.running


async def test_aclose_checkpoints(make_comp):
    comp = make_comp(StubClient([say("one")]))
    session = AgentSession(comp)
    await run_once(session)
    await asyncio.wait_for(session.aclose(), 2.0)

    saved = persistence.load(comp.session.id, comp.sessions_dir)
    assert saved.total_cost == pytest.approx(0.01)


async def test_aclose_is_idempotent(session):
    await asyncio.wait_for(session.aclose(), 2.0)
    await asyncio.wait_for(session.aclose(), 2.0)
