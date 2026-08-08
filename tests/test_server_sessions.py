"""Phase 8, stage 2a — the fan-out and the replay buffer.

This file pins `AgentSession`'s *publish side* only: the transcript, the `seq`
counter, subscribe/replay, and the backpressure drop. The drive task (`start`,
`interrupt`, checkpointing, per-session isolation) is stage 2b and lands in a
second file — deliberately, because every property here is testable with plain
queues and a hand-driven `publish`, so when one breaks you are reading ten lines
instead of debugging an async generator.

The conversion this stage implements: both existing surfaces have the *consumer*
driving the agent (`tui/app.py:965`, `main.py:304`), so the agent's progress is
gated on someone pulling. A server cannot work that way — nobody may be
listening when a run starts, N may listen, they leave mid-run, and a slow one
must not slow the agent down. So events are *pushed* into a bounded queue per
subscriber, and the transcript is what makes dropping a slow subscriber safe
rather than lossy.

The two tests worth reading twice are `test_replay_is_not_bounded_by_the_queue`
(the trap that makes a late subscriber drop itself on arrival) and
`test_subscribe_is_synchronous` (the race that has no other guard).
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from conftest import ScriptedApprover, StubClient
from server.session import AgentSession

import main
import persistence
from events import StatusEvent
from server.wire import RunParams

# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def make_comp(tmp_path, monkeypatch):
    """Build a real `Composition` with a stubbed client and no side effects.

    Three things must be neutralised, and the second is the one that bites:

    1. `persistence.default_sessions_dir()` -> the user's real `~/.forge`.
    2. `main.load_mcp_manager(root)` reads `Path.home() / ".mcp.json"` — an
       *absolute* path, so `project_root=tmp_path` does not contain it. Left
       unpatched, every test in this file spawns the developer's actual MCP
       stdio subprocesses.
    3. API keys, so `make_client` builds without reaching for the environment.

    The client is swapped *after* construction, which is the pattern the TUI
    tests already use (`test_tui_session.py:130`) — `build_composition` owns
    client construction and there is no seam to inject one through.
    """
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: None)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")

    def _make(responses: list | None = None):
        comp = main.build_composition(
            RunParams(
                provider="groq",
                model="stub-model",
                project_root=tmp_path,
            ),
            approver=ScriptedApprover([]),
        )
        comp.agent.client = StubClient(responses or [])
        return comp

    return _make


@pytest.fixture
def session(make_comp):
    """A session with a *tiny* queue, so the overflow path is cheap to reach."""
    return AgentSession(make_comp(), queue_maxsize=4)


def status(i: int) -> StatusEvent:
    """A distinguishable event. `message` doubles as the expected ordinal."""
    return StatusEvent(type="status", message=f"m{i}")


async def take(sub, n: int, *, timeout: float = 1.0) -> list[dict]:
    """Pull exactly `n` frames, failing fast rather than hanging the suite."""
    return [await asyncio.wait_for(anext(sub), timeout) for _ in range(n)]


async def expect_end(sub, *, timeout: float = 1.0) -> None:
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(sub), timeout)


# --- the seq counter and the transcript ---------------------------------------


async def test_publish_assigns_monotonic_seq_from_zero(session):
    for i in range(3):
        session.publish(status(i))
    assert [f["seq"] for f in session.transcript] == [0, 1, 2]


async def test_transcript_holds_encoded_frames_not_events(session):
    """Decision D1: the buffer stores what goes on the wire, not `Event`s.

    Storing frames makes "the replayed frame equals the live frame" structural
    rather than a property you have to trust, assigns `seq` exactly once at
    ingest, and turns `subscribe(after_seq=n)` into a list slice. It is also
    what makes the stage-5 gate ("identical frames on SSE and WS") true by
    construction instead of by coincidence: both transports read this list.
    """
    session.publish(status(0))
    (frame,) = session.transcript
    assert isinstance(frame, dict)
    assert frame == {"type": "status", "message": "m0", "seq": 0}


async def test_seq_is_session_scoped_not_run_scoped(session):
    """The transcript spans turns, so the counter must not restart.

    Stage 2b threads a second goal into the same session; if `seq` reset per
    run, a `Last-Event-ID` reconnect would replay from the wrong offset and the
    browser would silently show turn 2 spliced over turn 1.
    """
    for i in range(3):
        session.publish(status(i))
    session.publish(status(99))
    assert [f["seq"] for f in session.transcript] == [0, 1, 2, 3]


# --- subscribe and replay -----------------------------------------------------


async def test_subscriber_receives_frames_published_after_it_joined(session):
    sub = session.subscribe()
    for i in range(3):
        session.publish(status(i))
    frames = await take(sub, 3)
    assert [f["message"] for f in frames] == ["m0", "m1", "m2"]


async def test_late_subscriber_replays_the_whole_transcript(session):
    for i in range(3):
        session.publish(status(i))
    sub = session.subscribe()
    frames = await take(sub, 3)
    assert [f["seq"] for f in frames] == [0, 1, 2]


async def test_replayed_frames_keep_their_original_seq(session):
    """Replay must not renumber. `seq` identifies a position in the transcript,
    so a client that reconnects twice has to see the same id for the same
    event — otherwise `Last-Event-ID` addresses a moving target."""
    session.publish(status(0))
    first = await take(session.subscribe(), 1)
    second = await take(session.subscribe(), 1)
    assert first == second


async def test_after_seq_replays_only_the_tail(session):
    for i in range(5):
        session.publish(status(i))
    sub = session.subscribe(after_seq=2)
    frames = await take(sub, 2)
    assert [f["seq"] for f in frames] == [3, 4]


async def test_after_seq_past_the_end_replays_nothing(session):
    """A client reconnecting with the newest id it already holds must not be
    re-sent anything, and must not error."""
    session.publish(status(0))
    sub = session.subscribe(after_seq=99)
    session.publish(status(1))
    frames = await take(sub, 1)
    assert [f["seq"] for f in frames] == [1]


async def test_late_subscriber_sees_no_duplicates_across_the_join(session):
    """The join is where an off-by-one shows up: snapshot-then-register loses
    an event published between the two, register-then-snapshot duplicates it."""
    for i in range(3):
        session.publish(status(i))
    sub = session.subscribe()
    for i in range(3, 5):
        session.publish(status(i))
    frames = await take(sub, 5)
    assert [f["seq"] for f in frames] == [0, 1, 2, 3, 4]


async def test_subscribe_is_synchronous(session):
    """Decision D3, and the only guard this race has.

    `subscribe` must snapshot the transcript tail and register the queue with
    no `await` between them. asyncio is single-threaded and cooperative, so a
    function containing no await point cannot be interrupted — that, and
    nothing else, is what closes the gap. An `async def subscribe` invites a
    later refactor to add an await in the middle and silently reopen it, and no
    behavioural test would catch that because it needs a publish to land in the
    exact window.
    """
    assert not inspect.iscoroutinefunction(AgentSession.subscribe)


async def test_two_subscribers_both_see_every_frame(session):
    a, b = session.subscribe(), session.subscribe()
    for i in range(3):
        session.publish(status(i))
    assert await take(a, 3) == await take(b, 3)


# --- backpressure -------------------------------------------------------------


async def test_slow_subscriber_is_dropped_with_overflow(session):
    """Decision D4: the agent must never block on a consumer.

    `queue_maxsize=4`, ten events, nothing drained. The subscriber keeps the
    four it buffered, then learns it was cut off — it does not silently receive
    a transcript with a hole in it.
    """
    sub = session.subscribe()
    for i in range(10):
        session.publish(status(i))

    frames = await take(sub, 4)
    assert [f["seq"] for f in frames] == [0, 1, 2, 3]

    (control,) = await take(sub, 1)
    assert control["type"] == "overflow"
    await expect_end(sub)


async def test_overflow_frame_carries_no_seq(session):
    """Control frames are not transcript entries.

    `seq` addresses a position in the transcript for `Last-Event-ID` to resume
    from. Giving a server-minted control frame a seq would hand the client an
    id that indexes nothing, and reconnecting on it would replay the wrong
    tail.
    """
    sub = session.subscribe()
    for i in range(10):
        session.publish(status(i))
    frames = await take(sub, 5)
    assert "seq" not in frames[-1]


async def test_overflow_does_not_enter_the_transcript(session):
    sub = session.subscribe()
    for i in range(10):
        session.publish(status(i))
    await take(sub, 5)
    assert len(session.transcript) == 10
    assert all(f["type"] == "status" for f in session.transcript)


async def test_dropping_one_subscriber_spares_the_others(session):
    """The whole point of per-subscriber queues: one dead browser tab must not
    cost the other tab its stream, and must not stop the run."""
    slow = session.subscribe()
    fast = session.subscribe()

    for i in range(10):
        session.publish(status(i))
        # Drain `fast` as we go so it never fills; `slow` is left to rot.
        await take(fast, 1)

    assert len(session.transcript) == 10
    frames = await take(slow, 5)
    assert frames[-1]["type"] == "overflow"


async def test_replay_is_not_bounded_by_the_queue(session):
    """The trap: a subscriber that overflows on the instant it arrives.

    With `queue_maxsize=4`, a client reconnecting to a 10-event session cannot
    have its replay pushed through the live queue — it would fill, mark itself
    overflowed, and be dropped before delivering a single frame. Replay is a
    separate, unbounded handoff; the bound applies only to the live stream.
    That is what makes the drop in `test_slow_subscriber_is_dropped_with_
    overflow` recoverable rather than terminal.
    """
    for i in range(10):
        session.publish(status(i))

    sub = session.subscribe()
    frames = await take(sub, 10)
    assert [f["seq"] for f in frames] == list(range(10))


# --- teardown -----------------------------------------------------------------


async def test_unsubscribe_ends_iteration_without_overflow(session):
    """A closed tab is a normal outcome. It gets no `overflow` frame — that
    would report a fault where there was none, and the stage-4 transport would
    surface it to a user who simply navigated away."""
    sub = session.subscribe()
    session.publish(status(0))
    frames = await take(sub, 1)
    assert frames[0]["type"] == "status"

    session.unsubscribe(sub)
    await expect_end(sub)


async def test_unsubscribe_stops_delivery_but_not_publishing(session):
    sub = session.subscribe()
    session.unsubscribe(sub)
    session.publish(status(0))
    assert len(session.transcript) == 1


async def test_unsubscribe_is_idempotent(session):
    """Stage 4 will call this from a transport's `finally`, which can run after
    the subscriber was already dropped for overflow. A second call must not
    raise."""
    sub = session.subscribe()
    session.unsubscribe(sub)
    session.unsubscribe(sub)


async def test_publish_with_no_subscribers_is_recorded(session):
    """The reason the drive task exists: a run started by `POST /goal` produces
    events before the client has opened its event stream. Those events must be
    in the transcript, or the first thing the browser does is miss the start of
    its own run."""
    session.publish(status(0))
    assert [f["seq"] for f in session.transcript] == [0]
