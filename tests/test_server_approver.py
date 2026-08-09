"""Phase 8, stage 3 — `ServerApprover`: the parked Future.

The only place data flows *back into* the agent. `agent.py:208` does
`decision = await self._approver.decide(request)` inside the async generator, so
the generator is suspended there and the whole run waits on a human. Over a
network that becomes: publish a confirm frame, park an `asyncio.Future`, resolve
it from a second request.

Why that does not deadlock, since it is the property the design rests on: while
the agent is suspended in `decide`, the drive task is suspended in `__anext__`
and nothing further is published — but the *transport* is a different task, and
frames already published are sitting in each subscriber's queue. So the client
receives everything up to and including the confirm, then goes quiet. Stage 2a's
per-subscriber queue is what buys this; a transport sharing the drive task's
coroutine would never deliver the confirm and the run would hang forever.

Two additions to `server/session.py` this file assumes, both small:

- `publish_frame(body: dict)` — publishes a **server-minted** frame (the confirm)
  through the same numbering and fan-out as `publish(event)`. Refactor `publish`
  to route through a shared `_publish_frame`; its external behaviour must not
  change, so all 104 existing tests stay green.

  The confirm frame therefore *does* get a `seq` and *does* enter the transcript,
  unlike `_OVERFLOW_FRAME`. That is deliberate: a browser that reconnects
  mid-confirm has to be re-shown the pending question, and replay is the only
  mechanism that can do it. Confirms whose `request_id` is no longer pending are
  the client's to discard — the server answers 409 for those anyway.

- `AgentSession(comp, *, approver=None)` — hold the approver explicitly rather
  than letting stage 4's decisions route reach through `agent._approver`, which
  would make a private attribute part of the HTTP layer's vocabulary.

Most of this file is unit tests against `ServerApprover` alone, with a recorder
standing in for `publish_frame`. Three integration tests at the bottom drive a
real agent run, because the thing that has to be *proved* rather than asserted is
that edited arguments reach `agent.py:218-241`.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import StubClient
from server.approver import ServerApprover, UnknownDecision

import main
import persistence
from approval import ApprovalRequest, Decision
from client import NormalizedResponse, TextBlock, ToolCallBlock
from server.session import AgentSession
from server.wire import RunParams
from tools.base import ToolKind

# --- doubles ------------------------------------------------------------------


class Recorder:
    """Stands in for `AgentSession.publish_frame`."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def __call__(self, frame: dict) -> None:
        self.frames.append(frame)


def request(
    *,
    tool_name: str = "run_shell",
    arguments: dict | None = None,
    kind: ToolKind = ToolKind.EXECUTE,
    danger_reasons: list[str] | None = None,
    diff: str | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        tool_name=tool_name,
        arguments=arguments if arguments is not None else {"command": "ls"},
        kind=kind,
        danger_reasons=danger_reasons or [],
        diff=diff,
    )


@pytest.fixture
def published():
    return Recorder()


@pytest.fixture
def approver(published):
    a = ServerApprover(timeout=1.0)
    a.bind(published)
    return a


async def park(approver, published, req=None):
    """Start a `decide` and wait until its confirm frame has been published.

    Returns `(task, request_id)`. Spins on `sleep(0)` rather than sleeping a
    fixed interval: the only thing being waited for is the event loop giving the
    new task a turn, and a real delay would make every test in this file slower
    for no added confidence.
    """
    task = asyncio.create_task(approver.decide(req if req is not None else request()))
    for _ in range(100):
        if published.frames:
            break
        await asyncio.sleep(0)
    else:  # pragma: no cover - only reached when the implementation is wrong
        task.cancel()
        raise AssertionError("decide() never published a confirm frame")
    return task, published.frames[-1]["request_id"]


# --- the confirm frame --------------------------------------------------------


async def test_decide_publishes_exactly_one_confirm_frame(approver, published):
    task, _ = await park(approver, published)
    assert [f["type"] for f in published.frames] == ["confirm"]
    task.cancel()


async def test_the_confirm_frame_carries_what_the_agent_event_cannot(
    approver, published
):
    """Why there are two frames rather than one.

    `agent.py:202-206` yields a `ConfirmRequestEvent` before awaiting, but it
    carries only `tool_name`, `arguments`, `reason` — no `kind`, no
    `danger_reasons`, no `diff`, and no correlation id. Those live on
    `ApprovalRequest`, which only the approver sees. A browser modal needs all
    of them: the id to answer with, `danger_reasons` to decide whether to offer
    "always", and the diff to show what a write would do.
    """
    req = request(
        tool_name="write_file",
        arguments={"path": "a.py", "content": "x = 1"},
        kind=ToolKind.WRITE,
        danger_reasons=["writes outside the repo root"],
        diff="--- a.py\n+++ a.py\n+x = 1",
    )
    task, request_id = await park(approver, published, req)
    (frame,) = published.frames

    assert frame["type"] == "confirm"
    assert frame["request_id"] == request_id
    assert frame["tool_name"] == "write_file"
    assert frame["arguments"] == {"path": "a.py", "content": "x = 1"}
    assert frame["kind"] == "write"
    assert frame["danger_reasons"] == ["writes outside the repo root"]
    assert frame["diff"] == "--- a.py\n+++ a.py\n+x = 1"
    task.cancel()


async def test_kind_encodes_by_name_not_value(approver, published):
    """`ToolKind` is `Enum(auto())`, so `.value` is a positional integer.

    Same rule the encoder already follows (`server/wire.py`): reordering the enum
    must not silently change the wire format for every deployed client.
    """
    task, _ = await park(approver, published, request(kind=ToolKind.EXECUTE))
    assert published.frames[0]["kind"] == "execute"
    task.cancel()


async def test_each_request_gets_a_distinct_id(approver, published):
    first, id_a = await park(approver, published)
    published.frames.clear()
    second, id_b = await park(approver, published)
    assert id_a != id_b
    first.cancel()
    second.cancel()


# --- offers_always ------------------------------------------------------------


async def test_offers_always_is_true_for_a_clean_call(approver, published):
    task, _ = await park(approver, published, request(danger_reasons=[]))
    assert published.frames[0]["offers_always"] is True
    task.cancel()


async def test_offers_always_is_false_when_danger_flagged(approver, published):
    """Computed server-side, not derived in the browser.

    Withholding "always" from a flagged call is a security rule
    (`tui/approver.py:76-84`), and a security rule enforced only in browser
    JavaScript is not enforced.
    """
    task, _ = await park(
        approver, published, request(danger_reasons=["deletes a directory tree"])
    )
    assert published.frames[0]["offers_always"] is False
    task.cancel()


# --- resolve ------------------------------------------------------------------


async def test_resolve_returns_an_approval(approver, published):
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=True)
    decision = await asyncio.wait_for(task, 1.0)
    assert decision == Decision(approved=True, reason=None, arguments=None)


async def test_resolve_returns_a_denial_with_its_reason(approver, published):
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=False, reason="not on main")
    decision = await asyncio.wait_for(task, 1.0)
    assert decision.approved is False
    assert decision.reason == "not on main"


async def test_resolve_passes_edited_arguments_through(approver, published):
    """The approver does not validate. `agent.py:218-241` already re-validates
    and re-runs both danger checks; a second copy here would diverge."""
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=True, arguments={"command": "ls -la"})
    decision = await asyncio.wait_for(task, 1.0)
    assert decision.arguments == {"command": "ls -la"}


async def test_resolve_is_synchronous(approver):
    """It runs from a request handler, and setting a Future's result never waits."""
    import inspect

    assert not inspect.iscoroutinefunction(ServerApprover.resolve)


# --- pending bookkeeping ------------------------------------------------------


async def test_the_request_is_pending_while_parked(approver, published):
    task, request_id = await park(approver, published)
    assert request_id in approver.pending
    task.cancel()


async def test_resolving_clears_pending(approver, published):
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=True)
    await asyncio.wait_for(task, 1.0)
    assert approver.pending == frozenset()


async def test_nothing_is_pending_before_the_first_confirm(approver):
    assert approver.pending == frozenset()


async def test_cancelling_decide_clears_pending(approver, published):
    """A cancelled run must not leave its question parked.

    `interrupt()` cancels the drive task, which throws `CancelledError` into the
    generator suspended at `decide`. Only a `finally` around the await removes
    the entry — which is why the pop cannot live after the await.
    """
    task, _ = await park(approver, published)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert approver.pending == frozenset()


async def test_repeated_timeouts_leak_nothing(published):
    """The leak this guards is unbounded and permanent.

    Every abandoned confirm would keep a dead Future keyed by a uuid nobody will
    ever send, in a process designed to run for weeks.
    """
    approver = ServerApprover(timeout=0.01)
    approver.bind(published)

    for _ in range(5):
        await approver.decide(request())

    assert approver.pending == frozenset()


# --- timeout ------------------------------------------------------------------


async def test_timeout_denies(published):
    """Invariant 6 applied to a human in the loop.

    Without this, a browser closed mid-confirm suspends the generator for the
    lifetime of the process — holding a session slot and an MCP process tree
    that the idle reaper cannot claim, because the run never ends.
    """
    approver = ServerApprover(timeout=0.05)
    approver.bind(published)

    decision = await asyncio.wait_for(approver.decide(request()), 2.0)

    assert decision.approved is False
    assert "timed out" in (decision.reason or "")


async def test_timeout_clears_pending(published):
    approver = ServerApprover(timeout=0.05)
    approver.bind(published)
    await asyncio.wait_for(approver.decide(request()), 2.0)
    assert approver.pending == frozenset()


async def test_resolving_after_a_timeout_raises(published):
    """The browser reconnects and answers a question that already expired. That
    is a 409, not a crash and not a silent success."""
    approver = ServerApprover(timeout=0.05)
    approver.bind(published)
    await asyncio.wait_for(approver.decide(request()), 2.0)
    request_id = published.frames[0]["request_id"]

    with pytest.raises(UnknownDecision):
        approver.resolve(request_id, approved=True)


# --- 409s ---------------------------------------------------------------------


async def test_resolving_an_unknown_id_raises(approver):
    with pytest.raises(UnknownDecision):
        approver.resolve("nosuchid", approved=True)


async def test_resolving_twice_raises(approver, published):
    """A double-clicked button or a retried POST.

    `set_result` on a done Future raises `InvalidStateError`, which as an
    unhandled 500 is indistinguishable from a real bug.
    """
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=True)
    await asyncio.wait_for(task, 1.0)

    with pytest.raises(UnknownDecision):
        approver.resolve(request_id, approved=False)


# --- the session-scoped "always" set -----------------------------------------


async def test_remember_short_circuits_the_next_call(approver, published):
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=True, remember=True)
    await asyncio.wait_for(task, 1.0)
    published.frames.clear()

    decision = await asyncio.wait_for(approver.decide(request()), 1.0)

    assert decision.approved is True
    assert published.frames == []  # nobody was asked


async def test_remember_does_not_cover_a_danger_flagged_call(approver, published):
    """Ported from `tui/approver.py:189`, and the reason it is a named rule.

    "Always" granted on a harmless `ls` must not silence the prompt for a later
    call that tripped a danger check — that is the one case where being asked
    every time is the entire point.
    """
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=True, remember=True)
    await asyncio.wait_for(task, 1.0)
    published.frames.clear()

    flagged = request(danger_reasons=["deletes a directory tree"])
    task2, _ = await park(approver, published, flagged)

    assert published.frames[0]["type"] == "confirm"  # asked anyway
    task2.cancel()


async def test_remember_is_scoped_to_the_tool_name(approver, published):
    task, request_id = await park(approver, published)
    approver.resolve(request_id, approved=True, remember=True)
    await asyncio.wait_for(task, 1.0)
    published.frames.clear()

    other, _ = await park(approver, published, request(tool_name="write_file"))
    assert published.frames[0]["tool_name"] == "write_file"
    other.cancel()


async def test_two_approvers_do_not_share_the_always_set(published):
    """Why `SessionManager` takes a *factory*. A shared instance would carry one
    browser tab's granted permission into every other tab."""
    a, b = ServerApprover(timeout=1.0), ServerApprover(timeout=1.0)
    frames_b = Recorder()
    a.bind(published)
    b.bind(frames_b)

    task, request_id = await park(a, published)
    a.resolve(request_id, approved=True, remember=True)
    await asyncio.wait_for(task, 1.0)

    task_b, _ = await park(b, frames_b)
    assert frames_b.frames[0]["type"] == "confirm"
    task_b.cancel()


# --- bind ---------------------------------------------------------------------


async def test_bind_twice_raises():
    """A silently double-bound approver publishes confirms into the wrong
    session — the worst bug available in a multi-session server, and it presents
    as a UI glitch rather than an error."""
    approver = ServerApprover()
    approver.bind(Recorder())
    with pytest.raises(RuntimeError):
        approver.bind(Recorder())


async def test_decide_before_bind_raises():
    """Loud, rather than a confirm nobody will ever see and a run that hangs
    until the timeout."""
    approver = ServerApprover(timeout=1.0)
    with pytest.raises(RuntimeError):
        await approver.decide(request())


# --- integration: through a real agent run -----------------------------------


def say(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="end_turn",
    )


def call_shell(command: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[
            ToolCallBlock(id="c1", name="run_shell", arguments={"command": command})
        ],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="tool_use",
    )


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A real session whose approver is a `ServerApprover`, wired as stage 3 wires it.

    `approval_mode="on-request"` is what makes the policy return `Verdict.ASK`
    for an EXECUTE call (`policy.py:26-28`) — under the default AUTO mode a clean
    shell command is auto-approved and the approver is never consulted.
    """
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: None)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")

    def _make(responses: list):
        approver = ServerApprover(timeout=2.0)
        comp = main.build_composition(
            RunParams(
                provider="groq",
                model="stub-model",
                project_root=tmp_path,
                approval_mode="on-request",
            ),
            approver=approver,
        )
        comp.agent.client = StubClient(responses)
        session = AgentSession(comp, approver=approver)
        approver.bind(session.publish_frame)
        return session, approver

    return _make


async def wait_for_confirm(session, approver, *, timeout: float = 2.0) -> str:
    """Spin until the run parks on a confirm."""
    async with asyncio.timeout(timeout):
        while not approver.pending:
            await asyncio.sleep(0)
    return next(iter(approver.pending))


async def test_a_confirm_lands_in_the_transcript_and_the_run_continues(live):
    """The confirm frame is numbered and replayable, unlike `overflow`.

    A browser reconnecting mid-confirm has to be re-shown the pending question,
    and replay is the only mechanism that can do it.
    """
    session, approver = live([call_shell("echo hi"), say("all done")])
    session.start("run something")

    request_id = await wait_for_confirm(session, approver)
    confirms = [f for f in session.transcript if f["type"] == "confirm"]
    assert len(confirms) == 1
    assert "seq" in confirms[0]

    approver.resolve(request_id, approved=True)
    await asyncio.wait_for(session.wait(), 3.0)

    assert session.transcript[-1]["type"] == "terminal"
    assert session.transcript[-1]["reason"] == "completed"


async def test_edited_arguments_are_re_checked_by_the_agent(live):
    """The §4.5 hole, closed — and proved rather than asserted.

    Approve a harmless `echo`, then edit it over the wire into `rm -rf /`. The
    agent must re-validate and re-run the danger checks (`agent.py:218-241`) and
    refuse. If it did not, the approval prompt would be a bypass for the schema,
    the danger checks, and the path guard all at once.
    """
    session, approver = live([call_shell("echo hi"), say("all done")])
    session.start("run something")

    request_id = await wait_for_confirm(session, approver)
    approver.resolve(request_id, approved=True, arguments={"command": "rm -rf /"})
    await asyncio.wait_for(session.wait(), 3.0)

    (decision,) = [f for f in session.transcript if f["type"] == "approval_decision"]
    assert decision["approved"] is False

    (result,) = [f for f in session.transcript if f["type"] == "tool_result"]
    assert result["result"].startswith("DENIED:")
    assert "rejected" in result["result"]


async def test_denying_stops_the_tool(live):
    session, approver = live([call_shell("echo hi"), say("all done")])
    session.start("run something")

    request_id = await wait_for_confirm(session, approver)
    approver.resolve(request_id, approved=False, reason="not today")
    await asyncio.wait_for(session.wait(), 3.0)

    (result,) = [f for f in session.transcript if f["type"] == "tool_result"]
    assert result["result"] == "DENIED: not today"


async def test_an_interrupt_mid_confirm_does_not_leave_a_parked_future(live):
    """Closing the tab is the common case, and it must not leak.

    The timeout would eventually clear it, but that is 300 seconds of a held
    session slot and an MCP process tree the reaper cannot claim.
    """
    session, approver = live([call_shell("echo hi"), say("all done")])
    session.start("run something")

    await wait_for_confirm(session, approver)
    await asyncio.wait_for(session.interrupt(), 3.0)

    assert approver.pending == frozenset()
    assert not session.running
