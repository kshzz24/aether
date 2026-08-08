"""Phase 8, stage 2c — `SessionManager`.

One dict of `id -> AgentSession`, plus the four policies that dict needs and
nothing else can own: a concurrency cap, MCP connection, the live ∪ on-disk
listing, and teardown.

Two additions to `server/session.py` this file assumes:

- `AgentSession.id` — a property returning `self._comp.session.id`. The manager
  needs it as its dict key, and reaching through `_comp` from a transport would
  make the session's internals part of the HTTP layer's vocabulary.
- nothing else; `AgentSession` is finished after 2b.

`create` is the only async method, and the only reason is `connect_mcp`. Tests
patch `server.sessions.connect_mcp`, which assumes the module imports it by name
(`from main import connect_mcp`) — the same style `server/session.py` already
uses for `checkpoint`.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
import server.sessions as sessions_mod
from conftest import ScriptedApprover, StubClient
from server.sessions import SessionLimitReached, SessionManager

import main
import persistence
from client import Message, NormalizedResponse, TextBlock
from main import CompositionError
from server.wire import RunParams

# --- helpers ------------------------------------------------------------------


def say(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="end_turn",
    )


def saved_session(
    session_id: str,
    *,
    goal: str = "an earlier goal",
    updated_at: str = "2026-08-01T12:00:00",
    total_cost: float = 0.05,
    messages: list[Message] | None = None,
) -> persistence.Session:
    """A checkpoint as it would exist on disk from a previous process."""
    return persistence.Session(
        id=session_id,
        goal=goal,
        provider="groq",
        model="stub-model",
        created_at="2026-08-01T11:00:00",
        updated_at=updated_at,
        total_cost=total_cost,
        messages=messages
        if messages is not None
        else [
            Message(role="user", blocks=[TextBlock(text=goal)]),
            Message(role="assistant", blocks=[TextBlock(text="an earlier reply")]),
        ],
    )


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def neutralise(tmp_path, monkeypatch):
    """Same three neutralisations as the other server test files.

    `load_mcp_manager` is the load-bearing one: it reads `Path.home()/.mcp.json`,
    an absolute path `project_root` does not contain, so unpatched every `create`
    here spawns the developer's real stdio subprocesses.
    """
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: None)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")
    return tmp_path


@pytest.fixture
def manager(neutralise):
    return SessionManager(
        sessions_dir=neutralise,
        make_approver=lambda: ScriptedApprover([]),
    )


def params(tmp_path, **kwargs) -> RunParams:
    return RunParams(
        provider="groq",
        model="stub-model",
        project_root=tmp_path,
        **kwargs,
    )


async def run_one_turn(session, text: str = "hello") -> None:
    """Drive a session once, so it has a checkpoint and a nonzero cost."""
    session._comp.agent.client = StubClient([say(text)])
    session.start("do it")
    await asyncio.wait_for(session.wait(), 2.0)


# --- create -------------------------------------------------------------------


async def test_create_is_async(manager):
    """The only async method, and `connect_mcp` is the only reason.

    `AgentSession.__init__` stays sync; if `create` ever becomes sync it means
    the per-session MCP manager stopped being connected, which is silent — the
    session works, it just has no MCP tools.
    """
    assert inspect.iscoroutinefunction(SessionManager.create)


async def test_create_returns_a_registered_session(manager, tmp_path):
    session = await manager.create(params(tmp_path))
    assert manager.get(session.id) is session


async def test_create_gives_each_session_its_own_id(manager, tmp_path):
    a = await manager.create(params(tmp_path))
    b = await manager.create(params(tmp_path))
    assert a.id != b.id


async def test_create_gives_each_session_its_own_composition(manager, tmp_path):
    """Isolation is per-`Composition`, not per-id. Sharing an `Agent` would give
    two conversations one message list — the bug a shared class attribute
    produces and a shared factory produces just as easily."""
    a = await manager.create(params(tmp_path))
    b = await manager.create(params(tmp_path))

    assert a._comp is not b._comp
    assert a._comp.agent is not b._comp.agent
    assert a._comp.registry is not b._comp.registry


async def test_create_gives_each_session_its_own_approver(manager, tmp_path):
    """`make_approver` is a factory, not an instance.

    A single shared approver would carry one session's "always allowed" set into
    every other session — a permission granted in one browser tab silently
    applying in another.
    """
    a = await manager.create(params(tmp_path))
    b = await manager.create(params(tmp_path))
    assert a._comp.agent._approver is not b._comp.agent._approver


async def test_create_connects_mcp(manager, tmp_path, monkeypatch):
    calls = []

    async def fake_connect(comp):
        calls.append(comp)
        return ""

    monkeypatch.setattr(sessions_mod, "connect_mcp", fake_connect)
    session = await manager.create(params(tmp_path))

    assert calls == [session._comp]


async def test_a_session_whose_mcp_failed_is_not_registered(
    manager, tmp_path, monkeypatch
):
    """Build, connect, *then* insert — or a failed connect leaves a half-built
    session in the dict, holding a slot and answering requests with no tools."""

    async def boom(comp):
        raise RuntimeError("stdio server died")

    monkeypatch.setattr(sessions_mod, "connect_mcp", boom)

    with pytest.raises(RuntimeError):
        await manager.create(params(tmp_path))

    assert manager.list() == []


async def test_create_beyond_the_cap_raises(neutralise, tmp_path):
    """`max_sessions` is a real limit, not a knob.

    Each session owns an MCP manager, so it owns a process tree. Without this
    guard, opening browser tabs is a fork bomb — which is why §10.2 calls the
    idle reaper (stage 6) mandatory rather than polish.
    """
    manager = SessionManager(
        sessions_dir=neutralise,
        make_approver=lambda: ScriptedApprover([]),
        max_sessions=2,
    )
    await manager.create(params(tmp_path))
    await manager.create(params(tmp_path))

    with pytest.raises(SessionLimitReached):
        await manager.create(params(tmp_path))


# --- get ----------------------------------------------------------------------


async def test_get_unknown_id_raises_key_error(manager):
    """KeyError, so stage 4's handler maps it to 404 without inventing a type."""
    with pytest.raises(KeyError):
        manager.get("20260101-000000-dead")


async def test_get_is_synchronous(manager):
    assert not inspect.iscoroutinefunction(SessionManager.get)


# --- resume -------------------------------------------------------------------


async def test_resume_seeds_history_from_the_saved_session(manager, tmp_path):
    """The point of `AgentSession.__init__` doing `list(comp.history or [])`.

    Without it a resumed session's first turn discards the loaded conversation
    silently: `_next_history` sees an empty list, returns None, and the agent
    seeds fresh from the goal. Resume would appear to work and lose everything.
    """
    persistence.save(saved_session("20260801-120000-aaaa"), tmp_path)

    session = await manager.create(params(tmp_path, resume="20260801-120000-aaaa"))

    assert [m.role for m in session._history] == ["user", "assistant"]
    assert session._history[-1].blocks[0].text == "an earlier reply"


async def test_resume_keeps_the_saved_session_id(manager, tmp_path):
    """A resumed session must not be given a fresh id, or the next checkpoint
    writes a second file and the original is orphaned."""
    persistence.save(saved_session("20260801-120000-bbbb"), tmp_path)
    session = await manager.create(params(tmp_path, resume="20260801-120000-bbbb"))
    assert session.id == "20260801-120000-bbbb"


async def test_resume_unknown_id_raises_composition_error(manager, tmp_path):
    """`build_composition` already turns a missing file into `CompositionError`
    (`main.py:177`); the manager passes it through rather than re-wrapping."""
    with pytest.raises(CompositionError):
        await manager.create(params(tmp_path, resume="20260101-000000-dead"))


async def test_a_failed_resume_does_not_consume_a_slot(neutralise, tmp_path):
    manager = SessionManager(
        sessions_dir=neutralise,
        make_approver=lambda: ScriptedApprover([]),
        max_sessions=1,
    )
    with pytest.raises(CompositionError):
        await manager.create(params(tmp_path, resume="20260101-000000-dead"))

    await manager.create(params(tmp_path))  # must not raise SessionLimitReached


# --- list ---------------------------------------------------------------------


async def test_list_includes_live_sessions(manager, tmp_path):
    """A session that has never checkpointed is not on disk yet, and must still
    appear — otherwise a browser that just created one cannot see it."""
    session = await manager.create(params(tmp_path))
    assert [m.id for m in manager.list()] == [session.id]


async def test_list_includes_sessions_only_on_disk(manager, tmp_path):
    persistence.save(saved_session("20260801-120000-cccc"), tmp_path)
    assert [m.id for m in manager.list()] == ["20260801-120000-cccc"]


async def test_list_reports_the_saved_goal_and_cost(manager, tmp_path):
    persistence.save(
        saved_session("20260801-120000-dddd", goal="fix the parser", total_cost=0.25),
        tmp_path,
    )
    (meta,) = manager.list()
    assert meta.goal == "fix the parser"
    assert meta.total_cost == pytest.approx(0.25)
    assert meta.turns == 2


async def test_list_does_not_duplicate_a_session_that_is_both(manager, tmp_path):
    """A live session appears in the dict *and* on disk once it checkpoints.
    Listing both sources naively shows it twice."""
    session = await manager.create(params(tmp_path))
    await run_one_turn(session)

    ids = [m.id for m in manager.list()]
    assert ids.count(session.id) == 1


async def test_list_prefers_the_live_session_over_the_file(manager, tmp_path):
    """The dict is fresher than the last checkpoint by definition — the file is
    at best one turn behind, and stale between turns."""
    session = await manager.create(params(tmp_path))
    await run_one_turn(session)

    # A stale file for the same id, as a crashed earlier process would leave.
    persistence.save(
        saved_session(session.id, goal="stale goal", total_cost=99.0), tmp_path
    )

    (meta,) = [m for m in manager.list() if m.id == session.id]
    assert meta.total_cost != pytest.approx(99.0)
    assert meta.goal != "stale goal"


async def test_list_is_newest_first(manager, tmp_path):
    persistence.save(
        saved_session("20260801-120000-old", updated_at="2026-08-01T09:00:00"),
        tmp_path,
    )
    persistence.save(
        saved_session("20260801-120000-new", updated_at="2026-08-03T09:00:00"),
        tmp_path,
    )
    assert [m.id for m in manager.list()] == [
        "20260801-120000-new",
        "20260801-120000-old",
    ]


async def test_list_on_an_empty_manager_is_empty(manager):
    assert manager.list() == []


# --- delete -------------------------------------------------------------------


async def test_delete_evicts_the_session(manager, tmp_path):
    session = await manager.create(params(tmp_path))
    await manager.delete(session.id)

    with pytest.raises(KeyError):
        manager.get(session.id)


async def test_delete_checkpoints_before_evicting(manager, tmp_path):
    """Eviction must not lose the conversation — the session stays resumable
    from disk after the browser deletes it."""
    session = await manager.create(params(tmp_path))
    await run_one_turn(session)
    await manager.delete(session.id)

    saved = persistence.load(session.id, tmp_path)
    assert saved.total_cost == pytest.approx(0.01)


async def test_delete_ends_every_subscriber(manager, tmp_path):
    """A deleted session's watchers must be released, or an SSE handler blocks
    on `await queue.get()` for a session that no longer exists."""
    session = await manager.create(params(tmp_path))
    sub = session.subscribe()
    await manager.delete(session.id)

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(sub), 1.0)


async def test_delete_cancels_an_in_flight_run(manager, tmp_path):
    session = await manager.create(params(tmp_path))
    session._comp.agent.client = StubClient([say("hi")])
    session.start("do it")
    await manager.delete(session.id)
    assert not session.running


async def test_delete_frees_a_slot(neutralise, tmp_path):
    manager = SessionManager(
        sessions_dir=neutralise,
        make_approver=lambda: ScriptedApprover([]),
        max_sessions=1,
    )
    first = await manager.create(params(tmp_path))
    await manager.delete(first.id)
    await manager.create(params(tmp_path))  # must not raise


async def test_delete_unknown_id_raises_key_error(manager):
    with pytest.raises(KeyError):
        await manager.delete("20260101-000000-dead")


async def test_a_deleted_session_is_still_listed_from_disk(manager, tmp_path):
    """Deleting evicts from memory; it does not delete the checkpoint. The
    session leaves the live set and rejoins the list as a resumable one."""
    session = await manager.create(params(tmp_path))
    await run_one_turn(session)
    await manager.delete(session.id)

    assert [m.id for m in manager.list()] == [session.id]


# --- shutdown -----------------------------------------------------------------


async def test_aclose_all_closes_every_session(manager, tmp_path):
    a = await manager.create(params(tmp_path))
    b = await manager.create(params(tmp_path))
    sub_a, sub_b = a.subscribe(), b.subscribe()

    await manager.aclose_all()

    for sub in (sub_a, sub_b):
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(sub), 1.0)


async def test_aclose_all_checkpoints_every_session(manager, tmp_path):
    """The shutdown gate: kill the server mid-session and nothing is lost."""
    a = await manager.create(params(tmp_path))
    b = await manager.create(params(tmp_path))
    await run_one_turn(a)
    await run_one_turn(b)

    await manager.aclose_all()

    for session in (a, b):
        assert persistence.load(session.id, tmp_path).id == session.id


async def test_aclose_all_empties_the_live_set(manager, tmp_path):
    a = await manager.create(params(tmp_path))
    await manager.aclose_all()

    with pytest.raises(KeyError):
        manager.get(a.id)


async def test_aclose_all_is_idempotent(manager, tmp_path):
    """A lifespan teardown can run after an explicit shutdown call."""
    await manager.create(params(tmp_path))
    await manager.aclose_all()
    await manager.aclose_all()


async def test_aclose_all_on_an_empty_manager_is_a_no_op(manager):
    await manager.aclose_all()


# --- isolation ----------------------------------------------------------------


async def test_two_managed_sessions_do_not_bleed(manager, tmp_path):
    """Stage 2's gate, now through the manager rather than by hand."""
    a = await manager.create(params(tmp_path))
    b = await manager.create(params(tmp_path))

    await run_one_turn(a, "from a")
    await run_one_turn(b, "from b")

    texts_a = [f["text"] for f in a.transcript if f["type"] == "text"]
    texts_b = [f["text"] for f in b.transcript if f["type"] == "text"]
    assert texts_a == ["from a"]
    assert texts_b == ["from b"]
    assert a.total_cost == pytest.approx(0.01)
    assert b.total_cost == pytest.approx(0.01)


async def test_each_session_checkpoints_to_its_own_file(manager, tmp_path):
    a = await manager.create(params(tmp_path))
    b = await manager.create(params(tmp_path))
    await run_one_turn(a)
    await run_one_turn(b)

    assert (tmp_path / f"{a.id}.json").exists()
    assert (tmp_path / f"{b.id}.json").exists()
