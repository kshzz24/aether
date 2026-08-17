"""Phase 8, stage 6 — what happens when a session ends without anyone asking.

Three things that only matter once the process outlives a single run: MCP
subprocesses have to be closed on *every* eviction path, sessions nobody is
watching have to be swept, and shutdown has to checkpoint rather than vanish.

The MCP close is a real bug fix, not new scope: `main.py:313-314` closes the
manager in the CLI's `finally`, and `AgentSession.aclose` did not — so before this
stage every DELETE, every reap and every shutdown leaked a process tree.
"""

from __future__ import annotations

import asyncio
import json
import time

from asgi_client import running

import main
import persistence
from client import NormalizedResponse, TextBlock
from server.app import reap_once


def say(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="end_turn",
    )


class StubMCP:
    """Enough of `MCPManager` for `connect_mcp` and `aclose` to run."""

    def __init__(self) -> None:
        self.closed = 0

    async def connect_all(self) -> None:
        pass

    def register_into(self, registry) -> int:
        return 0

    def statuses(self) -> list:
        return []

    async def aclose(self) -> None:
        self.closed += 1


# --- the MCP leak -------------------------------------------------------------


async def test_delete_closes_mcp(forge_app, auth, monkeypatch):
    """Every session spawns its own stdio subprocesses. Somebody has to reap them."""
    mcp = StubMCP()
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: mcp)

    async with running(forge_app()) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        assert mcp.closed == 0
        session_id = created.json()["session_id"]
        await client.delete(f"/api/sessions/{session_id}", headers=auth)

    assert mcp.closed == 1


async def test_shutdown_closes_mcp(forge_app, auth, monkeypatch):
    """The fix lives in `aclose`, so all three eviction paths inherit it."""
    mcp = StubMCP()
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: mcp)

    async with running(forge_app()) as client:
        await client.post("/api/sessions", json={}, headers=auth)

    assert mcp.closed == 1


async def test_closing_twice_is_harmless(forge_app, auth, monkeypatch):
    """`aclose` is documented idempotent, and `aclose_all` can follow a DELETE."""
    mcp = StubMCP()
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: mcp)

    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session = app.state.manager.get(created.json()["session_id"])
        await session.aclose()
        await session.aclose()

    assert mcp.closed >= 2


# --- the reaper ---------------------------------------------------------------


async def make_session(app, client, auth, *, goal=None):
    body = {} if goal is None else {"goal": goal}
    created = await client.post("/api/sessions", json=body, headers=auth)
    return app.state.manager.get(created.json()["session_id"])


async def test_an_idle_unwatched_session_is_reaped(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        session = await make_session(app, client, auth)
        session.last_activity = time.monotonic() - 10_000

        assert await reap_once(app.state.manager, 900.0) == [session.id]
        assert app.state.manager.live == []


async def test_a_fresh_session_is_left_alone(forge_app, auth):
    """Reaping on age alone would kill the tab you left open over lunch."""
    app = forge_app()
    async with running(app) as client:
        session = await make_session(app, client, auth)

        assert await reap_once(app.state.manager, 900.0) == []
        assert app.state.manager.live == [session]


async def test_a_watched_session_is_left_alone(forge_app, auth):
    """A subscriber means a browser is looking at it right now."""
    app = forge_app()
    async with running(app) as client:
        session = await make_session(app, client, auth)
        session.last_activity = time.monotonic() - 10_000
        sub = session.subscribe()

        assert await reap_once(app.state.manager, 900.0) == []

        session.unsubscribe(sub)
        assert await reap_once(app.state.manager, 900.0) == [session.id]


async def test_a_running_session_is_left_alone(forge_app, auth, stub_responses):
    """Reaping a running session kills work in progress."""
    stub_responses([say("done")])
    app = forge_app()
    async with running(app) as client:
        session = await make_session(app, client, auth, goal="work")
        session.last_activity = time.monotonic() - 10_000

        assert session.running is True
        assert await reap_once(app.state.manager, 900.0) == []

        await asyncio.wait_for(session.wait(), 3.0)


async def test_activity_resets_the_clock(forge_app, auth, stub_responses):
    stub_responses([say("done")])
    app = forge_app()
    async with running(app) as client:
        session = await make_session(app, client, auth)
        session.last_activity = time.monotonic() - 10_000

        await client.post(
            f"/api/sessions/{session.id}/goal", json={"text": "hi"}, headers=auth
        )
        await asyncio.wait_for(session.wait(), 3.0)

        assert await reap_once(app.state.manager, 900.0) == []


async def test_one_failing_session_does_not_stop_the_sweep(forge_app, auth):
    """A reaper that dies on the first error is worse than none: it looks fine."""
    app = forge_app()
    async with running(app) as client:
        broken = await make_session(app, client, auth)
        healthy = await make_session(app, client, auth)
        for session in (broken, healthy):
            session.last_activity = time.monotonic() - 10_000

        async def _explode() -> None:
            raise RuntimeError("teardown blew up")

        broken.aclose = _explode

        reaped = await reap_once(app.state.manager, 900.0)

    assert reaped == [healthy.id]


async def test_the_reaper_task_is_cancelled_on_shutdown(forge_app):
    app = forge_app(reap_interval_sec=0.01)
    async with running(app):
        await asyncio.sleep(0.05)
        reaper = app.state.reaper
        assert reaper.done() is False

    assert reaper.cancelled() is True


async def test_the_reaper_survives_its_own_sweeps(forge_app, auth):
    """The loop keeps running: an eviction is not the end of the reaper's life."""
    app = forge_app(reap_interval_sec=0.01, idle_timeout_sec=0.0)
    async with running(app) as client:
        await make_session(app, client, auth)
        async with asyncio.timeout(3.0):
            while app.state.manager.live:
                await asyncio.sleep(0.01)
        assert app.state.reaper.done() is False


# --- shutdown ----------------------------------------------------------------


async def test_shutdown_checkpoints_every_session(forge_app, auth, stub_responses,
                                                  tmp_path):
    """A killed server must lose at most the in-flight turn, never the session."""
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        session = await make_session(app, client, auth, goal="say hello")
        await asyncio.wait_for(session.wait(), 3.0)
        session_id = session.id

    saved = json.loads((tmp_path / f"{session_id}.json").read_text(encoding="utf-8"))
    assert saved["goal"] == "say hello"
    assert saved["messages"]
    assert app.state.manager.live == []


async def test_a_session_created_over_http_is_named_on_disk(forge_app, auth,
                                                            stub_responses, tmp_path):
    """The web path creates first and names later, so `list_sessions` needs this."""
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        session = await make_session(app, client, auth)
        await client.post(
            f"/api/sessions/{session.id}/goal",
            json={"text": "port the parser"},
            headers=auth,
        )
        await asyncio.wait_for(session.wait(), 3.0)
        session_id = session.id

    (meta,) = persistence.list_sessions(tmp_path)
    assert (meta.id, meta.goal) == (session_id, "port the parser")
