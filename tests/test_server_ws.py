"""Phase 8, stage 5 — WebSocket: the same core, duplex.

Two tests here are the reason the transport exists rather than being a nicety.
`test_the_frames_match_sse_exactly` proves the frame vocabulary lives in one place
— if it ever fails, encoding has leaked out of `server/wire.py` into a transport.
`test_a_confirm_answered_over_ws_resumes_the_run` is the deadlock test: a single
loop alternating receive and send passes every simple echo test and hangs the
moment a confirm appears, because it can only ever be waiting for one of the two.
"""

from __future__ import annotations

import asyncio

from asgi_client import SSEStream, WSSession, running
from conftest import SERVER_TOKEN

from client import NormalizedResponse, TextBlock, ToolCallBlock


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


async def wait_for_confirm(approver, timeout: float = 2.0) -> str:
    async with asyncio.timeout(timeout):
        while not approver.pending:
            await asyncio.sleep(0)
    return next(iter(approver.pending))


def ws_path(session_id: str, **params) -> str:
    query = "&".join(f"{k}={v}" for k, v in {"token": SERVER_TOKEN, **params}.items())
    return f"/ws/sessions/{session_id}?{query}"


# --- the same frames ----------------------------------------------------------


async def test_ready_arrives_before_anything_else(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        async with WSSession(app, ws_path(created.json()["session_id"])) as socket:
            assert await socket.frame() == {"type": "ready"}


async def test_the_frames_match_sse_exactly(forge_app, auth, stub_responses):
    """One vocabulary, two sockets. Neither transport encodes anything itself."""
    stub_responses([say("all done")])
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        over_ws = await client.post("/api/sessions", json={}, headers=auth)
        ws_id = over_ws.json()["session_id"]
        async with WSSession(app, ws_path(ws_id)) as socket:
            await socket.frame()  # ready
            await client.post(
                f"/api/sessions/{ws_id}/goal", json={"text": "hi"}, headers=auth
            )
            ws_session = app.state.manager.get(ws_id)
            await asyncio.wait_for(ws_session.wait(), 3.0)
            ws_frames = await socket.frames(len(ws_session.transcript))

        over_sse = await client.post("/api/sessions", json={}, headers=auth)
        sse_id = over_sse.json()["session_id"]
        async with SSEStream(
            app, f"/api/sessions/{sse_id}/events", headers=auth
        ) as stream:
            await stream.frame()  # ready
            await client.post(
                f"/api/sessions/{sse_id}/goal", json={"text": "hi"}, headers=auth
            )
            sse_session = app.state.manager.get(sse_id)
            await asyncio.wait_for(sse_session.wait(), 3.0)
            sse_frames = [
                frame for _, frame in await stream.frames(len(sse_session.transcript))
            ]

    assert ws_frames == sse_frames


async def test_after_seq_replays_the_tail(forge_app, auth, stub_responses):
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions", json={"goal": "hi"}, headers=auth
        )
        session = app.state.manager.get(created.json()["session_id"])
        await asyncio.wait_for(session.wait(), 3.0)
        last = session.transcript[-1]["seq"]

        async with WSSession(app, ws_path(session.id, after_seq=last - 1)) as socket:
            assert await socket.frame() == {"type": "ready"}
            assert await socket.frame() == session.transcript[-1]


# --- inbound ------------------------------------------------------------------


async def test_a_goal_sent_over_ws_starts_the_run(forge_app, auth, stub_responses):
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session = app.state.manager.get(created.json()["session_id"])

        async with WSSession(app, ws_path(session.id)) as socket:
            await socket.frame()  # ready
            await socket.send_json({"kind": "goal", "text": "say hello"})
            await asyncio.wait_for(session.wait(), 3.0)

    assert session.transcript[-1]["reason"] == "completed"


async def test_a_confirm_answered_over_ws_resumes_the_run(
    forge_app, auth, stub_responses
):
    """The deadlock test.

    While the agent is parked in `decide`, the drive task is parked too and
    nothing further is published. The confirm still reaches this socket because
    the writer is a *separate task* draining its own queue, and the answer still
    reaches the approver because the reader is another. Collapse those into one
    loop and this test hangs.
    """
    stub_responses([call_shell("echo hi"), say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions",
            json={"goal": "run something", "approval_mode": "on-request"},
            headers=auth,
        )
        session = app.state.manager.get(created.json()["session_id"])

        async with WSSession(app, ws_path(session.id)) as socket:
            request_id = await wait_for_confirm(session.approver)

            confirms = []
            while not confirms:
                frame = await socket.frame()
                if frame["type"] == "confirm":
                    confirms.append(frame)
            assert confirms[0]["request_id"] == request_id

            await socket.send_json(
                {"kind": "decision", "request_id": request_id, "approved": True}
            )
            await asyncio.wait_for(session.wait(), 3.0)

    assert session.transcript[-1]["reason"] == "completed"


async def test_interrupt_over_ws_stops_the_run(forge_app, auth, stub_responses):
    stub_responses([call_shell("echo hi"), say("done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions",
            json={"goal": "run something", "approval_mode": "on-request"},
            headers=auth,
        )
        session = app.state.manager.get(created.json()["session_id"])

        async with WSSession(app, ws_path(session.id)) as socket:
            await socket.frame()  # ready
            await wait_for_confirm(session.approver)
            await socket.send_json({"kind": "interrupt"})
            async with asyncio.timeout(3.0):
                while session.running:
                    await asyncio.sleep(0)

    assert session.transcript[-1]["message"] == "interrupted"


# --- failures are frames, not closes -----------------------------------------


async def test_a_busy_session_sends_an_error_and_stays_open(
    forge_app, auth, stub_responses
):
    """The 409 equivalent. Dropping the socket would take the transcript with it."""
    stub_responses([call_shell("echo hi"), say("done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions",
            json={"goal": "first", "approval_mode": "on-request"},
            headers=auth,
        )
        session = app.state.manager.get(created.json()["session_id"])
        await wait_for_confirm(session.approver)

        async with WSSession(app, ws_path(session.id)) as socket:
            await socket.frame()  # ready
            await socket.send_json({"kind": "goal", "text": "second"})

            errors = []
            while not errors:
                frame = await socket.frame()
                if frame["type"] == "error":
                    errors.append(frame)
            assert "already in flight" in errors[0]["detail"]

            # Still usable: the same socket answers the pending confirm.
            request_id = next(iter(session.approver.pending))
            await socket.send_json(
                {"kind": "decision", "request_id": request_id, "approved": False}
            )
            await asyncio.wait_for(session.wait(), 3.0)

    assert session.transcript[-1]["type"] == "terminal"


async def test_a_stale_decision_is_an_error_frame(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        async with WSSession(app, ws_path(created.json()["session_id"])) as socket:
            await socket.frame()  # ready
            await socket.send_json(
                {"kind": "decision", "request_id": "gone", "approved": True}
            )
            frame = await socket.frame()

    assert frame["type"] == "error"
    assert "gone" in frame["detail"]


async def test_an_unknown_kind_is_an_error_frame(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        async with WSSession(app, ws_path(created.json()["session_id"])) as socket:
            await socket.frame()  # ready
            await socket.send_json({"kind": "teleport"})
            frame = await socket.frame()

    assert frame == {"type": "error", "detail": "unknown frame kind 'teleport'"}


async def test_a_malformed_frame_is_an_error_frame(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        async with WSSession(app, ws_path(created.json()["session_id"])) as socket:
            await socket.frame()  # ready
            await socket.send_json({"kind": "goal"})  # no text
            frame = await socket.frame()

    assert frame["type"] == "error"
    assert "text" in frame["detail"]


async def test_a_closed_socket_unsubscribes(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session = app.state.manager.get(created.json()["session_id"])

        async with WSSession(app, ws_path(session.id)) as socket:
            await socket.frame()
            assert session.subscriber_count == 1

        assert session.subscriber_count == 0
