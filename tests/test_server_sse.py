"""Phase 8, stage 4 — SSE: the default transport, and the one with subtleties.

Three of them, all invisible until a browser is watching: which frames may carry
an `id:`, which offset wins when the header and the query string disagree, and
whether a dropped connection unsubscribes. The first two decide whether reconnect
works at all; the third decides whether a server survives a day of browser tabs.
"""

from __future__ import annotations

import asyncio

from asgi_client import SSEStream, running

from client import NormalizedResponse, TextBlock
from server.transports import sse


def say(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="end_turn",
    )


async def _finished_session(client, app, auth, *, goal="say hello"):
    """A session whose run has completed, so its transcript is stable."""
    created = await client.post("/api/sessions", json={"goal": goal}, headers=auth)
    session = app.state.manager.get(created.json()["session_id"])
    await asyncio.wait_for(session.wait(), 3.0)
    return session


# --- framing ------------------------------------------------------------------


def test_a_sequenced_frame_carries_an_id_line():
    assert sse.encode({"type": "text", "text": "hi", "seq": 7}) == (
        'id: 7\ndata: {"type": "text", "text": "hi", "seq": 7}\n\n'
    )


def test_control_frames_without_a_seq_carry_no_id_line():
    """`ready` and `overflow` are not transcript positions.

    Giving either an `id:` would make the browser store an offset that indexes
    nothing, and its next automatic reconnect would ask to resume from there.
    """
    assert sse.encode({"type": "ready"}) == 'data: {"type": "ready"}\n\n'
    assert sse.encode({"type": "overflow"}) == 'data: {"type": "overflow"}\n\n'


def test_the_header_beats_the_query_parameter():
    """`EventSource` retries the same URL, so the two coexist and disagree.

    `?after_seq=` was written once, when the stream was first mounted;
    `Last-Event-ID` is the browser's own record of the last frame it actually
    received. If the query won, every reconnect would replay from the original
    mount point and the client would see the tail twice.
    """
    assert sse.resolve_offset(3, "11") == 11
    assert sse.resolve_offset(3, None) == 3
    assert sse.resolve_offset(-1, "") == -1


def test_a_malformed_last_event_id_falls_back_rather_than_guessing():
    assert sse.resolve_offset(4, "not-a-number") == 4


# --- the stream ---------------------------------------------------------------


async def test_ready_comes_first_then_the_transcript_in_order(forge_app, auth,
                                                              stub_responses):
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        session = await _finished_session(client, app, auth)

        path = f"/api/sessions/{session.id}/events"
        async with SSEStream(app, path, headers=auth) as stream:
            first_id, first = await stream.frame()
            assert (first_id, first) == (None, {"type": "ready"})

            replayed = await stream.frames(len(session.transcript))

    ids = [event_id for event_id, _ in replayed]
    assert ids == [frame["seq"] for frame in session.transcript]
    assert ids == sorted(ids)
    assert [frame for _, frame in replayed] == session.transcript


async def test_after_seq_replays_only_the_tail(forge_app, auth, stub_responses):
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        session = await _finished_session(client, app, auth)
        last = session.transcript[-1]["seq"]

        path = f"/api/sessions/{session.id}/events?after_seq={last - 1}"
        async with SSEStream(app, path, headers=auth) as stream:
            await stream.frame()  # ready
            event_id, frame = await stream.frame()

    assert event_id == last
    assert frame == session.transcript[-1]


async def test_last_event_id_replays_only_the_tail(forge_app, auth, stub_responses):
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        session = await _finished_session(client, app, auth)
        last = session.transcript[-1]["seq"]

        headers = {**auth, "Last-Event-ID": str(last - 1)}
        path = f"/api/sessions/{session.id}/events"
        async with SSEStream(app, path, headers=headers) as stream:
            await stream.frame()
            event_id, frame = await stream.frame()

    assert event_id == last
    assert frame == session.transcript[-1]


async def test_last_event_id_beats_a_stale_after_seq(forge_app, auth, stub_responses):
    """The reconnect case, end to end.

    The URL still says `after_seq=-1` because that is the URL the browser was
    constructed with; the header says it already has everything but the last
    frame. Only one of those is current.
    """
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        session = await _finished_session(client, app, auth)
        last = session.transcript[-1]["seq"]

        headers = {**auth, "Last-Event-ID": str(last - 1)}
        path = f"/api/sessions/{session.id}/events?after_seq=-1"
        async with SSEStream(app, path, headers=headers) as stream:
            await stream.frame()
            event_id, _ = await stream.frame()

    assert event_id == last


async def test_frames_published_after_the_stream_opens_arrive_live(
    forge_app, auth, stub_responses
):
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        path = f"/api/sessions/{session_id}/events"
        async with SSEStream(app, path, headers=auth) as stream:
            assert await stream.frame() == (None, {"type": "ready"})

            await client.post(
                f"/api/sessions/{session_id}/goal", json={"text": "hi"}, headers=auth
            )
            _, first = await stream.frame()

    assert first["type"] == "status"


async def test_a_dropped_connection_unsubscribes(forge_app, auth):
    """Missing the `finally` leaks a subscriber per dropped connection.

    Nothing breaks immediately — `publish` just keeps filling a queue nobody
    drains, and the symptom shows up much later as unexplained `overflow` frames
    on connections that were behaving perfectly.
    """
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session = app.state.manager.get(created.json()["session_id"])

        path = f"/api/sessions/{session.id}/events"
        async with SSEStream(app, path, headers=auth) as stream:
            await stream.frame()
            assert session.subscriber_count == 1

        assert session.subscriber_count == 0


async def test_two_streams_on_one_session_both_get_everything(
    forge_app, auth, stub_responses
):
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]
        path = f"/api/sessions/{session_id}/events"

        async with SSEStream(app, path, headers=auth) as left:
            async with SSEStream(app, path, headers=auth) as right:
                await left.frame()
                await right.frame()
                await client.post(
                    f"/api/sessions/{session_id}/goal",
                    json={"text": "hi"},
                    headers=auth,
                )
                assert await left.frame() == await right.frame()


async def test_a_quiet_session_gets_a_heartbeat(forge_app, auth, monkeypatch):
    """An intermediary cannot tell a deliberately silent stream from a dead one."""
    monkeypatch.setattr(sse, "HEARTBEAT_SEC", 0.01)
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        path = f"/api/sessions/{session_id}/events"
        async with SSEStream(app, path, headers=auth) as stream:
            await stream.event()  # ready
            assert await stream.event(timeout=2.0) == ": ping"
