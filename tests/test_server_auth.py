"""Phase 8, stage 4 — the token, on every route without exception.

SSE and WS take the token differently from the other seven routes, which makes
them the two most likely to ship unguarded: the natural way to add `?token=`
support is to add it to the one route that needed it, and then the WebSocket goes
out unauthenticated. So this file enumerates *every* route rather than a
representative sample, and `test_every_route_requires_a_token` fails if a new one
is added without one.
"""

from __future__ import annotations

import pytest
from asgi_client import SSEStream, WSSession, running
from conftest import SERVER_TOKEN

from server.app import create_app

# (method, path). Every `/api` route the app exposes; `/` and `/dashboard` are
# deliberately absent — they serve the empty shell, which must load before the
# user can hand it a token.
REST_ROUTES = [
    ("POST", "/api/sessions"),
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions/nope/goal"),
    ("POST", "/api/sessions/nope/decisions"),
    ("POST", "/api/sessions/nope/interrupt"),
    ("DELETE", "/api/sessions/nope"),
    ("GET", "/api/sessions/nope/events"),
    ("GET", "/api/stats"),
]


async def test_the_app_refuses_to_start_without_a_token(monkeypatch, tmp_path):
    """No degraded mode. This process runs shell commands as whoever launched it."""
    monkeypatch.delenv("FORGE_SERVER_TOKEN", raising=False)
    app = create_app(root=tmp_path)
    with pytest.raises(RuntimeError, match="FORGE_SERVER_TOKEN"):
        async with running(app):
            pass  # pragma: no cover - the lifespan raises before the body runs


@pytest.mark.parametrize(("method", "path"), REST_ROUTES)
async def test_every_route_requires_a_token(forge_app, method, path):
    async with running(forge_app()) as client:
        response = await client.request(method, path, json={})
    assert response.status_code == 401


@pytest.mark.parametrize(("method", "path"), REST_ROUTES)
async def test_a_wrong_token_is_a_401_not_a_404(forge_app, method, path):
    """Auth is checked before the session lookup.

    Every path here names a session that does not exist. If the lookup ran first,
    an unauthenticated caller could probe which session ids are live by reading
    404 against 401.
    """
    async with running(forge_app()) as client:
        response = await client.request(
            method, path, json={}, headers={"Authorization": "Bearer wrong"}
        )
    assert response.status_code == 401


async def test_the_right_token_gets_through(forge_app, auth):
    async with running(forge_app()) as client:
        response = await client.get("/api/sessions", headers=auth)
    assert response.status_code == 200
    assert response.json() == []


async def test_a_query_token_is_accepted_too(forge_app):
    """Accepted on every route, not only on the one that needs it.

    `EventSource` cannot set headers, which is the whole reason this form exists.
    A check that varies per route is a check that gets forgotten on the route
    added next.
    """
    async with running(forge_app()) as client:
        response = await client.get(f"/api/sessions?token={SERVER_TOKEN}")
    assert response.status_code == 200


async def test_a_bearer_prefix_is_required(forge_app):
    """A bare token in the header is not a bearer token."""
    async with running(forge_app()) as client:
        response = await client.get(
            "/api/sessions", headers={"Authorization": SERVER_TOKEN}
        )
    assert response.status_code == 401


async def test_sse_takes_the_token_in_the_query_string(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        path = f"/api/sessions/{session_id}/events?token={SERVER_TOKEN}"
        async with SSEStream(app, path) as stream:
            assert stream.status == 200
            assert stream.headers["content-type"].startswith("text/event-stream")


async def test_sse_without_a_token_never_opens_a_stream(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        async with SSEStream(app, f"/api/sessions/{session_id}/events") as stream:
            assert stream.status == 401


async def test_ws_is_rejected_before_accept(forge_app, auth):
    """The handshake is refused, not accepted-then-closed.

    There is no HTTP response left to carry a 401 once the handshake is under way,
    so the close has to come *before* `accept()`. A route that accepts first has
    already told the browser it was authorised, and a client that trusts `onopen`
    would believe it.
    """
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        async with WSSession(app, f"/ws/sessions/{session_id}?token=wrong") as socket:
            assert socket.accepted is False
            assert socket.close_code == 1008


async def test_ws_with_the_right_token_connects(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        path = f"/ws/sessions/{session_id}?token={SERVER_TOKEN}"
        async with WSSession(app, path) as socket:
            assert socket.accepted is True
            assert await socket.frame() == {"type": "ready"}


async def test_ws_for_an_unknown_session_closes(forge_app):
    app = forge_app()
    async with running(app):
        path = f"/ws/sessions/missing?token={SERVER_TOKEN}"
        async with WSSession(app, path) as socket:
            assert socket.accepted is False
