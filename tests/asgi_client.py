"""Test doubles for the two halves of ASGI the ordinary clients cannot reach.

`httpx.ASGITransport` is fine for request/response, and `running()` below wraps
it. It is no use for an event stream, and `httpx` speaks no WebSocket at all.
`starlette.testclient` can do both, but only by running the app in its own event
loop on another thread — and these tests routinely reach into a session from the
test's loop (`session.wait()`, `approver.pending`), which across loops is
undefined behaviour.

So the two long-lived transports are driven by calling the ASGI app directly, in
this loop, with queues standing in for the socket. It is about eighty lines and it
removes every ambiguity about who is running what where.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import httpx


@contextlib.asynccontextmanager
async def running(app):
    """Run the app's lifespan and hand back a client, all in this event loop."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://forge.test"
        ) as client:
            yield client


def _scope(kind: str, path: str, headers: dict[str, str] | None) -> dict:
    raw_path, _, query = path.partition("?")
    return {
        "type": kind,
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "http" if kind == "http" else "ws",
        "path": raw_path,
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "client": ("test", 1),
        "server": ("forge.test", 80),
    }


class SSEStream:
    """One live `text/event-stream` response, read event by event."""

    def __init__(self, app, path: str, *, headers: dict[str, str] | None = None):
        self._app = app
        self._scope = _scope("http", path, headers) | {"method": "GET"}
        self._inbound: asyncio.Queue[dict] = asyncio.Queue()
        self._outbound: asyncio.Queue[dict] = asyncio.Queue()
        self._buffer = ""
        self._task: asyncio.Task | None = None
        self.status: int | None = None
        self.headers: dict[str, str] = {}

    async def __aenter__(self) -> SSEStream:
        self._task = asyncio.create_task(
            self._app(self._scope, self._inbound.get, self._outbound.put)
        )
        start = await asyncio.wait_for(self._outbound.get(), 5.0)
        assert start["type"] == "http.response.start", start
        self.status = start["status"]
        self.headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
        return self

    async def __aexit__(self, *exc) -> None:
        # What a closed browser looks like from inside the app. Sent explicitly so
        # the generator's `finally` runs and `unsubscribe` is observable.
        await self._inbound.put({"type": "http.disconnect"})
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def event(self, timeout: float = 2.0) -> str:
        """The next complete event: everything up to a blank line."""
        while "\n\n" not in self._buffer:
            message = await asyncio.wait_for(self._outbound.get(), timeout)
            if message["type"] != "http.response.body":
                raise AssertionError(f"unexpected message: {message}")
            self._buffer += message.get("body", b"").decode()
            if not message.get("more_body", False) and "\n\n" not in self._buffer:
                raise AssertionError("stream ended without another event")
        head, _, rest = self._buffer.partition("\n\n")
        self._buffer = rest
        return head

    async def frame(self, timeout: float = 2.0) -> tuple[int | None, dict]:
        """The next `data:` frame and its `id:`, skipping heartbeat comments."""
        while True:
            lines = (await self.event(timeout)).splitlines()
            if all(line.startswith(":") for line in lines):
                continue
            event_id = next(
                (int(x[3:].strip()) for x in lines if x.startswith("id:")), None
            )
            data = next(x[5:].strip() for x in lines if x.startswith("data:"))
            return event_id, json.loads(data)

    async def frames(self, count: int, timeout: float = 2.0) -> list[tuple]:
        return [await self.frame(timeout) for _ in range(count)]


class WSSession:
    """One WebSocket, driven directly. `accepted` is False when it was rejected."""

    def __init__(self, app, path: str, *, headers: dict[str, str] | None = None):
        self._app = app
        self._scope = _scope("websocket", path, headers) | {"subprotocols": []}
        self._inbound: asyncio.Queue[dict] = asyncio.Queue()
        self._outbound: asyncio.Queue[dict] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.accepted = False
        self.close_code: int | None = None

    async def __aenter__(self) -> WSSession:
        self._task = asyncio.create_task(
            self._app(self._scope, self._inbound.get, self._outbound.put)
        )
        await self._inbound.put({"type": "websocket.connect"})
        first = await asyncio.wait_for(self._outbound.get(), 5.0)
        if first["type"] == "websocket.close":
            # Rejected before accept — which is the only correct way to refuse a
            # handshake, and so the thing the auth tests assert on.
            self.close_code = first.get("code")
        else:
            assert first["type"] == "websocket.accept", first
            self.accepted = True
        return self

    async def __aexit__(self, *exc) -> None:
        await self._inbound.put({"type": "websocket.disconnect", "code": 1000})
        if self._task is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._task, 2.0)
            self._task.cancel()

    async def send_json(self, payload: Any) -> None:
        await self._inbound.put(
            {"type": "websocket.receive", "text": json.dumps(payload)}
        )

    async def frame(self, timeout: float = 2.0) -> dict:
        message = await asyncio.wait_for(self._outbound.get(), timeout)
        if message["type"] == "websocket.close":
            raise AssertionError(f"closed with {message.get('code')}")
        return json.loads(message["text"])

    async def frames(self, count: int, timeout: float = 2.0) -> list[dict]:
        return [await self.frame(timeout) for _ in range(count)]
