"""WebSocket: the same frames out, plus a way in.

Duplex, so it replaces four REST routes at once — but it carries *byte-identical*
frames, because both transports read the same `Subscriber` and neither encodes
anything itself. `tests/test_server_ws.py` asserts that equality directly; if it
ever fails, frame-building has leaked out of `server/wire.py`.

Outbound frames are discriminated on `type`, inbound on `kind`, so a frame's
direction is legible from its shape alone (`web/src/api/transports/ws.ts`).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect

from server.approver import ServerApprover, UnknownDecision
from server.session import AgentSession, SessionBusy, Subscriber

logger = logging.getLogger(__name__)

_READY_FRAME = {"type": "ready"}


async def serve(
    session: AgentSession, websocket: WebSocket, *, after_seq: int = -1
) -> None:
    """Run one accepted socket until either side ends it.

    Two tasks, never one loop that alternates receive and send — and it is the
    same lesson as the parked approver. A single loop can only be waiting for one
    of the two: park on `receive()` and the confirm frame never goes out; park on
    the subscriber and the human's answer never comes in. Either way the confirm
    round-trip deadlocks on WS while working perfectly over SSE, and a naive echo
    test would not notice.

    `TaskGroup` also gives the right teardown for nothing: whichever side ends
    first cancels the other, so a closed browser cannot leave a writer parked on a
    queue forever.
    """
    sub = session.subscribe(after_seq=after_seq)
    # Two tasks share one socket. Each `send_json` is a single await that can
    # suspend mid-message, so without this an error frame from the reader could
    # interleave with a transcript frame from the writer and produce two
    # corrupt ones.
    lock = asyncio.Lock()
    try:
        await _send(websocket, lock, _READY_FRAME)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_writer(websocket, lock, sub))
            try:
                # The reader runs inline rather than as a third task, so its
                # ending is directly observable here.
                await _reader(websocket, lock, session)
            finally:
                # A returning reader *is* a disconnect: Starlette's `iter_json`
                # swallows `WebSocketDisconnect`, so a gone client looks like a
                # normal return. Nothing more will ever be read, and the writer
                # is parked on the subscriber's queue — release it through the
                # session's own close path rather than cancelling it, so the
                # TaskGroup sees a clean exit.
                session.unsubscribe(sub)
    except* WebSocketDisconnect:
        # Belt and braces: `receive`-shaped calls outside `iter_json` still raise.
        pass
    finally:
        # Idempotent, and reached on the paths the inner `finally` is not — an
        # exception out of `_send` before the group is even entered.
        session.unsubscribe(sub)


async def _send(websocket: WebSocket, lock: asyncio.Lock, frame: dict) -> None:
    async with lock:
        await websocket.send_json(frame)


async def _writer(
    websocket: WebSocket, lock: asyncio.Lock, sub: Subscriber
) -> None:
    async for frame in sub:
        try:
            await _send(websocket, lock, frame)
        except RuntimeError:
            # Starlette's wording for "already closed". A close the reader task
            # noticed first is not an error worth propagating.
            return


async def _reader(
    websocket: WebSocket, lock: asyncio.Lock, session: AgentSession
) -> None:
    """Dispatch inbound frames onto the session.

    Failures here become `error` *frames*, not closes. A stale `request_id` or a
    double-clicked send is recoverable — the REST equivalent is a 409 — and
    dropping the connection over one would take the transcript stream down with
    it.
    """
    async for message in websocket.iter_json():
        kind = message.get("kind")
        try:
            if kind == "goal":
                session.start(message["text"])
            elif kind == "decision":
                _resolve(session, message)
            elif kind == "interrupt":
                await session.interrupt()
            else:
                await _error(websocket, lock, f"unknown frame kind {kind!r}")
        except KeyError as exc:
            await _error(websocket, lock, f"{kind} frame is missing {exc}")
        except (SessionBusy, UnknownDecision) as exc:
            await _error(websocket, lock, str(exc))


def _resolve(session: AgentSession, message: dict) -> None:
    approver = session.approver
    if not isinstance(approver, ServerApprover):
        raise UnknownDecision("this session does not take decisions over the wire")
    approver.resolve(
        message["request_id"],
        approved=bool(message.get("approved")),
        reason=message.get("reason"),
        arguments=message.get("arguments"),
        remember=bool(message.get("remember", False)),
    )


async def _error(websocket: WebSocket, lock: asyncio.Lock, detail: str) -> None:
    await _send(websocket, lock, {"type": "error", "detail": detail})
