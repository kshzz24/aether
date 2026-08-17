"""Server-Sent Events: one long-lived GET per subscriber.

The whole protocol is three rules — `data: <json>\\n\\n` per frame, an optional
`id: <n>` line before it, and `: <text>\\n\\n` for a comment. That is why there is
no `sse-starlette` dependency here: the framing is four lines, and the parts that
are actually easy to get wrong (which frames carry an `id`, where `unsubscribe`
goes) are not the parts a library would do for you.

Reconnect is the browser's job, not ours. `EventSource` stores the last `id:` it
saw and replays it as the `Last-Event-ID` header on its own automatic retry, with
no application code involved — so the only thing this module owes it is an honest
`id`, and only on frames that have a transcript position.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from server.session import AgentSession, Subscriber

logger = logging.getLogger(__name__)

# Long enough not to chatter, short enough to beat the usual 30-60s idle timeout
# on a proxy that has no idea this connection is deliberately silent.
HEARTBEAT_SEC = 15.0

_READY_FRAME = {"type": "ready"}

_HEADERS = {
    # An intermediary that buffers an event stream turns it into a single reply
    # delivered at the end, which looks exactly like the server being broken.
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def resolve_offset(after_seq: int, last_event_id: str | None) -> int:
    """Where to resume from. The header wins over the query parameter.

    Both exist and they *will* disagree. `EventSource` retries the same URL it
    was constructed with, so the `?after_seq=` a fresh mount put there is still
    in the query string on every later reconnect, by which time it is stale;
    `Last-Event-ID` is the browser's own record of what it actually received. If
    the query won, every reconnect would replay the transcript from the original
    mount point and the client would see each frame twice.
    """
    if last_event_id:
        try:
            return int(last_event_id)
        except ValueError:
            # A client that sends garbage gets a full replay rather than an
            # error: resuming from an unknown offset is the one thing we must
            # not guess at.
            logger.warning("ignoring malformed Last-Event-ID %r", last_event_id)
    return after_seq


def encode(frame: dict) -> str:
    """One frame as an SSE event.

    The `id:` line is emitted only for frames that have a `seq`. `ready` and
    `overflow` do not: they are not transcript positions, and giving them an `id`
    would make the browser store an offset that indexes nothing and then ask to
    resume from it.
    """
    prefix = f"id: {frame['seq']}\n" if "seq" in frame else ""
    return f"{prefix}data: {json.dumps(frame)}\n\n"


async def frames(session: AgentSession, offset: int) -> AsyncIterator[str]:
    """Subscribe, replay from `offset`, then stream until the client goes away.

    `subscribe` is called here rather than in the route, and that is safe only
    because it filters the transcript synchronously: any frame published between
    the request arriving and this generator's first step is already in the replay
    buffer, so nothing can fall through the gap.
    """
    sub = session.subscribe(after_seq=offset)
    try:
        yield encode(_READY_FRAME)
        async for chunk in _pump(sub):
            yield chunk
    finally:
        # Not optional. A closed browser surfaces in here as a `CancelledError`
        # or a broken pipe, and without this the subscriber stays in the
        # session's fan-out set forever — `publish` keeps filling a queue nobody
        # drains, until it overflows and starts dropping other clients' frames.
        session.unsubscribe(sub)


async def _pump(sub: Subscriber) -> AsyncIterator[str]:
    """Yield frames, or a heartbeat comment when the session goes quiet.

    Cancelling `__anext__` on the timeout loses nothing, though it is exactly the
    shape that usually does: the only await inside it is `asyncio.Queue.get`,
    which removes its own waiter on cancellation and re-wakes the next getter.
    Nothing has been dequeued at the point the timeout fires.
    """
    while True:
        try:
            frame = await asyncio.wait_for(sub.__anext__(), HEARTBEAT_SEC)
        except StopAsyncIteration:
            return
        except TimeoutError:
            yield ": ping\n\n"
            continue
        yield encode(frame)


def stream(
    session: AgentSession, *, after_seq: int = -1, last_event_id: str | None = None
) -> StreamingResponse:
    return StreamingResponse(
        frames(session, resolve_offset(after_seq, last_event_id)),
        media_type="text/event-stream",
        headers=_HEADERS,
    )
