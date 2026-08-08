from __future__ import annotations

import asyncio
import logging
from collections import deque

from events import Event
from main import Composition
from server import wire

logger = logging.getLogger(__name__)


_SENTINEL = object()

_OVERFLOW_FRAME = {"type": "overflow"}


class AgentSession:
    """One conversation: its transcript, its `seq` counter, its subscribers."""

    def __init__(self, comp: Composition, *, queue_maxsize: int = 256) -> None:
        self._comp = comp
        self._queue_maxsize = queue_maxsize
        self._seq = 0
        self.transcript: list[dict] = []
        self._subs: set[Subscriber] = set()

    def publish(self, event: Event) -> None:
        """Record one event and hand it to every live subscriber.

        Synchronous on purpose: containing no `await` makes it atomic against the
        event loop, so a publish can never interleave with a `subscribe`.
        """
        frame = wire.frame(event=event, seq=self._seq)
        self._seq += 1
        self.transcript.append(frame)

        # Iterate a *copy*: the set is mutated below when a subscriber overflows,
        # and mutating during iteration raises RuntimeError — under exactly the
        # condition that is hardest to reach by hand.
        for sub in list(self._subs):
            try:
                # `put_nowait`, never `await put`: awaiting a full queue would
                # suspend the drive task on the slowest client, which is the
                # failure this whole design exists to prevent.
                sub._queue.put_nowait(frame)
            except asyncio.QueueFull:
                sub._overflowed = True
                self._subs.discard(sub)
                logger.warning(
                    "subscriber overflowed at seq=%d (queue_maxsize=%d); dropping",
                    frame["seq"],
                    self._queue_maxsize,
                )

    def subscribe(self, after_seq: int = -1) -> Subscriber:

        replay = [f for f in self.transcript if f["seq"] > after_seq]
        sub = Subscriber(
            replay=deque(replay),
            queue=asyncio.Queue(self._queue_maxsize),
        )
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        """End a subscriber's iteration cleanly, with no overflow frame.

        Idempotent: a transport's `finally` can run after `publish` already
        dropped this subscriber for overflow.
        """
        self._subs.discard(sub)
        sub._closed = True
        try:
            sub._queue.put_nowait(_SENTINEL)
        except asyncio.QueueFull:
            # No room for the wake-up sentinel. Harmless: a full queue means the
            # subscriber has frames buffered, so it cannot be parked in
            # `await get()`, and `_closed` ends it once those drain.
            pass


class Subscriber:
    """One connected client's view of a session.

    Async-iterable. Built by `AgentSession.subscribe`, never directly. Frames are
    shared with the transcript and with sibling subscribers, so a consumer must
    treat them as read-only — mutating one corrupts it for everybody.
    """

    def __init__(
        self, replay: deque[dict], queue: asyncio.Queue[dict | object]
    ) -> None:
        self._replay = replay
        self._queue = queue
        self._overflowed = False
        self._overflow_sent = False
        self._closed = False

    def __aiter__(self) -> Subscriber:
        return self

    async def __anext__(self) -> dict:
        """Replay, then buffered live frames, then a close reason, then wait.

        The order is the contract. In particular, buffered frames are drained
        *before* either close reason is honoured: a subscriber that overflowed
        still owns the frames it managed to buffer, and discarding them would
        leave its next reconnect asking to resume from the wrong `seq`.
        """
        if self._replay:
            return self._replay.popleft()

        try:
            item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        else:
            return self._unwrap(item)

        if self._overflowed:
            # Exactly once, then end — without the flag the iterator would
            # re-yield the notice forever and never terminate.
            if not self._overflow_sent:
                self._overflow_sent = True
                return _OVERFLOW_FRAME
            raise StopAsyncIteration

        # Checked after overflow: a transport's `finally` unsubscribes an
        # already-overflowed subscriber, and overflow is the informative reason.
        if self._closed:
            raise StopAsyncIteration

        # Open and momentarily empty. The only await in this method, and what
        # holds a connection alive between turns — a TerminalEvent ends the run,
        # not the stream.
        return self._unwrap(await self._queue.get())

    @staticmethod
    def _unwrap(item: dict | object) -> dict:
        """Turn the close sentinel into the end of iteration."""
        if item is _SENTINEL:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]
