from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from collections import deque

import persistence
from approval import Approver
from client import Message, TextBlock
from events import CostEvent, Event, StatusEvent, TerminalEvent, TerminalReason
from main import Composition, checkpoint
from server import wire
from tracing import traced

logger = logging.getLogger(__name__)


_SENTINEL = object()

_OVERFLOW_FRAME = {"type": "overflow"}


class SessionBusy(RuntimeError):
    """One run in flight per session. Becomes HTTP 409 at stage 4."""


class AgentSession:
    """One conversation: its transcript, its `seq` counter, its subscribers."""

    def __init__(
        self,
        comp: Composition,
        *,
        approver: Approver | None = None,
        queue_maxsize: int = 256,
    ) -> None:
        self._comp = comp
        # Held explicitly rather than reached for through `agent._approver`, which
        # would make a private attribute part of the HTTP layer's vocabulary.
        self._approver = approver
        self._queue_maxsize = queue_maxsize
        self._seq = 0
        self.transcript: list[dict] = []
        self._subs: set[Subscriber] = set()
        self._running = False
        self._task: asyncio.Task | None = None
        self._history: list[Message] = list(comp.history or [])
        self.total_cost = 0.0
        # `monotonic`, not `time()`: the idle reaper measures an elapsed interval,
        # and a wall-clock jump (NTP, DST) must not make a session look ancient.
        self.last_activity = time.monotonic()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def subscriber_count(self) -> int:
        """How many transports are attached. Zero is what the reaper looks for."""
        return len(self._subs)

    @property
    def approver(self) -> Approver | None:
        """Where stage 4's decisions route sends a human's answer."""
        return self._approver

    @property
    def id(self) -> str:
        return self._comp.session.id

    def start(self, goal: str) -> None:
        if self._running:
            raise SessionBusy("a run is already in flight for this session")
        # Name the session from its first goal. A web client creates the session
        # before it has anything to say, so `build_composition` had no goal to
        # name it with and left it "" — which would show as a blank column in
        # `GET /api/sessions` forever. Done here, not in the HTTP layer, because
        # "a session is named by what it was first asked to do" is session state.
        if not self._comp.session.goal:
            self._comp.session.goal = goal
        self._running = True
        self.last_activity = time.monotonic()
        self._task = asyncio.create_task(self._drive(goal))

    async def _drive(self, goal: str) -> None:

        stream = traced(
            self._comp.agent.run(goal, history=self._next_history(goal)),
            trace_id=self._comp.session.id,
            prompt_hash=hashlib.sha256(self._comp.agent.system.encode()).hexdigest()[
                :12
            ],
        )
        try:
            async for event in stream:
                self.publish(event=event)
        except asyncio.CancelledError:
            self.publish(StatusEvent(type="status", message="interrupted"))
            raise
        except Exception as exc:
            logger.exception("drive task failed")
            self.publish(TerminalEvent(reason=TerminalReason.ERROR, detail=str(exc)))
        finally:
            self._history = list(self._comp.agent.messages)
            self._running = False
            self._checkpoint()

    def _next_history(self, goal: str) -> list[Message] | None:
        if not self._history:
            return None
        return [*self._history, Message(role="user", blocks=[TextBlock(text=goal)])]

    def publish(self, event: Event) -> None:
        """Record one event off the agent's stream and hand it to every subscriber."""
        if isinstance(event, CostEvent):
            self.total_cost += event.cost_usd
        self._publish_frame(wire.encode(event))

    def publish_frame(self, body: dict) -> None:
        """Publish a frame the *server* minted, not one the agent yielded.

        Only `ServerApprover`'s confirm uses this. It goes through the same
        numbering and fan-out as `publish`, so it lands in the transcript with a
        `seq` and replays on reconnect — which is the only mechanism that can
        re-show a pending question to a browser that reconnected mid-confirm.
        """
        self._publish_frame(body)

    def _publish_frame(self, body: dict) -> None:
        """The shared tail of both publish paths: number, record, fan out.

        Synchronous on purpose: containing no `await` makes it atomic against the
        event loop, so a publish can never interleave with a `subscribe`.
        """
        frame = {**body, "seq": self._seq}

        self.last_activity = time.monotonic()
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

    def _checkpoint(self) -> None:
        comp = self._comp
        checkpoint(comp)
        comp.session.total_cost = self.total_cost
        persistence.save(comp.session, comp.sessions_dir)

    async def wait(self) -> None:
        task = self._task
        if task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def interrupt(self) -> None:
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._running = False

    async def aclose(self) -> None:
        """Stop the run, persist, release every subscriber, close MCP. Idempotent."""
        await self.interrupt()
        self._checkpoint()
        for sub in list(self._subs):
            self.unsubscribe(sub)
        # Every session builds its own `Composition` and so spawns its own stdio
        # subprocesses. `main.py:313-314` closes them in the CLI's `finally`; here
        # there is no `finally` around a whole process lifetime, so the close
        # belongs at the one point every eviction path passes through — `DELETE`,
        # the idle reaper, and shutdown all land in `aclose`. Without it each
        # evicted session leaks a process tree that outlives the server.
        # `MCPManager.aclose` swallows teardown noise and resets its exit stack,
        # so calling it twice is safe.
        if self._comp.mcp is not None:
            await self._comp.mcp.aclose()


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
