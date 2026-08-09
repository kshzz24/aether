"""The parked Future: the one place data flows *back into* the agent.

`agent.py:208` does `decision = await self._approver.decide(request)` inside the
async generator, so the generator suspends there and the whole run waits on a
human. Over a network that becomes three moves: publish a confirm frame, park an
`asyncio.Future`, resolve it from a second HTTP request.

**Why this does not deadlock**, since it is the property the design rests on:
while the agent is suspended in `decide`, the drive task is suspended in
`__anext__` and nothing further is published — but the *transport* is a different
task, and frames already published are sitting in each subscriber's queue. So the
client receives everything up to and including the confirm, then goes quiet.
Stage 2a's per-subscriber queue is what buys this; a transport sharing the drive
task's coroutine would never deliver the confirm and the run would hang forever.

**This module does not validate.** `agent.py:218-241` already re-validates edited
arguments against the tool's schema and re-runs both danger checks. Edited
arguments therefore pass through untouched — a second copy of a security check
here would be a second copy that diverges.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from uuid import uuid4

from approval import ApprovalRequest, Decision
from server import wire

logger = logging.getLogger(__name__)

# Five minutes. Invariant 6 ("every run is bounded") applied to a human in the
# loop: without a bound, a browser closed mid-confirm suspends the generator for
# the lifetime of the process, holding a session slot and an MCP process tree
# that the idle reaper cannot claim, because the run never ends.
APPROVAL_TIMEOUT = 300.0

# (decision, remember-this-tool-for-the-session). Same shape the TUI resolves
# with (`tui/approver.py:39`): "remember this" is a property of a *surface*, not
# of the agent's decision, so `approval.py` deliberately keeps it out of
# `Decision` and each approver holds its own set.
ConfirmOutcome = tuple[Decision, bool]


class UnknownDecision(RuntimeError):
    """No such pending request, or it was already answered. HTTP 409 at stage 4."""


class ServerApprover:
    """Implements the `Approver` Protocol (`approval.py:47`) across a network."""

    def __init__(self, *, timeout: float = APPROVAL_TIMEOUT) -> None:
        self._timeout = timeout
        self._publish: Callable[[dict], None] | None = None
        self._pending: dict[str, tuple[asyncio.Future[ConfirmOutcome], ApprovalRequest]]
        self._pending = {}
        # Tools the human said "always" to, for this session only. Never
        # persisted: an approval that outlives the session you granted it in is a
        # policy change disguised as a keystroke.
        self._always: set[str] = set()

    def bind(self, publish: Callable[[dict], None]) -> None:
        """Late-bind the session's `publish_frame`, closing the construction cycle.

        `build_composition` needs the approver, the approver needs the session,
        and the session needs the composition — so `SessionManager.create` is the
        only place that can see all three, and this is how it joins them.

        Rebinding raises. A silently double-bound approver publishes confirms into
        the wrong session, which is the worst bug available in a multi-session
        server and presents as a UI glitch rather than an error.
        """
        if self._publish is not None:
            raise RuntimeError("this ServerApprover is already bound to a session")
        self._publish = publish

    @property
    def pending(self) -> frozenset[str]:
        """The ids a decision would currently be accepted for."""
        return frozenset(self._pending)

    async def decide(self, request: ApprovalRequest) -> Decision:
        if self._publish is None:
            # Loud, rather than a confirm nobody will ever see and a run that
            # hangs until the timeout.
            raise RuntimeError("ServerApprover.decide() called before bind()")

        # A remembered tool is still re-checked for danger (`tui/approver.py:189`):
        # "always" granted on a harmless `ls` must not cover a later call that
        # tripped a danger check — the one case where being asked every time is
        # the entire point.
        if request.tool_name in self._always and not request.danger_reasons:
            return Decision(approved=True, reason="always allowed this session")

        request_id = uuid4().hex
        # `loop.create_future()` rather than the bare `asyncio.Future()`
        # constructor, which reaches for the running loop by a deprecated path.
        future: asyncio.Future[ConfirmOutcome]
        future = asyncio.get_running_loop().create_future()

        # Park *before* publishing. Publish first and a fast client could resolve
        # an id that is not registered yet, which would 409 a valid answer.
        self._pending[request_id] = (future, request)
        try:
            self._publish(wire.confirm_frame(request, request_id))
            decision, remember = await asyncio.wait_for(future, self._timeout)
        except TimeoutError:
            logger.info("approval %s timed out after %.1fs", request_id, self._timeout)
            return Decision(approved=False, reason="approval timed out")
        finally:
            # In a `finally`, so it runs on timeout and on cancellation too — an
            # interrupt throws `CancelledError` into the generator suspended right
            # here. Every abandoned confirm that skipped this would leave a dead
            # Future keyed by a uuid nobody will ever send, in a process designed
            # to run for weeks.
            self._pending.pop(request_id, None)

        if remember:
            self._always.add(request.tool_name)
        return decision

    def resolve(
        self,
        request_id: str,
        *,
        approved: bool,
        reason: str | None = None,
        arguments: dict | None = None,
        remember: bool = False,
    ) -> None:
        """Answer a parked confirm. Synchronous — setting a result never waits."""
        entry = self._pending.get(request_id)
        if entry is None:
            # Unknown id, or already answered and popped: the browser reconnected
            # and answered a question that expired. A 409, not a crash and not a
            # silent success.
            raise UnknownDecision(f"no approval pending for {request_id!r}")

        future, request = entry
        if future.done():
            # A double-clicked button or a retried POST. `set_result` on a done
            # Future raises `InvalidStateError`, which as an unhandled 500 is
            # indistinguishable from a real bug.
            raise UnknownDecision(f"approval {request_id!r} was already answered")

        # The same rule `offers_always` encodes, enforced again on the way in: the
        # client was never offered "always" for a flagged call, so a request that
        # asks for it anyway is not one to honour.
        if remember and request.danger_reasons:
            logger.warning(
                "refusing 'always' for danger-flagged %s", request.tool_name
            )
            remember = False

        decision = Decision(approved=approved, reason=reason, arguments=arguments)
        future.set_result((decision, remember))
