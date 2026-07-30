import asyncio
import random
import time
from typing import Any


class BreakerOpen(Exception):
    """Raised when a call is refused because the breaker is open (cooling down)."""


class CircuitBreaker:
    """Per-provider failure gate: closed -> open -> half_open -> closed.

    Closed: calls pass; consecutive failures are counted.
    Open: calls are refused until `cooldown_sec` elapses.
    Half-open: one probe call is allowed; success closes, failure re-opens.
    """

    def __init__(
        self, *, fail_threshold: int, cooldown_sec: float, clock=time.monotonic
    ):
        self._fail_threshold = fail_threshold
        self._cooldown_sec = cooldown_sec
        self._clock = clock          # injectable so tests advance time without sleeping
        self._state = "closed"
        self._opened_at = 0.0        # when we last transitioned to open
        self._failures = 0           # consecutive failures while closed

    def allow(self) -> bool:
        """True if a call may proceed. Drives the open -> half_open transition."""
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._clock() - self._opened_at >= self._cooldown_sec:
                self._state = "half_open"   # cooldown elapsed: let one probe through
                return True
            return False
        # half_open: a probe is already permitted
        return True

    def record_success(self) -> None:
        """A call succeeded: reset to fully closed."""
        self._state = "closed"
        self._failures = 0

    def record_failure(self) -> None:
        """A call failed: trip open at the threshold, or re-open a failed probe."""
        if self._state == "half_open":
            self._state = "open"
            self._opened_at = self._clock()
            return
        self._failures += 1
        if self._failures >= self._fail_threshold:
            self._state = "open"
            self._opened_at = self._clock()

    @property
    def state(self) -> str:
        return self._state


async def call_with_resilience(
    breaker: CircuitBreaker,
    fn,
    *,
    max_attempts: int,
    base_delay_sec: float,
    transient: tuple[type[Exception], ...],
) -> Any:
    """Run `await fn()` behind the breaker, retrying transient errors with backoff.

    Refuses immediately with BreakerOpen if the breaker is open. Retries only
    `transient` exceptions (exp backoff + jitter) up to `max_attempts`; any other
    exception propagates on the first try and never trips the breaker.
    """
    if not breaker.allow():
        raise BreakerOpen()

    for attempt in range(max_attempts):
        try:
            result = await fn()
            breaker.record_success()
            return result
        except transient:
            breaker.record_failure()
            if attempt == max_attempts - 1:
                raise
            delay = base_delay_sec * (2 ** attempt) * (0.5 + random.random())
            await asyncio.sleep(delay)
