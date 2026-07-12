"""Task 6 — circuit breaker + retry/backoff.

Pure in-memory logic. The clock is injected so state transitions are tested by
advancing a fake time, never by sleeping. `call_with_resilience` uses
base_delay_sec=0.0 so its backoff sleeps are no-ops.
"""

from __future__ import annotations

import pytest

from gateway.breaker import BreakerOpen, CircuitBreaker, call_with_resilience


# --- state machine -----------------------------------------------------------

def test_starts_closed_and_allows():
    b = CircuitBreaker(fail_threshold=3, cooldown_sec=10, clock=lambda: 0)
    assert b.state == "closed"
    assert b.allow() is True


def test_opens_after_threshold():
    b = CircuitBreaker(fail_threshold=3, cooldown_sec=10, clock=lambda: 0)
    for _ in range(3):
        b.record_failure()
    assert b.state == "open"
    assert b.allow() is False


def test_below_threshold_stays_closed():
    b = CircuitBreaker(fail_threshold=3, cooldown_sec=10, clock=lambda: 0)
    b.record_failure()
    b.record_failure()
    assert b.state == "closed"
    assert b.allow() is True


def test_success_resets_failure_count():
    b = CircuitBreaker(fail_threshold=2, cooldown_sec=10, clock=lambda: 0)
    b.record_failure()
    b.record_success()      # resets the streak
    b.record_failure()      # only 1 consecutive failure now, not 2
    assert b.state == "closed"


def test_half_open_then_close_on_success():
    t = {"now": 0.0}
    b = CircuitBreaker(fail_threshold=1, cooldown_sec=5, clock=lambda: t["now"])
    b.record_failure()
    assert b.allow() is False        # open, still cooling
    t["now"] = 6.0                   # past cooldown
    assert b.allow() is True         # half-open probe allowed
    assert b.state == "half_open"
    b.record_success()
    assert b.state == "closed"


def test_half_open_failure_reopens():
    t = {"now": 0.0}
    b = CircuitBreaker(fail_threshold=1, cooldown_sec=5, clock=lambda: t["now"])
    b.record_failure()               # open at t=0
    t["now"] = 6.0
    assert b.allow() is True         # half-open probe
    assert b.state == "half_open"
    b.record_failure()               # probe failed -> reopen, cooldown restarts at t=6
    assert b.state == "open"
    assert b.allow() is False        # cooling again


# --- call_with_resilience ----------------------------------------------------

async def test_call_retries_transient_then_succeeds():
    b = CircuitBreaker(fail_threshold=5, cooldown_sec=1)
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError()
        return "ok"

    out = await call_with_resilience(
        b, flaky, max_attempts=3, base_delay_sec=0.0, transient=(TimeoutError,)
    )
    assert out == "ok"
    assert calls["n"] == 3


async def test_call_exhausts_attempts_and_raises_transient():
    b = CircuitBreaker(fail_threshold=5, cooldown_sec=1)
    calls = {"n": 0}

    async def always_times_out():
        calls["n"] += 1
        raise TimeoutError()

    with pytest.raises(TimeoutError):
        await call_with_resilience(
            b, always_times_out, max_attempts=3, base_delay_sec=0.0,
            transient=(TimeoutError,),
        )
    assert calls["n"] == 3           # tried exactly max_attempts times


async def test_non_transient_reraises_immediately_without_tripping():
    b = CircuitBreaker(fail_threshold=1, cooldown_sec=5)
    calls = {"n": 0}

    async def bad_request():
        calls["n"] += 1
        raise ValueError("400")      # NOT in the transient set

    with pytest.raises(ValueError):
        await call_with_resilience(
            b, bad_request, max_attempts=3, base_delay_sec=0.0,
            transient=(TimeoutError,),
        )
    assert calls["n"] == 1           # no retry
    assert b.state == "closed"       # did not trip the breaker


async def test_call_raises_breaker_open_when_open():
    b = CircuitBreaker(fail_threshold=1, cooldown_sec=99)
    b.record_failure()               # -> open, long cooldown

    async def should_not_run():
        raise AssertionError("fn must not be called when breaker is open")

    with pytest.raises(BreakerOpen):
        await call_with_resilience(
            b, should_not_run, max_attempts=1, base_delay_sec=0.0,
            transient=(TimeoutError,),
        )
