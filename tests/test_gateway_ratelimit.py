"""Task 3 — per-key token-bucket rate limiter over Redis.

Hermetic: runs against fakeredis (with lupa for Lua EVAL), so no container is
needed. Time is injected via `now=` so refill is tested deterministically
instead of with sleeps.
"""

from __future__ import annotations

import fakeredis.aioredis as fakeredis
import pytest_asyncio

from gateway.ratelimit import TokenBucket


@pytest_asyncio.fixture
async def rds():
    r = fakeredis.FakeRedis(decode_responses=True)
    await r.flushall()
    yield r
    await r.aclose()


async def test_allows_up_to_capacity(rds):
    # capacity 3, no refill: first 3 allowed, 4th denied.
    b = TokenBucket(rds, capacity=3, refill_per_sec=0, now=lambda: 1000.0)
    results = [await b.allow("k") for _ in range(4)]
    assert results == [True, True, True, False]


async def test_refills_over_time(rds):
    clock = {"t": 1000.0}
    b = TokenBucket(rds, capacity=1, refill_per_sec=1.0, now=lambda: clock["t"])
    assert await b.allow("k") is True     # spend the one token
    assert await b.allow("k") is False    # bucket empty
    clock["t"] += 1.0                     # one second passes -> +1 token
    assert await b.allow("k") is True     # refilled, allowed again


async def test_never_exceeds_capacity_on_refill(rds):
    # idle a long time; the bucket caps at capacity, doesn't accrue infinite tokens.
    clock = {"t": 0.0}
    b = TokenBucket(rds, capacity=2, refill_per_sec=1.0, now=lambda: clock["t"])
    await b.allow("k")                    # touch the key so state exists
    clock["t"] += 10_000                  # huge idle
    assert await b.allow("k") is True
    assert await b.allow("k") is True
    assert await b.allow("k") is False    # only 2 tokens, not 10_000


async def test_keys_are_isolated(rds):
    b = TokenBucket(rds, capacity=1, refill_per_sec=0, now=lambda: 0.0)
    assert await b.allow("a") is True
    assert await b.allow("b") is True     # different key, its own bucket
    assert await b.allow("a") is False    # a's bucket independently empty
