"""Task 4 — exact response cache over Redis.

Hermetic: runs against fakeredis, so no container is needed. `canonical_key`
tests are pure (no Redis); the cache tests use the fakeredis fixture.
"""

from __future__ import annotations

import fakeredis.aioredis as fakeredis
import pytest_asyncio

from gateway.cache import ExactCache, canonical_key
from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)


@pytest_asyncio.fixture
async def rds():
    r = fakeredis.FakeRedis(decode_responses=True)
    await r.flushall()
    yield r
    await r.aclose()


def _req(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": text}]
    )


def _resp(text: str = "hi") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="cmpl-1",
        created=1,
        model="m",
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_key_is_stable_across_construction():
    # Two independently built but identical requests hash the same.
    assert canonical_key(_req("hi")) == canonical_key(_req("hi"))


def test_key_differs_on_content():
    assert canonical_key(_req("hi")) != canonical_key(_req("bye"))


def test_key_is_a_sha256_hexdigest():
    key = canonical_key(_req("hi"))
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


async def test_put_then_get_roundtrips(rds):
    c = ExactCache(rds, ttl_sec=60)
    resp = _resp()
    await c.put("k", resp)
    assert await c.get("k") == resp


async def test_miss_returns_none(rds):
    assert await ExactCache(rds, ttl_sec=60).get("absent") is None


async def test_put_overwrites(rds):
    # A second put on the same key replaces the value (no write-if-absent).
    c = ExactCache(rds, ttl_sec=60)
    await c.put("k", _resp("first"))
    await c.put("k", _resp("second"))
    got = await c.get("k")
    assert got is not None
    assert got.choices[0].message.content == "second"


async def test_put_sets_ttl(rds):
    # The stored key carries the configured TTL, not a persistent (-1) key.
    c = ExactCache(rds, ttl_sec=60)
    await c.put("k", _resp())
    ttl = await rds.ttl("cache:k")
    assert 0 < ttl <= 60


async def test_canonical_key_roundtrips_through_cache(rds):
    # The real usage: key comes from canonical_key, and get/put agree on it.
    c = ExactCache(rds, ttl_sec=60)
    key = canonical_key(_req("hello"))
    resp = _resp()
    await c.put(key, resp)
    assert await c.get(key) == resp
