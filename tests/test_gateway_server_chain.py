"""Task 8 -- the gate chain wired into POST /v1/chat/completions.

These tests exercise the *ordering and control flow* of the chain
(rate-limit -> exact cache -> breaker -> provider), not the provider or the DB.
The provider is a fake (no network), Redis is faked, and the ledger writes are
stubbed to no-ops -- so the tests are hermetic and assert on the HTTP status and
on *how often the provider was actually called*.

Setup note: the real `lifespan` connects to Postgres and Redis on startup. We
replace it with a no-op and set `app.state` by hand, then drive the app through
`TestClient` as a context manager. The context manager is what gives every
request one persistent event loop -- `fakeredis.aioredis` binds its connections
to a loop, so a fresh loop per request would break it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import fakeredis.aioredis as fakeredis
import pytest
from fastapi.testclient import TestClient

from client import NormalizedResponse, TextBlock
from gateway import server as srv
from gateway.cache import ExactCache
from gateway.config import BreakerCfg, GatewayConfig, RateLimitCfg, RetryCfg
from gateway.ratelimit import TokenBucket

REQ = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


def _ok_response() -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text="hello")],
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        stop_reason="end_turn",
    )


class FakeClient:
    """Stands in for a provider client. `behavior()` returns a response or raises.

    `calls` counts how many times `create` actually ran -- the assertion that
    proves a cache hit (or an open breaker) skipped the upstream entirely.
    """

    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = 0

    async def create(self, *, messages, tools, system):
        self.calls += 1
        return self._behavior()


@asynccontextmanager
async def _noop_lifespan(app):
    # Stand-in for the real startup (Postgres + Redis). Does nothing; the fixture
    # populates app.state itself.
    yield


@pytest.fixture
def build(monkeypatch):
    """Factory: wire app.state + the fakes, return (TestClient, FakeClient)."""
    entered: list[TestClient] = []

    def _build(
        *,
        behavior,
        capacity: int = 60,
        refill: float = 0.0,
        fail_threshold: int = 5,
        cooldown: float = 99.0,
        max_attempts: int = 1,
        transient: tuple[type[Exception], ...] | None = None,
    ):
        r = fakeredis.FakeRedis(decode_responses=True)

        srv.app.state.redis = r
        srv.app.state.prices = {}
        srv.app.state.pool = None  # unused: the ledger writes are stubbed below
        srv.app.state.gwcfg = GatewayConfig(
            ratelimit=RateLimitCfg(capacity=capacity, refill_per_sec=refill),
            breaker=BreakerCfg(fail_threshold=fail_threshold, cooldown_sec=cooldown),
            retry=RetryCfg(max_attempts=max_attempts, base_delay_sec=0.0),
        )
        srv.app.state.bucket = TokenBucket(
            r, capacity=capacity, refill_per_sec=refill
        )
        srv.app.state.exact = ExactCache(r, ttl_sec=3600)
        srv.app.state.semantic = None
        srv.app.state.breakers = {}

        fake = FakeClient(behavior)
        monkeypatch.setattr(srv, "make_client", lambda **kw: fake)

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(srv.ledger, "record", _noop)
        monkeypatch.setattr(srv.ledger, "record_failure", _noop)
        monkeypatch.setattr(srv.app.router, "lifespan_context", _noop_lifespan)
        if transient is not None:
            monkeypatch.setattr(srv, "TRANSIENT", transient)

        tc = TestClient(srv.app)
        tc.__enter__()
        entered.append(tc)
        return tc, fake

    yield _build
    for tc in entered:
        tc.__exit__(None, None, None)


def test_second_identical_request_is_cache_hit(build):
    client, fake = build(behavior=_ok_response)

    r1 = client.post("/v1/chat/completions", json=REQ)
    r2 = client.post("/v1/chat/completions", json=REQ)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert fake.calls == 1  # second request served from the exact cache


def test_rate_limit_returns_429(build):
    # capacity 2, no refill: two requests pass, the third is over budget.
    client, fake = build(behavior=_ok_response, capacity=2, refill=0.0)

    assert client.post("/v1/chat/completions", json=REQ).status_code == 200
    assert client.post("/v1/chat/completions", json=REQ).status_code == 200
    assert client.post("/v1/chat/completions", json=REQ).status_code == 429


def test_breaker_opens_and_returns_503(build):
    class Boom(Exception):
        pass

    def boom():
        raise Boom()

    # fail_threshold=1 + max_attempts=1: one failed request opens the breaker.
    client, fake = build(
        behavior=boom, fail_threshold=1, max_attempts=1, transient=(Boom,)
    )

    # 1st: provider fails (transient) -> breaker trips -> 502 upstream error.
    assert client.post("/v1/chat/completions", json=REQ).status_code == 502
    # 2nd: breaker is open -> refused fast as 503, provider NOT called again.
    assert client.post("/v1/chat/completions", json=REQ).status_code == 503
    assert fake.calls == 1
