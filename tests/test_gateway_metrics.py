"""Task 7 — /stats aggregation over the ledger (+ Redis cache-hit counters).

`compute_stats` aggregates the WHOLE ledger table, so unlike the tagged-row
ledger tests this one must control the entire table state: it truncates first,
then seeds known rows. Needs the live Postgres test container; Redis is faked.
"""

from __future__ import annotations

import os

import fakeredis.aioredis as fakeredis
import pytest_asyncio

from gateway import ledger
from gateway.metrics import compute_stats

DSN = os.environ.get(
    "FORGE_LEDGER_DSN", "postgresql://forge:forge@localhost:5433/forge"
)


@pytest_asyncio.fixture
async def clean_pool():
    # Defense in depth (conftest already redirects to a *_test DB): never TRUNCATE
    # a non-test database, even if this module is run with a stray prod DSN.
    assert DSN.rsplit("/", 1)[-1].endswith("_test"), (
        f"refusing to TRUNCATE non-test database: {DSN}"
    )
    p = await ledger.init_pool(DSN)
    await p.execute("TRUNCATE ledger")   # whole-table aggregation needs a known state
    yield p
    await p.close()


@pytest_asyncio.fixture
async def rds():
    r = fakeredis.FakeRedis(decode_responses=True)
    await r.flushall()
    yield r
    await r.aclose()


async def _seed(pool):
    await ledger.record(
        pool, provider="openai", model="m1",
        input_tokens=1, output_tokens=1, cost_usd=1.0, latency_ms=100,
    )
    await ledger.record(
        pool, provider="openai", model="m1",
        input_tokens=1, output_tokens=1, cost_usd=2.0, latency_ms=300,
    )
    await ledger.record_failure(
        pool, provider="openai", model="m1", latency_ms=50, error="x",
    )


async def test_empty_ledger_returns_zeros(clean_pool, rds):
    s = await compute_stats(clean_pool, rds)
    assert s["requests_total"] == 0
    assert s["error_rate"] == 0.0          # avg over 0 rows is NULL -> coerced to 0
    assert s["p95_latency_ms"] == 0.0
    assert s["cost_by_model"] == {}
    assert s["cache_hit_rate"] == 0.0


async def test_stats_over_seeded_ledger(clean_pool, rds):
    await _seed(clean_pool)
    s = await compute_stats(clean_pool, rds)

    assert s["requests_total"] == 3
    assert abs(s["error_rate"] - 1 / 3) < 1e-6
    assert abs(s["cost_by_model"]["m1"] - 3.0) < 1e-6


async def test_p95_ignores_failed_rows(clean_pool, rds):
    # ok latencies {100, 300}; the failed row's 50ms must NOT count.
    # percentile_cont(0.95) over [100, 300] = 100 + 0.95*200 = 290.
    await _seed(clean_pool)
    s = await compute_stats(clean_pool, rds)
    assert abs(s["p95_latency_ms"] - 290.0) < 1e-6


async def test_cache_hit_rate_from_redis_counters(clean_pool, rds):
    await rds.set("forge:cache:hits", 3)
    await rds.set("forge:cache:misses", 1)
    s = await compute_stats(clean_pool, rds)
    assert abs(s["cache_hit_rate"] - 0.75) < 1e-6
