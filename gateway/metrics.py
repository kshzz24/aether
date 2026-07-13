

from __future__ import annotations

import asyncpg
from redis.asyncio import Redis


async def compute_stats(pool: asyncpg.Pool, redis: Redis) -> dict:
    requests_total = await pool.fetchval("SELECT count(*) FROM ledger")

    error_rate = await pool.fetchval(
        "SELECT avg((status = 'error')::int) FROM ledger"
    )

    p95_latency_ms = await pool.fetchval(
        "SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) "
        "FROM ledger WHERE status = 'ok'"
    )

    rows = await pool.fetch(
        "SELECT model, sum(cost_usd) AS cost FROM ledger GROUP BY model"
    )
    cost_by_model = {row["model"]: float(row["cost"]) for row in rows}

    hits = int(await redis.get("forge:cache:hits") or 0)
    misses = int(await redis.get("forge:cache:misses") or 0)
    total = hits + misses
    cache_hit_rate = hits / total if total else 0.0

    return {
        "requests_total": requests_total,
        "error_rate": float(error_rate or 0),
        "p95_latency_ms": float(p95_latency_ms or 0),
        "cost_by_model": cost_by_model,
        "cache_hit_rate": cache_hit_rate,
    }
