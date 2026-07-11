"""Append-only request ledger (Postgres).

Phase 3 metering: one immutable row per gateway request. The table is
INSERT-only -- never UPDATE or DELETE. A correction is a new row, not an edit
(the append-only / audit-log discipline). Money is stored as NUMERIC, not float,
so running-cost sums don't accumulate rounding error.
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS ledger (
    id            BIGSERIAL      PRIMARY KEY,
    ts            TIMESTAMPTZ    NOT NULL DEFAULT now(),
    provider      TEXT           NOT NULL,
    model         TEXT           NOT NULL,
    input_tokens  INTEGER        NOT NULL,
    output_tokens INTEGER        NOT NULL,
    cost_usd      NUMERIC(12, 6) NOT NULL,
    latency_ms    INTEGER        NOT NULL
    -- TODO(phase-14): correlation_id for request tracing
);
"""

_INSERT = (
    "INSERT INTO ledger "
    "(provider, model, input_tokens, output_tokens, cost_usd, latency_ms) "
    "VALUES ($1, $2, $3, $4, $5, $6)"
)


async def init_pool(dsn: str) -> asyncpg.Pool:
    """Create the connection pool and ensure the ledger table exists.

    Called once at gateway startup; the server owns the pool's lifecycle
    (create here, ``await pool.close()`` on shutdown).
    """
    pool = await asyncpg.create_pool(dsn)
    async with pool.acquire() as con:
        await con.execute(_DDL)
    return pool


async def record(
    pool: asyncpg.Pool,
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    timeout: float = 2.0,
) -> None:
    """Write one request row, bounded by ``timeout``.

    Blocking-with-timeout: the caller awaits this before responding, so the row
    is durable on the happy path (consistency). But a slow or unreachable ledger
    is logged and dropped rather than propagated -- metering is not on the
    request's liveness path, and losing a row must never fail a user's turn.
    """
    try:
        await asyncio.wait_for(
            _insert(
                pool,
                provider,
                model,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_ms,
            ),
            timeout,
        )
    except Exception as e:  # asyncio.TimeoutError is a subclass, so it's caught too
        log.warning("ledger write dropped: %s", e)


async def _insert(
    pool: asyncpg.Pool,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
) -> None:
    async with pool.acquire() as con:
        await con.execute(
            _INSERT,
            provider,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
            latency_ms,
        )
