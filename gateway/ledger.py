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
    latency_ms    INTEGER        NOT NULL,
    status        TEXT           NOT NULL DEFAULT 'ok'
    -- TODO(phase-14): correlation_id for request tracing
);
"""

# Existing databases predate `status`; CREATE IF NOT EXISTS won't touch them, so
# migrate the column in explicitly. Idempotent -- safe to run every startup.
_MIGRATE = (
    "ALTER TABLE ledger "
    "ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ok'"
)

_INSERT = (
    "INSERT INTO ledger "
    "(provider, model, input_tokens, output_tokens, cost_usd, latency_ms, status) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7)"
)


async def init_pool(dsn: str) -> asyncpg.Pool:
    """Create the connection pool and ensure the ledger table exists.

    Called once at gateway startup; the server owns the pool's lifecycle
    (create here, ``await pool.close()`` on shutdown).
    """
    pool = await asyncpg.create_pool(dsn)
    async with pool.acquire() as con:
        await con.execute(_DDL)
        await con.execute(_MIGRATE)
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
    status: str = "ok",
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
                status,
            ),
            timeout,
        )
    except Exception as e:  # asyncio.TimeoutError is a subclass, so it's caught too
        log.warning("ledger write dropped: %s", e)


async def record_failure(
    pool: asyncpg.Pool,
    *,
    provider: str,
    model: str,
    latency_ms: int,
    error: str,
    timeout: float = 2.0,
) -> None:
    """Record a failed request: status='error', zero tokens, zero cost.

    The error text is logged, not stored -- the `status` column is all the
    error-rate metric needs. Add an `error` column only when something reads it.
    """
    log.warning("gateway request failed (%s/%s): %s", provider, model, error)
    await record(
        pool,
        provider=provider,
        model=model,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0,
        latency_ms=latency_ms,
        status="error",
        timeout=timeout,
    )


async def _insert(
    pool: asyncpg.Pool,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    status: str,
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
            status,
        )
