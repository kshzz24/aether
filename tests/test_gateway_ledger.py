"""Task 2 — ledger gains a request `status` and a failure-recording path.

Needs a live Postgres. DSN comes from FORGE_LEDGER_DSN, falling back to the
local `forge-pg` container. Tests tag their rows with a unique model name and
query by it, so they never depend on (or wipe) existing rows — the ledger is
append-only.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from gateway import ledger

DSN = os.environ.get(
    "FORGE_LEDGER_DSN", "postgresql://forge:forge@localhost:5433/forge"
)


@pytest_asyncio.fixture
async def pool():
    p = await ledger.init_pool(DSN)
    yield p
    await p.close()


async def test_init_pool_adds_status_column(pool):
    # The existing table predates `status`; init_pool must migrate it in
    # (ALTER TABLE ... ADD COLUMN IF NOT EXISTS), not just define it in CREATE.
    dtype = await pool.fetchval(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = 'ledger' AND column_name = 'status'"
    )
    assert dtype is not None


async def test_record_defaults_status_ok(pool):
    tag = f"test-{uuid.uuid4()}"
    await ledger.record(
        pool,
        provider="p",
        model=tag,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        latency_ms=5,
    )
    status = await pool.fetchval("SELECT status FROM ledger WHERE model = $1", tag)
    assert status == "ok"


async def test_record_failure_marks_status_error(pool):
    tag = f"test-{uuid.uuid4()}"
    await ledger.record_failure(
        pool, provider="p", model=tag, latency_ms=12, error="boom"
    )
    row = await pool.fetchrow(
        "SELECT status, cost_usd, input_tokens, output_tokens "
        "FROM ledger WHERE model = $1",
        tag,
    )
    assert row["status"] == "error"
    assert row["cost_usd"] == 0       # a failed request cost nothing
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
