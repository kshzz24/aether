"""Test-suite safety net: never let tests touch the live ledger.

Several gateway tests need a real Postgres, and one of them (`test_gateway_metrics`)
TRUNCATEs the whole `ledger` table to get a known state. Left unguarded those
tests default to the *production* DB (``localhost:5433/forge``) and wipe it.

This conftest runs at collection time -- before any test module evaluates its
module-level ``DSN = os.environ.get("FORGE_LEDGER_DSN", ...)`` -- and redirects
the entire suite to a dedicated ``<db>_test`` database, creating it if needed.
A hard assertion refuses to proceed if the resolved target is not a ``_test``
database, so a destructive test can never point at production again.

Override with ``FORGE_TEST_LEDGER_DSN`` if you want an explicit test DSN; even
then the ``_test`` suffix is enforced.
"""

from __future__ import annotations

import asyncio
import os
import urllib.parse

_DEFAULT = "postgresql://forge:forge@localhost:5433/forge"


def _resolve_test_dsn() -> str:
    """Derive a `<db>_test` DSN from the explicit test DSN or the prod DSN."""
    base = os.environ.get("FORGE_TEST_LEDGER_DSN") or os.environ.get(
        "FORGE_LEDGER_DSN", _DEFAULT
    )
    parts = urllib.parse.urlsplit(base)
    db = parts.path.lstrip("/") or "forge"
    if not db.endswith("_test"):
        db = f"{db}_test"
    return urllib.parse.urlunsplit(parts._replace(path=f"/{db}"))


async def _ensure_db(dsn: str) -> None:
    """Create the test database if it doesn't exist (via the `postgres` admin DB)."""
    import asyncpg

    parts = urllib.parse.urlsplit(dsn)
    db = parts.path.lstrip("/")
    admin_dsn = urllib.parse.urlunsplit(parts._replace(path="/postgres"))
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{db}"')
    finally:
        await conn.close()


_TEST_DSN = _resolve_test_dsn()

# Hard guard: the suite must NEVER point at a non-`_test` database.
_target_db = urllib.parse.urlsplit(_TEST_DSN).path.lstrip("/")
assert _target_db.endswith("_test"), (
    f"refusing to run tests against {_target_db!r}: not a *_test database"
)

# Best-effort create. If Postgres is down, DB-backed tests fail loudly on connect
# -- but the env already points at the test DB, so prod is never touched.
try:
    asyncio.run(_ensure_db(_TEST_DSN))
except Exception:  # noqa: BLE001 -- collection must not crash when PG is absent
    pass

# Redirect the whole suite. Set before any test module reads FORGE_LEDGER_DSN.
os.environ["FORGE_LEDGER_DSN"] = _TEST_DSN
