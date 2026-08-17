"""Phase 8, stage 4 — `/api/stats` degrades instead of failing.

The chat server has to run with the whole Phase-3 gateway stack down; that is the
condition on most development machines and the source of this suite's standing
Postgres/Redis errors. A 500 here would break the dashboard route for everyone who
has not stood up a ledger, so an absent ledger is an *answer*, not an error.
"""

from __future__ import annotations

from asgi_client import running

from server import app as app_module


async def test_no_ledger_is_available_false_not_a_500(forge_app, auth, monkeypatch):
    monkeypatch.delenv("FORGE_LEDGER_DSN", raising=False)
    async with running(forge_app()) as client:
        response = await client.get("/api/stats", headers=auth)

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "FORGE_LEDGER_DSN" in body["detail"]


async def test_a_connection_failure_is_also_available_false(
    forge_app, auth, monkeypatch
):
    """Any failure reaching the ledger is the same answer to the client."""

    async def _boom(state):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(app_module, "_ledger", _boom)
    async with running(forge_app()) as client:
        response = await client.get("/api/stats", headers=auth)

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "detail": "connection refused",
    }


async def test_a_live_ledger_reports_available_true(forge_app, auth, monkeypatch):
    """The shape `web/src/components/Dashboard.tsx` switches on."""

    async def _ledger(state):
        return object(), object()

    async def _compute_stats(pool, conn):
        return {
            "requests_total": 3,
            "error_rate": 0.0,
            "p95_latency_ms": 12.0,
            "cost_by_model": {"stub-model": 0.03},
            "cache_hit_rate": 0.5,
        }

    monkeypatch.setattr(app_module, "_ledger", _ledger)
    monkeypatch.setattr(app_module, "compute_stats", _compute_stats)

    async with running(forge_app()) as client:
        response = await client.get("/api/stats", headers=auth)

    body = response.json()
    assert body["available"] is True
    assert body["requests_total"] == 3
    assert body["cost_by_model"] == {"stub-model": 0.03}
