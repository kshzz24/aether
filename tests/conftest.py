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


import pytest  # noqa: E402

from approval import ApprovalRequest, Decision, Verdict  # noqa: E402
from events import (  # noqa: E402
    ApprovalDecisionEvent,
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    SubagentEvent,
    TerminalEvent,
    TerminalReason,
    TextDeltaEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tools.base import ToolKind  # noqa: E402


class ScriptedApprover:
    """Test double: returns queued Decisions in order; records requests seen."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = list(decisions)
        self.seen: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> Decision:
        self.seen.append(request)
        return self._decisions.pop(0)


@pytest.fixture
def ScriptedApprover_():
    return ScriptedApprover


class StubClient:
    """A fake LLMClient that returns scripted responses, no network."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.received: list[list] = []  # snapshot of messages per create() call

    async def create(self, messages, tools, system):
        self.received.append(list(messages))  # copy: agent mutates the list
        return self._responses.pop(0)


def sample_events() -> list[Event]:
    """One instance of every member of the `Event` union.

    Shared by every surface that consumes the stream (`cli/renderer.py` and
    `tui/transcript.py`). Adding a variant to the union without adding it here
    fails `test_sample_covers_every_event_in_union`, which in turn forces both
    surfaces to grow a branch for it. One list, two gates.
    """
    return [
        StatusEvent(type="status", message="working"),
        TextDeltaEvent(type="text_delta", text="hel"),
        TextEvent(type="text", text="hello"),
        ToolCallEvent(type="tool_call", name="run_shell", arguments={"cmd": "ls"}),
        ToolResultEvent(type="tool_result", name="run_shell", result="ok", flags=[]),
        CostEvent(
            type="cost",
            cost_usd=0.01,
            total_cost_usd=0.02,
            input_tokens=100,
            output_tokens=50,
        ),
        ConfirmRequestEvent(
            tool_name="run_shell",
            arguments={"cmd": "rm -rf /"},
            reason="destructive shell command",
        ),
        TerminalEvent(reason=TerminalReason.COMPLETED, detail=""),
        ApprovalDecisionEvent(
            type="approval_decision",
            tool_name="write_file",
            kind=ToolKind.WRITE,
            danger_reasons=[],
            verdict=Verdict.AUTO_APPROVE,
            approved=True,
            source="policy",
        ),
        SubagentEvent(type="subagent", task="explore the repo", phase="started"),
    ]


# --- phase 8, stages 4-6: the HTTP layer --------------------------------------
#
# Shared by five test modules, so it lives here rather than being copied. Nothing
# below touches the network: `StubClient` answers every model call and
# `load_mcp_manager` is stubbed out, so an app built by `forge_app` can run a
# whole session with no provider and no subprocesses.

SERVER_TOKEN = "test-token"


@pytest.fixture
def forge_app(tmp_path, monkeypatch):
    """A factory for a `create_app` rooted at a temp dir, with a known token.

    `idle_timeout_sec`/`reap_interval_sec` are exposed because the lifecycle tests
    need a reaper that fires inside a test's patience.
    """
    import main
    import persistence
    from server.app import create_app

    monkeypatch.setenv("FORGE_SERVER_TOKEN", SERVER_TOKEN)
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(main, "load_mcp_manager", lambda root: None)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")

    def _make(**kwargs):
        kwargs.setdefault("root", tmp_path)
        return create_app(**kwargs)

    return _make


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {SERVER_TOKEN}"}


@pytest.fixture
def stub_responses(monkeypatch):
    """Make every session the app builds talk to a `StubClient`.

    Patching `build_composition` is the only seam: the app builds compositions
    itself, several requests deep, and a test cannot reach in to swap the client
    afterwards without racing the run it just started.
    """
    from server import sessions as sessions_module

    real = sessions_module.build_composition
    scripts: list[list] = []

    def _script(responses: list) -> None:
        scripts.append(list(responses))

    def _patched(params, **kwargs):
        comp = real(params, **kwargs)
        if scripts:
            comp.agent.client = StubClient(scripts.pop(0))
        return comp

    monkeypatch.setattr(sessions_module, "build_composition", _patched)
    return _script
