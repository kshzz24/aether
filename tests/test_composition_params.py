"""Stage-1 gate for the `Namespace` -> `RunParams` refactor (Phase 8).

Why this file exists. `build_composition` used to read an `argparse.Namespace`
plus three process-wide globals: a CWD-relative `open("prices.toml")`, a
`find_repo_root(Path.cwd())`, and a `Path.cwd()`-rooted `.mcp.json` /
`.forge/config.toml` lookup. All three are invisible while exactly one surface
exists and that surface is started by a human standing in their own repo.

A server breaks every one of those assumptions at once: it has no argv, it is
started from wherever systemd happened to put it, and it builds N compositions
for N different repos inside one process. So the refactor replaces implicit
process state with two explicit inputs — a `RunParams` and a `project_root` —
and these tests pin that the replacement is total. A single surviving
`Path.cwd()` would pass every existing test and fail in production only.

`test_prices_load_from_a_foreign_cwd` and `test_project_root_drives_config_and_
repo_root` are the two that would have caught the original bugs; the rest guard
the mechanical parts of the translation.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest
from conftest import ScriptedApprover

import main as composition_root
import persistence
from client import Message, TextBlock
from main import CompositionError, build_composition
from server.wire import RunParams


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A composition that touches nothing real.

    Sessions land in a temp dir, and a dummy key is set because the provider
    SDKs raise from their *constructor* when one is absent — no request is ever
    made by these tests.
    """
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    return tmp_path


def _approver() -> ScriptedApprover:
    """No decision is ever requested here; the approver only has to exist."""
    return ScriptedApprover([])


def _namespace(**over) -> argparse.Namespace:
    """The exact attribute set `main.main()`'s parser produces."""
    base = dict(
        goal=None,
        provider="groq",
        model="stub-model",
        max_iterations=None,
        max_cost_usd=None,
        gateway_url=None,
        resume=None,
        list_sessions=False,
        tui=False,
        setup=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------
# RunParams itself — pure, no I/O
# --------------------------------------------------------------------------


def test_run_params_is_constructible_with_no_arguments():
    """Every field defaults to None.

    Not cosmetic: `RunParams()` is what the server will build for a session
    that names only a model, and `dataclasses.replace` (which the TUI's
    `_rebuild` now uses) can only ever produce a *complete* instance if the
    original one was constructible in the first place.
    """
    params = RunParams()
    assert params.goal is None
    assert params.provider is None
    assert params.project_root is None


def test_from_namespace_maps_every_cli_flag():
    """The translation is the whole point of the classmethod, so pin it field
    by field. A mis-mapped name here is silent: the flag is simply ignored and
    the config default wins, which looks like "my --model flag does nothing"."""
    args = _namespace(
        goal="do the thing",
        provider="anthropic",
        model="claude-opus-4-8",
        max_iterations=7,
        max_cost_usd=2.5,
        gateway_url="http://localhost:8000/v1",
        resume="20260101-000000-abcd",
    )
    params = RunParams.from_namespace(args)

    assert params.goal == "do the thing"
    assert params.provider == "anthropic"
    assert params.model == "claude-opus-4-8"
    assert params.max_iterations == 7
    assert params.max_cost_usd == 2.5
    assert params.gateway_url == "http://localhost:8000/v1"
    assert params.resume == "20260101-000000-abcd"
    # argv cannot express a project root: the CLI is always rooted at its CWD,
    # and `build_composition` is what resolves None -> Path.cwd().
    assert params.project_root is None


def test_from_namespace_tolerates_a_partial_namespace():
    """The TUI's tests build partial namespaces, and `--setup` short-circuits
    argument parsing in ways that leave attributes unset. Missing attributes
    must read as None, not raise."""
    params = RunParams.from_namespace(argparse.Namespace())
    assert params == RunParams()


def test_approval_mode_field_matches_the_config_key():
    """`_rebuild` passes `{"approval_mode": ...}` straight through from
    `tui/commands.py`, and `build_composition` copies it into the config merge
    unchanged. If the field were named `approval`, `ForgeConfig`'s
    `extra="forbid"` would reject the merge at runtime — so the name is a
    contract, not a preference."""
    assert "approval_mode" in RunParams.__dataclass_fields__


# --------------------------------------------------------------------------
# build_composition — the two paths agree
# --------------------------------------------------------------------------


def test_namespace_and_explicit_params_build_the_same_composition(isolated):
    """The CLI (argv -> RunParams) and the server (RunParams direct) must be
    two spellings of one path, not two paths."""
    via_argv = build_composition(
        RunParams.from_namespace(
            _namespace(provider="groq", model="stub-model", max_iterations=9)
        ),
        approver=_approver(),
    )
    direct = build_composition(
        RunParams(provider="groq", model="stub-model", max_iterations=9),
        approver=_approver(),
    )

    assert via_argv.agent.model == direct.agent.model
    assert via_argv.agent.max_iterations == direct.agent.max_iterations == 9
    assert via_argv.agent.max_cost_usd == direct.agent.max_cost_usd
    assert via_argv.config.provider == direct.config.provider == "groq"
    assert via_argv.agent.repo_root == direct.agent.repo_root


def test_isolated_compositions_do_not_share_mutable_state(isolated):
    """Two sessions in one process is the entire premise of Phase 8. If the
    todo store or the message list were shared, session B would read session
    A's work — the classic per-connection-state-that-was-a-global bug."""
    first = build_composition(RunParams(provider="groq"), approver=_approver())
    second = build_composition(RunParams(provider="groq"), approver=_approver())

    assert first.todos is not second.todos
    assert first.agent is not second.agent
    assert first.agent.messages is not second.agent.messages
    assert first.session.id != second.session.id


# --------------------------------------------------------------------------
# The three CWD globals
# --------------------------------------------------------------------------


def test_prices_load_from_a_foreign_cwd(isolated, tmp_path, monkeypatch):
    """`open("prices.toml")` resolved against the process CWD.

    A server is not started from the FORGE checkout, so this raised
    FileNotFoundError before a single event existed. The fix is
    `Path(__file__).parent / "prices.toml"` — data that ships with the code is
    located relative to the code.
    """
    monkeypatch.chdir(tmp_path)
    comp = build_composition(RunParams(provider="groq"), approver=_approver())
    assert comp.agent is not None


def test_project_root_drives_config_and_repo_root(isolated, tmp_path, monkeypatch):
    """One field, both derivations.

    `repo_root` bounds the agent's path-escape checks and `.forge/config.toml`
    sets its budget. Reading either from `Path.cwd()` means every server
    session silently shares the server's directory — session A's agent could
    write into session B's repo.
    """
    workspace = tmp_path / "workspace"
    (workspace / ".forge").mkdir(parents=True)
    (workspace / ".forge" / "config.toml").write_text(
        "max_iterations = 3\n", encoding="utf-8"
    )
    # Deliberately somewhere else: if anything still reads the CWD, it reads
    # this directory, which has no config and is not the workspace.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    comp = build_composition(
        RunParams(provider="groq", project_root=workspace), approver=_approver()
    )

    assert comp.agent.max_iterations == 3
    assert comp.agent.repo_root == workspace.resolve()


def test_repo_root_prefers_the_enclosing_git_root(isolated, tmp_path, monkeypatch):
    """`project_root` is where the session is rooted; `repo_root` is the git
    boundary above it. A session opened in `src/` must still be allowed to
    touch `tests/`, so the derivation walks up."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    comp = build_composition(
        RunParams(provider="groq", project_root=nested), approver=_approver()
    )
    assert comp.agent.repo_root == repo.resolve()


def test_mcp_config_is_read_from_project_root(isolated, tmp_path, monkeypatch):
    """`.mcp.json` was the third CWD read. Per-session MCP is what makes tool
    federation isolated in Phase 8, so it has to follow the session's root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"demo": {"command": "echo"}}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    comp = build_composition(
        RunParams(provider="groq", project_root=workspace), approver=_approver()
    )

    assert comp.mcp is not None
    assert "demo" in comp.mcp.configs


def test_a_project_without_mcp_config_gets_no_manager(isolated, tmp_path, monkeypatch):
    """The inverse, so the test above cannot pass by accident on a machine that
    happens to have a user-scope `.mcp.json`."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "nohome")

    comp = build_composition(
        RunParams(provider="groq", project_root=workspace), approver=_approver()
    )
    assert comp.mcp is None


# --------------------------------------------------------------------------
# Resume, and the CLI wiring
# --------------------------------------------------------------------------


def test_unreadable_resume_raises_composition_error(isolated):
    """Guards a specific slip: the resume branch still reading `args.resume`
    after the parameter was renamed. That is a NameError, not a
    CompositionError — the surface would show a traceback instead of
    'cannot resume ...'."""
    with pytest.raises(CompositionError) as exc:
        build_composition(
            RunParams(provider="groq", resume="no-such-session"),
            approver=_approver(),
        )
    assert "no-such-session" in str(exc.value)


def test_resumed_session_supplies_goal_and_history(isolated, tmp_path):
    """Resume must win over params: the saved goal continues, it does not
    restart. Same rule as before the refactor — pinned here because the
    resume branch is the one that changed."""
    history = [Message(role="user", blocks=[TextBlock(text="hi")])]
    saved = persistence.Session(
        id="20260101-000000-abcd",
        goal="the original goal",
        provider="groq",
        model="stub-model",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        total_cost=0.0,
        messages=history,
    )
    persistence.save(saved, tmp_path)

    comp = build_composition(
        RunParams(goal="a different goal", resume=saved.id, provider="groq"),
        approver=_approver(),
    )

    assert comp.goal == "the original goal"
    assert comp.history == saved.messages


def test_main_builds_params_before_dispatching_to_run(isolated, monkeypatch):
    """The one-shot CLI path end to end.

    `main()` parses argv, translates once, and hands a RunParams to `_run`.
    Building the params inside a branch that returns early leaves the name
    unbound at the dispatch — a NameError on every ordinary `forge "<goal>"`
    invocation, which no unit test of `build_composition` can see.
    """
    seen: list[RunParams] = []

    def _fake_run(params):
        # Deliberately sync: `main` only has to *reach* the call with a bound
        # name. Awaiting a real coroutine would drag the whole agent in.
        seen.append(params)

    monkeypatch.setattr(composition_root, "_run", _fake_run)
    monkeypatch.setattr(asyncio, "run", lambda _coro: None)
    monkeypatch.setattr("sys.argv", ["forge", "do the thing", "--provider", "groq"])

    composition_root.main()

    assert len(seen) == 1
    assert seen[0].goal == "do the thing"
    assert seen[0].provider == "groq"
