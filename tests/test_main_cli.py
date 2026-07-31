import sys

import pytest

import main
import persistence


class _FakeApp:
    """Stands in for ForgeApp so argv routing is testable without a terminal."""

    launched: list[object] = []

    def __init__(self, args) -> None:
        self.args = args

    def run(self) -> None:
        _FakeApp.launched.append(self.args)


@pytest.fixture
def fake_tui(monkeypatch):
    """`main` imports ForgeApp lazily, so patching the module attribute works."""
    import tui

    _FakeApp.launched = []
    monkeypatch.setattr(tui, "ForgeApp", _FakeApp)
    return _FakeApp


def test_bare_invocation_opens_the_tui(monkeypatch, fake_tui):
    # With no goal there is nothing for the one-shot path to do, so `forge`
    # alone is the interactive surface.
    monkeypatch.setattr(sys, "argv", ["forge"])
    main.main()
    assert len(fake_tui.launched) == 1


def test_an_explicit_goal_still_runs_one_shot(monkeypatch, fake_tui):
    # `forge "<goal>"` must NOT open the TUI -- it stays scriptable and pipeable.
    called = []
    monkeypatch.setattr(sys, "argv", ["forge", "do the thing"])
    monkeypatch.setattr(main.asyncio, "run", lambda coro: called.append(coro) or
                        coro.close())
    main.main()
    assert fake_tui.launched == []
    assert called


def test_tui_flag_opens_the_tui_even_with_a_goal(monkeypatch, fake_tui):
    monkeypatch.setattr(sys, "argv", ["forge", "--tui", "do the thing"])
    main.main()
    assert len(fake_tui.launched) == 1
    assert fake_tui.launched[0].goal == "do the thing"


def test_list_sessions_wins_over_the_tui(monkeypatch, tmp_path, fake_tui):
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["forge", "--list-sessions"])
    main.main()
    assert fake_tui.launched == []


def test_list_sessions_renders_and_exits_without_running(monkeypatch, tmp_path, capsys):
    # Seed one saved session, then --list-sessions should print it and return
    # without touching the network / building an agent.
    session = persistence.Session(
        id="20260729-120000-abcd", goal="explore the repo", provider="groq",
        model="openai/gpt-oss-120b", created_at="2026-07-29T12:00:00",
        updated_at="2026-07-29T12:05:00", total_cost=0.02, messages=[],
    )
    persistence.save(session, tmp_path)
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["forge", "--list-sessions"])

    main.main()

    out = capsys.readouterr().out
    assert "20260729-120000-abcd" in out
    assert "explore the repo" in out


def test_list_sessions_empty_says_none(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["forge", "--list-sessions"])

    main.main()

    assert "no saved sessions" in capsys.readouterr().out


def test_resume_with_missing_id_notices_and_returns(monkeypatch, tmp_path, capsys):
    # Resuming a non-existent id must surface a notice, not crash.
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["forge", "--resume", "nope-1234"])

    main.main()

    assert "cannot resume" in capsys.readouterr().out
