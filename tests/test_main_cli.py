import sys

import pytest

import main
import persistence


def test_goal_required_without_resume_or_list(monkeypatch):
    # A bare invocation with no goal, no --resume, no --list-sessions must error.
    monkeypatch.setattr(sys, "argv", ["forge"])
    with pytest.raises(SystemExit):
        main.main()


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
