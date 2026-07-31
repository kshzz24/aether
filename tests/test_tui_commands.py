"""Slash-command dispatch — pure, no app, no async.

Every one of these runs without booting Textual. That is the payoff for keeping
`dispatch` a free function over a context struct instead of a method on the app:
the command surface can grow without the suite getting slower or flakier.
"""

from __future__ import annotations

from datetime import datetime

import persistence
from config import ForgeConfig
from tools.base import Tool, ToolKind
from tools.registry import ToolRegistry
from tui.commands import COMMANDS, CommandContext, dispatch


def _tool(name: str, kind: ToolKind = ToolKind.READ) -> Tool:
    async def run(args: dict) -> str:
        return "ok"

    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        kind=kind,
        run=run,
    )


def _ctx(tmp_path, *, config=None, tools=(), goal="do a thing") -> CommandContext:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    now = datetime.now().isoformat(timespec="seconds")
    session = persistence.Session(
        id="sess-abc123",
        goal=goal,
        provider="groq",
        model="llama-3.3",
        created_at=now,
        updated_at=now,
        total_cost=0.0,
        messages=[],
    )
    return CommandContext(
        config=config or ForgeConfig(),
        registry=registry,
        session=session,
        sessions_dir=tmp_path,
        total_cost=0.0123,
        turns=3,
    )


# --------------------------------------------------------------------------
# Is it a command at all?
# --------------------------------------------------------------------------


def test_a_bare_goal_is_not_a_command(tmp_path):
    assert dispatch("read config.py and summarize it", _ctx(tmp_path)) is None


def test_leading_whitespace_still_parses_as_a_command(tmp_path):
    assert dispatch("   /help", _ctx(tmp_path)) is not None


def test_unknown_command_reports_rather_than_becoming_a_goal(tmp_path):
    """A typo must not be silently forwarded to the model — that costs money."""
    result = dispatch("/toolz", _ctx(tmp_path))
    assert result is not None
    assert "unknown command" in result.text
    assert "/help" in result.text


def test_a_word_containing_a_slash_is_not_a_command(tmp_path):
    assert dispatch("fix src/main.py", _ctx(tmp_path)) is None


# --------------------------------------------------------------------------
# The commands
# --------------------------------------------------------------------------


def test_help_names_every_registered_command(tmp_path):
    """/help is generated from COMMANDS, so the two can never drift apart."""
    text = dispatch("/help", _ctx(tmp_path)).text
    for name in COMMANDS:
        assert name in text


def test_config_shows_provider_and_model(tmp_path):
    cfg = ForgeConfig(provider="groq", model="llama-3.3-70b")
    text = dispatch("/config", _ctx(tmp_path, config=cfg)).text
    assert "groq" in text
    assert "llama-3.3-70b" in text


def test_config_renders_an_unset_allowlist_as_all_tools(tmp_path):
    text = dispatch("/config", _ctx(tmp_path)).text
    assert "all tools" in text


def test_tools_lists_names_and_kinds(tmp_path):
    ctx = _ctx(
        tmp_path, tools=[_tool("read_file"), _tool("run_shell", ToolKind.EXECUTE)]
    )
    text = dispatch("/tools", ctx).text
    assert "read_file" in text
    assert "run_shell" in text
    assert "execute" in text


def test_tools_handles_an_empty_registry(tmp_path):
    assert "no tools" in dispatch("/tools", _ctx(tmp_path)).text


def test_stats_shows_cost_and_turns(tmp_path):
    text = dispatch("/stats", _ctx(tmp_path)).text
    assert "0.0123" in text
    assert "3" in text


def test_mcp_reports_the_unbuilt_phase_6_state(tmp_path):
    """The seam exists now so Phase 6 fills it in without restructuring."""
    assert "no MCP servers" in dispatch("/mcp", _ctx(tmp_path)).text


def test_quit_sets_the_quit_flag(tmp_path):
    result = dispatch("/quit", _ctx(tmp_path))
    assert result.quit is True


def test_commands_are_case_insensitive(tmp_path):
    assert dispatch("/HELP", _ctx(tmp_path)).text.startswith("commands:")


# --------------------------------------------------------------------------
# The ones that touch disk
# --------------------------------------------------------------------------


def test_save_then_sessions_round_trips(tmp_path):
    ctx = _ctx(tmp_path, goal="build the thing")

    saved = dispatch("/save", ctx)
    assert "sess-abc123" in saved.text

    listed = dispatch("/sessions", ctx).text
    assert "sess-abc123" in listed
    assert "build the thing" in listed


def test_sessions_reports_an_empty_directory(tmp_path):
    assert "no saved sessions" in dispatch("/sessions", _ctx(tmp_path)).text


def test_resume_returns_the_id_for_the_app_to_act_on(tmp_path):
    ctx = _ctx(tmp_path)
    dispatch("/save", ctx)

    result = dispatch("/resume sess-abc123", ctx)
    assert result.resume_id == "sess-abc123"


def test_resume_with_an_unknown_id_errors_without_raising(tmp_path):
    result = dispatch("/resume nope-does-not-exist", _ctx(tmp_path))
    assert result.resume_id is None
    assert "cannot resume" in result.text


def test_resume_without_an_id_shows_usage(tmp_path):
    result = dispatch("/resume", _ctx(tmp_path))
    assert result.resume_id is None
    assert "usage" in result.text
