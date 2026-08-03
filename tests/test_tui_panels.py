"""The panels and overlays: sidebar, shortcuts, pickers, and the new commands.

These are the features that only exist once the app is running — the sidebar
reads `UndoStack.touched()`, `/undo` needs the hooks to have been wired into the
composition, and the pickers are modal screens. Everything testable without an
app already lives in the pure test modules.
"""

from __future__ import annotations

import argparse

import pytest
from textual.widgets import Input, OptionList

import persistence
from approval import ApprovalMode
from tui.app import ForgeApp
from tui.commands import COMMANDS
from tui.help import HelpScreen
from tui.pickers import FuzzyPicker
from tui.prompt import PromptArea
from tui.sidebar import Sidebar
from tui.transcript import TranscriptView


def _args(**over) -> argparse.Namespace:
    base = dict(
        goal=None,
        gateway_url=None,
        resume=None,
        list_sessions=False,
        tui=True,
        setup=False,
        provider="groq",
        model="stub-model",
        max_iterations=None,
        max_cost_usd=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key-not-used")
    return tmp_path


async def _command(pilot, text: str) -> None:
    prompt = pilot.app.query_one(PromptArea)
    prompt.text = text
    prompt.post_message(PromptArea.Submitted(text))
    await pilot.pause()
    await pilot.pause()


def _transcript_text(app) -> str:
    return app.query_one(TranscriptView).text


def _tree_labels(app, selector: str = "#changed-tree") -> list[str]:
    """Every node label in a Tree, flattened."""
    labels: list[str] = []

    def walk(node) -> None:
        for child in node.children:
            labels.append(str(child.label))
            walk(child)

    walk(app.query_one(selector).root)
    return labels


# --------------------------------------------------------------------------
# The sidebar (ctrl+b)
# --------------------------------------------------------------------------


async def test_the_sidebar_is_hidden_on_startup(sessions_dir):
    """An empty pane on startup is just lost width."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(Sidebar).display is False


async def test_ctrl_b_shows_the_sidebar(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app.query_one(Sidebar).display is True


async def test_ctrl_b_toggles_back(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app.query_one(Sidebar).display is False


async def test_an_untouched_session_says_so(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert _tree_labels(app) == ["nothing yet"]


async def test_a_changed_file_appears_in_the_sidebar(sessions_dir, tmp_path):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        target = tmp_path / "touched.py"
        target.write_text("x", encoding="utf-8")
        args = {"path": str(target), "content": "y"}
        app._undo.before_tool("write_file", args)
        app._undo.after_tool("write_file", args, "ok")

        await pilot.press("ctrl+b")
        await pilot.pause()
        assert "touched.py" in _tree_labels(app)


# --------------------------------------------------------------------------
# The shortcuts overlay
# --------------------------------------------------------------------------


async def test_f1_opens_the_shortcuts_overlay(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


async def test_escape_closes_the_overlay(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)


async def test_question_mark_in_the_prompt_types_a_question_mark(sessions_dir):
    """`?` opens the overlay only from the transcript. Bound app-wide it would
    make the prompt unable to type the character."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(PromptArea).focus()
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()

        assert not isinstance(app.screen, HelpScreen)
        assert "?" in app.query_one(PromptArea).text


async def test_question_mark_on_the_transcript_opens_the_overlay(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(TranscriptView).focus()
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


# --------------------------------------------------------------------------
# ctrl+r history search
# --------------------------------------------------------------------------


async def test_ctrl_r_with_no_history_does_not_open_a_picker(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert not isinstance(app.screen, FuzzyPicker)


async def test_ctrl_r_lists_what_you_have_typed(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/help")
        await _command(pilot, "/tools")

        await pilot.press("ctrl+r")
        await pilot.pause()
        assert isinstance(app.screen, FuzzyPicker)
        assert app.screen.query_one(OptionList).option_count == 2


async def test_the_history_picker_offers_the_newest_first(sessions_dir):
    """An empty filter should offer what you just ran."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/help")
        await _command(pilot, "/tools")

        await pilot.press("ctrl+r")
        await pilot.pause()
        first = app.screen.query_one(OptionList).get_option_at_index(0)
        assert "/tools" in str(first.prompt)


async def test_filtering_narrows_the_history(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/help")
        await _command(pilot, "/tools")

        await pilot.press("ctrl+r")
        await pilot.pause()
        app.screen.query_one(Input).value = "hlp"
        await pilot.pause()
        assert app.screen.query_one(OptionList).option_count == 1


async def test_choosing_from_history_fills_the_prompt_without_sending(sessions_dir):
    """There is no undo on a sent goal, so a recalled prompt lands in the box."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/help")

        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one(PromptArea).text == "/help"


async def test_escaping_the_history_picker_changes_nothing(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/help")

        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert app.query_one(PromptArea).text == ""


# --------------------------------------------------------------------------
# /undo
# --------------------------------------------------------------------------


async def test_undo_restores_a_file_the_agent_changed(sessions_dir, tmp_path):
    """End to end: the hooks reach the agent through build_composition, the
    snapshot is taken, and /undo puts the bytes back."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        target = tmp_path / "a.py"
        target.write_text("original", encoding="utf-8")

        args = {"path": str(target), "content": "replaced"}
        app.comp.agent.hooks.before_tool("write_file", args)
        target.write_text("replaced", encoding="utf-8")
        app.comp.agent.hooks.after_tool("write_file", args, "ok")

        await _command(pilot, "/undo")
        assert target.read_text(encoding="utf-8") == "original"


async def test_undo_with_nothing_to_revert_says_so(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/undo")
        assert "nothing to undo" in _transcript_text(app)


async def test_redo_puts_it_back(sessions_dir, tmp_path):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        target = tmp_path / "a.py"
        target.write_text("original", encoding="utf-8")
        args = {"path": str(target), "content": "agent version"}
        app.comp.agent.hooks.before_tool("write_file", args)
        target.write_text("agent version", encoding="utf-8")
        app.comp.agent.hooks.after_tool("write_file", args, "ok")

        await _command(pilot, "/undo")
        assert target.read_text(encoding="utf-8") == "original"
        await _command(pilot, "/redo")
        assert target.read_text(encoding="utf-8") == "agent version"


async def test_u_on_the_transcript_undoes(sessions_dir, tmp_path):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        target = tmp_path / "a.py"
        target.write_text("original", encoding="utf-8")
        args = {"path": str(target), "content": "changed"}
        app._undo.before_tool("write_file", args)
        target.write_text("changed", encoding="utf-8")
        app._undo.after_tool("write_file", args, "ok")

        app.query_one(TranscriptView).focus()
        await pilot.pause()
        await pilot.press("u")
        await pilot.pause()
        assert target.read_text(encoding="utf-8") == "original"


async def test_u_in_the_prompt_types_a_u(sessions_dir):
    """`u` is printable; bound app-wide it would revert the agent's work every
    time someone typed the word "update"."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(PromptArea).focus()
        await pilot.press("u")
        await pilot.pause()
        assert app.query_one(PromptArea).text == "u"


# --------------------------------------------------------------------------
# /yolo and the /resume picker
# --------------------------------------------------------------------------


async def test_yolo_switches_to_auto_approval(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/yolo")
        assert app.comp.agent.policy.mode is ApprovalMode.AUTO


async def test_yolo_says_what_it_gave_up(sessions_dir):
    """Auto-approve drops the danger checks too; that has to be said out loud."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/yolo")
        assert "without asking" in _transcript_text(app)


async def test_resume_with_no_sessions_says_so(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/resume")
        assert "no saved sessions" in _transcript_text(app)


async def test_bare_resume_opens_a_picker(sessions_dir):
    """Beats reading an id off /sessions and typing it back in."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/save")

        await _command(pilot, "/resume")
        assert isinstance(app.screen, FuzzyPicker)
        assert app.screen.query_one(OptionList).option_count >= 1
        await pilot.press("escape")
        await pilot.pause()


async def test_resume_with_an_id_still_works(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        session_id = app.comp.session.id
        await _command(pilot, "/save")
        await _command(pilot, f"/resume {session_id}")
        assert not isinstance(app.screen, FuzzyPicker)
        assert app.comp.session.id == session_id


async def test_the_agents_hooks_are_the_undo_stacks(sessions_dir):
    """The Phase-2 seam is what carries this; if the wiring is dropped, undo
    silently becomes a no-op rather than failing."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.comp.agent.hooks.before_tool == app._undo.before_tool


# --------------------------------------------------------------------------
# /files and /context
# --------------------------------------------------------------------------


async def test_files_reports_an_untouched_session(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/files")
        assert "has not changed any files" in _transcript_text(app)


async def test_files_lists_what_changed(sessions_dir, tmp_path):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        target = tmp_path / "changed.py"
        args = {"path": str(target), "content": "x"}
        app._undo.before_tool("write_file", args)
        app._undo.after_tool("write_file", args, "ok")

        await _command(pilot, "/files")
        assert "changed.py" in _transcript_text(app)


async def test_context_shows_a_meter(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/context")
        assert "tokens" in _transcript_text(app)


async def test_the_status_bar_carries_a_context_meter(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "%" in str(app.query_one("#status-context").content)


async def test_the_status_bar_names_the_project_rules(sessions_dir):
    """A stale CLAUDE.md quietly steering the agent is a confusing afternoon."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "CLAUDE.md" in str(app.query_one("#status-rules").content)


# --------------------------------------------------------------------------
# /bell and /prompt
# --------------------------------------------------------------------------


async def test_bell_toggles(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._bell_enabled is True
        await _command(pilot, "/bell")
        assert app._bell_enabled is False


async def test_prompt_without_templates_points_at_the_directory(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/prompt")
        assert "no saved prompts" in _transcript_text(app)


async def test_prompt_fills_the_box_from_a_template(
    sessions_dir, tmp_path, monkeypatch
):
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "review.md").write_text("review this carefully", encoding="utf-8")
    monkeypatch.setattr("tui.commands.TEMPLATES_DIR", directory)
    monkeypatch.setattr("tui.app.TEMPLATES_DIR", directory)

    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/prompt review")
        assert app.query_one(PromptArea).text == "review this carefully"


async def test_an_unknown_template_lists_the_real_ones(
    sessions_dir, tmp_path, monkeypatch
):
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "review.md").write_text("body", encoding="utf-8")
    monkeypatch.setattr("tui.commands.TEMPLATES_DIR", directory)

    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/prompt nope")
        assert "review" in _transcript_text(app)


async def test_bare_prompt_opens_the_picker(sessions_dir, tmp_path, monkeypatch):
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "review.md").write_text("body", encoding="utf-8")
    monkeypatch.setattr("tui.commands.TEMPLATES_DIR", directory)
    monkeypatch.setattr("tui.app.TEMPLATES_DIR", directory)

    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await _command(pilot, "/prompt")
        assert isinstance(app.screen, FuzzyPicker)
        await pilot.press("escape")
        await pilot.pause()


# --------------------------------------------------------------------------
# Documentation stays honest
# --------------------------------------------------------------------------


async def test_every_command_dispatches_to_something(sessions_dir):
    """`/help` is the list people trust; a documented command that falls through
    to "unknown command" is worse than one that does not exist."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        from tui.commands import dispatch

        ctx = app._command_context()
        for name in COMMANDS:
            if name == "/quit":
                continue
            result = dispatch(name, ctx)
            assert result is not None, f"{name} was treated as a goal"
            assert "unknown command" not in result.text, f"{name} is not implemented"
