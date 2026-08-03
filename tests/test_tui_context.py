"""The context meter and the project-rules chip.

The token estimate is `len // 4`, not a tokenizer — see `tui/context.py`. These
tests pin the properties that matter for a ten-character bar (monotonic, counts
every block kind, clamps) rather than an accuracy the estimate never claimed.
"""

from __future__ import annotations

from client import Message, TextBlock, ToolCallBlock, ToolResultBlock
from tui.context import estimate_tokens, find_project_rules, meter, rules_label


def _user(text: str) -> Message:
    return Message(role="user", blocks=[TextBlock(text=text)])


# --------------------------------------------------------------------------
# Token estimate
# --------------------------------------------------------------------------


def test_no_messages_is_no_tokens():
    assert estimate_tokens([]) == 0


def test_text_is_counted():
    assert estimate_tokens([_user("a" * 400)]) == 100


def test_the_estimate_grows_with_the_conversation():
    one = estimate_tokens([_user("a" * 400)])
    two = estimate_tokens([_user("a" * 400), _user("b" * 400)])
    assert two > one


def test_tool_results_are_counted():
    """Tool output is usually the largest thing in the window; a meter that
    ignored it would read near-empty right before a compaction."""
    message = Message(
        role="user",
        blocks=[ToolResultBlock(tool_call_id="1", content="x" * 400)],
    )
    assert estimate_tokens([message]) == 100


def test_tool_calls_are_counted():
    message = Message(
        role="assistant",
        blocks=[ToolCallBlock(id="1", name="read_file", arguments={"path": "a.py"})],
    )
    assert estimate_tokens([message]) > 0


# --------------------------------------------------------------------------
# The bar
# --------------------------------------------------------------------------


def test_an_empty_context_draws_an_empty_bar():
    assert meter(0, 1000, width=10).startswith("░" * 10)


def test_a_half_full_context_draws_a_half_bar():
    bar = meter(500, 1000, width=10)
    assert bar.count("█") == 5
    assert "50%" in bar


def test_the_bar_is_always_the_requested_width():
    """A bar that grew past its width would push the cost readout off the
    status line — a rendering bug that gets reported as a cost bug."""
    for used in (0, 250, 999, 1000, 5000):
        bar = meter(used, 1000, width=12).split(" ")[0]
        assert len(bar) == 12


def test_overflow_clamps_to_full():
    assert meter(9999, 1000, width=10).count("█") == 10


def test_a_zero_budget_is_not_a_division_by_zero():
    assert meter(10, 0) == "—"


# --------------------------------------------------------------------------
# Project rules
# --------------------------------------------------------------------------


def test_no_rules_files_means_no_chip(tmp_path):
    assert find_project_rules(tmp_path) == []
    assert rules_label(tmp_path) == ""


def test_a_claude_md_is_found(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("rules", encoding="utf-8")
    assert [p.name for p in find_project_rules(tmp_path)] == ["CLAUDE.md"]


def test_several_rules_files_are_all_named(tmp_path):
    """Which files are steering the agent should be visible; a stale one
    quietly changing its behaviour is a confusing afternoon."""
    (tmp_path / "CLAUDE.md").write_text("a", encoding="utf-8")
    (tmp_path / "FORGE.md").write_text("b", encoding="utf-8")
    label = rules_label(tmp_path)
    assert "CLAUDE.md" in label
    assert "FORGE.md" in label


def test_a_directory_named_like_a_rules_file_is_ignored(tmp_path):
    (tmp_path / "CLAUDE.md").mkdir()
    assert find_project_rules(tmp_path) == []
