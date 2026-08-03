"""Saved prompts loaded from `~/.forge/prompts/*.md`.

Small surface, but it reads user-controlled files at startup of a command, so
the failure modes matter more than the happy path: a missing directory is
normal, and one bad file must not hide the rest.
"""

from __future__ import annotations

from tui.templates import load_templates


def test_a_missing_directory_is_not_an_error(tmp_path):
    """Most people never create one; the picker should say "no saved prompts"
    rather than fail."""
    assert load_templates(tmp_path / "nope") == {}


def test_an_empty_directory_yields_nothing(tmp_path):
    assert load_templates(tmp_path) == {}


def test_a_template_is_keyed_by_its_filename(tmp_path):
    (tmp_path / "review.md").write_text("look carefully", encoding="utf-8")
    assert load_templates(tmp_path) == {"review": "look carefully"}


def test_surrounding_whitespace_is_stripped(tmp_path):
    (tmp_path / "a.md").write_text("\n\n  body  \n\n", encoding="utf-8")
    assert load_templates(tmp_path)["a"] == "body"


def test_internal_structure_is_preserved(tmp_path):
    """Templates are markdown; a multi-paragraph prompt must survive intact."""
    body = "line one\n\n- a\n- b"
    (tmp_path / "a.md").write_text(body, encoding="utf-8")
    assert load_templates(tmp_path)["a"] == body


def test_non_markdown_files_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("not a template", encoding="utf-8")
    (tmp_path / "real.md").write_text("template", encoding="utf-8")
    assert list(load_templates(tmp_path)) == ["real"]


def test_an_empty_template_is_skipped(tmp_path):
    """Filling the prompt box with nothing looks like the command failed."""
    (tmp_path / "blank.md").write_text("   \n", encoding="utf-8")
    assert load_templates(tmp_path) == {}


def test_a_directory_named_like_a_template_does_not_break_the_load(tmp_path):
    (tmp_path / "folder.md").mkdir()
    (tmp_path / "good.md").write_text("body", encoding="utf-8")
    assert list(load_templates(tmp_path)) == ["good"]


def test_templates_come_back_sorted(tmp_path):
    for name in ("zebra", "apple", "mango"):
        (tmp_path / f"{name}.md").write_text("x", encoding="utf-8")
    assert list(load_templates(tmp_path)) == ["apple", "mango", "zebra"]
