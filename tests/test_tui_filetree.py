"""Nesting a flat path list into a tree.

Two consumers depend on this being right: the `ctrl+b` sidebar mounts it as a
`Tree`, and `/files` plus the end-of-run summary render it as text. It lives in
a widget-free module precisely so it can be tested like this.
"""

from __future__ import annotations

from pathlib import Path

from tui.filetree import group_paths, relative, sorted_items, tree_lines


def _root(tmp_path) -> Path:
    return tmp_path


# --------------------------------------------------------------------------
# relative
# --------------------------------------------------------------------------


def test_a_path_inside_the_repo_is_shown_relative(tmp_path):
    assert relative(tmp_path / "tui" / "app.py", tmp_path) == "tui/app.py"


def test_a_path_outside_the_repo_keeps_its_absolute_form(tmp_path):
    """A tool can legitimately write to /tmp; `../../../tmp/x` would be less
    honest than the real location."""
    outside = Path("/somewhere/else/x.txt")
    assert relative(outside, tmp_path) == outside.as_posix()


def test_separators_are_normalised(tmp_path):
    """Windows paths must render with the same separator as everything else."""
    assert "\\" not in relative(tmp_path / "a" / "b" / "c.py", tmp_path)


# --------------------------------------------------------------------------
# group_paths
# --------------------------------------------------------------------------


def test_a_single_file_is_one_leaf(tmp_path):
    assert group_paths([tmp_path / "a.py"], tmp_path) == {"a.py": None}


def test_files_nest_under_their_directory(tmp_path):
    tree = group_paths([tmp_path / "tui" / "app.py"], tmp_path)
    assert tree == {"tui": {"app.py": None}}


def test_siblings_share_a_parent(tmp_path):
    tree = group_paths(
        [tmp_path / "tui" / "app.py", tmp_path / "tui" / "theme.py"], tmp_path
    )
    assert set(tree["tui"]) == {"app.py", "theme.py"}


def test_a_name_used_as_both_file_and_directory_becomes_a_directory(tmp_path):
    """Otherwise the second path would try to descend into None and crash."""
    tree = group_paths(
        [tmp_path / "tui", tmp_path / "tui" / "app.py"], tmp_path
    )
    assert isinstance(tree["tui"], dict)


def test_no_paths_is_an_empty_tree(tmp_path):
    assert group_paths([], tmp_path) == {}


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------


def test_directories_sort_before_files():
    """Same convention as `tree` and `ls --group-directories-first`."""
    ordered = sorted_items({"z.py": None, "alpha": {}})
    assert [name for name, _ in ordered] == ["alpha", "z.py"]


def test_ordering_ignores_case():
    ordered = sorted_items({"Zebra.py": None, "apple.py": None})
    assert [name for name, _ in ordered] == ["apple.py", "Zebra.py"]


# --------------------------------------------------------------------------
# tree_lines
# --------------------------------------------------------------------------


def test_no_paths_renders_nothing(tmp_path):
    assert tree_lines([], tmp_path) == []


def test_the_last_entry_uses_the_corner_glyph(tmp_path):
    lines = tree_lines([tmp_path / "a.py", tmp_path / "b.py"], tmp_path)
    assert lines[0].startswith("├── ")
    assert lines[-1].startswith("└── ")


def test_nested_files_are_indented_under_their_directory(tmp_path):
    lines = tree_lines([tmp_path / "tui" / "app.py"], tmp_path)
    assert lines == ["└── tui", "    └── app.py"]


def test_a_continuing_branch_draws_its_vertical(tmp_path):
    lines = tree_lines(
        [tmp_path / "tui" / "app.py", tmp_path / "z.py"], tmp_path
    )
    assert any(line.startswith("│   ") for line in lines)


def test_every_file_appears_once(tmp_path):
    paths = [tmp_path / "a" / "one.py", tmp_path / "a" / "two.py", tmp_path / "b.py"]
    rendered = "\n".join(tree_lines(paths, tmp_path))
    for name in ("one.py", "two.py", "b.py"):
        assert rendered.count(name) == 1


def test_a_very_long_list_is_capped_and_says_so(tmp_path):
    """The summary is a glance at what changed, not a file manager."""
    paths = [tmp_path / f"f{n}.py" for n in range(300)]
    assert "more" in tree_lines(paths, tmp_path)[-1]
