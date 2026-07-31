"""`@file` completion — pure, no app.

Ranking is the whole feature. An index of every repo file is easy; putting the
one the user meant first is the part worth testing.
"""

from __future__ import annotations

from tui.files import build_file_index, complete_mention, match_paths, split_mention

_INDEX = [
    "main.py",
    "config.py",
    "tui/app.py",
    "tui/config.py",
    "gateway/config.py",
    "gateway/server.py",
    "tests/test_gateway_cache.py",
]


# --------------------------------------------------------------------------
# Finding the mention
# --------------------------------------------------------------------------


def test_a_line_with_no_at_sign_is_not_a_mention():
    assert split_mention("read the config") is None


def test_a_trailing_mention_is_found():
    assert split_mention("look at @conf") == ("look at ", "conf")


def test_a_bare_at_sign_is_a_mention_with_an_empty_partial():
    assert split_mention("@") == ("", "")


def test_a_completed_mention_followed_by_a_space_is_not_active():
    """`@a.py and then` must not keep completing — the mention is finished."""
    assert split_mention("read @main.py and then") is None


def test_the_last_mention_wins():
    assert split_mention("@main.py @conf") == ("@main.py ", "conf")


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_a_full_path_prefix_ranks_first():
    assert match_paths("@gateway/con", _INDEX)[0] == "gateway/config.py"


def test_a_prefix_match_beats_a_basename_match():
    """`@tui` should reach tui/... before a file merely containing "tui"."""
    matches = match_paths("@tui", _INDEX)
    assert matches[0].startswith("tui/")


def test_a_basename_match_beats_a_substring_match():
    """Typing `@gateway` means the gateway/ package before a test file that
    merely mentions the word."""
    matches = match_paths("@gateway", _INDEX)
    assert matches[0].startswith("gateway/")
    assert matches.index("gateway/server.py") < matches.index(
        "tests/test_gateway_cache.py"
    )


def test_matching_is_case_insensitive():
    assert match_paths("@MAIN", _INDEX)[0] == "main.py"


def test_a_bare_at_offers_the_whole_index():
    assert match_paths("@", _INDEX)[:2] == _INDEX[:2]


def test_no_match_returns_nothing():
    assert match_paths("@zzzznope", _INDEX) == []


def test_text_without_a_mention_matches_nothing():
    assert match_paths("plain text", _INDEX) == []


# --------------------------------------------------------------------------
# Completing
# --------------------------------------------------------------------------


def test_completion_preserves_the_text_before_the_mention():
    assert complete_mention("please read @mai", _INDEX) == "please read @main.py"


def test_completion_of_a_second_mention_leaves_the_first_alone():
    completed = complete_mention("@main.py and @gateway/ser", _INDEX)
    assert completed == "@main.py and @gateway/server.py"


def test_completion_returns_none_when_nothing_matches():
    assert complete_mention("@zzzznope", _INDEX) is None


def test_completion_returns_none_without_a_mention():
    assert complete_mention("no mention here", _INDEX) is None


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------


def test_the_index_is_repo_relative_posix(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x", encoding="utf-8")
    assert "pkg/mod.py" in build_file_index(tmp_path)


def test_the_index_skips_gitignored_paths(tmp_path):
    """Offering a path the agent's tools would refuse to read is worse than
    offering nothing.

    `iter_files` resolves .gitignore relative to the *repo* root, so this needs
    a .git marker — outside a repo there is no ignore file to honour.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("secret/\n", encoding="utf-8")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "keys.txt").write_text("x", encoding="utf-8")
    (tmp_path / "visible.py").write_text("x", encoding="utf-8")

    index = build_file_index(tmp_path)
    assert "visible.py" in index
    assert "secret/keys.txt" not in index


def test_shorter_paths_come_first(tmp_path):
    """`@config` should offer the top-level file before a nested one."""
    (tmp_path / "config.py").write_text("x", encoding="utf-8")
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "config.py").write_text("x", encoding="utf-8")

    index = build_file_index(tmp_path)
    assert index.index("config.py") < index.index("deep/config.py")


def test_the_index_is_bounded(tmp_path):
    for i in range(30):
        (tmp_path / f"f{i}.py").write_text("x", encoding="utf-8")
    assert len(build_file_index(tmp_path, limit=10)) == 10
