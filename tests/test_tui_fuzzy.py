"""The fuzzy scorer behind ctrl+r, /prompt, and any future picker.

Ranking is the whole product here — a matcher that returns the right *set* in
the wrong order is a picker you have to read instead of one you can trust the
first row of. So most of these tests are about order, not membership.
"""

from __future__ import annotations

from tui.fuzzy import rank, score

# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_a_subsequence_matches():
    assert score("cfg", "config.toml") is not None


def test_a_non_subsequence_does_not_match():
    assert score("xyz", "config.toml") is None


def test_order_matters():
    """Fuzzy is a subsequence, not a bag of characters."""
    assert score("gfc", "config.toml") is None


def test_matching_is_case_insensitive():
    assert score("TUI", "tui/app.py") is not None


def test_an_empty_needle_matches_everything():
    """An empty search box should show the full list, not an empty one."""
    assert score("", "anything") == 0


def test_no_match_is_none_not_zero():
    """The caller filters on None and sorts on the number; conflating them
    would put every non-match at the top of the list."""
    assert score("zzzz", "abc") is None
    assert score("abc", "abc") != 0


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_contiguous_beats_scattered():
    assert score("app", "app.py") > score("app", "a-p-p.py")


def test_a_word_boundary_beats_the_middle_of_a_word():
    assert score("tc", "tui/commands.py") > score("tc", "attach.py")


def test_a_shorter_candidate_wins_a_tie():
    assert score("test", "test.py") > score("test", "test_of_a_much_longer_name.py")


def test_an_early_match_beats_a_late_one():
    assert score("cfg", "cfg_loader.py") > score("cfg", "a/very/long/path/cfg.py")


def test_rank_drops_non_matches():
    assert rank("cfg", ["config.toml", "readme.md", "cfg.py"]) == [
        "cfg.py",
        "config.toml",
    ]


def test_rank_puts_the_obvious_answer_first():
    items = ["tui/transcript.py", "tui/commands.py", "tests/test_tui_commands.py"]
    assert rank("tuicmd", items)[0] == "tui/commands.py"


def test_rank_is_stable_for_equal_scores():
    """Two identical candidates must not swap places between keystrokes."""
    assert rank("a", ["xa", "ya"]) == ["xa", "ya"]


def test_rank_with_an_empty_needle_preserves_input_order():
    items = ["b", "a", "c"]
    assert rank("", items) == items


def test_rank_of_nothing_is_nothing():
    assert rank("x", []) == []
