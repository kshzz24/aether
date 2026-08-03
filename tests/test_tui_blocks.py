"""Splitting model prose into prose and code, and spotting diffs.

The transcript mounts fenced code as its own widget so it can carry a copy
button. That makes fence parsing load-bearing: get it wrong and either the copy
button hands you prose, or a paragraph gets syntax-highlighted as Python.
"""

from __future__ import annotations

from tui.blocks import Code, Prose, is_long, looks_like_diff, split_blocks

# --------------------------------------------------------------------------
# Fence splitting
# --------------------------------------------------------------------------


def test_plain_prose_is_one_block():
    assert split_blocks("just a sentence") == [Prose("just a sentence")]


def test_a_fence_becomes_a_code_block():
    blocks = split_blocks("before\n```\nx = 1\n```\nafter")
    assert blocks == [Prose("before"), Code("x = 1"), Prose("after")]


def test_the_info_string_becomes_the_language():
    assert split_blocks("```python\nx = 1\n```") == [Code("x = 1", "python")]


def test_the_language_survives_extra_fence_metadata():
    """Models write ```python title=foo.py often enough to matter."""
    blocks = split_blocks("```python title=foo.py\nx = 1\n```")
    assert isinstance(blocks[0], Code)
    assert blocks[0].language.startswith("python")


def test_an_unclosed_fence_still_yields_its_code():
    """A streamed response is routinely cut mid-block; dropping the half-written
    code would lose exactly the part the user is waiting for."""
    assert split_blocks("intro\n```py\nx = 1") == [Prose("intro"), Code("x = 1", "py")]


def test_tildes_open_a_fence_too():
    assert split_blocks("~~~\nx = 1\n~~~") == [Code("x = 1")]


def test_a_backtick_fence_inside_a_tilde_fence_stays_literal():
    blocks = split_blocks("~~~\n```\nx = 1\n```\n~~~")
    assert blocks == [Code("```\nx = 1\n```")]


def test_an_empty_code_block_is_kept():
    """It is still a thing the model emitted; silently dropping it makes the
    transcript disagree with what was said."""
    assert split_blocks("```\n```") == [Code("")]


def test_blank_prose_between_two_fences_is_dropped():
    blocks = split_blocks("```\na\n```\n\n```\nb\n```")
    assert blocks == [Code("a"), Code("b")]


def test_multiple_blocks_keep_their_order():
    blocks = split_blocks("one\n```\na\n```\ntwo\n```\nb\n```\nthree")
    assert [type(b) for b in blocks] == [Prose, Code, Prose, Code, Prose]


def test_a_deeply_indented_fence_is_not_a_fence():
    """Four spaces is an indented code block in markdown, not a fence."""
    assert split_blocks("    ```\n    x = 1") == [Prose("    ```\n    x = 1")]


def test_empty_text_yields_no_blocks():
    assert split_blocks("") == []


# --------------------------------------------------------------------------
# Diff detection
# --------------------------------------------------------------------------


def test_a_hunk_header_marks_a_diff():
    assert looks_like_diff("@@ -1,3 +1,4 @@\n-old\n+new") is True


def test_a_file_header_pair_marks_a_diff():
    assert looks_like_diff("--- a/x.py\n+++ b/x.py\n-old\n+new") is True


def test_prose_with_dashes_is_not_a_diff():
    """Tool output is full of bullet lists; painting one red and green would be
    a lie about what changed."""
    assert looks_like_diff("- first item\n- second item\n+ a plus sign") is False


def test_a_lone_old_file_header_is_not_a_diff():
    assert looks_like_diff("--- a/x.py\nsome text") is False


def test_empty_text_is_not_a_diff():
    assert looks_like_diff("") is False


# --------------------------------------------------------------------------
# Folding
# --------------------------------------------------------------------------


def test_short_prose_is_not_long():
    assert is_long("one\ntwo\nthree") is False


def test_prose_past_the_threshold_is_long():
    assert is_long("\n".join(str(n) for n in range(40))) is True
