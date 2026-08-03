"""Undo for the agent's file writes.

The subtle requirement, and the one most of these tests are about: a snapshot is
*pending* until `after_tool` confirms the write landed. The agent turns every
tool failure into an observation string rather than raising (`agent.py:236`), so
a denied or errored write looks exactly like a successful one from the outside
except for that string. Get this wrong and `/undo` "restores" a file nobody
changed, clobbering whatever is actually there.
"""

from __future__ import annotations

from tui.undo import MAX_SNAPSHOT_BYTES, UndoStack


def _write(stack: UndoStack, path, content: str, *, result: str = "ok") -> None:
    """One successful write_file round-trip through the hooks."""
    args = {"path": str(path), "content": content}
    stack.before_tool("write_file", args)
    if not (result.startswith("ERROR") or result.startswith("DENIED")):
        path.write_text(content, encoding="utf-8")
    stack.after_tool("write_file", args, result)


# --------------------------------------------------------------------------
# Capturing
# --------------------------------------------------------------------------


def test_undoing_an_edit_restores_the_previous_content(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("original", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "replaced")
    assert target.read_text(encoding="utf-8") == "replaced"

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "original"


def test_undoing_a_newly_created_file_deletes_it(tmp_path):
    """There is no previous content to restore; leaving the file behind would
    make undo a lie."""
    target = tmp_path / "new.py"
    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "hello")

    result = stack.undo_last()
    assert not target.exists()
    assert target in result.deleted


def test_edit_file_is_snapshotted_too(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("original", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    args = {"path": str(target), "old_string": "original", "new_string": "changed"}
    stack.before_tool("edit_file", args)
    target.write_text("changed", encoding="utf-8")
    stack.after_tool("edit_file", args, "ok")

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "original"


def test_a_read_only_tool_is_not_snapshotted(tmp_path):
    stack = UndoStack()
    stack.start_batch()
    stack.before_tool("read_file", {"path": str(tmp_path / "a.py")})
    stack.after_tool("read_file", {"path": str(tmp_path / "a.py")}, "contents")
    assert stack.depth == 0


def test_a_shell_command_is_not_claimed_as_undoable(tmp_path):
    """`run_shell` can do anything; pretending its effects are revertible would
    be the most dangerous kind of wrong."""
    stack = UndoStack()
    stack.start_batch()
    args = {"command": "rm -rf build"}
    stack.before_tool("run_shell", args)
    stack.after_tool("run_shell", args, "done")
    assert stack.touched() == []


def test_binary_content_survives_a_round_trip(tmp_path):
    """Snapshots are bytes: an undo that silently transcodes a file is a worse
    bug than no undo at all."""
    target = tmp_path / "logo.png"
    original = b"\x89PNG\r\n\x1a\n\xff\xfe"
    target.write_bytes(original)

    stack = UndoStack()
    stack.start_batch()
    args = {"path": str(target), "content": "clobbered"}
    stack.before_tool("write_file", args)
    target.write_text("clobbered", encoding="utf-8")
    stack.after_tool("write_file", args, "ok")

    stack.undo_last()
    assert target.read_bytes() == original


# --------------------------------------------------------------------------
# Failures must not become undo entries
# --------------------------------------------------------------------------


def test_a_denied_write_is_not_undoable(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("original", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "never applied", result="DENIED: user declined")

    assert stack.depth == 0
    assert stack.undo_last().changed == 0
    assert target.read_text(encoding="utf-8") == "original"


def test_a_failed_write_is_not_undoable(tmp_path):
    stack = UndoStack()
    stack.start_batch()
    _write(stack, tmp_path / "a.py", "x", result="ERROR: permission denied")
    assert stack.depth == 0


def test_an_unreadable_path_does_not_raise(tmp_path):
    """A snapshot failing must never stop the agent's write."""
    stack = UndoStack()
    stack.start_batch()
    stack.before_tool("write_file", {"path": str(tmp_path)})  # a directory
    stack.after_tool("write_file", {"path": str(tmp_path)}, "ok")
    assert stack.depth == 0


def test_a_missing_path_argument_is_ignored():
    stack = UndoStack()
    stack.start_batch()
    stack.before_tool("write_file", {"content": "x"})
    stack.after_tool("write_file", {"content": "x"}, "ok")
    assert stack.depth == 0


def test_a_huge_file_is_recorded_but_not_snapshotted(tmp_path):
    """Holding hundreds of megabytes in memory to make undo work is a bad
    trade — but the file must still show up as touched, and the skip must be
    reported rather than silently doing nothing."""
    target = tmp_path / "big.bin"
    target.write_bytes(b"0" * (MAX_SNAPSHOT_BYTES + 1))

    stack = UndoStack()
    stack.start_batch()
    args = {"path": str(target), "content": "x"}
    stack.before_tool("write_file", args)
    stack.after_tool("write_file", args, "ok")

    assert target in stack.touched()
    assert target in stack.undo_last().skipped


# --------------------------------------------------------------------------
# Batching — /undo reverts a turn, not a single write
# --------------------------------------------------------------------------


def test_one_undo_reverts_every_write_in_the_turn(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("a0", encoding="utf-8")
    b.write_text("b0", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, a, "a1")
    _write(stack, b, "b1")

    stack.undo_last()
    assert a.read_text(encoding="utf-8") == "a0"
    assert b.read_text(encoding="utf-8") == "b0"


def test_undo_only_reverts_the_newest_turn(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "v1")
    stack.start_batch()
    _write(stack, target, "v2")

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "v1"
    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "v0"


def test_two_edits_to_one_file_unwind_to_the_state_before_both(tmp_path):
    """Within a turn the writes must unwind in reverse order, or the file ends
    up at the intermediate state instead of the original."""
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "v1")
    _write(stack, target, "v2")

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "v0"


def test_empty_turns_are_skipped(tmp_path):
    """Turns where the agent only read files must not make /undo a no-op that
    the user has to press repeatedly."""
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "v1")
    stack.start_batch()
    stack.start_batch()

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "v0"


def test_undo_with_nothing_recorded_is_harmless():
    result = UndoStack().undo_last()
    assert result.changed == 0
    assert result.summary() == "nothing to undo"


# --------------------------------------------------------------------------
# Redo
# --------------------------------------------------------------------------


def test_redo_puts_the_agents_version_back(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("original", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "agent version")

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "original"
    stack.redo_last()
    assert target.read_text(encoding="utf-8") == "agent version"


def test_redo_of_a_created_file_recreates_it(tmp_path):
    target = tmp_path / "new.py"
    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "hello")

    stack.undo_last()
    assert not target.exists()
    stack.redo_last()
    assert target.read_text(encoding="utf-8") == "hello"


def test_undo_and_redo_can_alternate(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")
    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "v1")

    for _ in range(3):
        stack.undo_last()
        assert target.read_text(encoding="utf-8") == "v0"
        stack.redo_last()
        assert target.read_text(encoding="utf-8") == "v1"


def test_redo_with_nothing_undone_is_harmless():
    result = UndoStack().redo_last()
    assert result.changed == 0
    assert result.summary("nothing to redo") == "nothing to redo"


def test_new_work_invalidates_the_redo_stack(tmp_path):
    """Redoing onto a file the agent has since rewritten would silently discard
    the newer version — the same rule as an editor's undo history."""
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "v1")
    stack.undo_last()
    assert stack.redo_depth == 1

    stack.start_batch()
    assert stack.redo_depth == 0
    _write(stack, target, "v2")
    stack.redo_last()
    assert target.read_text(encoding="utf-8") == "v2", "redo clobbered newer work"


def test_redo_restores_a_whole_turn(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("a0", encoding="utf-8")
    b.write_text("b0", encoding="utf-8")

    stack = UndoStack()
    stack.start_batch()
    _write(stack, a, "a1")
    _write(stack, b, "b1")

    stack.undo_last()
    stack.redo_last()
    assert a.read_text(encoding="utf-8") == "a1"
    assert b.read_text(encoding="utf-8") == "b1"


def test_a_redone_turn_can_be_undone_again(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")
    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "v1")

    stack.undo_last()
    stack.redo_last()
    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "v0"


def test_history_is_bounded(tmp_path):
    """Undo is a safety net, not version control — git is right there."""
    stack = UndoStack(max_batches=3)
    for n in range(10):
        stack.start_batch()
        _write(stack, tmp_path / f"f{n}.py", "x")
    assert stack.depth <= 3


def test_a_write_before_any_batch_is_still_captured(tmp_path):
    """A goal supplied on argv starts before the first StatusEvent."""
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")

    stack = UndoStack()
    _write(stack, target, "v1")

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "v0"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_touched_lists_the_most_recent_first(tmp_path):
    stack = UndoStack()
    stack.start_batch()
    _write(stack, tmp_path / "a.py", "x")
    _write(stack, tmp_path / "b.py", "x")
    assert [p.name for p in stack.touched()] == ["b.py", "a.py"]


def test_touched_does_not_repeat_a_file(tmp_path):
    stack = UndoStack()
    stack.start_batch()
    _write(stack, tmp_path / "a.py", "x")
    _write(stack, tmp_path / "a.py", "y")
    assert len(stack.touched()) == 1


def test_the_summary_names_what_happened(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")
    stack = UndoStack()
    stack.start_batch()
    _write(stack, target, "v1")
    assert "restored 1" in stack.undo_last().summary()


def test_the_hooks_bundle_is_wired_to_the_stack(tmp_path):
    """build_composition takes a Hooks; this is the adapter that makes the
    Phase-2 seam carry the snapshotter."""
    stack = UndoStack()
    hooks = stack.hooks()
    target = tmp_path / "a.py"
    target.write_text("v0", encoding="utf-8")

    args = {"path": str(target), "content": "v1"}
    hooks.before_tool("write_file", args)
    target.write_text("v1", encoding="utf-8")
    hooks.after_tool("write_file", args, "ok")

    stack.undo_last()
    assert target.read_text(encoding="utf-8") == "v0"
