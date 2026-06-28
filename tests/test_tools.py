import asyncio

import pytest


def drive(coro):
    """Run a single coroutine to completion and return its result."""
    return asyncio.run(coro)


def test_read_file_returns_contents(tmp_path):
    from tools import read_file

    f = tmp_path / "hello.txt"
    f.write_text("hi there", encoding="utf-8")
    result = drive(read_file.run({"path": str(f)}))
    assert result == "hi there"


def test_read_file_missing_raises(tmp_path):
    from tools import read_file

    with pytest.raises(FileNotFoundError):
        drive(read_file.run({"path": str(tmp_path / "nope.txt")}))


def test_write_file_creates_file_and_parent_dirs(tmp_path):
    from tools import write_file

    target = tmp_path / "sub" / "dir" / "out.txt"
    msg = drive(write_file.run({"path": str(target), "content": "data"}))
    assert target.read_text(encoding="utf-8") == "data"
    assert target.parent.is_dir()
    assert "out.txt" in msg


def test_run_shell_captures_output_and_zero_exit():
    from tools import run_shell

    result = drive(run_shell.run({"command": "echo hello"}))
    assert "hello" in result
    assert "exit_code: 0" in result


def test_run_shell_nonzero_exit_is_data_not_error():
    from tools import run_shell

    # `exit 3` works under both cmd.exe and sh; no dependency on PATH.
    result = drive(run_shell.run({"command": "exit 3"}))
    assert "exit_code: 3" in result  # returned as data, NOT raised


def test_edit_file_unique_replace_changes_file_and_returns_diff(tmp_path):
    from tools import edit_file

    f = tmp_path / "code.py"
    f.write_text("a = 1\nb = 2\n", encoding="utf-8")
    diff = drive(edit_file.run(
        {"path": str(f), "old_string": "b = 2", "new_string": "b = 3"}
    ))
    assert f.read_text(encoding="utf-8") == "a = 1\nb = 3\n"
    assert "-b = 2" in diff
    assert "+b = 3" in diff
    assert "-a = 1" not in diff  # the unchanged line must NOT appear as removed


def test_edit_file_not_found_raises(tmp_path):
    from tools import edit_file

    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        drive(edit_file.run(
            {"path": str(f), "old_string": "zzz", "new_string": "y"}
        ))


def test_edit_file_non_unique_raises(tmp_path):
    from tools import edit_file

    f = tmp_path / "code.py"
    f.write_text("dup\ndup\n", encoding="utf-8")
    with pytest.raises(ValueError):
        drive(edit_file.run(
            {"path": str(f), "old_string": "dup", "new_string": "y"}
        ))


def test_build_tools_returns_four_named_tools():
    from agent import Tool
    from tools import build_tools

    tools = build_tools()
    assert set(tools) == {"read_file", "write_file", "run_shell", "edit_file"}
    assert all(isinstance(t, Tool) for t in tools.values())
    assert all(key == tool.schema["name"] for key, tool in tools.items())
