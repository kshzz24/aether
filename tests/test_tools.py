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


def test_run_shell_accepts_numeric_string_timeout():
    from tools import run_shell

    # Models routinely emit numbers as strings; a well-formed one must not
    # blow up in asyncio — it should be coerced and the command should run.
    result = drive(run_shell.run({"command": "echo hi", "timeout": "5"}))
    assert "hi" in result
    assert "exit_code: 0" in result


def test_run_shell_rejects_non_numeric_timeout():
    from tools import run_shell

    # Junk timeout is a malformed arg -> clear ValueError (agent turns it into
    # an observation), NOT a cryptic TypeError from deep inside asyncio.
    with pytest.raises(ValueError):
        drive(run_shell.run({"command": "echo hi", "timeout": "soon"}))


def test_run_shell_rejects_non_positive_timeout():
    from tools import run_shell

    with pytest.raises(ValueError):
        drive(run_shell.run({"command": "echo hi", "timeout": 0}))


def test_run_shell_timeout_raises_without_hanging():
    import sys
    import time

    from tools import run_shell

    # A command that outlives its timeout must be killed and surface a clean
    # TimeoutError fast, not hang the agent. Python is portable across shells.
    cmd = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        drive(run_shell.run({"command": cmd, "timeout": 1}))
    elapsed = time.monotonic() - start
    # The agent must not block for the command's full lifetime. Killing only
    # the shell leaves the child holding the pipes and wait() blocks ~30s.
    assert elapsed < 10, f"run_shell hung for {elapsed:.1f}s after timeout"


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
