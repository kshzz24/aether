import pytest

from tools import glob, grep, list_dir
from tools.traversal import iter_files


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / ".gitignore").write_text("ignored/\n*.log\n")
    (tmp_path / "a.py").write_text("import os\ndef foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    pass\n")
    (tmp_path / "notes.log").write_text("secret\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "c.py").write_text("def baz(): pass\n")
    src = tmp_path / "src"
    src.mkdir()
    (src / "d.py").write_text("def foo():\n    return 2\n")
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02foo")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_iter_files_excludes_git_and_ignored(repo):
    names = {p.name for p in iter_files(repo)}
    assert "a.py" in names
    assert "d.py" in names
    assert "notes.log" not in names   # *.log ignored
    assert "c.py" not in names        # ignored/ dir pruned
    assert all(".git" not in p.parts for p in iter_files(repo))


async def test_grep_regex_matches(repo):
    out = await grep.run({"pattern": r"def foo"})
    assert "a.py:2: def foo():" in out
    assert "src/d.py:1: def foo():" in out
    assert "notes.log" not in out  # ignored file never searched


async def test_grep_ignore_case(repo):
    out = await grep.run({"pattern": "IMPORT", "ignore_case": True})
    assert "a.py:1: import os" in out


async def test_grep_include_filter(repo):
    out = await grep.run({"pattern": "def", "include": "b.py"})
    assert "b.py:1: def bar():" in out
    assert "a.py" not in out


async def test_grep_no_matches(repo):
    assert await grep.run({"pattern": "zzz_nope"}) == "no matches"


async def test_grep_invalid_regex(repo):
    with pytest.raises(ValueError):
        await grep.run({"pattern": "("})


async def test_grep_skips_binary(repo):
    out = await grep.run({"pattern": "foo"})
    assert "bin.dat" not in out


async def test_grep_per_file_cap(repo):
    (repo / "many.py").write_text("\n".join("x" for _ in range(80)))
    out = await grep.run({"pattern": "x", "include": "many.py"})
    numbered = [
        ln for ln in out.splitlines()
        if ln.startswith("many.py:") and ln.split(":", 2)[1].isdigit()
    ]
    assert len(numbered) == 50
    assert "truncated" in out


async def test_glob_recursive(repo):
    lines = set((await glob.run({"pattern": "**/*.py"})).splitlines())
    assert "a.py" in lines
    assert "src/d.py" in lines
    assert "ignored/c.py" not in lines  # gitignored, never returned


async def test_glob_sorted(repo):
    lines = (await glob.run({"pattern": "*.py"})).splitlines()
    assert lines == sorted(lines)


async def test_glob_no_match(repo):
    assert await glob.run({"pattern": "**/*.rs"}) == "no files match"


async def test_list_dir_non_recursive(repo):
    lines = (await list_dir.run({"path": str(repo)})).splitlines()
    assert "src/" in lines
    assert "a.py" in lines
    assert "notes.log" not in lines            # gitignored
    assert ".git/" not in lines                # .git dir skipped (.gitignore file is fine)
    assert "d.py" not in lines                 # nested: not listed
    assert lines.index("src/") < lines.index("a.py")  # dirs first


async def test_list_dir_empty(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.chdir(tmp_path)
    assert await list_dir.run({"path": str(d)}) == "(empty)"


async def test_list_dir_not_a_dir(repo):
    with pytest.raises(ValueError):
        await list_dir.run({"path": str(repo / "a.py")})
