# Agentic Search Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three cross-platform, read-only search builtins — `grep`, `glob`, `list_dir` — so the agent can look around a repo without shelling out through `run_shell`.

**Architecture:** A single gitignore-aware traversal helper (`tools/traversal.py`) does the walking; the three tools are thin modules over it, each following the existing `SCHEMA`/`KIND`/`run` module contract and registered in `tools/__init__.py`. All ignore/`.git` logic lives in the helper (DRY).

**Tech Stack:** Python 3.11+, `pathspec` for `.gitignore` matching, stdlib `os.walk`/`re`, `pytest` + `pytest-asyncio` (already configured, `asyncio_mode=auto`).

## Global Constraints

- Python **3.11+**, full type hints, small single-purpose functions (CLAUDE.md conventions).
- All three tools are `ToolKind.READ`; they must never write to stdout (return strings only — Invariant 1).
- Malformed input raises `ValueError`; the loop turns it into an observation (Invariant 5). No bare `except: pass` that hides real bugs.
- Ignore behavior: **always skip `.git/`**; respect the **repository-root** `.gitignore` via `pathspec` `gitwildmatch`. Nested `.gitignore` files are out of scope.
- Caps (verbatim): grep **50 matches/file**, **1,000 total**; glob **1,000** paths; list_dir **500** entries. Skip binary files (null-byte sniff) and files **> 5 MB** in grep.
- Module contract: each tool module exposes top-level `KIND`, `SCHEMA` (with `name`/`description`/`parameters`), and `async def run(args: dict) -> str`.
- Run all blocking traversal via `asyncio.to_thread` so it never blocks the event loop.

---

## File Structure

- `pyproject.toml` — add `pathspec>=0.12` to `[project].dependencies`.
- `tools/traversal.py` — **create.** `find_repo_root`, `load_ignore_spec`, `iter_files`.
- `tools/grep.py` — **create.** Regex content search.
- `tools/glob.py` — **create.** Path-pattern file discovery.
- `tools/list_dir.py` — **create.** Non-recursive directory listing.
- `tools/__init__.py` — **modify.** Register the three new modules.
- `tests/test_search_tools.py` — **create.** Fixture + tests for all four units.

---

## Task 1: Dependency + gitignore-aware traversal helper

**Files:**
- Modify: `pyproject.toml` (dependencies list)
- Create: `tools/traversal.py`
- Test: `tests/test_search_tools.py`

**Interfaces:**
- Produces:
  - `find_repo_root(start: Path) -> Path | None`
  - `load_ignore_spec(repo_root: Path | None) -> pathspec.PathSpec`
  - `iter_files(root: Path) -> Iterator[Path]` — yields absolute file paths under `root`, skipping `.git/` and gitignored paths.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to `[project].dependencies`:
```toml
    "pathspec>=0.12",
```
Then install:
```bash
pip install "pathspec>=0.12"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_search_tools.py`:
```python
from pathlib import Path

import pytest

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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_search_tools.py::test_iter_files_excludes_git_and_ignored -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.traversal'`.

- [ ] **Step 4: Write the implementation**

Create `tools/traversal.py`:
```python
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pathspec

_ALWAYS_SKIP = {".git"}


def find_repo_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` containing a ``.git`` entry, else None."""
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def load_ignore_spec(repo_root: Path | None) -> pathspec.PathSpec:
    """Build a PathSpec from ``<repo_root>/.gitignore`` (empty if absent)."""
    if repo_root is None:
        return pathspec.PathSpec.from_lines("gitwildmatch", [])
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return pathspec.PathSpec.from_lines("gitwildmatch", [])
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def iter_files(root: Path) -> Iterator[Path]:
    """Yield files under ``root``, skipping ``.git/`` and gitignored paths.

    Ignored directories are pruned during the walk so their subtrees are never
    descended into. Paths are matched relative to the repo root (git semantics),
    falling back to ``root`` when ``root`` is not inside a git repo.
    """
    root = root.resolve()
    repo_root = find_repo_root(root)
    spec = load_ignore_spec(repo_root)
    base = repo_root or root

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        kept = []
        for d in dirnames:
            if d in _ALWAYS_SKIP:
                continue
            rel = (current / d).relative_to(base).as_posix()
            if spec.match_file(rel) or spec.match_file(rel + "/"):
                continue
            kept.append(d)
        dirnames[:] = kept  # in-place prune: os.walk won't descend removed dirs

        for f in filenames:
            rel = (current / f).relative_to(base).as_posix()
            if spec.match_file(rel):
                continue
            yield current / f
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_search_tools.py::test_iter_files_excludes_git_and_ignored -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tools/traversal.py tests/test_search_tools.py
git commit -m "tools: gitignore-aware traversal helper + pathspec dep"
```

---

## Task 2: `grep` — regex content search

**Files:**
- Create: `tools/grep.py`
- Test: `tests/test_search_tools.py` (append)

**Interfaces:**
- Consumes: `tools.traversal.iter_files`.
- Produces: module `tools.grep` with `KIND`, `SCHEMA`, `async def run(args: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_tools.py`:
```python
from tools import grep


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search_tools.py -k grep -v`
Expected: FAIL — `ImportError: cannot import name 'grep' from 'tools'`.

- [ ] **Step 3: Write the implementation**

Create `tools/grep.py`:
```python
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pathspec

from tools.base import ToolKind
from tools.traversal import iter_files

KIND = ToolKind.READ
SCHEMA = {
    "name": "grep",
    "description": (
        "Search file contents with a Python regular expression. Returns matching "
        "lines grouped by file as 'path:line: text'. Respects .gitignore, skips "
        ".git, binary files, and files over 5 MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"},
            "path": {"type": "string", "description": "file or directory (default cwd)"},
            "ignore_case": {"type": "boolean"},
            "include": {
                "type": "string",
                "description": "glob to scope files searched, e.g. *.py",
            },
        },
        "required": ["pattern"],
    },
}

_MAX_PER_FILE = 50
_MAX_TOTAL = 1000
_MAX_BYTES = 5 * 1024 * 1024


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(1024)
    except OSError:
        return True


def _display_path(f: Path) -> str:
    try:
        return f.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return f.as_posix()


def _candidates(root: Path, include: str | None) -> list[Path]:
    files = [root] if root.is_file() else list(iter_files(root))
    if include:
        spec = pathspec.PathSpec.from_lines("gitwildmatch", [include])
        files = [f for f in files if spec.match_file(f.as_posix())]
    return sorted(files)


def _search(regex: re.Pattern[str], root: Path, include: str | None) -> str:
    out: list[str] = []
    total = 0
    for f in _candidates(root, include):
        if total >= _MAX_TOTAL:
            break
        try:
            if f.stat().st_size > _MAX_BYTES or _is_binary(f):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = _display_path(f)
        per_file = 0
        for i, line in enumerate(text.splitlines(), start=1):
            if not regex.search(line):
                continue
            if per_file >= _MAX_PER_FILE:
                out.append(f"... truncated (per-file cap reached in {rel})")
                break
            out.append(f"{rel}:{i}: {line}")
            per_file += 1
            total += 1
            if total >= _MAX_TOTAL:
                out.append("... truncated (1000-match total cap reached)")
                break
    return "\n".join(out) if out else "no matches"


async def run(args: dict) -> str:
    flags = re.IGNORECASE if args.get("ignore_case") else 0
    try:
        regex = re.compile(args["pattern"], flags)
    except re.error as e:
        raise ValueError(f"invalid regex {args['pattern']!r}: {e}") from None

    root = Path(args.get("path") or ".")
    if not root.exists():
        raise ValueError(f"path does not exist: {root}")

    return await asyncio.to_thread(_search, regex, root, args.get("include"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search_tools.py -k grep -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/grep.py tests/test_search_tools.py
git commit -m "tools: add grep (regex content search)"
```

---

## Task 3: `glob` — path-pattern file discovery

**Files:**
- Create: `tools/glob.py`
- Test: `tests/test_search_tools.py` (append)

**Interfaces:**
- Consumes: `tools.traversal.iter_files`.
- Produces: module `tools.glob` with `KIND`, `SCHEMA`, `async def run(args: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_tools.py`:
```python
from tools import glob


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search_tools.py -k glob -v`
Expected: FAIL — `ImportError: cannot import name 'glob' from 'tools'`.

- [ ] **Step 3: Write the implementation**

Create `tools/glob.py`:
```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pathspec

from tools.base import ToolKind
from tools.traversal import iter_files

KIND = ToolKind.READ
SCHEMA = {
    "name": "glob",
    "description": (
        "Find files by path glob (e.g. '**/*.py', 'src/**/*.ts'). Returns matching "
        "paths sorted, one per line. Respects .gitignore and skips .git."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "path glob, e.g. **/*.py"},
            "path": {"type": "string", "description": "root directory (default cwd)"},
        },
        "required": ["pattern"],
    },
}

_MAX = 1000


def _rel(f: Path, root: Path) -> str:
    try:
        return f.relative_to(root).as_posix()
    except ValueError:
        return f.as_posix()


def _glob(pattern: str, root: Path) -> str:
    spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
    matches = sorted(
        rel for f in iter_files(root) if spec.match_file(rel := _rel(f, root))
    )
    if not matches:
        return "no files match"
    out = matches[:_MAX]
    if len(matches) > _MAX:
        out.append(f"... truncated ({len(matches) - _MAX} more)")
    return "\n".join(out)


async def run(args: dict) -> str:
    root = Path(args.get("path") or ".").resolve()
    if not root.exists():
        raise ValueError(f"path does not exist: {root}")
    return await asyncio.to_thread(_glob, args["pattern"], root)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search_tools.py -k glob -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/glob.py tests/test_search_tools.py
git commit -m "tools: add glob (path-pattern file discovery)"
```

---

## Task 4: `list_dir` — non-recursive directory listing

**Files:**
- Create: `tools/list_dir.py`
- Test: `tests/test_search_tools.py` (append)

**Interfaces:**
- Consumes: `tools.traversal.find_repo_root`, `tools.traversal.load_ignore_spec`.
- Produces: module `tools.list_dir` with `KIND`, `SCHEMA`, `async def run(args: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_tools.py`:
```python
from tools import list_dir


async def test_list_dir_non_recursive(repo):
    lines = (await list_dir.run({"path": str(repo)})).splitlines()
    assert "src/" in lines
    assert "a.py" in lines
    assert "notes.log" not in lines            # gitignored
    assert not any(".git" in ln for ln in lines)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search_tools.py -k list_dir -v`
Expected: FAIL — `ImportError: cannot import name 'list_dir' from 'tools'`.

- [ ] **Step 3: Write the implementation**

Create `tools/list_dir.py`:
```python
from __future__ import annotations

import asyncio
from pathlib import Path

from tools.base import ToolKind
from tools.traversal import find_repo_root, load_ignore_spec

KIND = ToolKind.READ
SCHEMA = {
    "name": "list_dir",
    "description": (
        "List the immediate contents of a directory (non-recursive). Directories "
        "are suffixed with '/'. Respects .gitignore and skips .git."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "directory to list (default cwd)"},
        },
        "required": [],
    },
}

_MAX = 500


def _list(root: Path) -> str:
    repo_root = find_repo_root(root)
    spec = load_ignore_spec(repo_root)
    base = repo_root or root
    dirs: list[str] = []
    files: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.name == ".git":
            continue
        rel = entry.relative_to(base).as_posix()
        if entry.is_dir():
            if spec.match_file(rel) or spec.match_file(rel + "/"):
                continue
            dirs.append(entry.name + "/")
        else:
            if spec.match_file(rel):
                continue
            files.append(entry.name)
    combined = dirs + files
    if not combined:
        return "(empty)"
    out = combined[:_MAX]
    if len(combined) > _MAX:
        out.append(f"... truncated ({len(combined) - _MAX} more)")
    return "\n".join(out)


async def run(args: dict) -> str:
    root = Path(args.get("path") or ".").resolve()
    if not root.exists():
        raise ValueError(f"path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    return await asyncio.to_thread(_list, root)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search_tools.py -k list_dir -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/list_dir.py tests/test_search_tools.py
git commit -m "tools: add list_dir (non-recursive directory listing)"
```

---

## Task 5: Register the tools as builtins

**Files:**
- Modify: `tools/__init__.py`
- Test: `tests/test_search_tools.py` (append)

**Interfaces:**
- Consumes: `tools.grep`, `tools.glob`, `tools.list_dir`, `config.ForgeConfig`, `tools.build_registry`.
- Produces: nothing new — makes the three tools discoverable through `build_registry`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_search_tools.py`:
```python
from config import ForgeConfig
from tools import build_registry
from tools.base import ToolKind


def test_search_tools_registered(tmp_path):
    config = ForgeConfig(user_tools_dir=tmp_path / "no_user_tools")
    registry = build_registry(config)
    for name in ("grep", "glob", "list_dir"):
        assert registry.get(name).kind is ToolKind.READ
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_tools.py::test_search_tools_registered -v`
Expected: FAIL — registry raises on unknown tool `grep` (or `KeyError`).

- [ ] **Step 3: Wire them in**

In `tools/__init__.py`, update the import and the builtin list:
```python
from tools import edit_file, glob, grep, list_dir, read_file, run_shell, write_file

_BUILTIN_MODULES = [
    read_file,
    write_file,
    run_shell,
    edit_file,
    grep,
    glob,
    list_dir,
]
```

- [ ] **Step 4: Run the full suite to verify everything is green**

Run: `pytest tests/test_search_tools.py -v`
Expected: PASS (all tests).
Then the whole suite: `pytest -q`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py tests/test_search_tools.py
git commit -m "tools: register grep/glob/list_dir as builtins"
```

---

## Self-Review

**Spec coverage:**
- §4 ignore behavior (skip `.git`, root `.gitignore`, pathspec) → Task 1.
- §5 traversal helper (`find_repo_root`, `load_ignore_spec`, `iter_files`, `to_thread`) → Task 1 + used with `to_thread` in Tasks 2–4.
- §6 grep (regex, `path`/`ignore_case`/`include`, caps, binary + 5 MB skip, `"no matches"`, `ValueError`) → Task 2.
- §7 glob (`**` via gitwildmatch, sorted, 1,000 cap, `"no files match"`) → Task 3.
- §8 list_dir (non-recursive, dirs-first with `/`, 500 cap, `"(empty)"`, `ValueError` on non-dir) → Task 4.
- §9 module contract + registration + `pathspec` dep → Task 1 (dep) + Task 5 (register).
- §10 testing (fixture with `.git`, `.gitignore`, binary; all asserted behaviors) → Tasks 1–5.

**Placeholder scan:** none — every code step contains complete, runnable code.

**Type consistency:** `iter_files`, `find_repo_root`, `load_ignore_spec` names/signatures match between Task 1 and their consumers in Tasks 2–4; every tool exposes exactly `KIND`/`SCHEMA`/`run(args: dict) -> str` as `Tool.from_module` expects.

**Deferred (per spec, intentionally not in any task):** path-escape enforcement (Phase 5 Approver) and nested `.gitignore`.
