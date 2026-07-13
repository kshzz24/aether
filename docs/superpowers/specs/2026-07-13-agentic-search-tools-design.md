# Agentic Search Tools — Design Spec

**Date:** 2026-07-13
**Status:** Approved, ready for implementation plan
**Phase context:** Pre–Phase 5 tool top-up. These three tools are the
**Floor** of FORGE's Retrieval Ladder (§6), which pins agentic `grep`/`glob`
search to Phase 5. They are added *before* the Phase 5 Approver work because they
are independent (`ToolKind.READ`, no control-flow changes) and make the eventual
`codebase-investigator` skill and the Approver's policy surface more real.

---

## 1. Motivation

Today the agent can only search a repo by shelling out through `run_shell`. That
is wrong on two axes:

1. `run_shell` is `ToolKind.EXECUTE` — once the Phase 5 Approver lands, every
   search prompts for confirmation.
2. It is not cross-platform (`grep`/`find` vs. Windows).

This adds three pure-Python, cross-platform, read-only tools so "look around the
codebase" is a first-class, zero-friction capability: **`grep`**, **`glob`**,
**`list_dir`**.

## 2. Invariants honored

- **No stdout (Invariant 1):** every tool returns a string; the renderer remains
  the only writer.
- **Errors are data (Invariant 5):** malformed input (bad regex, missing path)
  raises `ValueError`; the loop already catches tool exceptions and returns them
  as observations the model can self-correct. This mirrors `run_shell`'s handling
  of a bad `timeout`.
- **Read kind:** all three are `ToolKind.READ`, so they require no approval and
  add no policy branches to the (upcoming) Approver.

## 3. Scope

**In:** `grep`, `glob`, `list_dir` as builtin tools; a shared gitignore-aware
traversal helper; `pathspec` as a runtime dependency; unit tests.

**Out / deferred seams:**
- **Path-escape enforcement.** A READ tool can currently be handed an absolute
  `path` outside the working directory. The always-on path-escape check is
  **Phase 5 Approver** work (Model M2: expose the seam now, fill the policy in
  its phase). We do *not* build that policy here.
- **Nested per-directory `.gitignore`.** We honor the repository-root
  `.gitignore` only. Nested `.gitignore` files are a documented limitation.
- **Semantic / structural search.** `repo_map` (Phase 5.5) and embeddings
  (Phase 10) are explicitly out — this is the Floor rung, deliberately cheap.

## 4. Ignore behavior (shared decision)

All traversal:
- **Always skips `.git/`.**
- **Respects the repository-root `.gitignore`,** matched with `pathspec`
  (`gitwildmatch`). Rationale: the agent should see what git sees. This repo's
  `.venv/` alone holds ~4,364 Python files; without ignore handling the tools are
  unusable.
- If the search path is not inside a git repo (no ancestor `.git`), there is no
  ignore file to apply — traversal still skips any `.git/` it encounters.

`pathspec` was chosen over a hand-rolled matcher because correct `.gitignore`
semantics (globstars, negation, anchoring, directory-only patterns, rule
precedence) are a rabbit hole with **no curriculum payoff** — it is plumbing, not
one of Phase 5's concepts.

## 5. Shared helper — `tools/traversal.py`

Single home for the ignore logic so the three tools stay DRY.

- `find_repo_root(start: Path) -> Path | None`
  Nearest ancestor of `start` containing `.git`; else `None`.
- `load_ignore_spec(repo_root: Path | None) -> pathspec.PathSpec`
  Build a `PathSpec` from `<repo_root>/.gitignore`; empty spec if the root is
  `None` or the file is absent.
- `iter_files(root: Path) -> Iterator[Path]`
  Walk `root`; **always skip `.git/`**; skip any path the spec ignores (matched
  relative to the repo root); yield file paths. Directory pruning happens during
  the walk so ignored subtrees (e.g. `.venv/`) are never descended into.

**Concurrency:** callers invoke the walk via `asyncio.to_thread(...)` so a large
traversal never blocks the event loop (same "don't block the agent" care as
`run_shell`).

## 6. `grep` — `tools/grep.py`

- **Kind:** `READ`
- **Args:**
  | arg | type | required | default | meaning |
  |---|---|---|---|---|
  | `pattern` | string | yes | — | Python regex |
  | `path` | string | no | cwd | file or directory to search |
  | `ignore_case` | boolean | no | `false` | case-insensitive match |
  | `include` | string | no | — | glob (e.g. `*.py`) scoping which files are searched |
- **Behavior:** resolve `path`; if a directory, walk via `iter_files`; if
  `include` is set, keep only files whose relative path matches it (via
  `pathspec` `gitwildmatch`). For each **text** file, scan lines against the
  compiled regex.
  - Skip binary files (null-byte sniff on a leading chunk).
  - Skip files larger than **5 MB**.
- **Output:** grouped by file:
  ```
  path/to/file.py:12: the matched line text
  path/to/file.py:40: another match
  ```
  Caps: **50 matches per file**, **1,000 matches total**; on hitting a cap append
  `... truncated (N more)`. No matches → the literal string `"no matches"`.
- **Errors:** a malformed regex raises `ValueError` (returned to the model as an
  observation, not a crash).

## 7. `glob` — `tools/glob.py`

- **Kind:** `READ`
- **Args:**
  | arg | type | required | default | meaning |
  |---|---|---|---|---|
  | `pattern` | string | yes | — | path glob, e.g. `**/*.py`, `src/**/*.ts` |
  | `path` | string | no | cwd | root directory to search from |
- **Behavior:** walk `path` via `iter_files` (already gitignore-aware); keep
  files whose repo-relative path matches `pattern`, compiled with `pathspec`'s
  `gitwildmatch` so `**` behaves correctly and the same engine is reused.
- **Output:** matching relative paths, **sorted**, one per line. Cap **1,000**,
  then `... truncated (N more)`. None → `"no files match"`.

## 8. `list_dir` — `tools/list_dir.py`

- **Kind:** `READ`
- **Args:**
  | arg | type | required | default | meaning |
  |---|---|---|---|---|
  | `path` | string | no | cwd | directory to list |
- **Behavior:** **non-recursive** — immediate children only (recursion is
  `glob`'s job; this keeps the tools cleanly separated). Apply gitignore
  filtering; skip `.git`. Directories are suffixed with `/`. Sort **directories
  first, then files**, each alphabetically. Cap **500** entries.
- **Output:** newline-separated listing. Empty (or fully-ignored) directory →
  `"(empty)"`. A `path` that is not a directory raises `ValueError`.

## 9. Module contract & registration

Each tool follows the existing module shape (cf. `tools/read_file.py`):

```python
KIND = ToolKind.READ
SCHEMA = {"name": ..., "description": ..., "parameters": {...}}
async def run(args: dict) -> str: ...
```

- Register by adding `grep`, `glob`, `list_dir` to `_BUILTIN_MODULES` in
  `tools/__init__.py`.
- Add `pathspec>=0.12` to `[project].dependencies` in `pyproject.toml`.

## 10. Testing (pytest — ≥1 per tool, per CLAUDE.md)

Fixture: a `tmp_path` fake repo containing a `.git/` directory, a `.gitignore`
(ignoring e.g. `ignored/` and `*.log`), and a mix of tracked + ignored files
(including one binary and one > 5 MB stub for the skip paths).

- **traversal:** `.git/` and gitignored paths are excluded; ignored subtrees are
  not descended.
- **grep:** regex match; `ignore_case`; `include` filter; per-file and total
  caps produce the truncation line; binary + oversize files skipped; `"no
  matches"` on miss; malformed regex raises `ValueError`.
- **glob:** `**` matches across directories; sorted output; `"no files match"`
  on miss; gitignored files never returned.
- **list_dir:** non-recursive (nested files absent); dirs-first ordering with `/`
  suffix; `"(empty)"` for an empty dir; `ValueError` for a non-directory path.

## 11. Order of implementation

1. `pathspec` dependency + `tools/traversal.py` + its tests.
2. `grep` + tests.
3. `glob` + tests.
4. `list_dir` + tests.
5. Register all three in `tools/__init__.py`; full suite green.
