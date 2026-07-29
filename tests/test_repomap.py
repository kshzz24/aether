from pathlib import Path

from repomap.tags import Tag, extract_tags, language_for

_SRC = b"""class Widget:
    def render(self):
        helper()


def helper():
    return Widget()
"""


def test_extract_tags_finds_defs_and_refs():
    tags = extract_tags("w.py", _SRC)
    defs = {t.name for t in tags if t.kind == "def"}
    refs = {t.name for t in tags if t.kind == "ref"}
    assert {"Widget", "render", "helper"} <= defs
    assert {"helper", "Widget"} <= refs  # helper() and Widget() calls
    widget_def = next(t for t in tags if t.name == "Widget" and t.kind == "def")
    assert widget_def.line == 1  # 1-based


def test_extract_tags_unsupported_language_is_empty():
    assert extract_tags("readme.md", b"# hello") == []


def test_language_for_maps_extension():
    assert language_for("a/b/c.py") == "python"
    assert language_for("x.md") is None


from repomap.rank import rank_files  # noqa: E402


def _defs(path, *names):
    return [Tag(n, "def", path, 1) for n in names]


def _refs(path, *names):
    return [Tag(n, "ref", path, 1) for n in names]


def test_widely_referenced_file_ranks_above_a_leaf():
    # core.py defines core_fn, referenced by a.py and b.py; leaf.py is unused.
    tags = (
        _defs("core.py", "core_fn")
        + _defs("a.py", "a_fn") + _refs("a.py", "core_fn")
        + _defs("b.py", "b_fn") + _refs("b.py", "core_fn")
        + _defs("leaf.py", "leaf_fn")
    )
    scores = rank_files(tags)
    assert scores["core.py"] > scores["leaf.py"]
    assert scores["core.py"] > scores["a.py"]


def test_focus_files_biases_ranking():
    tags = (
        _defs("core.py", "core_fn")
        + _defs("a.py", "a_fn") + _refs("a.py", "core_fn")
        + _defs("leaf.py", "leaf_fn")
    )
    base = rank_files(tags)
    focused = rank_files(tags, focus_files=["leaf.py"])
    assert focused["leaf.py"] > base["leaf.py"]


def test_rank_files_empty_is_empty():
    assert rank_files([]) == {}


def test_ubiquitous_symbol_names_do_not_manufacture_edges():
    # `get` is defined in 4 files, so a call to it says nothing about who depends
    # on whom. leaf.py's only real dependency is core.py's unique core_fn; if the
    # `get` edges survived, they would split leaf's rank 5 ways and core.py would
    # merely tie with the get-definers.
    tags = _defs("core.py", "core_fn") + _refs("leaf.py", "core_fn", "get")
    for f in ("a.py", "b.py", "c.py", "d.py"):
        tags += _defs(f, "get")
    scores = rank_files(tags)
    assert scores["core.py"] > scores["a.py"]


def test_dunder_defs_are_not_graph_targets():
    # __init__ is a definition everywhere and a dependency nowhere.
    tags = (
        _defs("a.py", "__init__")
        + _defs("core.py", "core_fn")
        + _refs("caller.py", "__init__", "core_fn")
    )
    scores = rank_files(tags)
    assert scores["core.py"] > scores["a.py"]


from repomap.cache import TagCache  # noqa: E402


def test_cache_hit_on_matching_mtime(tmp_path):
    cache = TagCache(tmp_path / "c.sqlite")
    tags = [Tag("Foo", "def", "f.py", 1)]
    cache.put("f.py", 123.0, tags)
    assert cache.get("f.py", 123.0) == tags


def test_cache_miss_on_changed_mtime_or_unknown(tmp_path):
    cache = TagCache(tmp_path / "c.sqlite")
    cache.put("f.py", 123.0, [Tag("Foo", "def", "f.py", 1)])
    assert cache.get("f.py", 999.0) is None       # mtime changed -> stale
    assert cache.get("other.py", 123.0) is None    # unknown file


from repomap.render import render_map  # noqa: E402


def _file_defs(path, *names):
    return [Tag(n, "def", path, i + 1) for i, n in enumerate(names)]


def test_render_respects_budget_and_drops_low_ranked():
    tags = _file_defs("top.py", "A", "B") + _file_defs("bottom.py", "C", "D")
    out = render_map(["top.py", "bottom.py"], tags, max_tokens=5)  # tiny budget
    assert "top.py" in out
    assert "bottom.py" not in out


def test_render_includes_all_within_large_budget():
    tags = _file_defs("top.py", "A") + _file_defs("bottom.py", "C")
    out = render_map(["top.py", "bottom.py"], tags, max_tokens=10_000)
    assert "top.py" in out and "bottom.py" in out


def test_build_repo_map_over_a_small_repo(tmp_path):
    (tmp_path / "core.py").write_text(
        "def core_fn():\n    return 1\n", encoding="utf-8"
    )
    (tmp_path / "app.py").write_text(
        "from core import core_fn\n\n\ndef app():\n    return core_fn()\n",
        encoding="utf-8",
    )
    from repomap import build_repo_map

    out = build_repo_map(tmp_path, max_tokens=1000)
    assert "core.py" in out and "core_fn" in out


def test_build_repo_map_empty_repo_is_blank(tmp_path):
    from repomap import build_repo_map

    assert build_repo_map(tmp_path, max_tokens=1000) == ""


def test_repomap_imports_standalone():
    # `tools/__init__` imports the repo_map tool, which imports this package, so a
    # module-level `tools` import in repomap is a cycle. Run in a fresh
    # interpreter: an in-process import would find `tools` already loaded.
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-c", "import repomap; assert repomap.build_repo_map"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
