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
