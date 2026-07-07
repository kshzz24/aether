from tools.base import ToolKind
from tools.loader import load_user_tools

GOOD = '''
from tools.base import ToolKind

KIND = ToolKind.READ
SCHEMA = {
    "name": "custom",
    "description": "a user tool",
    "parameters": {"type": "object", "properties": {}},
}


async def run(args):
    return "custom ok"
'''


def test_nonexistent_dir_returns_empty(tmp_path):
    tools, errors = load_user_tools(tmp_path / "does-not-exist")
    assert tools == []
    assert errors == []


def test_good_user_tool_loads(tmp_path):
    (tmp_path / "custom.py").write_text(GOOD, encoding="utf-8")
    tools, errors = load_user_tools(tmp_path)
    assert [t.name for t in tools] == ["custom"]
    assert tools[0].kind is ToolKind.READ
    assert errors == []


def test_syntactically_broken_tool_is_quarantined(tmp_path):
    (tmp_path / "good.py").write_text(GOOD, encoding="utf-8")
    (tmp_path / "broken.py").write_text("this is not python !!!", encoding="utf-8")
    tools, errors = load_user_tools(tmp_path)
    # the good one still loads; the broken one is recorded, not raised
    assert [t.name for t in tools] == ["custom"]
    assert len(errors) == 1
    assert "broken.py" in errors[0]


def test_tool_missing_required_fields_is_quarantined(tmp_path):
    (tmp_path / "nofields.py").write_text("x = 1\n", encoding="utf-8")
    tools, errors = load_user_tools(tmp_path)
    assert tools == []
    assert len(errors) == 1
    assert "nofields.py" in errors[0]


def test_underscore_files_are_skipped(tmp_path):
    (tmp_path / "_helper.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "custom.py").write_text(GOOD, encoding="utf-8")
    tools, errors = load_user_tools(tmp_path)
    assert [t.name for t in tools] == ["custom"]  # _helper.py skipped, no error
    assert errors == []
