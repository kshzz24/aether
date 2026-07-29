import pytest

from skills.base import Skill, SkillRegistry, parse_skill, render_skill_menu

_VALID = """---
name: writing-tests
description: Use when adding a feature — write a failing test first.
---
# Writing tests

Write the failing test before the implementation.
"""


def test_parse_skill_reads_frontmatter_and_body():
    skill = parse_skill(_VALID, "builtin")
    assert skill.name == "writing-tests"
    assert skill.description.startswith("Use when adding a feature")
    assert "Write the failing test" in skill.body
    assert skill.source == "builtin"


def test_parse_skill_missing_fence_raises():
    with pytest.raises(ValueError):
        parse_skill("no frontmatter here", "user")


def test_parse_skill_missing_required_field_raises():
    text = "---\nname: x\n---\nbody"
    with pytest.raises(ValueError):
        parse_skill(text, "user")


def test_registry_incumbent_wins_on_duplicate_name():
    reg = SkillRegistry()
    a = Skill("dup", "first", "body A", "builtin")
    b = Skill("dup", "second", "body B", "user")
    reg.register(a)
    reg.register(b)  # ignored: incumbent (builtin) wins
    assert reg.get("dup").body == "body A"
    assert [s.name for s in reg.list()] == ["dup"]


def test_render_skill_menu_lists_name_and_description():
    skills = [
        Skill("debugging", "Systematic debugging.", "b1", "builtin"),
        Skill("writing-tests", "TDD first.", "b2", "builtin"),
    ]
    menu = render_skill_menu(skills)
    assert "## Available skills" in menu
    assert "debugging: Systematic debugging." in menu
    assert "writing-tests: TDD first." in menu


def test_render_skill_menu_empty_is_blank():
    assert render_skill_menu([]) == ""


from pathlib import Path  # noqa: E402

from skills.loader import load_skills  # noqa: E402


def _write(dir_path: Path, filename: str, name: str, desc: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / filename).write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\nbody of {name}\n",
        encoding="utf-8",
    )


def test_load_skills_discovers_builtin_and_user_skips_underscore(tmp_path):
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    _write(builtin, "a.md", "alpha", "builtin alpha")
    _write(user, "b.md", "beta", "user beta")
    _write(user, "_wip.md", "wip", "should be skipped")

    skills, errors = load_skills(builtin, user)

    names = [s.name for s in skills]
    assert names == ["alpha", "beta"]        # builtins first, then user
    assert "wip" not in names                 # underscore-prefixed skipped
    assert errors == []
    assert skills[0].source == "builtin" and skills[1].source == "user"


def test_load_skills_quarantines_malformed_but_keeps_good(tmp_path):
    builtin = tmp_path / "builtin"
    _write(builtin, "good.md", "good", "fine")
    (builtin / "bad.md").write_text("no frontmatter", encoding="utf-8")

    skills, errors = load_skills(builtin, tmp_path / "missing_user")

    assert [s.name for s in skills] == ["good"]   # good one still loaded
    assert len(errors) == 1 and "bad.md" in errors[0]


def test_load_skills_missing_dirs_return_empty(tmp_path):
    skills, errors = load_skills(tmp_path / "nope", tmp_path / "nada")
    assert skills == [] and errors == []


import asyncio  # noqa: E402

from skills.tool import build_skill_tool  # noqa: E402
from tools.base import ToolKind  # noqa: E402


def _registry_with(*skills):
    reg = SkillRegistry()
    for s in skills:
        reg.register(s)
    return reg


def test_load_skill_tool_returns_body_for_known_name():
    reg = _registry_with(Skill("debugging", "d", "THE FULL BODY", "builtin"))
    tool = build_skill_tool(registry=reg)

    assert tool.name == "load_skill"
    assert tool.kind is ToolKind.READ
    result = asyncio.run(tool.run({"name": "debugging"}))
    assert result == "THE FULL BODY"


def test_load_skill_tool_unknown_name_is_error_observation():
    reg = _registry_with(Skill("debugging", "d", "body", "builtin"))
    tool = build_skill_tool(registry=reg)

    result = asyncio.run(tool.run({"name": "nope"}))
    assert result.startswith("ERROR")
    assert "debugging" in result   # lists what IS available


from skills import build_skill_registry  # noqa: E402


def test_build_skill_registry_loads_the_three_seed_skills():
    builtin = Path(__file__).resolve().parent.parent / "skills" / "builtin"
    reg = build_skill_registry(builtin, Path("/nonexistent-user-skills"))
    names = {s.name for s in reg.list()}
    assert {"writing-tests", "debugging", "committing"} <= names
