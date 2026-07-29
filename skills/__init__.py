from skills.base import Skill, SkillRegistry, parse_skill, render_skill_menu
from skills.loader import load_skills
from skills.tool import build_skill_tool

__all__ = [
    "Skill",
    "SkillRegistry",
    "parse_skill",
    "render_skill_menu",
    "load_skills",
    "build_skill_tool",
    "build_skill_registry",
]


def build_skill_registry(builtin_dir, user_dir) -> SkillRegistry:
    """Load builtin then user skills into a registry (builtins win collisions)."""
    registry = SkillRegistry()
    skills, _errors = load_skills(builtin_dir, user_dir)
    for skill in skills:
        registry.register(skill)
    return registry
