from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

Source = Literal["builtin", "user"]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    source: Source


def parse_skill(text: str, source: Source) -> Skill:
    """Parse a skill markdown file: a ``---`` frontmatter block, then the body.

    Frontmatter is flat ``key: value`` lines; only ``name`` and ``description``
    are read. Raises ValueError on a malformed file (the loader quarantines it).
    """
    lines = text.lstrip().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("skill missing opening '---' formatter fence")

    close = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if close is None:
        raise ValueError("skill formatter not terminated by '---'")

    meta: dict[str, str] = {}
    for line in lines[1:close]:
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    if "name" not in meta or "description" not in meta:
        raise ValueError("skill frontmatter requires 'name' and 'description'")
    body = "\n".join(lines[close + 1 :]).strip()
    return Skill(
        name=meta["name"],
        description=meta["description"],
        body=body,
        source=source,
    )


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            logging.warning(
                "skill %r already registered; keeping incumbent, ignoring duplicate",
                skill.name,
            )
            return
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def list(self) -> list[Skill]:
        return list(self._skills.values())


def render_skill_menu(skills: list[Skill]) -> str:
    """The system-prompt menu: one line per skill. Empty string if no skills."""
    if not skills:
        return ""
    lines = [
        "## Available skills",
        "Call load_skill(name) to load a skill's full instructions when one is "
        "relevant to the task.",
        "",
    ]
    for skill in sorted(skills, key=lambda s: s.name):
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)
