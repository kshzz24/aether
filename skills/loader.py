from __future__ import annotations

import logging
from pathlib import Path

from skills.base import Skill, parse_skill


def load_skills(builtin_dir: Path, user_dir: Path) -> tuple[list[Skill], list[dir]]:

    skills = []
    errors = []

    for src, dir in (("builtin", builtin_dir), ("user", user_dir)):
        if not dir.exists():
            continue

        for path in sorted(dir.glob("*.md")):
            if path.name.startswith("_"):
                continue
            try:
                skills.append(parse_skill(path.read_text(encoding="utf-8"), src))
            except Exception as e:
                logging.warning("failed to load skill %s: %s", path.name, e)
                errors.append(f"failed to load skill from {path.name} due to {e}")

    return skills, errors