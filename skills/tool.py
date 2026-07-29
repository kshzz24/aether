from __future__ import annotations

from skills.base import SkillRegistry
from tools.base import Tool, ToolKind

_PARAMS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The name of a skill from the '## Available skills' menu.",
        }
    },
    "required": ["name"],
}

_DESCRIPTION = (
    "Load a skill's full instructions by name. Consult the '## Available skills' "
    "menu in your system prompt and call this when a skill is relevant to the task."
)


def build_skill_tool(*, registry: SkillRegistry) -> Tool:
    async def run(args: dict) -> str:
        name = args["name"]
        try:
            skill = registry.get(name)
        except KeyError:
            available = ", ".join(sorted(s.name for s in registry.list())) or "(none)"
            return f"ERROR: no skill named {name!r}. Available: {available}"
        return skill.body

    return Tool(
        name="load_skill",
        description=_DESCRIPTION,
        parameters=_PARAMS,
        kind=ToolKind.READ,
        run=run,
    )
