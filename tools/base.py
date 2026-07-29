from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto


class ToolKind(Enum):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    AGENT = auto()


@dataclass(frozen=True)
class PreviewResult:
    diff: str
    base_hash: str | None = None
    hashed_path: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    kind: ToolKind
    run: Callable[[dict], Awaitable[str]]
    preview: Callable[[dict], Awaitable[PreviewResult]] | None = None

    def to_wire(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @staticmethod
    def from_module(module) -> Tool:
        name = module.SCHEMA["name"]
        description = module.SCHEMA["description"]
        parameters = module.SCHEMA["parameters"]
        kind = module.KIND
        run = module.run
        preview = getattr(module, "PREVIEW", None)

        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            kind=kind,
            run=run,
            preview=preview,
        )
