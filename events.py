from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class StatusEvent:
    type: Literal["status"]
    message: str


@dataclass(frozen=True)
class TextEvent:
    type: Literal["text"]
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    type: Literal["tool_call"]
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolResultEvent:
    type: Literal["tool_result"]
    name: str
    result: str


@dataclass(frozen=True)
class CostEvent:
    type: Literal["cost"]
    cost_usd: float
    total_cost_usd: float


Event: TypeAlias = StatusEvent | TextEvent | ToolCallEvent | ToolResultEvent | CostEvent
