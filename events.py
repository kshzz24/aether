from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, TypeAlias


class TerminalReason(Enum):
    COMPLETED = auto()
    MAX_ITERATIONS = auto()
    MAX_COST = auto()
    LOOP_DETECTED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class TerminalEvent:
    reason: TerminalReason
    detail: str = ""


@dataclass(frozen=True)
class ConfirmRequestEvent:
    tool_name: str
    arguments: dict[str, Any]
    reason: str


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


Event: TypeAlias = (
    StatusEvent
    | TextEvent
    | ToolCallEvent
    | ToolResultEvent
    | CostEvent
    | TerminalEvent
    | ConfirmRequestEvent
)
