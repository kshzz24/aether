from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Protocol, runtime_checkable

from tools.base import ToolKind


# str-valued so Pydantic (config) coerces "never" -> ApprovalMode.NEVER by value
# with no extra validator. `.name` is still AUTO / ON_REQUEST / NEVER.
class ApprovalMode(StrEnum):
    AUTO = "auto"
    ON_REQUEST = "on-request"
    NEVER = "never"


class Verdict(Enum):
    AUTO_APPROVE = auto()
    AUTO_DENY = auto()
    ASK = auto()


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: dict
    kind: ToolKind
    danger_reasons: list[str]
    diff: str | None = None


@dataclass(frozen=True)
class Decision:
    approved: bool
    reason: str | None = None


@runtime_checkable
class Approver(Protocol):
    async def decide(self, request: ApprovalRequest) -> Decision: ...
