from approval import ApprovalMode, Verdict
from tools.base import ToolKind


class PolicyEngine:
    def __init__(self, mode: ApprovalMode) -> None:
        self.mode = mode

    def evaluate(
        self, kind: ToolKind, danger_reasons: list[str]
    ) -> tuple[Verdict, str | None]:

        if kind is ToolKind.READ:
            return Verdict.AUTO_APPROVE, None

        if kind is ToolKind.AGENT:
            return Verdict.AUTO_APPROVE, None

        if self.mode is ApprovalMode.NEVER:
            reason = "; ".join(danger_reasons) or f"mode=never denies {kind.name}"
            return Verdict.AUTO_DENY, reason

        if danger_reasons:
            return Verdict.ASK, None

        if self.mode is ApprovalMode.AUTO:
            return Verdict.AUTO_APPROVE, None
        return Verdict.ASK, None
