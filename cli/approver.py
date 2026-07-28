import asyncio
from collections.abc import Callable

from approval import ApprovalRequest, Decision


class CliApprover:

    def __init__(self, input_fn: Callable[[], str] = input) -> None:
        self._input_fn = input_fn

    async def decide(self, request: ApprovalRequest) -> Decision:
        lines = [f"Allow {request.tool_name}({request.arguments})?"]
        if request.danger_reasons:
            lines.append(f"  ⚠ {'; '.join(request.danger_reasons)}")
        if request.diff is not None:
            lines.append(request.diff)
        print("\n".join(lines))
        answer = await asyncio.to_thread(self._input_fn)
        approved = answer.strip().lower() in ("y", "yes")

        return Decision(approved=approved, reason=None if approved else "user declined")
