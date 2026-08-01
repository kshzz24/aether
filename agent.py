import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import safety
from approval import ApprovalMode, ApprovalRequest, Approver, Verdict
from client import (
    LLMClient,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolCallingUnsupportedError,
    ToolResultBlock,
)
from context.compactor import compact, needs_compaction
from context.loop_detector import LoopDetector
from events import (
    ApprovalDecisionEvent,
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    SubagentEvent,
    TerminalEvent,
    TerminalReason,
    TextDeltaEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from policy import PolicyEngine
from tools.base import ToolKind
from tools.hooks import Hooks
from tools.registry import ToolRegistry

CONTEXT_WINDOW_SIZE = 128000


def _hash_file(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


class Agent:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        registry: ToolRegistry,
        system: str,
        max_iterations: int,
        max_cost_usd: float,
        policy: PolicyEngine | None = None,
        approver: Approver | None = None,
        repo_root: Path | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.system = system
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.policy = policy if policy is not None else PolicyEngine(ApprovalMode.AUTO)
        self.repo_root = repo_root or Path.cwd()
        self.max_context_token = CONTEXT_WINDOW_SIZE
        self.hooks = hooks or Hooks()
        self._approver = approver
        # Live, resumable run state. Held as plain attributes so the driver can
        # snapshot them for checkpointing; the agent itself never persists them
        # (Invariant 1 — the agent does no I/O).
        self.messages: list[Message] = []
        self.total_cost = 0.0

    async def run(
        self, goal: str, *, history: list[Message] | None = None
    ) -> AsyncIterator[Event]:
        self.hooks.before_run(goal)
        try:
            async for event in self._run(goal, history):
                yield event
        finally:
            self.hooks.after_run(goal)

    async def _run(
        self, goal: str, history: list[Message] | None = None
    ) -> AsyncIterator[Event]:
        # Resume: the saved history already contains the goal as its first
        # message, so it is not re-appended. Fresh run: seed with the goal.
        if history:
            self.messages = list(history)
        else:
            self.messages = [Message(role="user", blocks=[TextBlock(text=goal)])]
        tools_schemas = self.registry.wire_schemas()

        self.total_cost = 0.0
        detector = LoopDetector()

        for _ in range(self.max_iterations):
            yield StatusEvent(type="status", message="thinking")

            # Streaming is an *optional* client capability (`StreamingClient` in
            # client.py), probed rather than required. A client without it —
            # GatewayClient, a test double — takes the buffered path below and
            # produces a byte-identical event sequence.
            make_stream = getattr(self.client, "stream", None)
            try:
                if make_stream is None:
                    response = await self.client.create(
                        messages=self.messages, tools=tools_schemas, system=self.system
                    )
                else:
                    turn = make_stream(
                        messages=self.messages, tools=tools_schemas, system=self.system
                    )
                    async for delta in turn:
                        yield TextDeltaEvent(type="text_delta", text=delta)
                    response = turn.response
            except ToolCallingUnsupportedError as e:
                yield TerminalEvent(reason=TerminalReason.ERROR, detail=str(e))
                return

            # this turn's cost.
            current_cost = response.cost_usd
            self.total_cost += current_cost
            yield CostEvent(
                type="cost", cost_usd=current_cost, total_cost_usd=self.total_cost
            )

            # Surface the model's text BEFORE the stop check, so a final
            # end_turn answer is still emitted.
            for block in response.blocks:
                if isinstance(block, TextBlock):
                    yield TextEvent(type="text", text=block.text)

            self.messages.append(Message(role="assistant", blocks=response.blocks))

            if response.stop_reason != "tool_use":
                yield TerminalEvent(reason=TerminalReason.COMPLETED)
                return

            if self.total_cost >= self.max_cost_usd:
                yield TerminalEvent(
                    reason=TerminalReason.MAX_COST,
                    detail="stopped: cost limit reached",
                )
                return

            tool_results = []

            for block in response.blocks:
                if isinstance(block, ToolCallBlock):
                    yield ToolCallEvent(
                        type="tool_call",
                        name=block.name,
                        arguments=block.arguments,
                    )

                    self.hooks.before_tool(block.name, block.arguments)

                    # Validate the call against the registered schema (and the
                    # allowlist) BEFORE dispatch. An invalid or disallowed call
                    # is data, not a crash: return the error as an observation.
                    try:
                        self.registry.validate_call(block.name, block.arguments)
                        tool = self.registry.get(block.name)
                    except Exception as e:
                        result_str = f"ERROR: {e}"
                        self.hooks.on_error(e)
                    else:
                        danger = []

                        if "command" in block.arguments:
                            danger += safety.dangerous_command(
                                block.arguments["command"]
                            )
                        danger += safety.path_escape(
                            block.arguments, repo_root=self.repo_root
                        )

                        preview = (
                            await tool.preview(block.arguments)
                            if tool.preview
                            else None
                        )

                        verdict, deny_reason = self.policy.evaluate(
                            tool.kind, danger_reasons=danger
                        )

                        request = ApprovalRequest(
                            block.name,
                            block.arguments,
                            tool.kind,
                            danger,
                            preview.diff if preview else None,
                        )

                        if verdict is Verdict.ASK:
                            yield ConfirmRequestEvent(
                                tool_name=block.name,
                                arguments=block.arguments,
                                reason="approval required",
                            )

                            decision = await self._approver.decide(request)
                            approved = decision.approved

                            deny_reason = decision.reason or "user declined"
                            source = "human"

                            # An edited call is a NEW call, and gets checked
                            # like one. Trusting it because a human typed it
                            # would make the approval prompt a bypass for the
                            # schema, the danger checks and the path guard.
                            if approved and decision.arguments is not None:
                                block.arguments = decision.arguments
                                try:
                                    self.registry.validate_call(
                                        block.name, block.arguments
                                    )
                                except Exception as e:
                                    approved = False
                                    deny_reason = f"edited call is invalid: {e}"
                                else:
                                    danger = []
                                    if "command" in block.arguments:
                                        danger += safety.dangerous_command(
                                            block.arguments["command"]
                                        )
                                    danger += safety.path_escape(
                                        block.arguments, repo_root=self.repo_root
                                    )
                                    if danger:
                                        approved = False
                                        deny_reason = (
                                            "edited call was rejected: "
                                            + "; ".join(danger)
                                        )
                        else:
                            approved = verdict is Verdict.AUTO_APPROVE
                            source = "policy"

                        yield ApprovalDecisionEvent(
                            type="approval_decision",
                            tool_name=block.name,
                            kind=tool.kind,
                            danger_reasons=danger,
                            verdict=verdict,
                            approved=approved,
                            source=source,
                        )

                        if approved:
                            stale = (
                                preview is not None
                                and preview.base_hash is not None
                                and _hash_file(preview.hashed_path) != preview.base_hash
                            )
                            if stale:
                                result_str = (
                                    "ERROR: file changed since approval; not applied"
                                )
                            else:
                                is_agent = tool.kind is ToolKind.AGENT
                                if is_agent:
                                    yield SubagentEvent(
                                        type="subagent",
                                        task=str(
                                            block.arguments.get("prompt", ""),
                                        ),
                                        phase="started",
                                    )
                                try:
                                    result_str = await tool.run(block.arguments)
                                except Exception as e:
                                    result_str = f"ERROR: {e}  "
                                    self.hooks.on_error(e)

                                if is_agent:
                                    yield SubagentEvent(
                                        type="subagent",
                                        task=str(block.arguments.get("prompt", "")),
                                        phase="completed",
                                    )
                        else:
                            result_str = f"DENIED: {deny_reason}"

                    self.hooks.after_tool(block.name, block.arguments, result_str)
                    detector.record(block.name, block.arguments, result_str)

                    tool_results.append(
                        ToolResultBlock(tool_call_id=block.id, content=result_str)
                    )

                    yield ToolResultEvent(
                        type="tool_result", name=block.name, result=result_str
                    )

            if tool_results:
                self.messages.append(Message(role="user", blocks=tool_results))

            if detector.is_looping():
                yield TerminalEvent(reason=TerminalReason.LOOP_DETECTED)
                return

            if needs_compaction(response.input_tokens, self.max_context_token):
                compacted, summary_cost = await compact(
                    self.client, self.messages, keep_recent=6
                )
                # Mutate in place so the reference the driver snapshots stays live.
                self.messages[:] = compacted
                self.total_cost += summary_cost
                yield CostEvent(
                    type="cost",
                    cost_usd=summary_cost,
                    total_cost_usd=self.total_cost,
                )
                yield StatusEvent(type="status", message="compacted context")

        yield TerminalEvent(reason=TerminalReason.MAX_ITERATIONS)
