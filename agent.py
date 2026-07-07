from collections.abc import AsyncIterator

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
    ConfirmRequestEvent,
    CostEvent,
    Event,
    StatusEvent,
    TerminalEvent,
    TerminalReason,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tools.base import ToolKind
from tools.hooks import Hooks
from tools.registry import ToolRegistry

CONTEXT_WINDOW_SIZE = 128000


class Agent:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        registry: ToolRegistry,
        system: str,
        max_iterations: int,
        max_cost_usd: float,
        auto_approve: bool = True,
        hooks: Hooks | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.system = system
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.max_context_token = CONTEXT_WINDOW_SIZE
        self._auto_approve = auto_approve
        self.hooks = hooks or Hooks()

    def _resolve_confirmation(self, tool_name: str, arguments: dict) -> bool:
        # TODO(phase-5): the Approver replaces this stub with real policy modes.
        return self._auto_approve

    async def run(self, goal: str) -> AsyncIterator[Event]:
        self.hooks.before_run(goal)
        try:
            async for event in self._run(goal):
                yield event
        finally:
            self.hooks.after_run(goal)

    async def _run(self, goal: str) -> AsyncIterator[Event]:
        initial_text = TextBlock(text=goal)
        messages = [Message(role="user", blocks=[initial_text])]
        tools_schemas = self.registry.wire_schemas()

        total_cost = 0.0
        detector = LoopDetector()

        for _ in range(self.max_iterations):
            yield StatusEvent(type="status", message="thinking")

            try:
                response = await self.client.create(
                    messages=messages, tools=tools_schemas, system=self.system
                )
            except ToolCallingUnsupportedError as e:
                yield TerminalEvent(reason=TerminalReason.ERROR, detail=str(e))
                return

            # this turn's cost.
            current_cost = response.cost_usd
            total_cost += current_cost
            yield CostEvent(
                type="cost", cost_usd=current_cost, total_cost_usd=total_cost
            )

            # Surface the model's text BEFORE the stop check, so a final
            # end_turn answer is still emitted.
            for block in response.blocks:
                if isinstance(block, TextBlock):
                    yield TextEvent(type="text", text=block.text)

            messages.append(Message(role="assistant", blocks=response.blocks))

            if response.stop_reason != "tool_use":
                yield TerminalEvent(reason=TerminalReason.COMPLETED)
                return

            if total_cost >= self.max_cost_usd:
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
                        is_dangerous = tool.kind is ToolKind.EXECUTE
                        if is_dangerous:
                            yield ConfirmRequestEvent(
                                tool_name=block.name,
                                arguments=block.arguments,
                                reason="dangerous tool",
                            )

                        if is_dangerous and not self._resolve_confirmation(
                            block.name, block.arguments
                        ):
                            result_str = "DENIED: user declined to run this tool"
                        else:
                            try:
                                result_str = await tool.run(block.arguments)
                            except Exception as e:
                                result_str = f"ERROR: {e}  "
                                self.hooks.on_error(e)

                    self.hooks.after_tool(block.name, block.arguments, result_str)
                    detector.record(block.name, block.arguments, result_str)

                    tool_results.append(
                        ToolResultBlock(tool_call_id=block.id, content=result_str)
                    )

                    yield ToolResultEvent(
                        type="tool_result", name=block.name, result=result_str
                    )

            if tool_results:
                messages.append(Message(role="user", blocks=tool_results))

            if detector.is_looping():
                yield TerminalEvent(reason=TerminalReason.LOOP_DETECTED)
                return

            if needs_compaction(response.input_tokens, self.max_context_token):
                messages, summary_cost = await compact(
                    self.client, messages, keep_recent=6
                )
                total_cost += summary_cost
                yield CostEvent(
                    type="cost",
                    cost_usd=summary_cost,
                    total_cost_usd=total_cost,
                )
                yield StatusEvent(type="status", message="compacted context")

        yield TerminalEvent(reason=TerminalReason.MAX_ITERATIONS)
