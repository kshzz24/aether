from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

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

CONTEXT_WINDOW_SIZE = 128000

DANGEROUS_TOOLS = {"run_shell"}


@dataclass
class Tool:
    schema: dict
    run: Callable[[dict], Awaitable[str]]


class Agent:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        tools: dict[str, Tool],
        system: str,
        max_iterations: int,
        max_cost_usd: float,
        auto_approve: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.system = system
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.max_context_token = CONTEXT_WINDOW_SIZE
        self._auto_approve = auto_approve

    def _resolve_confirmation(self, tool_name: str, arguments: dict) -> bool:

        return self._auto_approve

    async def run(self, goal: str) -> AsyncIterator[Event]:

        initial_text = TextBlock(text=goal)
        messages = [Message(role="user", blocks=[initial_text])]
        tools_schemas = [t.schema for t in self.tools.values()]

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

                    is_dangerous = block.name in DANGEROUS_TOOLS
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
                            current_tool_needed = self.tools[block.name]
                            result_str = await current_tool_needed.run(block.arguments)
                        except Exception as e:
                            result_str = f"ERROR: {e}  "

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
