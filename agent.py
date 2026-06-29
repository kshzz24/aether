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
from events import (
    CostEvent,
    Event,
    StatusEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)


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
        pricing: dict[str, tuple[float, float]],
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.system = system
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.pricing = pricing

    def _price(self, input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = self.pricing[self.model]
        return input_tokens * in_rate + out_rate * output_tokens

    async def run(self, goal: str) -> AsyncIterator[Event]:

        initial_text = TextBlock(text=goal)
        messages = [Message(role="user", blocks=[initial_text])]
        tools_schemas = [t.schema for t in self.tools.values()]

        total_cost = 0.0

        for _ in range(self.max_iterations):

            yield StatusEvent(type="status", message="thinking")

            try:
                response = await self.client.create(
                    messages=messages, tools=tools_schemas, system=self.system
                )
            except ToolCallingUnsupportedError as e:
                yield StatusEvent(type="status", message=f"stopped: {e}")
                return

            # this turn's cost.
            current_cost = self._price(response.input_tokens, response.output_tokens)
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
                return

            if total_cost >= self.max_cost_usd:
                yield StatusEvent(type="status", message="stopped: cost limit reached")
                return

            tool_results = []

            for block in response.blocks:
                if isinstance(block, ToolCallBlock):

                    yield ToolCallEvent(
                        type="tool_call",
                        name=block.name,
                        arguments=block.arguments,
                    )

                    try:
                        current_tool_needed = self.tools[block.name]
                        result_str = await current_tool_needed.run(block.arguments)
                    except Exception as e:
                        result_str = f"ERROR: {e}  "

                    tool_results.append(
                        ToolResultBlock(tool_call_id=block.id, content=result_str)
                    )

                    yield ToolResultEvent(
                        type="tool_result", name=block.name, result=result_str
                    )

            if tool_results:
                messages.append(Message(role="user", blocks=tool_results))

        yield StatusEvent(type="status", message="stopped: max iterations reached")
