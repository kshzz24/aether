from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

from client import (
    LLMClient,
    Message,
    NormalizedResponse,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from events import (
    Event,
    StatusEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    CostEvent,
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
        blocks = [initial_text]
        messages = [Message(role="user", blocks=blocks)]

        for _ in range(self.max_iterations):

            response = await self.client.create(
                messages=messages, tools=self.tools, system=self.system
            )

            for block in response.blocks:
                if isinstance(block, TextBlock):
                    pass
                elif isinstance(block, ToolCallBlock):
                    pass
