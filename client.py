from dataclasses import dataclass
from typing import Literal, Protocol
from anthropic import AsyncAnthropic
import json
from openai import AsyncOpenAI


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolCallBlock:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass
class ToolResultBlock:
    tool_call_id: str
    content: str


@dataclass
class Message:
    role: Literal["user", "assistant"]
    blocks: list[TextBlock | ToolCallBlock | ToolResultBlock]


@dataclass
class NormalizedResponse:
    blocks: list[TextBlock | ToolCallBlock]
    input_tokens: int
    output_tokens: int
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop"]


class LLMClient(Protocol):
    async def create(
        self, messages: list[Message], tools: list[dict], system: str
    ) -> NormalizedResponse:
        pass


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    def _to_anthropic(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            content = []
            for b in m.blocks:
                if isinstance(b, TextBlock):
                    content.append({"type": "text", "text": b.text})
                elif isinstance(b, ToolCallBlock):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": b.id,
                            "name": b.name,
                            "input": b.arguments,
                        }
                    )
                elif isinstance(b, ToolResultBlock):
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b.tool_call_id,
                            "content": b.content,
                        }
                    )
            out.append({"role": m.role, "content": content})
        return out

    async def create(self, messages, tools, system) -> NormalizedResponse:

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=self._to_anthropic(messages=messages),
            tools=tools,
        )

        blocks = []

        for block in response.content:

            if block.type == "text":
                blocks.append(TextBlock(text=block.text))

            elif block.type == "tool_use":
                blocks.append(
                    ToolCallBlock(id=block.id, name=block.name, arguments=block.input)
                )

        return NormalizedResponse(
            blocks=blocks,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


class OpenAICompatibleClient(LLMClient):

    def __init__(self, model: str, api_key: str, base_url: str):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def _to_openai(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            if m.role == "assistant":
                text = "".join(b.text for b in m.blocks if isinstance(b, TextBlock))
                tool_calls = [
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {
                            "name": b.name,
                            "arguments": json.dumps(b.arguments),
                        },
                    }
                    for b in m.blocks
                    if isinstance(b, ToolCallBlock)
                ]
                msg = {"role": "assistant", "content": text or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls

                out.append(msg)
            else:
                for b in m.blocks:
                    if isinstance(b, TextBlock):
                        out.append({"role": "user", "content": b.text})
                    elif isinstance(b, ToolResultBlock):
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b.tool_call_id,
                                "content": b.content,
                            }
                        )
        return out

    async def create(self, messages, tools, system) -> NormalizedResponse:

        full_messages = [
            {"role": "system", "content": system},
            *self._to_openai(messages),
        ]

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            tools=openai_tools or None,
        )

        message = response.choices[0].message
        blocks = []

        if message.content:
            blocks.append(TextBlock(text=message.content))

        for tc in message.tool_calls or []:
            blocks.append(
                ToolCallBlock(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
            )

        finish = response.choices[0].finish_reason
        stop_reason = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }.get(finish, "end_turn")

        return NormalizedResponse(
            blocks=blocks,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            stop_reason=stop_reason,
        )


_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "together": "https://api.together.xyz/v1",
}


def make_client(provider, model, api_key="", base_url=None) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key)
    return OpenAICompatibleClient(
        model=model, api_key=api_key, base_url=base_url or _URLS[provider]
    )
