import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from anthropic import AsyncAnthropic
from openai import APIError, AsyncOpenAI, BadRequestError


class ToolCallingUnsupportedError(Exception):
    """The model could not emit a valid tool call.

    Raised when the provider rejects the generation because the model produced
    malformed tool-call output (e.g. Groq's ``tool_use_failed``). Typically
    means the model does not reliably support tool calling. Normalized here so
    the agent loop never sees a provider-specific SDK exception.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"model '{model}' could not produce a valid tool call; "
            "it likely does not support tool calling"
        )


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
    cost_usd: float
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop"]


def cost(rates, model, in_token, out_token) -> float:

    row = rates.get(model)
    if row is None:
        logging.warning("no price for model %r; metering as $0", model)
        return 0.0
    return (in_token * row["input"] + out_token * row["output"]) / 1_000_000


# OpenAI finish_reason -> our stop_reason. Shared by the buffered and streaming
# paths so the two can never disagree about what ended a turn.
_STOP_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


class StreamNotFinishedError(RuntimeError):
    """`.response` was read before the stream was exhausted."""


def _is_tool_call_failure(exc: Exception) -> bool:
    """True when a provider error means "the model botched a tool call".

    Providers say this in at least two ways: Groq's `tool_use_failed` code, and
    a plain message about unparseable tool-call arguments. Both mean the same
    thing to us, and both should reach the user as one explained error rather
    than a raw SDK exception.
    """
    if getattr(exc, "code", None) == "tool_use_failed":
        return True
    text = str(exc).lower()
    return "tool call" in text and ("parse" in text or "json" in text)


class _StreamBase:
    """Shared bookkeeping for a streaming turn."""

    def __init__(self, model: str, rates: dict[str, dict]) -> None:
        self._model = model
        self._rates = rates
        self._response: NormalizedResponse | None = None

    @property
    def response(self) -> NormalizedResponse:
        if self._response is None:
            raise StreamNotFinishedError(
                "read .response before the stream finished; iterate it first"
            )
        return self._response

    def _meter(self, input_tokens: int | None, output_tokens: int | None) -> tuple:
        """Token counts and cost, tolerating a provider that reports neither.

        Some OpenAI-compatible providers drop `usage` from streamed responses.
        Metering that as $0 *silently* would quietly disable the cost governor,
        so it is logged as loudly as an unpriced model is.
        """
        if input_tokens is None or output_tokens is None:
            logging.warning(
                "model %r reported no usage for a streamed turn; metering as $0",
                self._model,
            )
            return 0, 0, 0.0
        return (
            input_tokens,
            output_tokens,
            cost(self._rates, self._model, input_tokens, output_tokens),
        )


class LLMClient(Protocol):
    async def create(
        self, messages: list[Message], tools: list[dict], system: str
    ) -> NormalizedResponse:
        pass


class TurnStream(Protocol):
    """One streaming turn: text deltas now, the normalized result at the end.

    Iterating yields plain `str` — never a provider chunk object. That is what
    keeps Invariant 3 intact while streaming: provider shape still stops inside
    this module, and the agent loop sees only strings it can put in an Event.
    """

    def __aiter__(self) -> AsyncIterator[str]: ...

    @property
    def response(self) -> NormalizedResponse:
        """The finished turn. Only valid once iteration has completed."""
        ...


@runtime_checkable
class StreamingClient(Protocol):
    """An *optional* capability, deliberately separate from `LLMClient`.

    Making `create` itself streaming would have been a breaking change to a
    contract other code implements — `GatewayClient`, the test doubles, and
    anything users write. As a second protocol it is purely additive: a client
    that cannot stream simply does not have `stream`, the agent probes for it,
    and every existing client keeps working with no edits at all.

    `GatewayClient` not implementing it is the intended outcome, not an
    oversight: a gateway turn degrades to a single non-streamed response, which
    is exactly the graceful-degradation posture Phase 3 established.
    """

    def stream(
        self, messages: list[Message], tools: list[dict], system: str
    ) -> TurnStream: ...


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str, rates: dict[str, dict]):
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._provider = "anthropic"
        self._rates = rates

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

    def _to_anthropic_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    def _request(self, messages, tools, system) -> dict:
        return {
            "model": self._model,
            "max_tokens": 4096,
            "system": system,
            "messages": self._to_anthropic(messages=messages),
            "tools": self._to_anthropic_tools(tools),
        }

    def stream(self, messages, tools, system) -> TurnStream:
        """Satisfies `StreamingClient`. Same request, delivered incrementally."""
        return _AnthropicStream(
            self._client,
            self._model,
            self._rates,
            self._request(messages, tools, system),
        )

    async def create(self, messages, tools, system) -> NormalizedResponse:

        response = await self._client.messages.create(
            **self._request(messages, tools, system)
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        blocks = []
        cost_usd = cost(
            self._rates, self._model, in_token=input_tokens, out_token=output_tokens
        )
        for block in response.content:

            if block.type == "text":
                blocks.append(TextBlock(text=block.text))

            elif block.type == "tool_use":
                blocks.append(
                    ToolCallBlock(id=block.id, name=block.name, arguments=block.input)
                )

        return NormalizedResponse(
            blocks=blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=response.stop_reason,
            cost_usd=cost_usd,
        )


class _AnthropicStream(_StreamBase):
    """Anthropic's streaming turn, normalized to `str` deltas."""

    def __init__(self, client, model, rates, request: dict) -> None:
        super().__init__(model, rates)
        self._client = client
        self._request = request

    async def __aiter__(self) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._request) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()

        blocks: list[TextBlock | ToolCallBlock] = []
        for block in final.content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(
                    ToolCallBlock(id=block.id, name=block.name, arguments=block.input)
                )

        usage = getattr(final, "usage", None)
        input_tokens, output_tokens, cost_usd = self._meter(
            getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
        )
        self._response = NormalizedResponse(
            blocks=blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=final.stop_reason,
            cost_usd=cost_usd,
        )


class _OpenAIStream(_StreamBase):
    """An OpenAI-compatible streaming turn, reassembled into one response.

    Tool calls arrive as *fragments* — a name in one chunk, argument JSON split
    across several more, keyed only by an index. They have to be accumulated and
    parsed at the end; a partially-received `arguments` string is not valid JSON.
    """

    def __init__(self, client, model, rates, request: dict, fallback=None) -> None:
        super().__init__(model, rates)
        self._client = client
        self._request = request
        # A buffered retry of the same turn. Used when the stream fails before
        # it has shown anything, so degrading costs the user nothing visible.
        self._fallback = fallback

    async def _open(self):
        """Start the stream, retrying without `stream_options` if refused.

        `include_usage` is how a streamed OpenAI turn reports token counts at
        all, but not every compatible provider implements it — and losing
        streaming entirely over a metering flag is the wrong trade.
        """
        try:
            return await self._client.chat.completions.create(
                **self._request,
                stream=True,
                stream_options={"include_usage": True},
            )
        except BadRequestError as exc:
            if _is_tool_call_failure(exc):
                raise ToolCallingUnsupportedError(self._model) from exc
            logging.warning(
                "provider rejected stream_options; streaming without usage: %s", exc
            )
            return await self._client.chat.completions.create(
                **self._request, stream=True
            )

    async def __aiter__(self) -> AsyncIterator[str]:
        text_parts: list[str] = []
        # index -> {"id", "name", "arguments"}, accumulated across chunks.
        partial_calls: dict[int, dict] = {}
        finish_reason = "stop"
        input_tokens = output_tokens = None

        try:
            stream = await self._open()
            async for chunk in stream:  # noqa: B007
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                # The usage-only final chunk carries no choices.
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    text_parts.append(delta.content)
                    yield delta.content

                for fragment in delta.tool_calls or []:
                    call = partial_calls.setdefault(
                        fragment.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if fragment.id:
                        call["id"] = fragment.id
                    if fragment.function is None:
                        continue
                    if fragment.function.name:
                        call["name"] = fragment.function.name
                    if fragment.function.arguments:
                        call["arguments"] += fragment.function.arguments
        except APIError as exc:
            # Providers validate tool-call JSON server-side and can reject a
            # turn *mid-stream* — Groq answers "Failed to parse tool call
            # arguments as JSON" for models that emit malformed calls. Catching
            # only BadRequestError let that escape as a raw APIError.
            if text_parts or self._fallback is None:
                # Words are already on screen; silently restarting the turn
                # would repeat them. Report instead.
                if _is_tool_call_failure(exc):
                    raise ToolCallingUnsupportedError(self._model) from exc
                raise
            logging.warning(
                "streaming turn failed before any output (%s); "
                "retrying without streaming",
                exc,
            )
            self._response = await self._fallback()
            return

        blocks: list[TextBlock | ToolCallBlock] = []
        if text_parts:
            blocks.append(TextBlock(text="".join(text_parts)))
        for _, call in sorted(partial_calls.items()):
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                # Truncated argument JSON means the model failed to produce a
                # usable call — the same condition the buffered path reports.
                raise ToolCallingUnsupportedError(self._model) from exc
            blocks.append(
                ToolCallBlock(id=call["id"], name=call["name"], arguments=arguments)
            )

        input_tokens, output_tokens, cost_usd = self._meter(input_tokens, output_tokens)
        self._response = NormalizedResponse(
            blocks=blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=_STOP_REASONS.get(finish_reason, "end_turn"),
            cost_usd=cost_usd,
        )


class OpenAICompatibleClient(LLMClient):

    def __init__(self, model: str, api_key: str, base_url: str, rates: dict[str, dict]):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._rates = rates

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

    def _request(self, messages, tools, system) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                *self._to_openai(messages),
            ],
            "tools": [
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
            or None,
        }

    def stream(self, messages, tools, system) -> TurnStream:
        """Satisfies `StreamingClient`. Same request, delivered incrementally.

        The buffered `create` is handed over as a fallback: a stream that dies
        before producing any text degrades to a normal turn rather than losing
        it, which is the same posture the gateway takes when it is unreachable.
        """
        return _OpenAIStream(
            self._client,
            self._model,
            self._rates,
            self._request(messages, tools, system),
            fallback=lambda: self.create(messages, tools, system),
        )

    async def create(self, messages, tools, system) -> NormalizedResponse:

        try:
            response = await self._client.chat.completions.create(
                **self._request(messages, tools, system)
            )
        except APIError as e:
            if _is_tool_call_failure(e):
                raise ToolCallingUnsupportedError(self._model) from e
            raise

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
        stop_reason = _STOP_REASONS.get(finish, "end_turn")

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        cost_usd = cost(
            self._rates, self._model, in_token=input_tokens, out_token=output_tokens
        )

        return NormalizedResponse(
            blocks=blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            cost_usd=cost_usd,
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


def make_client(provider, model, api_key="", base_url=None, rates=None) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key, rates=rates)
    return OpenAICompatibleClient(
        model=model, api_key=api_key, base_url=base_url or _URLS[provider], rates=rates
    )
