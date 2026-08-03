"""Streaming turns, normalized.

The design constraint these pin down: a stream yields plain `str`, and the
finished turn is the *same* `NormalizedResponse` the buffered path produces.
Provider chunk shape must not escape `client.py` (Invariant 3), and the agent
must not be able to tell which path produced a response.

The hard part is not the text — it is that OpenAI delivers tool calls as
fragments keyed by index, with the argument JSON split across chunks, so a
partially-received call is not parseable until the stream ends.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from openai import APIError

from client import (
    OpenAICompatibleClient,
    StreamNotFinishedError,
    TextBlock,
    ToolCallBlock,
    ToolCallingUnsupportedError,
)

_RATES = {"m": {"input": 1_000_000, "output": 2_000_000}}  # $1 / $2 per token


def _chunk(*, content=None, tool_calls=None, finish=None, usage=None):
    """One OpenAI streaming chunk, shaped like the SDK's."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choices = [] if (content is None and tool_calls is None and finish is None) else [
        SimpleNamespace(delta=delta, finish_reason=finish)
    ]
    return SimpleNamespace(choices=choices, usage=usage)


def _fragment(index, *, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def _usage(prompt=3, completion=5):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


def _client(chunks, *, on_create=None):
    """An OpenAICompatibleClient whose SDK returns `chunks`."""
    client = OpenAICompatibleClient(
        model="m", api_key="k", base_url="http://x", rates=_RATES
    )

    async def create(**kwargs):
        if on_create is not None:
            on_create(kwargs)
        return _FakeStream(chunks)

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


async def _drain(stream) -> list[str]:
    return [delta async for delta in stream]


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


async def test_deltas_arrive_in_order():
    stream = _client(
        [_chunk(content="Hel"), _chunk(content="lo"), _chunk(usage=_usage())]
    ).stream([], [], "sys")
    assert await _drain(stream) == ["Hel", "lo"]


async def test_deltas_are_plain_strings():
    """Provider chunk shape must stop inside client.py (Invariant 3)."""
    stream = _client([_chunk(content="hi"), _chunk(usage=_usage())]).stream([], [], "s")
    assert all(isinstance(d, str) for d in await _drain(stream))


async def test_the_deltas_concatenate_to_the_final_text():
    stream = _client(
        [_chunk(content="Hel"), _chunk(content="lo"), _chunk(usage=_usage())]
    ).stream([], [], "sys")
    deltas = await _drain(stream)
    assert stream.response.blocks == [TextBlock(text="".join(deltas))]


async def test_a_turn_with_no_text_has_no_text_block():
    stream = _client([_chunk(usage=_usage())]).stream([], [], "sys")
    await _drain(stream)
    assert stream.response.blocks == []


# --------------------------------------------------------------------------
# Tool calls, which arrive in fragments
# --------------------------------------------------------------------------


async def test_a_tool_call_split_across_chunks_is_reassembled():
    """The argument JSON is split mid-string; it is not parseable until the end."""
    stream = _client(
        [
            _chunk(tool_calls=[_fragment(0, call_id="c1", name="read_file")]),
            _chunk(tool_calls=[_fragment(0, arguments='{"path": ')]),
            _chunk(tool_calls=[_fragment(0, arguments='"a.py"}')]),
            _chunk(finish="tool_calls"),
            _chunk(usage=_usage()),
        ]
    ).stream([], [], "sys")
    await _drain(stream)

    assert stream.response.blocks == [
        ToolCallBlock(id="c1", name="read_file", arguments={"path": "a.py"})
    ]


async def test_two_tool_calls_are_kept_apart_by_index():
    stream = _client(
        [
            _chunk(
                tool_calls=[
                    _fragment(0, call_id="c1", name="read_file", arguments="{}"),
                    _fragment(1, call_id="c2", name="list_dir", arguments="{}"),
                ]
            ),
            _chunk(finish="tool_calls"),
            _chunk(usage=_usage()),
        ]
    ).stream([], [], "sys")
    await _drain(stream)

    assert [b.name for b in stream.response.blocks] == ["read_file", "list_dir"]


async def test_a_tool_call_with_no_arguments_becomes_an_empty_dict():
    stream = _client(
        [
            _chunk(tool_calls=[_fragment(0, call_id="c1", name="repo_map")]),
            _chunk(finish="tool_calls"),
            _chunk(usage=_usage()),
        ]
    ).stream([], [], "sys")
    await _drain(stream)
    assert stream.response.blocks[0].arguments == {}


async def test_truncated_argument_json_is_reported_as_unsupported():
    """The same condition the buffered path reports — a model that cannot
    produce a usable call — rather than a raw JSONDecodeError."""
    stream = _client(
        [
            _chunk(tool_calls=[_fragment(0, call_id="c1", name="f", arguments='{"a')]),
            _chunk(finish="tool_calls"),
            _chunk(usage=_usage()),
        ]
    ).stream([], [], "sys")
    with pytest.raises(ToolCallingUnsupportedError):
        await _drain(stream)


async def test_text_and_a_tool_call_can_arrive_in_one_turn():
    stream = _client(
        [
            _chunk(content="let me look"),
            _chunk(tool_calls=[_fragment(0, call_id="c1", name="f", arguments="{}")]),
            _chunk(finish="tool_calls"),
            _chunk(usage=_usage()),
        ]
    ).stream([], [], "sys")
    assert await _drain(stream) == ["let me look"]
    assert [type(b).__name__ for b in stream.response.blocks] == [
        "TextBlock",
        "ToolCallBlock",
    ]


# --------------------------------------------------------------------------
# Normalization: the agent must not be able to tell the paths apart
# --------------------------------------------------------------------------


async def test_stop_reason_is_normalized():
    stream = _client([_chunk(finish="tool_calls"), _chunk(usage=_usage())]).stream(
        [], [], "sys"
    )
    await _drain(stream)
    assert stream.response.stop_reason == "tool_use"


async def test_an_ordinary_finish_is_end_turn():
    stream = _client([_chunk(finish="stop"), _chunk(usage=_usage())]).stream(
        [], [], "sys"
    )
    await _drain(stream)
    assert stream.response.stop_reason == "end_turn"


async def test_an_unknown_finish_reason_falls_back_to_end_turn():
    stream = _client([_chunk(finish="weird"), _chunk(usage=_usage())]).stream(
        [], [], "sys"
    )
    await _drain(stream)
    assert stream.response.stop_reason == "end_turn"


async def test_usage_and_cost_are_carried_through():
    stream = _client([_chunk(content="x"), _chunk(usage=_usage(3, 5))]).stream(
        [], [], "sys"
    )
    await _drain(stream)
    assert (stream.response.input_tokens, stream.response.output_tokens) == (3, 5)
    assert stream.response.cost_usd == pytest.approx(3 * 1 + 5 * 2)


async def test_a_provider_reporting_no_usage_meters_zero_and_warns(caplog):
    """A cost governor silently reading $0 is the failure to avoid — some
    OpenAI-compatible providers drop usage from streamed responses."""
    stream = _client([_chunk(content="x"), _chunk(finish="stop")]).stream([], [], "s")
    with caplog.at_level("WARNING"):
        await _drain(stream)
    assert stream.response.cost_usd == 0.0
    assert "no usage" in caplog.text


# --------------------------------------------------------------------------
# The request, and lifecycle
# --------------------------------------------------------------------------


async def test_usage_is_requested_explicitly():
    """Without stream_options a streamed OpenAI turn reports no tokens at all."""
    seen: list[dict] = []
    stream = _client([_chunk(usage=_usage())], on_create=seen.append).stream(
        [], [], "sys"
    )
    await _drain(stream)
    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}


async def test_the_streamed_request_matches_the_buffered_one():
    """Same messages, same tools, same system — only the delivery differs."""
    client = OpenAICompatibleClient(
        model="m", api_key="k", base_url="http://x", rates=_RATES
    )
    tools = [{"name": "f", "description": "d", "parameters": {"type": "object"}}]
    request = client._request([], tools, "the system prompt")
    assert request["model"] == "m"
    assert request["messages"][0] == {
        "role": "system",
        "content": "the system prompt",
    }
    assert json.dumps(request["tools"])  # serializable, i.e. no SDK objects


async def test_reading_the_response_early_is_an_error_not_a_wrong_answer():
    stream = _client([_chunk(content="x"), _chunk(usage=_usage())]).stream(
        [], [], "sys"
    )
    with pytest.raises(StreamNotFinishedError):
        _ = stream.response


def test_a_client_advertises_the_streaming_capability():
    from client import StreamingClient

    client = OpenAICompatibleClient(
        model="m", api_key="k", base_url="http://x", rates=_RATES
    )
    assert isinstance(client, StreamingClient)


# --------------------------------------------------------------------------
# When the provider rejects a turn mid-stream
# --------------------------------------------------------------------------


class _FakeAPIError(APIError):
    """An SDK error without the SDK's constructor requirements."""

    def __init__(self, message: str, code: str | None = None) -> None:
        Exception.__init__(self, message)
        self.message = message
        self.code = code


def _exploding_client(exc, *, before: list | None = None, buffered=None):
    """A client whose stream raises `exc`, optionally after some chunks."""
    client = OpenAICompatibleClient(
        model="m", api_key="k", base_url="http://x", rates=_RATES
    )

    class _Boom:
        def __aiter__(self):
            async def gen():
                for chunk in before or []:
                    yield chunk
                raise exc

            return gen()

    async def create(**kwargs):
        return _Boom()

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    if buffered is not None:
        client.create = buffered
    return client


def test_a_tool_call_failure_is_recognised_by_code():
    from client import _is_tool_call_failure

    assert _is_tool_call_failure(_FakeAPIError("x", code="tool_use_failed"))


def test_a_tool_call_failure_is_recognised_by_message():
    """Groq answers this in prose rather than a code."""
    from client import _is_tool_call_failure

    assert _is_tool_call_failure(
        _FakeAPIError("Failed to parse tool call arguments as JSON")
    )


def test_an_unrelated_api_error_is_not_a_tool_call_failure():
    from client import _is_tool_call_failure

    assert not _is_tool_call_failure(_FakeAPIError("rate limit exceeded"))


async def test_a_stream_that_dies_before_any_output_degrades_to_buffered():
    """The user loses streaming for that turn, not the turn."""
    from client import NormalizedResponse

    async def buffered(*_a, **_k):
        return NormalizedResponse(
            blocks=[TextBlock(text="recovered")],
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            stop_reason="end_turn",
        )

    client = _exploding_client(
        _FakeAPIError("Failed to parse tool call arguments as JSON"),
        buffered=buffered,
    )
    stream = client.stream([], [], "sys")

    assert await _drain(stream) == []
    assert stream.response.blocks == [TextBlock(text="recovered")]


async def test_a_stream_that_dies_after_output_does_not_silently_restart():
    """Retrying once words are on screen would print the answer twice."""
    client = _exploding_client(
        _FakeAPIError("Failed to parse tool call arguments as JSON"),
        before=[_chunk(content="already shown")],
    )
    stream = client.stream([], [], "sys")
    with pytest.raises(ToolCallingUnsupportedError):
        await _drain(stream)


async def test_an_unrelated_error_after_output_propagates_unchanged():
    client = _exploding_client(
        _FakeAPIError("rate limit exceeded"),
        before=[_chunk(content="shown")],
    )
    stream = client.stream([], [], "sys")
    with pytest.raises(APIError):
        await _drain(stream)


def test_the_gateway_client_does_not_advertise_streaming():
    """Not an oversight: a gateway turn degrades to one non-streamed response,
    which is the graceful-degradation posture Phase 3 established."""
    from client import StreamingClient
    from gateway.client import GatewayClient

    client = GatewayClient(
        gateway_url="http://x", model="m", fallback=None, rates=_RATES
    )
    assert not isinstance(client, StreamingClient)
