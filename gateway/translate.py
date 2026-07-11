import time
import json
from uuid import uuid4

from client import (
    Message,
    NormalizedResponse,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    FunctionCall,
    ToolCall,
    Usage,
)


def to_internal(req: ChatCompletionRequest) -> tuple[list[Message], list[dict], str]:
    """OpenAI wire request -> the (messages, tools, system) that client.create wants.

    Inverse of OpenAICompatibleClient._to_openai. User text and tool results are
    separate rows on the wire but belong in one internal user turn, so they are
    buffered and flushed when an assistant turn (or the end) closes the group --
    consecutive single-block user messages would otherwise trip Anthropic's
    role-alternation rules upstream.
    """
    system = ""
    messages: list[Message] = []
    pending_user: list[TextBlock | ToolResultBlock] = []

    def flush_user() -> None:
        nonlocal pending_user
        if pending_user:
            messages.append(Message(role="user", blocks=pending_user))
            pending_user = []

    for m in req.messages:
        if m.role == "system":
            system = m.content or ""

        elif m.role == "user":
            if m.content is not None:
                pending_user.append(TextBlock(text=m.content))

        elif m.role == "tool":
            pending_user.append(
                ToolResultBlock(tool_call_id=m.tool_call_id, content=m.content or "")
            )

        elif m.role == "assistant":
            flush_user()
            blocks: list[TextBlock | ToolCallBlock] = []
            if m.content:
                blocks.append(TextBlock(text=m.content))
            for tc in m.tool_calls or []:
                blocks.append(
                    ToolCallBlock(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
            messages.append(Message(role="assistant", blocks=blocks))

    flush_user()

    tools = [
        {
            "name": t.function.name,
            "description": t.function.description,
            "parameters": t.function.parameters,
        }
        for t in (req.tools or [])
    ]

    return messages, tools, system


def to_wire(resp: NormalizedResponse, model: str) -> ChatCompletionResponse:

    stop_reason = resp.stop_reason
    out_token = resp.output_tokens
    inp_token = resp.input_tokens
    blocks = resp.blocks

    content = (
        "".join(block.text for block in blocks if isinstance(block, TextBlock)) or None
    )
    tool_calls = [
        ToolCall(
            id=block.id,
            type="function",
            function=FunctionCall(
                name=block.name, arguments=json.dumps(block.arguments)
            ),
        )
        for block in blocks
        if isinstance(block, ToolCallBlock)
    ]
    usage = Usage(
        prompt_tokens=inp_token,
        completion_tokens=out_token,
        total_tokens=inp_token + out_token,
    )

    message = ChatMessage(role="assistant", content=content, tool_calls=tool_calls)
    finish_reason = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "stop": "stop",
        "max_tokens": "length",
    }.get(stop_reason, "stop")

    chat_completion = ChatCompletionResponse(
        id=f"chat-{uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        usage=usage,
    )

    return chat_completion
