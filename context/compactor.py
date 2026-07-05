import logging

from client import LLMClient, Message, TextBlock, ToolCallBlock, ToolResultBlock

log = logging.getLogger(__name__)


def needs_compaction(input_tokens, max_context_tokens) -> bool:

    return input_tokens >= int(0.8 * max_context_tokens)


def _render_middle(messages: list[Message]) -> str:

    res = ""
    for message in messages:
        role = message.role
        for block in message.blocks:
            if isinstance(block, TextBlock):
                res += f"{role}:{block.text}"
            elif isinstance(block, ToolCallBlock):
                res += f"assistant called {block.name}({block.arguments})"
            elif isinstance(block, ToolResultBlock):
                res += f"tool result: {block.content} then"

    return res


async def compact(
    client: LLMClient, messages: list[Message], keep_recent: int
) -> tuple[list[Message], float]:

    cut = len(messages) - keep_recent
    while cut > 1 and messages[cut].role != "assistant":
        cut -= 1
    # cut <= 1 means keep_recent covers the whole transcript (or all but the
    # goal): nothing can be dropped while keeping the goal and whole turns
    # intact. Bail before a negative/short cut lands `recent` on an orphaned
    # tool_result (its tool_call swept into the summarized middle).
    if cut <= 1:
        return messages, 0.0
    goal, middle, recent = messages[0], messages[1:cut], messages[cut:]

    if not middle:
        return messages, 0.0

    rendered = _render_middle(middle)
    resp = await client.create(
        messages=[Message(role="user", blocks=[TextBlock(rendered)])],
        tools=[],
        system=(
            "You compress an agent transcript. Preserve decisions, "
            "file paths, and open problems. Be terse."
        ),
    )
    summary_text = "".join(b.text for b in resp.blocks if isinstance(b, TextBlock))
    new_goal = Message(
        role="user",
        blocks=[*goal.blocks, TextBlock(text=summary_text)],
    )
    log.info("compacted %d messages", len(middle))
    return [new_goal, *recent], resp.cost_usd
