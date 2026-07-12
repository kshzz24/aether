"""Task 5 — guarded semantic cache + the poisoning demo.

No network: the embedder is injected as a deterministic fake over a tiny 2-D
vector space. The point under test is the cache mechanism (cosine + threshold
gate), not any real embedding model.
"""

from __future__ import annotations

import numpy as np

from gateway.cache import SemanticCache, request_text
from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)

# A hand-built vector space:
#   "list files"   and "show files" are ~0.99 cosine — near-identical direction.
#   "delete files" is orthogonal to both.
_VECS = {
    "list files": np.array([1.0, 0.0]),
    "show files": np.array([0.99, 0.14]),
    "delete files": np.array([0.0, 1.0]),
}


async def fake_embed(text: str) -> np.ndarray:
    return _VECS[text]


def _resp(tag: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"cmpl-{tag}",
        created=1,
        model="m",
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=tag),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


RESP_LIST = _resp("list")
RESP_SHOW = _resp("show")


def test_request_text_uses_last_user_message():
    req = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ],
    )
    assert request_text(req) == "second"


async def test_empty_cache_misses():
    c = SemanticCache(threshold=0.97, embedder=fake_embed)
    assert await c.get("list files") is None


async def test_hit_above_threshold():
    c = SemanticCache(threshold=0.97, embedder=fake_embed)
    await c.put("list files", RESP_LIST)
    # cosine("show files", "list files") ~= 0.99 >= 0.97 -> hit
    assert await c.get("show files") == RESP_LIST


async def test_miss_below_threshold():
    c = SemanticCache(threshold=0.97, embedder=fake_embed)
    await c.put("list files", RESP_LIST)
    # "delete files" is orthogonal -> cosine 0 -> miss
    assert await c.get("delete files") is None


async def test_exact_text_hits():
    c = SemanticCache(threshold=0.97, embedder=fake_embed)
    await c.put("list files", RESP_LIST)
    # identical text -> cosine 1.0 -> always a hit
    assert await c.get("list files") == RESP_LIST


async def test_returns_nearest_of_several():
    c = SemanticCache(threshold=0.97, embedder=fake_embed)
    await c.put("delete files", _resp("delete"))  # orthogonal, far
    await c.put("list files", RESP_LIST)          # near "show files"
    got = await c.get("show files")
    assert got == RESP_LIST                        # picks the closest, not just any


async def test_poisoning_demo():
    """A loose threshold serves a WRONG answer for a near-identical prompt that
    actually needs a different one; the strict default refuses the fuzzy match.

    This is the concrete anti-pattern the phase is teaching — keep it.
    """
    loose = SemanticCache(threshold=0.90, embedder=fake_embed)
    await loose.put("list files", RESP_LIST)
    # POISONED: a "show files" query is served the "list files" answer.
    assert await loose.get("show files") == RESP_LIST

    strict = SemanticCache(threshold=0.999, embedder=fake_embed)
    await strict.put("list files", RESP_LIST)
    # Strict refuses the ~0.99 match -> no wrong answer served.
    assert await strict.get("show files") is None
