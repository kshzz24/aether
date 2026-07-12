from email import message
import hashlib
import json
import numpy as np
from redis.asyncio import Redis

from gateway.models import ChatCompletionRequest, ChatCompletionResponse


def canonical_key(req: ChatCompletionRequest) -> str:

    dump = json.dumps(req.model_dump(), sort_keys=True, separators=(",", ":"))
    hashed_key = hashlib.sha256(dump.encode()).hexdigest()
    return hashed_key


def request_text(req: ChatCompletionRequest) -> str:
    messages = req.messages
    n = len(messages)

    last_user_text = ""
    for i in range(n - 1, -1, -1):
        if messages[i].role == "user":
            last_user_text = messages[i].content
            break

    return last_user_text or ""


async def embed(text: str, *, model: str, client) -> np.ndarray:
    embedding = await client.embeddings.create(model=model, input=text)

    return np.array(embedding.data[0].embedding)


class ExactCache:

    def __init__(self, redis: Redis, *, ttl_sec: int):
        self._redis = redis
        self._ttl_sec = ttl_sec

    async def get(self, key: str) -> ChatCompletionResponse | None:
        r = self._redis
        redis_key = f"cache:{key}"
        resp: str | None = await r.get(redis_key)
        if not resp:
            return None

        return ChatCompletionResponse.model_validate_json(resp)

    async def put(self, key: str, resp: ChatCompletionResponse) -> None:
        r = self._redis
        redis_key = f"cache:{key}"

        await r.set(redis_key, resp.model_dump_json(), ex=self._ttl_sec)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class SemanticCache:
    def __init__(self, *, threshold: float, embedder):
        self._threshold = threshold
        self._embedder = embedder
        self._store: list[tuple[np.ndarray, ChatCompletionResponse]] = []

    async def get(self, text: str) -> ChatCompletionResponse | None:
        if not self._store:  # FIX 1: empty-store guard (was a bogus
            return None
        embedding_array = await self._embedder(text=text)

        stored_vectors = self._store

        best_resp = None
        best_cos = float("-inf")
        for vec, resp in stored_vectors:
            cosine_value = cosine(vec, embedding_array)
            if cosine_value >= best_cos:
                best_resp = resp
                best_cos = cosine_value

        return best_resp if best_cos >= self._threshold else None

    async def put(self, text: str, resp: ChatCompletionResponse) -> None:
        embedding_array = await self._embedder(text=text)
        self._store.append((embedding_array, resp))
