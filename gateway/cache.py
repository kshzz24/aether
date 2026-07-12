import hashlib
import json

from redis.asyncio import Redis

from gateway.models import ChatCompletionRequest, ChatCompletionResponse


def canonical_key(req: ChatCompletionRequest) -> str:

    dump = json.dumps(req.model_dump(), sort_keys=True, separators=(",", ":"))
    hashed_key = hashlib.sha256(dump.encode()).hexdigest()
    return hashed_key


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
