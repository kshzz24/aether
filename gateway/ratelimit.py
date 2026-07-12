import time

# One atomic read-refill-consume-write. Running it as a single EVAL is what makes
# concurrent requests for the same key safe (no check-then-act race).
_SCRIPT = """
-- KEYS[1] = bucket key
-- ARGV = { capacity, refill_per_sec, now, cost }
local cap    = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now    = tonumber(ARGV[3])
local cost   = tonumber(ARGV[4])

local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1])
local ts     = tonumber(b[2])
if tokens == nil then          -- first sight of this key: start with a full bucket
  tokens = cap
  ts = now
end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(cap, tokens + elapsed * refill)   -- lazy refill, capped

local ok = 0
if tokens >= cost then
  tokens = tokens - cost
  ok = 1
end

redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], 3600)   -- reclaim idle keys after an hour
return ok
"""


class TokenBucket:
    def __init__(self, redis, *, capacity, refill_per_sec, now=time.time):
        self._redis = redis
        self._capacity = capacity
        self._refill = refill_per_sec
        self._now = now  # injectable clock (tests pass a fake)
        self._script = redis.register_script(_SCRIPT)  # register once, reuse

    async def allow(self, key: str, cost: int = 1) -> bool:
        now = self._now()
        ok = await self._script(
            keys=[f"ratelimit:{key}"],
            args=[self._capacity, self._refill, now, cost],
        )
        return bool(ok)
