# Phase 4 — Gateway Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **FORGE workflow note:** On this project **the user writes the implementation; Claude guides and reviews**. This plan is therefore a *guide*, not paste-ready code. Each task gives you the exact file, the exact interface signatures, and the **failing tests to make pass** (TDD target), plus the algorithm and gotchas. You write the implementation bodies. Reference code is shown only for fiddly infra (Redis Lua, cosine math). Ask Claude to review after each task's tests go green.

**Goal:** Make the Phase-3 gateway resilient and cost-aware — add a per-key rate limiter, a two-layer cache (exact + guarded semantic), a per-provider circuit breaker with retry/backoff, and a `/stats` metrics endpoint — with the agent loop unchanged.

**Architecture:** The endpoint `POST /v1/chat/completions` becomes a gate chain: rate-limit → exact cache → semantic cache → breaker → provider (with retry/backoff) → populate cache + ledger. Each gate is its own small module with one responsibility. Redis holds counter/equality state; numpy holds meaning-vectors; Postgres (the existing ledger) backs `/stats`.

**Tech Stack:** Python 3.11+, FastAPI, async `redis` client, `numpy`, existing `openai` SDK (embeddings), Postgres via `asyncpg`, pytest.

## Global Constraints

- Python 3.11+, full type hints, `dataclass` for value types. `ruff` clean.
- The agent core (`agent.py`) and `client.py` are **not touched**. All work is server-side under `gateway/`.
- A tool/infra failure is data, not a crash: a dead Redis or a stale semantic index degrades a feature, never the request path — except 429/503, which are the *correct* answers this phase introduces.
- Every request outcome (success **or** failure) is recorded in the ledger so `/stats` is computable.
- One+ pytest unit test per piece. Commit at each green step.
- Config lives in the `gateway.*` blocks (§7 of the spec); the gateway process loads them at startup (see Task 1).
- Spec of record: `docs/superpowers/specs/2026-07-12-phase-4-gateway-depth-design.md`.

---

## File structure (locked before tasks)

| File | Responsibility | Task |
|---|---|---|
| `gateway/config.py` (new) | Pydantic `GatewayConfig` for the `gateway.*` blocks; loaded at startup | 1 |
| `gateway/ledger.py` (modify) | add `status` column + failure-recording path | 2 |
| `gateway/ratelimit.py` (new) | `TokenBucket` over Redis (atomic Lua refill+consume) | 3 |
| `gateway/cache.py` (new) | `ExactCache` (Redis) + `SemanticCache` (numpy) + `embed()` | 4, 5 |
| `gateway/breaker.py` (new) | `CircuitBreaker` + `call_with_resilience()` retry/backoff | 6 |
| `gateway/metrics.py` (new) | ledger aggregation for `/stats` | 7 |
| `gateway/server.py` (modify) | wire the gate chain; add `GET /stats` | 8 |
| `tests/gateway/…` (new) | one test module per piece | each |

Dependencies to add (`pyproject.toml` / requirements): `redis`, `numpy`. `openai`, `asyncpg`, `fastapi` already present.

Redis for local dev: `docker run -p 6379:6379 redis:7-alpine` (document this in the task, don't assume it's running).

---

## Task 1: Gateway config + dependencies + Redis bring-up

**Files:**
- Create: `gateway/config.py`
- Modify: `pyproject.toml` (add `redis`, `numpy`), `gateway/server.py:16-25` (load config in `lifespan`)
- Test: `tests/gateway/test_config.py`

**Interfaces:**
- Produces:
  ```python
  # gateway/config.py
  class RedisCfg(BaseModel):     url: str = "redis://localhost:6379"
  class RateLimitCfg(BaseModel): capacity: int = 60; refill_per_sec: float = 1.0
  class CacheCfg(BaseModel):     exact_ttl_sec: int = 3600
  class SemanticCfg(BaseModel):  enabled: bool = False; threshold: float = 0.97; model: str = "text-embedding-3-small"
  class BreakerCfg(BaseModel):   fail_threshold: int = 5; cooldown_sec: float = 30
  class RetryCfg(BaseModel):     max_attempts: int = 3; base_delay_sec: float = 0.5
  class GatewayConfig(BaseModel):
      redis: RedisCfg = RedisCfg()
      ratelimit: RateLimitCfg = RateLimitCfg()
      cache: CacheCfg = CacheCfg()
      semantic: SemanticCfg = SemanticCfg()
      breaker: BreakerCfg = BreakerCfg()
      retry: RetryCfg = RetryCfg()

  def load_gateway_config(path: str | None = None) -> GatewayConfig: ...
  ```
  Loads the `[gateway]` table from the given TOML (default `.forge/config.toml`); missing file / missing keys → all defaults.

- [ ] **Step 1 — Write the failing tests** (`tests/gateway/test_config.py`):

```python
def test_defaults_when_no_file():
    cfg = load_gateway_config(path="does-not-exist.toml")
    assert cfg.semantic.enabled is False
    assert cfg.ratelimit.capacity == 60

def test_reads_gateway_table(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[gateway.semantic]\nenabled = true\nthreshold = 0.9\n')
    cfg = load_gateway_config(path=str(p))
    assert cfg.semantic.enabled is True
    assert cfg.semantic.threshold == 0.9
    assert cfg.ratelimit.capacity == 60  # untouched block keeps default
```

- [ ] **Step 2 — Run, verify fail:** `pytest tests/gateway/test_config.py -v` → FAIL (module missing).
- [ ] **Step 3 — Implement** `gateway/config.py`. Use `tomllib` to read, pull the `gateway` sub-table, feed it to `GatewayConfig(**table)`. Reuse the Phase-2 config style if one exists (`grep -rn "BaseModel\|ForgeConfig" .` first — follow that pattern).
- [ ] **Step 4 — Wire into `lifespan`** (`gateway/server.py`): alongside `app.state.prices`, add `app.state.gwcfg = load_gateway_config()` and `app.state.redis = redis.asyncio.from_url(app.state.gwcfg.redis.url)`; close it on shutdown.
- [ ] **Step 5 — Run, verify pass**, then `ruff check gateway/config.py`.
- [ ] **Step 6 — Commit:** `rtk git add gateway/config.py tests/gateway/test_config.py pyproject.toml gateway/server.py && rtk git commit -m "phase-4: gateway config + redis client"`

**Gotcha:** `redis.asyncio` is the async client (`from redis import asyncio as aioredis` or `redis.asyncio.from_url`). Don't use the sync client in an async server.

---

## Task 2: Ledger — status column + failure recording

**Files:**
- Modify: `gateway/ledger.py`
- Test: `tests/gateway/test_ledger_status.py`

**Interfaces:**
- Consumes: existing `init_pool(dsn)`, `record(pool, *, provider, model, input_tokens, output_tokens, cost_usd, latency_ms)`.
- Produces: `record(...)` gains `status: str = "ok"`; a helper `record_failure(pool, *, provider, model, latency_ms, error: str)` that inserts a row with `status="error"` and zero tokens/cost.
- Schema: table gains `status TEXT NOT NULL DEFAULT 'ok'` (and optionally `error TEXT`). The `CREATE TABLE IF NOT EXISTS` must include it; add an idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS status ...` for existing tables.

- [ ] **Step 1 — Failing test** (needs a live Postgres; mark with a `pg` marker or skip if `FORGE_LEDGER_DSN` unset):

```python
async def test_record_failure_marks_status(pool):
    await record_failure(pool, provider="openai", model="gpt-x", latency_ms=12, error="boom")
    row = await pool.fetchrow("SELECT status, cost_usd FROM ledger ORDER BY id DESC LIMIT 1")
    assert row["status"] == "error"
    assert row["cost_usd"] == 0
```

- [ ] **Step 2 — Run, verify fail** (column/function missing).
- [ ] **Step 3 — Implement:** add the column to the create DDL + an `ALTER ... IF NOT EXISTS`; add `status` param to `record`; write `record_failure`.
- [ ] **Step 4 — Run, verify pass.**
- [ ] **Step 5 — Commit:** `rtk git commit -am "phase-4: ledger records request status"`

**Gotcha:** keep the write append-only (INSERT only, never UPDATE) — that invariant is the point of the ledger.

---

## Task 3: Rate limiter — `TokenBucket` over Redis

**Files:**
- Create: `gateway/ratelimit.py`
- Test: `tests/gateway/test_ratelimit.py`

**Interfaces:**
- Produces:
  ```python
  class TokenBucket:
      def __init__(self, redis, *, capacity: int, refill_per_sec: float): ...
      async def allow(self, key: str, cost: int = 1) -> bool: ...
      # True if a token was consumed; False if bucket empty (→ caller returns 429)
  ```

**Algorithm:** classic token bucket. State per key in Redis: `tokens` (float), `ts` (last refill epoch). On `allow`: elapsed = now − ts; tokens = min(capacity, tokens + elapsed·refill); if tokens ≥ cost → tokens −= cost, allow; write back. **Do this in one atomic Lua `EVAL`** so concurrent requests can't double-spend. Reference script (adapt keys/args):

```lua
-- KEYS[1]=bucket key  ARGV[1]=capacity ARGV[2]=refill ARGV[3]=now ARGV[4]=cost
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[1])
local ts = tonumber(b[2]) or tonumber(ARGV[3])
local elapsed = math.max(0, tonumber(ARGV[3]) - ts)
tokens = math.min(tonumber(ARGV[1]), tokens + elapsed * tonumber(ARGV[2]))
local ok = 0
if tokens >= tonumber(ARGV[4]) then tokens = tokens - tonumber(ARGV[4]); ok = 1 end
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[3])
redis.call('EXPIRE', KEYS[1], 3600)
return ok
```

- [ ] **Step 1 — Failing tests** (use a real Redis, or `fakeredis.aioredis`; add `fakeredis` as a dev dep if you prefer no container in tests):

```python
async def test_allows_up_to_capacity(redis):
    b = TokenBucket(redis, capacity=3, refill_per_sec=0)
    assert [await b.allow("k") for _ in range(4)] == [True, True, True, False]

async def test_refills_over_time(redis, monkeypatch):
    b = TokenBucket(redis, capacity=1, refill_per_sec=1000)  # fast refill
    assert await b.allow("k") is True
    assert await b.allow("k") is False
    await asyncio.sleep(0.01)                                 # ~10 tokens refilled
    assert await b.allow("k") is True

async def test_keys_are_isolated(redis):
    b = TokenBucket(redis, capacity=1, refill_per_sec=0)
    assert await b.allow("a") is True
    assert await b.allow("b") is True   # different key, own bucket
```

- [ ] **Step 2 — Run, verify fail.**
- [ ] **Step 3 — Implement** using `redis.eval(SCRIPT, 1, key, capacity, refill, now, cost)`. Register the script once (`redis.register_script`) for efficiency.
- [ ] **Step 4 — Run, verify pass.**
- [ ] **Step 5 — Commit:** `rtk git commit -am "phase-4: per-key token-bucket rate limiter"`

**Gotcha:** compute `now` in Python (`time.time()`) and pass it in — don't use Redis `TIME` inside the script, it complicates testing and determinism.

---

## Task 4: Exact cache (Redis)

**Files:**
- Create: `gateway/cache.py` (the `ExactCache` half + the key function)
- Test: `tests/gateway/test_cache_exact.py`

**Interfaces:**
- Produces:
  ```python
  def canonical_key(req: ChatCompletionRequest) -> str: ...   # sha256 hex
  class ExactCache:
      def __init__(self, redis, *, ttl_sec: int): ...
      async def get(self, key: str) -> ChatCompletionResponse | None: ...
      async def put(self, key: str, resp: ChatCompletionResponse) -> None: ...
  ```

**Key:** `sha256` over a stable JSON dump of the output-affecting fields — `model`, `messages`, `tools`, and sampling params like `temperature`. Use `json.dumps(..., sort_keys=True, separators=(",",":"))` on `req.model_dump()` (drop volatile/non-output fields). Store the value as `resp.model_dump_json()`; rehydrate with `ChatCompletionResponse.model_validate_json`.

- [ ] **Step 1 — Failing tests:**

```python
def test_key_is_stable_across_field_order():
    a = ChatCompletionRequest(model="m", messages=[{"role":"user","content":"hi"}])
    b = ChatCompletionRequest(model="m", messages=[{"role":"user","content":"hi"}])
    assert canonical_key(a) == canonical_key(b)

def test_key_differs_on_content():
    a = ChatCompletionRequest(model="m", messages=[{"role":"user","content":"hi"}])
    b = ChatCompletionRequest(model="m", messages=[{"role":"user","content":"bye"}])
    assert canonical_key(a) != canonical_key(b)

async def test_put_then_get_roundtrips(redis, sample_response):
    c = ExactCache(redis, ttl_sec=60)
    await c.put("k", sample_response)
    got = await c.get("k")
    assert got == sample_response

async def test_miss_returns_none(redis):
    assert await ExactCache(redis, ttl_sec=60).get("absent") is None
```

- [ ] **Step 2 — Run, verify fail.**
- [ ] **Step 3 — Implement.** `put` uses `redis.set(key, json, ex=ttl_sec)`.
- [ ] **Step 4 — Run, verify pass.**
- [ ] **Step 5 — Commit:** `rtk git commit -am "phase-4: exact response cache (redis)"`

---

## Task 5: Semantic cache (numpy) — guarded, with the poisoning demo

**Files:**
- Modify: `gateway/cache.py` (add `embed()` + `SemanticCache`)
- Test: `tests/gateway/test_cache_semantic.py`

**Interfaces:**
- Produces:
  ```python
  async def embed(text: str, *, model: str, client) -> np.ndarray: ...  # provider embedding endpoint
  class SemanticCache:
      def __init__(self, *, threshold: float, embedder): ...  # embedder: async (str)->np.ndarray
      async def get(self, text: str) -> ChatCompletionResponse | None: ...
      async def put(self, text: str, resp: ChatCompletionResponse) -> None: ...
  ```
  Holds `list[tuple[np.ndarray, ChatCompletionResponse]]` in memory. `get` embeds `text`, cosine vs all stored, returns best **iff** `cos ≥ threshold`, else `None`.

**Cosine:** `float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))`.

**Request→text:** derive the text to embed from the last user message content (a helper `request_text(req) -> str`). Keep it simple; document the choice.

**Inject the embedder** so tests don't hit the network — pass a fake `embedder` in tests that maps known strings to fixed vectors.

- [ ] **Step 1 — Failing tests** (use a stub embedder — the point is the cache mechanism, not the model):

```python
def vecs():  # deterministic fake space
    return {"list files": np.array([1.0, 0.0]),
            "show files": np.array([0.99, 0.14]),   # ~0.99 cosine — near-identical
            "delete files": np.array([0.0, 1.0])}   # orthogonal

async def fake_embed(text): return vecs()[text]

async def test_hit_above_threshold():
    c = SemanticCache(threshold=0.97, embedder=fake_embed)
    await c.put("list files", RESP_A)
    assert await c.get("show files") == RESP_A     # 0.99 ≥ 0.97 → hit

async def test_miss_below_threshold():
    c = SemanticCache(threshold=0.97, embedder=fake_embed)
    await c.put("list files", RESP_A)
    assert await c.get("delete files") is None     # orthogonal → miss

async def test_poisoning_demo():
    """A loose threshold returns a WRONG answer for a near-identical prompt
    that actually needs a different one. The default 0.97 does not."""
    loose = SemanticCache(threshold=0.90, embedder=fake_embed)
    await loose.put("list files", RESP_LIST)
    assert await loose.get("show files") == RESP_LIST   # POISONED: served list-answer for a show-query
    strict = SemanticCache(threshold=0.999, embedder=fake_embed)
    await strict.put("list files", RESP_LIST)
    assert await strict.get("show files") is None       # strict refuses the fuzzy match
```

- [ ] **Step 2 — Run, verify fail.**
- [ ] **Step 3 — Implement** `SemanticCache` + `embed()` (real `embed` calls `client.embeddings.create(model=..., input=text)` and wraps `np.array(resp.data[0].embedding)`).
- [ ] **Step 4 — Run, verify pass.** The poisoning test is the deliverable that makes the anti-pattern concrete — keep it.
- [ ] **Step 5 — Commit:** `rtk git commit -am "phase-4: guarded semantic cache + poisoning demo"`

**Gotcha:** the cache stays **disabled** unless `gateway.semantic.enabled` is true — that wiring happens in Task 8; here it's a standalone, injectable unit.

---

## Task 6: Circuit breaker + retry/backoff

**Files:**
- Create: `gateway/breaker.py`
- Test: `tests/gateway/test_breaker.py`

**Interfaces:**
- Produces:
  ```python
  class BreakerOpen(Exception): ...
  class CircuitBreaker:
      def __init__(self, *, fail_threshold: int, cooldown_sec: float, clock=time.monotonic): ...
      def allow(self) -> bool: ...          # False when OPEN and still cooling
      def record_success(self) -> None: ...
      def record_failure(self) -> None: ...
      @property
      def state(self) -> str: ...           # "closed" | "open" | "half_open"

  async def call_with_resilience(breaker, fn, *, max_attempts, base_delay_sec,
                                 transient: tuple[type[Exception], ...]) -> Any: ...
  # raises BreakerOpen if breaker.allow() is False; else retries fn() on `transient`
  # errors with exp backoff+jitter up to max_attempts, updating breaker each way.
  ```

**State machine:** closed → (fail_threshold consecutive failures) → open → (after cooldown, `allow()` returns True once) → half_open → success → closed / failure → open. Inject `clock` so tests advance time without sleeping.

**Backoff:** delay = `base_delay_sec * 2**attempt * (0.5 + random())` (full jitter-ish). Only retry `transient` exceptions; re-raise others immediately (a 400 must **not** retry or trip the breaker).

- [ ] **Step 1 — Failing tests:**

```python
def test_opens_after_threshold():
    b = CircuitBreaker(fail_threshold=3, cooldown_sec=10, clock=lambda: 0)
    for _ in range(3): b.record_failure()
    assert b.state == "open"
    assert b.allow() is False

def test_half_open_then_close_on_success():
    t = {"now": 0.0}
    b = CircuitBreaker(fail_threshold=1, cooldown_sec=5, clock=lambda: t["now"])
    b.record_failure(); assert b.allow() is False
    t["now"] = 6.0                       # past cooldown
    assert b.allow() is True             # half-open probe allowed
    b.record_success(); assert b.state == "closed"

async def test_call_retries_transient_then_succeeds():
    b = CircuitBreaker(fail_threshold=5, cooldown_sec=1)
    calls = {"n": 0}
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3: raise TimeoutError()
        return "ok"
    out = await call_with_resilience(b, flaky, max_attempts=3,
                                     base_delay_sec=0.0, transient=(TimeoutError,))
    assert out == "ok" and calls["n"] == 3

async def test_call_raises_breaker_open_when_open():
    b = CircuitBreaker(fail_threshold=1, cooldown_sec=99)
    b.record_failure()
    with pytest.raises(BreakerOpen):
        await call_with_resilience(b, lambda: None, max_attempts=1,
                                   base_delay_sec=0.0, transient=(TimeoutError,))
```

- [ ] **Step 2 — Run, verify fail.**
- [ ] **Step 3 — Implement** the state machine + wrapper. Keep one breaker instance **per provider** (a `dict[str, CircuitBreaker]`), created in Task 8.
- [ ] **Step 4 — Run, verify pass.**
- [ ] **Step 5 — Commit:** `rtk git commit -am "phase-4: circuit breaker + retry/backoff"`

**Gotcha:** classify what's transient. `openai.APITimeoutError`, `openai.APIConnectionError`, and 5xx `openai.APIStatusError` are transient; `openai.BadRequestError` (400) is not. Map these in Task 8 where the provider call lives.

---

## Task 7: Metrics — `/stats` aggregation

**Files:**
- Create: `gateway/metrics.py`
- Test: `tests/gateway/test_metrics.py`

**Interfaces:**
- Produces:
  ```python
  async def compute_stats(pool, redis) -> dict: ...
  # {"p95_latency_ms", "error_rate", "cost_by_model", "requests_total", "cache_hit_rate"}
  ```

**Queries (single round-trips over the ledger):**
- `p95_latency_ms`: `SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FROM ledger WHERE status='ok'`
- `error_rate`: `SELECT avg((status='error')::int) FROM ledger` (0 if empty).
- `cost_by_model`: `SELECT model, sum(cost_usd) FROM ledger GROUP BY model`.
- `requests_total`: `SELECT count(*) FROM ledger`.
- `cache_hit_rate`: from Redis counters `forge:cache:hits` / (`hits`+`misses`), incremented in Task 8. 0 if no data.

- [ ] **Step 1 — Failing test** (seed rows, then assert):

```python
async def test_stats_over_seeded_ledger(pool, redis):
    await record(pool, provider="openai", model="m1", input_tokens=1, output_tokens=1, cost_usd=1.0, latency_ms=100)
    await record(pool, provider="openai", model="m1", input_tokens=1, output_tokens=1, cost_usd=2.0, latency_ms=300)
    await record_failure(pool, provider="openai", model="m1", latency_ms=50, error="x")
    s = await compute_stats(pool, redis)
    assert s["requests_total"] == 3
    assert abs(s["error_rate"] - 1/3) < 1e-6
    assert abs(s["cost_by_model"]["m1"] - 3.0) < 1e-6
```

- [ ] **Step 2 — Run, verify fail.**
- [ ] **Step 3 — Implement** `compute_stats`. Guard empty ledger (return zeros, not `None`).
- [ ] **Step 4 — Run, verify pass.**
- [ ] **Step 5 — Commit:** `rtk git commit -am "phase-4: /stats ledger aggregation"`

---

## Task 8: Wire the gate chain into `server.py` (+ `GET /stats`)

**Files:**
- Modify: `gateway/server.py`
- Test: `tests/gateway/test_server_chain.py` (FastAPI `TestClient`, upstream `create` mocked)

**Interfaces:**
- Consumes everything above. Builds, at startup (`lifespan`): the Redis client, `TokenBucket`, `ExactCache`, `SemanticCache` (only if `gwcfg.semantic.enabled`), and `breakers: dict[str, CircuitBreaker]` (lazily per provider).

**Endpoint order (this IS the phase):**
```python
@app.post("/v1/chat/completions")
async def chat_completions(req, request):            # need raw request for Authorization
    key = request.headers.get("authorization", "default")
    if not await bucket.allow(key):                  # 1. rate limit
        raise HTTPException(429, "rate limit exceeded")

    ckey = canonical_key(req)
    if hit := await exact.get(ckey):                 # 2. exact cache
        await redis.incr("forge:cache:hits"); return hit
    if semantic and (hit := await semantic.get(request_text(req))):  # 3. semantic (if enabled)
        await redis.incr("forge:cache:hits"); return hit
    await redis.incr("forge:cache:misses")

    provider = route(req.model)
    breaker = breakers.setdefault(provider, CircuitBreaker(**gwcfg.breaker.model_dump()))
    messages, tools, system = to_internal(req)
    client = make_client(provider=provider, model=req.model, api_key=..., rates=...)

    t0 = time.perf_counter()
    try:                                             # 4+5. breaker + retry + call
        resp = await call_with_resilience(
            breaker, lambda: client.create(messages=messages, tools=tools, system=system),
            max_attempts=gwcfg.retry.max_attempts, base_delay_sec=gwcfg.retry.base_delay_sec,
            transient=TRANSIENT)
    except BreakerOpen:
        raise HTTPException(503, f"{provider} circuit open")
    except ToolCallingUnsupportedError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        await ledger.record_failure(pool, provider=provider, model=req.model,
                                    latency_ms=int((time.perf_counter()-t0)*1000), error=str(e))
        raise HTTPException(502, f"upstream error:{e}")

    latency = int((time.perf_counter()-t0)*1000)
    await ledger.record(pool, provider=provider, model=req.model,
                        input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                        cost_usd=resp.cost_usd, latency_ms=latency, status="ok")
    wire = to_wire(resp, req.model)
    await exact.put(ckey, wire)
    if semantic: await semantic.put(request_text(req), wire)
    return wire

@app.get("/stats")
async def stats(): return await compute_stats(pool, redis)
```

- [ ] **Step 1 — Failing tests** (mock `client.create`; use `fakeredis`; a real or skipped Postgres):

```python
def test_second_identical_request_is_cache_hit(client, mock_create):
    r1 = client.post("/v1/chat/completions", json=REQ); assert r1.status_code == 200
    r2 = client.post("/v1/chat/completions", json=REQ); assert r2.status_code == 200
    assert mock_create.call_count == 1          # second served from cache, no upstream

def test_rate_limit_returns_429(client_low_capacity, mock_create):
    for _ in range(CAPACITY): assert client.post(...).status_code == 200
    assert client.post(...).status_code == 429

def test_breaker_opens_and_returns_503(client, failing_create):
    for _ in range(FAIL_THRESHOLD + 1):
        client.post("/v1/chat/completions", json=REQ)   # induce failures
    assert client.post("/v1/chat/completions", json=REQ).status_code == 503
    # while open, upstream is NOT called again
```

- [ ] **Step 2 — Run, verify fail.**
- [ ] **Step 3 — Implement** the wiring. Define `TRANSIENT = (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)` (verify exact names against the installed `openai` version). Move `route()`, `ENV_KEYS`, `rates` lookup as they are today.
- [ ] **Step 4 — Run, verify pass.** Then `ruff check gateway/`.
- [ ] **Step 5 — Commit:** `rtk git commit -am "phase-4: wire rate-limit/cache/breaker gate chain + /stats"`

**Gotcha — cache-hit billing:** a cache hit returns the wire response but writes **no ledger row** and is **not** re-billed. Only real upstream calls hit the ledger. `cache_hit_rate` comes from the Redis counters, exactly as designed.

**Gotcha — rate-limit before cache:** keep the order above. A cache hit still spends a token. That is the deliberate §4.1 tradeoff — leave a one-line comment in the code saying so, so a future reader doesn't "fix" it.

---

## Task 9: End-to-end acceptance (the Phase-4 gate)

**Files:** none new — a manual/scripted run. Optionally `tests/gateway/test_acceptance_phase4.py` behind a `live` marker.

**Preconditions:** Redis up (`docker run -p 6379:6379 redis:7-alpine`), Postgres up with `FORGE_LEDGER_DSN` set, provider key set, gateway running (`uvicorn gateway.server:app`).

- [ ] **Step 1 — Cache:** send the same request twice through the agent (`forge --gateway-url … "…"` or `curl`). Second is faster; `GET /stats` shows non-zero `cache_hit_rate`; ledger row count increased by 1, not 2.
- [ ] **Step 2 — Rate limit:** set a tiny `capacity`; fire past it; observe **429**.
- [ ] **Step 3 — Breaker:** point a model at a bad base URL / kill the provider; fire `fail_threshold+1` requests; observe the breaker **open** and return **503** without an upstream call; after `cooldown_sec`, a probe is allowed.
- [ ] **Step 4 — Stats:** `GET /stats` returns p95 latency, error rate, cost-by-model from the real runs.
- [ ] **Step 5 — Loop unchanged:** run the same agent goal **with and without** `--gateway-url`; behavior is byte-identical (Phase-3 gate still green).
- [ ] **Step 6 — Update `tracking.md`** §9: mark Phase 4 built, with the acceptance evidence (mirror the Phase-3 row's style). Commit: `rtk git commit -am "phase-4: gateway depth complete — acceptance gate green"`.

---

## Self-review (against the spec)

**Spec coverage:**
- §4 gate chain → Task 8. §4.1 ordering tradeoff → Task 8 comment. §4.2 canonical key → Task 4. §4.3 cache-hit = no ledger row → Task 8 gotcha.
- §5.1 rate limiter → Task 3. §5.2 caches → Tasks 4–5. §5.3 breaker → Task 6. §5.4 `/stats` → Task 7.
- §6 semantic guard (off by default, high threshold, poisoning test) → Task 5 + Task 8 enable-gate.
- §7 config → Task 1. §8 deps → Task 1. §9 tests → each task. §11 acceptance gate → Task 9.
- Ledger `status` column (needed by §5.4) → Task 2.

**Placeholder scan:** none — every task has concrete tests, signatures, and commands.

**Type consistency:** `canonical_key`, `ExactCache.get/put`, `SemanticCache.get/put`, `TokenBucket.allow`, `CircuitBreaker`/`call_with_resilience`/`BreakerOpen`, `compute_stats`, `record`/`record_failure` are named identically where produced (Tasks 1–7) and consumed (Task 8). `ChatCompletionRequest`/`ChatCompletionResponse` match the existing `gateway/models.py`.

**Gap check:** in-gateway failover is intentionally absent (spec decision 4, fail-fast). Streaming cache absent (scope-out). No task touches `agent.py`/`client.py` (invariant).
