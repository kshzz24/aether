# FORGE — Phase 4 Gateway Depth Design

**Date:** 2026-07-12
**Status:** Approved (design); implementation pending
**Scope:** Make the Phase-3 gateway resilient and cost-aware under failure and
load. Four capabilities added to the existing `POST /v1/chat/completions` path:
a per-key **rate limiter**, a two-layer **cache** (exact + semantic), a
per-provider **circuit breaker** with retry/backoff, and a read-only **`/stats`**
metrics endpoint over the ledger.

---

## 1. Purpose

Phase 3 turned the client into a real HTTP service that routes and meters. It has
one failure posture (degrade-to-passthrough in the *client*) and no protection
against load, repeated work, or a flapping upstream. Phase 4 makes the *gateway
itself* resilient and cost-aware: it stops hammering dead providers, serves
repeat work without paying for it twice, bounds per-key request rate, and exposes
the numbers that prove it.

The agent loop does not change. Everything here lives inside the gateway, behind
the same OpenAI-wire contract the `GatewayClient` already speaks.

## 2. Invariants honored

- **The agent core is untouched.** Every change is server-side, behind
  `POST /v1/chat/completions`. The loop remains byte-identical (Phase-3 gate still
  holds).
- **Provider shape stops at the client / server edge.** Cache values and ledger
  rows store already-normalized data; the gate chain never re-parses provider JSON.
- **A failure is data, not a crash.** A dead Redis, a stale semantic index, or a
  tripped breaker degrades a *feature*, never availability of the request path,
  except where the phase deliberately returns 429/503 as the correct answer.
- **Bounded and observable.** Retry counts are bounded; every request's outcome
  (success or failure) is recorded so `/stats` can be computed.

## 3. Decisions (settled in brainstorming)

1. **Semantic cache: build it, guarded.** Exact cache is always on; semantic cache
   is **opt-in, off by default**, behind a **high cosine threshold** (default
   `0.97`). This honors the spec's own SDE-3 warning that a semantic *response*
   cache is usually an anti-pattern for a coding agent, while still delivering the
   embeddings + cache-poisoning lesson.
2. **Exact cache + rate limiter → Redis.** Redis owns the equality lookup (exact
   cache) and the atomic counter state (token bucket). One new service.
3. **Semantic cache → in-memory numpy.** Vectors held in a Python list; cosine
   similarity computed with numpy. Zero infra; the mechanism is naked and
   inspectable. Volatile across restarts and single-instance — acceptable, the
   gateway is one process.
4. **Breaker action on OPEN → fail fast (503).** When a provider's breaker is
   open, the gateway returns 503 *without* calling the dead provider. It does
   **not** fail over to an alternate provider in-gateway. End-to-end resilience
   still exists: the Phase-3 `GatewayClient` degrades to a direct provider call
   when the gateway errors. This keeps the breaker lesson pure and adds no
   fallback-map config.
5. **Metrics → a JSON `GET /stats` endpoint** aggregating the existing ledger
   (p95 latency, error rate, cost per model). No Prometheus/OTel (that is Phase
   14). The Phase-7 dashboard will read the same ledger.
6. **Embeddings → provider endpoint** (`text-embedding-3-small` via the existing
   OpenAI SDK). No new heavyweight local-model dependency. Tradeoff accepted: the
   semantic path pays one embedding call per request to *maybe* save one
   completion call — tolerable because the semantic cache is opt-in and off by
   default.

## 4. The request path (the heart of the phase)

The endpoint becomes a gate chain. A request must clear each gate before it is
allowed to cost money. This ordering is the design.

```
POST /v1/chat/completions
  │
  ├─ 1. rate limit   (Redis token bucket, keyed on Authorization header) ── over? → 429
  │
  ├─ 2. exact cache  (Redis: sha256 of the canonical request)            ── hit?  → return, no upstream
  │
  ├─ 3. semantic cache (numpy cosine, only if enabled and ≥ threshold)   ── hit?  → return, no upstream
  │
  ├─ 4. breaker check (per-provider state)                               ── OPEN? → 503, no upstream
  │
  ├─ 5. call provider  (retry + exponential backoff + jitter on transient errors)
  │       ├─ success → record ledger (status=ok), populate exact + semantic cache → return
  │       └─ failure → record ledger (status=error), trip breaker → 5xx
```

### 4.1 The ordering tradeoff (named on purpose)

Rate limiting is placed **before** the cache, so a cache hit still consumes a
token from the bucket. The alternative — cache-first — makes cache hits "free"
against the limit. We put the gate at the door (canonical API-gateway order:
auth → limit → cache → origin) because the limiter protects the *gateway's own*
resources and enforces per-key fairness, not only the upstream provider's. This
choice is documented in code, not left implicit.

### 4.2 The cache key

The exact-cache key is `sha256` over a **canonical** serialization of the request:
`model` + `messages` + `tools` + `system` + sampling params that affect output
(e.g. `temperature`). Canonicalization (stable key order, no incidental
whitespace) ensures the same logical request maps to the same key. Streaming is
not cached (out of scope; the gateway is non-streaming).

### 4.3 What is cached

The **wire response** (`ChatCompletionResponse`) is what gets stored and replayed,
so a cache hit is indistinguishable from a fresh call to the `GatewayClient`. A
cache hit is *not* re-billed and produces **no ledger row** — it did no upstream
work. (A cache-hit counter may be surfaced in `/stats` from Redis, separate from
the ledger.)

## 5. New and changed modules (small, single-purpose)

| File | Owns | Primary concept |
|---|---|---|
| `gateway/ratelimit.py` | `TokenBucket` over Redis — atomic refill + consume via a Lua script | token-bucket state machine |
| `gateway/cache.py` | `ExactCache` (Redis) + `SemanticCache` (numpy) + `embed()` helper | invalidation, cosine similarity, poisoning |
| `gateway/breaker.py` | `CircuitBreaker` (per-provider) + `call_with_resilience()` retry/backoff wrapper | breaker state machine, backoff + jitter |
| `gateway/metrics.py` | `/stats` aggregation queries over the ledger | percentiles, read model |
| `gateway/server.py` | wires the gate chain into the endpoint; adds `GET /stats` | orchestration only |
| `gateway/ledger.py` | **add** a `status` column (`ok`/`error`) + a failure-recording path | needed for the error-rate metric |
| `gateway/config` (Pydantic) | new `gateway.*` config blocks | layered configuration |

Retry/backoff lives *with* the breaker because "call the provider resiliently" is
one concern: `call_with_resilience(provider, fn)` checks the breaker, retries
transient failures with `base * 2^n + jitter`, and records success/failure back
to the breaker.

### 5.1 Rate limiter — `TokenBucket`

- One bucket per **key** = the incoming `Authorization` header (or a fixed default
  when absent, for local use).
- State in Redis: `tokens`, `last_refill_ts`. Refill + consume run in a single
  **atomic Lua script** so concurrent requests can't double-spend.
- `capacity` and `refill_per_sec` come from config. Over budget → the endpoint
  returns **429** before any upstream work.

### 5.2 Cache — `ExactCache` + `SemanticCache`

- `ExactCache`: `get(key) -> ChatCompletionResponse | None`, `put(key, resp, ttl)`
  in Redis; TTL from config (`exact_ttl_sec`).
- `SemanticCache`: holds `list[(vector, ChatCompletionResponse)]` in memory.
  `get(text)` embeds `text`, computes cosine vs stored vectors with numpy, returns
  the best match **iff** similarity `≥ threshold`. `put(text, resp)` embeds and
  appends. Disabled entirely when `semantic.enabled = false` (the default).
- `embed(text) -> np.ndarray` calls the provider embedding endpoint
  (`text-embedding-3-small`) via the existing OpenAI SDK.

### 5.3 Circuit breaker — `CircuitBreaker`

- Per-provider state machine: **closed → open → half-open**.
- Closed: requests flow; consecutive failures counted. At `fail_threshold` → open.
- Open: requests short-circuit to **503**; after `cooldown_sec` → half-open.
- Half-open: a single probe request is allowed; success → closed, failure → open.
- Transient errors (timeouts, 5xx, connection errors) count toward the breaker and
  are retried by `call_with_resilience`; a 400 (bad request) does **not** trip the
  breaker — it's the caller's fault, not the provider's.

### 5.4 Metrics — `GET /stats`

Read-only aggregation over the ledger, returning JSON:

```json
{
  "p95_latency_ms": 812,
  "error_rate": 0.02,
  "cost_by_model": { "claude-...": 1.34, "gpt-...": 0.51 },
  "requests_total": 128,
  "cache_hit_rate": 0.19
}
```

- `p95_latency_ms` via Postgres `percentile_cont(0.95)`.
- `error_rate` = `count(status='error') / count(*)` — this is why the ledger gains
  a `status` column.
- `cache_hit_rate` read from Redis counters (hits vs total), separate from the
  ledger since hits produce no ledger row.

## 6. The semantic-cache guard (honoring the anti-pattern)

Three concrete guards so the "usually skip it" warning is respected while the
lesson still lands:

1. **Off by default** — `gateway.semantic.enabled = false`.
2. **High threshold** — default cosine `≥ 0.97`; the threshold is the
   precision/recall dial the lesson tunes.
3. **A deliberate poisoning test** — a unit test feeds two near-identical prompts
   that require *different* answers, and demonstrates the semantic cache returning
   the wrong (poisoned) answer at a loose threshold, then not returning it at the
   default threshold. The risk is made concrete in a test, never in production.

## 7. Config additions (`.forge/config.toml`, Pydantic, `extra="forbid"`)

```toml
[gateway.redis]
url = "redis://localhost:6379"

[gateway.ratelimit]
capacity = 60
refill_per_sec = 1.0

[gateway.cache]
exact_ttl_sec = 3600

[gateway.semantic]
enabled = false
threshold = 0.97
model = "text-embedding-3-small"

[gateway.breaker]
fail_threshold = 5
cooldown_sec = 30

[gateway.retry]
max_attempts = 3
base_delay_sec = 0.5
```

## 8. New dependencies

- `redis` (async client) — rate limiter + exact cache.
- `numpy` — cosine similarity for the semantic cache.
- Embeddings ride the **existing** `openai` SDK — no new dep.
- p95 uses Postgres `percentile_cont` — no new DB dep.

## 9. Testing (pytest; one+ unit test per piece)

- **token bucket**: refill math, exhaustion → 429, refill-over-time restores tokens.
- **breaker**: opens after `fail_threshold` failures; half-open probe; closes on
  success; a 400 does not trip it.
- **exact cache**: hit / miss / TTL expiry; canonical key stability (reordered
  fields → same key).
- **semantic cache**: hit above threshold, miss below, and **the poisoning demo**.
- **/stats**: aggregation correctness on a seeded ledger (p95, error rate,
  cost-by-model).
- **acceptance (end-to-end)**:
  a. a repeated request is served from cache with **zero** upstream calls,
  b. exceeding the bucket returns **429**,
  c. inducing repeated upstream failures **opens the breaker** and returns **503**
     fast (no upstream call while open).

## 10. Scope OUT (do not build this phase)

- Prometheus / OpenTelemetry / histograms — Phase 14.
- The metrics dashboard UI — Phase 7.
- Difficulty / cost-optimal routing — Phase 13.
- In-gateway failover to an alternate provider — chosen against (fail-fast).
- Semantic cache on by default, or semantic caching without the threshold guard.
- Caching of streaming responses (the gateway is non-streaming).

## 11. Acceptance gate for Phase 4

With Redis running and the gateway up:
1. Send the same request twice — the second returns from the exact cache with no
   new ledger row and a lower latency; `/stats` shows a non-zero `cache_hit_rate`.
2. Exceed the configured rate for a key — the gateway returns 429.
3. Force an upstream provider to fail repeatedly — the breaker opens and the
   gateway returns 503 without calling upstream; after `cooldown_sec` a probe is
   allowed.
4. `GET /stats` returns p95 latency, error rate, and cost-by-model from real runs.
5. The agent loop run through the gateway is still byte-identical to a direct run
   (Phase-3 gate unbroken).
