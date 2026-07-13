import os
import time
import tomllib
from contextlib import asynccontextmanager

import openai
import redis.asyncio
from fastapi import FastAPI, HTTPException, Request

from client import ToolCallingUnsupportedError, make_client
from gateway import ledger
from gateway.breaker import BreakerOpen, CircuitBreaker, call_with_resilience
from gateway.cache import ExactCache, canonical_key, request_text
from gateway.config import load_gateway_config
from gateway.metrics import compute_stats
from gateway.models import ChatCompletionRequest, ChatCompletionResponse
from gateway.ratelimit import TokenBucket
from gateway.translate import to_internal, to_wire

# from main import ENV_KEYS


ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}
TRANSIENT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await ledger.init_pool(os.environ["FORGE_LEDGER_DSN"])
    with open("prices.toml", "rb") as f:
        app.state.prices = tomllib.load(f)
    app.state.gwcfg = load_gateway_config()
    app.state.redis = redis.asyncio.from_url(
        app.state.gwcfg.redis.url, decode_responses=True
    )
    app.state.bucket = TokenBucket(
        app.state.redis,
        capacity=app.state.gwcfg.ratelimit.capacity,
        refill_per_sec=app.state.gwcfg.ratelimit.refill_per_sec,
    )
    app.state.exact = ExactCache(
        app.state.redis, ttl_sec=app.state.gwcfg.cache.exact_ttl_sec
    )
    app.state.breakers = {}
    app.state.semantic = None
    yield
    await app.state.redis.aclose()
    await app.state.pool.close()


app = FastAPI(lifespan=lifespan)


def route(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o1", "o3")):
        return "openai"
    # NOTE: "openai/gpt-oss-*" is an OpenAI open model hosted on Groq -- its name
    # collides with a real OpenAI/OpenRouter id, which is exactly why name-based
    # routing is a stopgap. Mapped to groq here for single-provider use.
    if model.startswith(("llama", "mixtral", "gemma", "qwen", "openai/gpt-oss")):
        return "groq"
    raise HTTPException(status_code=400, detail=f"no route for model {model!r}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse:

    st = request.app.state

    # ── GATE 1: rate limit ────────────────────────────────

    key = request.headers.get("authorization", "default")
    if not await st.bucket.allow(key):
        raise HTTPException(429, "rate limit exceeded")

    # ── GATE 2: exact cache ───────────────────────────────
    ckey = canonical_key(req)
    if hit := await st.exact.get(ckey):
        await st.redis.incr("forge:cache:hits")
        return hit

    # ── GATE 3: semantic cache (only if enabled) ──────────
    if st.semantic and (hit := await st.semantic.get(request_text(req))):
        await st.redis.incr("forge:cache:hits")
        return hit

    await st.redis.incr("forge:cache:misses")

    messages, tools, system = to_internal(req)
    provider = route(req.model)
    api_key = os.environ.get(ENV_KEYS.get(provider, ""), "")
    rates = st.prices.get(provider, {})

    client = make_client(
        provider=provider, model=req.model, api_key=api_key, rates=rates
    )

    # ── GATE 4+5: breaker + retry, then the provider call ─
    breaker = st.breakers.setdefault(
        provider,
        CircuitBreaker(
            fail_threshold=st.gwcfg.breaker.fail_threshold,
            cooldown_sec=st.gwcfg.breaker.cooldown_sec,
        ),
    )

    t0 = time.perf_counter()
    try:
        resp = await call_with_resilience(
            breaker,
            lambda: client.create(messages=messages, tools=tools, system=system),
            max_attempts=st.gwcfg.retry.max_attempts,
            base_delay_sec=st.gwcfg.retry.base_delay_sec,
            transient=TRANSIENT,
        )
    except BreakerOpen:
        raise HTTPException(
            status_code=503, detail=f"{provider} circuit open"
        ) from None
    except ToolCallingUnsupportedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # A real upstream failure: record it so /stats error_rate stays computable.
        await ledger.record_failure(
            st.pool,
            provider=provider,
            model=req.model,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error=str(e),
        )
        raise HTTPException(status_code=502, detail=f"upstream error:{e}") from e

    latency_ms = int((time.perf_counter() - t0) * 1000)
    await ledger.record(
        st.pool,
        provider=provider,
        model=req.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
        latency_ms=latency_ms,
        status="ok",
    )

    wire = to_wire(resp, req.model)
    await st.exact.put(ckey, wire)  # populate cache for next time
    if st.semantic:
        await st.semantic.put(request_text(req), wire)
    return wire


@app.get("/stats")
async def stats():
    return await compute_stats(app.state.pool, app.state.redis)
