import os
import time
import tomllib
from contextlib import asynccontextmanager

import redis.asyncio
from fastapi import FastAPI, HTTPException

from client import ToolCallingUnsupportedError, make_client
from gateway import ledger
from gateway.config import load_gateway_config
from gateway.models import ChatCompletionRequest, ChatCompletionResponse
from gateway.translate import to_internal, to_wire

# from main import ENV_KEYS


ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await ledger.init_pool(os.environ["FORGE_LEDGER_DSN"])
    with open("prices.toml", "rb") as f:
        app.state.prices = tomllib.load(f)
    app.state.gwcfg = load_gateway_config()
    app.state.redis = redis.asyncio.from_url(
        app.state.gwcfg.redis.url, decode_responses=True
    )
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
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    messages, tools, system = to_internal(req)
    provider = route(req.model)
    api_key = os.environ.get(ENV_KEYS.get(provider, ""), "")
    rates = app.state.prices.get(provider, {})

    client = make_client(
        provider=provider, model=req.model, api_key=api_key, rates=rates
    )
    t0 = time.perf_counter()
    try:
        resp = await client.create(messages=messages, tools=tools, system=system)
    except ToolCallingUnsupportedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error:{e}")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    await ledger.record(
        app.state.pool,
        provider=provider,
        model=req.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
        latency_ms=latency_ms,
    )

    return to_wire(resp, req.model)
