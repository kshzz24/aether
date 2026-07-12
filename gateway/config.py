"""Gateway configuration for Phase 4 (rate limit / cache / breaker / retry).

Loaded by the gateway process at startup from the [gateway] table of
`.forge/config.toml`. Each config group is a small nested model, so a TOML
sub-table like [gateway.redis] maps straight onto the `redis` field and
downstream code reads `gwcfg.redis.url`, `gwcfg.ratelimit.capacity`, etc.

Unlike the CLI's ForgeConfig this is NOT extra="forbid": an unknown gateway key
should not crash the server mid-phase.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


class RedisCfg(BaseModel):
    url: str = "redis://localhost:6379"


class RateLimitCfg(BaseModel):
    # token bucket (spec §5.1): a bucket of `capacity` tokens refilling at
    # `refill_per_sec`; each request spends one token.
    capacity: int = 60
    refill_per_sec: float = 1.0


class CacheCfg(BaseModel):
    exact_ttl_sec: int = 3600


class SemanticCfg(BaseModel):
    enabled: bool = False          # off by default — guards the anti-pattern
    threshold: float = 0.97        # high cosine bar; the precision/recall dial
    model: str = "text-embedding-3-small"


class BreakerCfg(BaseModel):
    fail_threshold: int = 5
    cooldown_sec: float = 30


class RetryCfg(BaseModel):
    max_attempts: int = 3
    base_delay_sec: float = 0.5


class GatewayConfig(BaseModel):
    # field name == TOML group name, so [gateway.<name>] hydrates the model.
    redis: RedisCfg = RedisCfg()
    ratelimit: RateLimitCfg = RateLimitCfg()
    cache: CacheCfg = CacheCfg()
    semantic: SemanticCfg = SemanticCfg()
    breaker: BreakerCfg = BreakerCfg()
    retry: RetryCfg = RetryCfg()


def load_gateway_config(path: str | None = None) -> GatewayConfig:
    """Load the [gateway] table into a validated GatewayConfig.

    A missing file is the normal case (returns all defaults), not an error.
    """
    p = Path(path) if path else Path(".forge/config.toml")
    if not p.exists():
        return GatewayConfig()
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return GatewayConfig(**data.get("gateway", {}))
