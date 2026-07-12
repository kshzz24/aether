"""Task 1 — GatewayConfig loading + CLI collision guard.

These encode the contract the rest of Phase 4 relies on:
- field names match the TOML group names (redis / ratelimit / cache /
  semantic / breaker / retry), so Pydantic maps [gateway.<group>] tables onto
  them and downstream code can say gwcfg.redis.url, gwcfg.ratelimit.capacity, …
- a missing file yields an all-defaults GatewayConfig (not {}).
- the CLI's load_config still works when a [gateway] table is present.
"""

from __future__ import annotations

from gateway.config import GatewayConfig, load_gateway_config


def test_defaults_when_file_missing():
    cfg = load_gateway_config(path="does-not-exist.toml")
    assert isinstance(cfg, GatewayConfig)          # NOT {} — a real config object
    assert cfg.semantic.enabled is False           # semantic cache off by default
    assert cfg.ratelimit.capacity == 60
    assert cfg.redis.url == "redis://localhost:6379"


def test_defaults_when_path_is_none():
    # path=None must not crash (str/None has no .exists()); falls back to the
    # default project path, which likely doesn't exist here -> all defaults.
    cfg = load_gateway_config()
    assert isinstance(cfg, GatewayConfig)
    assert cfg.breaker.fail_threshold == 5


def test_reads_gateway_table(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[gateway.semantic]\n"
        "enabled = true\n"
        "threshold = 0.9\n"
        "[gateway.ratelimit]\n"
        "capacity = 5\n"
    )
    cfg = load_gateway_config(path=str(p))
    assert cfg.semantic.enabled is True            # value from file
    assert cfg.semantic.threshold == 0.9
    assert cfg.ratelimit.capacity == 5
    assert cfg.cache.exact_ttl_sec == 3600         # untouched group keeps its default


def test_cli_config_tolerates_gateway_table(tmp_path):
    """The shared .forge/config.toml may hold a [gateway] table. The CLI's
    ForgeConfig has extra='forbid', so load_config must drop 'gateway' or it
    raises. This guards the merged.pop('gateway', None) line in config.py."""
    from config import load_config

    proj = tmp_path / "config.toml"
    proj.write_text(
        'provider = "anthropic"\n'
        "[gateway.redis]\n"
        'url = "redis://elsewhere:6379"\n'
    )
    absent = tmp_path / "no-system.toml"
    cfg = load_config(system_path=absent, project_path=proj)  # must not raise
    assert cfg.provider == "anthropic"
