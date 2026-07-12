"""Layered configuration for FORGE.

Precedence, lowest to highest:
    model defaults  <  system config.toml  <  project config.toml  <  CLI flags

The whole chain is a dict merge: each layer ``.update()``s over the previous, and
Pydantic fills any key nobody set with the model default. ``goal`` is NOT config
(it enters per-invocation via the channel, Invariant 4) and never lives here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ForgeConfig(BaseModel):
    # extra="forbid": a typo'd key in a config.toml is a loud error, not a
    # silently-ignored no-op (same "reject loudly" stance as the registry).
    model_config = ConfigDict(extra="forbid")

    provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    max_iterations: int = 25
    max_cost_usd: float = 1.0
    # None => allow every registered tool. A list in TOML coerces to a set.
    allowlist: set[str] | None = None
    user_tools_dir: Path = Path.home() / ".forge" / "tools"
    # TODO(phase-5): real approval modes replace this stub-carried flag.
    auto_approve: bool = True


def _read_toml(path: Path) -> dict:
    """Return a config file's contents, or {} if it isn't there.

    An absent config file is the normal case, not an error.
    """
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(
    cli_overrides: dict | None = None,
    *,
    system_path: Path | None = None,
    project_path: Path | None = None,
) -> ForgeConfig:
    """Merge the config layers into a validated ForgeConfig.

    ``cli_overrides`` must contain only flags the user *explicitly* set (the
    caller strips unset/None sentinels), otherwise defaults would clobber file
    config. Paths are injectable so tests can point at temp files.
    """
    if system_path is None:
        system_path = Path.home() / ".forge" / "config.toml"
    if project_path is None:
        project_path = Path.cwd() / ".forge" / "config.toml"

    merged: dict = {}
    merged.update(_read_toml(system_path))
    merged.update(_read_toml(project_path))
    merged.update(cli_overrides or {})
    merged.pop("gateway", None)
    return ForgeConfig(**merged)
