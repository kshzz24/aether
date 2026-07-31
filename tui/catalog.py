"""What providers and models are available, and which will actually work.

Pure functions over plain dicts — no widgets, no environment access beyond what
is handed in. That is what lets the setup wizard's *content* be tested without
booting an app, the same split that keeps `commands.dispatch` and
`files.match_paths` cheap to test.

Both data sources already exist: `prices.toml` lists the models FORGE can meter,
and `main.ENV_KEYS` maps a provider to the environment variable holding its key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """A model id and its $/Mtok rates. Rates are None when unpriced."""

    id: str
    input_rate: float | None = None
    output_rate: float | None = None

    @property
    def priced(self) -> bool:
        return self.input_rate is not None and self.output_rate is not None

    @property
    def free(self) -> bool:
        return self.priced and self.input_rate == 0 and self.output_rate == 0

    def rate_label(self) -> str:
        """Short, right-alignable price summary for a picker row."""
        if not self.priced:
            # An unpriced model meters as $0, which makes the cost governor
            # silently useless. Say so rather than showing a blank.
            return "unpriced · meters as $0"
        if self.free:
            return "free tier"
        return f"${self.input_rate:g} / ${self.output_rate:g} per Mtok"


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    env_var: str
    has_key: bool
    models: tuple[ModelInfo, ...] = ()

    def model_label(self) -> str:
        if not self.models:
            return "no priced models"
        return f"{len(self.models)} model{'s' if len(self.models) != 1 else ''}"

    def key_label(self) -> str:
        return "key set" if self.has_key else f"no {self.env_var}"


def _models_for(prices: Mapping[str, object]) -> tuple[ModelInfo, ...]:
    """Read one provider's table out of prices.toml's parsed form."""
    models: list[ModelInfo] = []
    for model_id, rates in prices.items():
        if isinstance(rates, dict):
            models.append(
                ModelInfo(
                    id=model_id,
                    input_rate=rates.get("input"),
                    output_rate=rates.get("output"),
                )
            )
    # Cheapest first: the person choosing usually wants the cheap one that
    # works, and free tiers sort to the top for free.
    models.sort(key=lambda m: (m.input_rate is None, m.input_rate or 0, m.id))
    return tuple(models)


def load_catalog(
    *,
    prices: Mapping[str, Mapping],
    env: Mapping[str, str],
    env_keys: Mapping[str, str],
) -> list[ProviderInfo]:
    """Every known provider, annotated with key presence and priced models.

    Providers whose key is actually set sort first — the entire point of the
    picker is surfacing what will work before the user picks something that
    cannot. Within each group, ties break on name so the order is stable.
    """
    providers = [
        ProviderInfo(
            name=name,
            env_var=env_var,
            has_key=bool(env.get(env_var, "").strip()),
            models=_models_for(prices.get(name, {})),
        )
        for name, env_var in env_keys.items()
    ]
    providers.sort(key=lambda p: (not p.has_key, p.name))
    return providers


def find(catalog: list[ProviderInfo], name: str) -> ProviderInfo | None:
    return next((p for p in catalog if p.name == name), None)
