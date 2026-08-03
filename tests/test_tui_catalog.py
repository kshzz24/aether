"""The provider/model catalog — pure, no app, no real environment.

This is the data behind the first-run wizard. Its job is to answer "what will
actually work?" before the user picks something that cannot, which is why key
presence drives the ordering rather than being a footnote.
"""

from __future__ import annotations

from tui.catalog import ModelInfo, ProviderInfo, find, load_catalog

_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_PRICES = {
    "groq": {
        "llama-3.3-70b": {"input": 0.59, "output": 0.79},
        "gpt-oss-120b": {"input": 0.15, "output": 0.75},
        "free-thing": {"input": 0, "output": 0},
    },
    "anthropic": {"claude-opus-4-8": {"input": 5, "output": 25}},
}


def _catalog(env=None):
    return load_catalog(prices=_PRICES, env=env or {}, env_keys=_ENV_KEYS)


# --------------------------------------------------------------------------
# Key detection and ordering
# --------------------------------------------------------------------------


def test_a_provider_with_its_key_set_is_marked():
    catalog = _catalog({"GROQ_API_KEY": "gsk_x"})
    assert find(catalog, "groq").has_key is True


def test_a_provider_without_its_key_is_not_marked():
    assert find(_catalog(), "groq").has_key is False


def test_a_whitespace_only_key_does_not_count():
    """An exported-but-empty variable is the classic half-configured state; it
    must not be advertised as ready."""
    catalog = _catalog({"GROQ_API_KEY": "   "})
    assert find(catalog, "groq").has_key is False


def test_providers_with_keys_sort_first():
    """The whole point of the picker is surfacing what will work."""
    catalog = _catalog({"OPENAI_API_KEY": "sk_x"})
    assert catalog[0].name == "openai"


def test_ordering_is_stable_within_each_group():
    catalog = _catalog({"GROQ_API_KEY": "x", "ANTHROPIC_API_KEY": "y"})
    assert [p.name for p in catalog] == ["anthropic", "groq", "openai"]


def test_every_known_provider_appears_even_without_a_key():
    """A provider missing from the list cannot be chosen, so none may be
    dropped just because the key is absent."""
    assert {p.name for p in _catalog()} == set(_ENV_KEYS)


def test_the_env_var_name_is_carried_for_error_messages():
    assert find(_catalog(), "groq").env_var == "GROQ_API_KEY"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def test_models_come_from_the_prices_table():
    ids = {m.id for m in find(_catalog(), "groq").models}
    assert ids == {"llama-3.3-70b", "gpt-oss-120b", "free-thing"}


def test_models_sort_cheapest_first():
    models = find(_catalog(), "groq").models
    assert [m.id for m in models] == ["free-thing", "gpt-oss-120b", "llama-3.3-70b"]


def test_a_provider_absent_from_prices_has_no_models_rather_than_crashing():
    """prices.toml covers 3 of 6 real providers — this is the normal path."""
    assert find(_catalog(), "openai").models == ()


def test_rates_are_carried_through():
    model = next(m for m in find(_catalog(), "groq").models if m.id == "gpt-oss-120b")
    assert (model.input_rate, model.output_rate) == (0.15, 0.75)


# --------------------------------------------------------------------------
# Labels — what the picker actually shows
# --------------------------------------------------------------------------


def test_a_priced_model_shows_its_rates():
    assert "0.15" in ModelInfo("m", 0.15, 0.75).rate_label()


def test_a_zero_rated_model_is_labelled_free():
    assert ModelInfo("m", 0, 0).rate_label() == "free tier"


def test_an_unpriced_model_says_it_meters_as_zero():
    """A cost governor silently reading $0 is worse than one that admits it."""
    label = ModelInfo("m").rate_label()
    assert "unpriced" in label
    assert "$0" in label


def test_key_label_names_the_missing_variable():
    """The label is the user's instruction for how to fix it."""
    assert "GROQ_API_KEY" in ProviderInfo("groq", "GROQ_API_KEY", False).key_label()


def test_key_label_is_short_when_the_key_is_present():
    assert ProviderInfo("groq", "GROQ_API_KEY", True).key_label() == "key set"


def test_model_label_pluralises():
    one = ProviderInfo("p", "K", True, (ModelInfo("a"),))
    two = ProviderInfo("p", "K", True, (ModelInfo("a"), ModelInfo("b")))
    assert one.model_label() == "1 model"
    assert two.model_label() == "2 models"


def test_model_label_handles_none():
    assert "no priced models" in ProviderInfo("p", "K", True).model_label()


def test_find_returns_none_for_an_unknown_provider():
    assert find(_catalog(), "nope") is None
