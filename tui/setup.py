"""First-run setup: pick a provider, then a model.

This screen exists because of a specific failure. `ForgeConfig` defaults to
`provider = "anthropic"`, so someone whose only key is `GROQ_API_KEY` used to get
a raw `TypeError: Could not resolve authentication method` from deep inside an
SDK. The fix is not better documentation — it is asking the question at startup,
and showing which keys are actually present so the answer is obvious.

Two steps over one `OptionList`. `escape` walks back a step before it cancels,
because picking the wrong provider should cost one keystroke, not a restart.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from tui.catalog import ProviderInfo, find

# Returned to the caller: the chosen provider and model.
SetupResult = tuple[str, str]


def provider_rows(catalog: list[ProviderInfo]) -> list[Option]:
    """One picker row per provider, key status first."""
    width = max((len(p.name) for p in catalog), default=0)
    rows = []
    for provider in catalog:
        mark = "✓" if provider.has_key else "·"
        rows.append(
            Option(
                f" {mark}  {provider.name:<{width}}   "
                f"{provider.key_label():<22} {provider.model_label()}",
                id=provider.name,
            )
        )
    return rows


def model_rows(provider: ProviderInfo) -> list[Option]:
    width = max((len(m.id) for m in provider.models), default=0)
    return [
        Option(f"    {m.id:<{width}}   {m.rate_label()}", id=m.id)
        for m in provider.models
    ]


class SetupScreen(ModalScreen[SetupResult | None]):
    """Asks for a provider, then a model; dismisses with the pair."""

    BINDINGS = [("escape", "back", "back")]

    def __init__(self, catalog: list[ProviderInfo]) -> None:
        super().__init__()
        self._catalog = catalog
        self.provider: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-body"):
            yield Label("Choose a provider", id="setup-title")
            yield Label(
                "a ✓ means the key for that provider is already in your "
                "environment",
                id="setup-hint",
            )
            yield OptionList(*provider_rows(self._catalog), id="setup-options")
            yield Input(
                placeholder="model id (e.g. gpt-4o-mini)",
                id="setup-model-input",
            )

    def on_mount(self) -> None:
        # The free-text fallback only appears for providers with no priced
        # models, so it starts hidden.
        self.query_one("#setup-model-input", Input).display = False
        self.query_one(OptionList).focus()

    # ------------------------------------------------------------------ steps

    def _show_models(self, provider: ProviderInfo) -> None:
        self.provider = provider.name
        self.query_one("#setup-title", Label).update(
            f"Choose a model · {provider.name}"
        )
        options = self.query_one(OptionList)
        options.clear_options()

        if not provider.models:
            # prices.toml covers 3 of 6 providers, so this is a normal path.
            self.query_one("#setup-hint", Label).update(
                f"no priced models for {provider.name} — type an id "
                "(it will meter as $0)"
            )
            options.display = False
            entry = self.query_one("#setup-model-input", Input)
            entry.display = True
            entry.focus()
            return

        hint = f"set {provider.env_var} before running" if not provider.has_key else ""
        self.query_one("#setup-hint", Label).update(hint)
        options.add_options(model_rows(provider))
        # `clear_options` leaves nothing highlighted, so enter would be a no-op
        # and the list would render with no visible cursor.
        options.highlighted = 0
        options.focus()

    def _back_to_providers(self) -> None:
        self.provider = None
        self.query_one("#setup-title", Label).update("Choose a provider")
        self.query_one("#setup-hint", Label).update(
            "a ✓ means the key for that provider is already in your environment"
        )
        entry = self.query_one("#setup-model-input", Input)
        entry.display = False
        entry.value = ""
        options = self.query_one(OptionList)
        options.display = True
        options.clear_options()
        options.add_options(provider_rows(self._catalog))
        options.highlighted = 0
        options.focus()

    # --------------------------------------------------------------- handlers

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        chosen = event.option.id or ""
        if self.provider is None:
            provider = find(self._catalog, chosen)
            if provider is not None:
                self._show_models(provider)
            return
        self.dismiss((self.provider, chosen))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        model = event.value.strip()
        if model and self.provider:
            self.dismiss((self.provider, model))

    def action_back(self) -> None:
        """One step back, then out. Escaping straight to the app on a mistyped
        provider would mean restarting to fix a one-key error."""
        if self.provider is None:
            self.dismiss(None)
        else:
            self._back_to_providers()
