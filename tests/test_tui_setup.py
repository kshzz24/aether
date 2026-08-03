"""The first-run wizard, the theme, and the branding.

The wizard exists because of one concrete failure: FORGE defaulted to
`anthropic` while the user's only key was `GROQ_API_KEY`, and the result was a
raw SDK traceback naming none of the things they had to change. These tests pin
the behaviour that replaces it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Input, Label, OptionList

import main as composition_root
import persistence
from tui.app import ForgeApp, _explain, _needs_setup, _write_user_config
from tui.branding import TIPS, WORDMARK, banner
from tui.catalog import ModelInfo, ProviderInfo
from tui.setup import SetupScreen
from tui.theme import DEFAULT_THEME, FORGE_DARK, FORGE_LIGHT, next_theme
from tui.transcript import TranscriptView

_CATALOG = [
    ProviderInfo(
        "groq",
        "GROQ_API_KEY",
        True,
        (ModelInfo("gpt-oss-120b", 0.15, 0.75), ModelInfo("llama-3.3", 0.59, 0.79)),
    ),
    ProviderInfo("anthropic", "ANTHROPIC_API_KEY", False, (ModelInfo("opus", 5, 25),)),
    ProviderInfo("openai", "OPENAI_API_KEY", False, ()),
]


def _args(**over) -> argparse.Namespace:
    base = dict(
        goal=None,
        gateway_url=None,
        resume=None,
        list_sessions=False,
        tui=True,
        setup=False,
        provider="groq",
        model="stub-model",
        max_iterations=None,
        max_cost_usd=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "default_sessions_dir", lambda: tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    return tmp_path


class _WizardHarness(App):
    """Hosts SetupScreen alone so the wizard is testable without the whole app."""

    CSS_PATH = Path(__file__).resolve().parents[1] / "tui" / "forge.tcss"

    def __init__(self) -> None:
        super().__init__()
        for theme in (FORGE_DARK, FORGE_LIGHT):
            self.register_theme(theme)
        self.theme = DEFAULT_THEME
        self.result: tuple | None = "unset"

    def on_mount(self) -> None:
        self.push_screen(SetupScreen(_CATALOG), self._done)

    def _done(self, result) -> None:
        self.result = result


def _options(app: App) -> OptionList:
    return app.screen.query_one(OptionList)


# --------------------------------------------------------------------------
# The wizard
# --------------------------------------------------------------------------


async def test_the_wizard_lists_every_provider():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert _options(app).option_count == len(_CATALOG)


async def test_a_provider_with_a_key_is_visibly_marked():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        first = _options(app).get_option_at_index(0)
        assert "✓" in str(first.prompt)
        assert "groq" in str(first.prompt)


async def test_a_keyless_provider_names_the_variable_to_set():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = [
            str(_options(app).get_option_at_index(i).prompt)
            for i in range(_options(app).option_count)
        ]
        assert any("ANTHROPIC_API_KEY" in row for row in rows)


async def test_selecting_a_provider_shows_its_models():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")          # groq
        await pilot.pause()

        rows = [
            str(_options(app).get_option_at_index(i).prompt)
            for i in range(_options(app).option_count)
        ]
        assert any("gpt-oss-120b" in row for row in rows)
        assert not any("anthropic" in row for row in rows)


async def test_model_rows_show_rates():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert "0.15" in str(_options(app).get_option_at_index(0).prompt)


async def test_selecting_a_model_dismisses_with_the_pair():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")          # provider: groq
        await pilot.pause()
        await pilot.press("enter")          # model: first row
        await pilot.pause()

    assert app.result == ("groq", "gpt-oss-120b")


async def test_escape_returns_to_the_provider_step_rather_than_cancelling():
    """Picking the wrong provider should cost one keystroke, not a restart."""
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert app.result == "unset", "escape cancelled instead of stepping back"
        assert _options(app).option_count == len(_CATALOG)


async def test_escape_on_the_first_step_cancels():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert app.result is None


async def test_a_provider_with_no_priced_models_offers_free_text():
    """prices.toml covers half the providers, so this is a normal path."""
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        options = _options(app)
        options.highlighted = 2                     # openai, no models
        await pilot.press("enter")
        await pilot.pause()

        entry = app.screen.query_one("#setup-model-input", Input)
        assert entry.display is True
        assert options.display is False


async def test_the_free_text_model_dismisses_with_the_pair():
    app = _WizardHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        _options(app).highlighted = 2
        await pilot.press("enter")
        await pilot.pause()

        entry = app.screen.query_one("#setup-model-input", Input)
        entry.value = "gpt-4o-mini"
        await pilot.press("enter")
        await pilot.pause()

    assert app.result == ("openai", "gpt-4o-mini")


# --------------------------------------------------------------------------
# When the wizard runs
# --------------------------------------------------------------------------


def test_setup_is_needed_when_nothing_is_configured(monkeypatch, tmp_path):
    import tui.app as app_module

    monkeypatch.setattr(app_module, "_USER_CONFIG", tmp_path / "none.toml")
    monkeypatch.setattr(app_module, "_PROJECT_CONFIG", tmp_path / "none2.toml")
    assert _needs_setup(_args(provider=None, model=None)) is True


def test_explicit_flags_suppress_the_wizard(monkeypatch, tmp_path):
    """Someone who passed --provider has already answered the question."""
    import tui.app as app_module

    monkeypatch.setattr(app_module, "_USER_CONFIG", tmp_path / "none.toml")
    monkeypatch.setattr(app_module, "_PROJECT_CONFIG", tmp_path / "none2.toml")
    assert _needs_setup(_args(provider="groq", model=None)) is False


def test_an_existing_config_suppresses_the_wizard(monkeypatch, tmp_path):
    import tui.app as app_module

    config = tmp_path / "config.toml"
    config.write_text('provider = "groq"\n', encoding="utf-8")
    monkeypatch.setattr(app_module, "_USER_CONFIG", config)
    monkeypatch.setattr(app_module, "_PROJECT_CONFIG", tmp_path / "none.toml")
    assert _needs_setup(_args(provider=None, model=None)) is False


def test_the_setup_flag_forces_the_wizard(monkeypatch, tmp_path):
    import tui.app as app_module

    config = tmp_path / "config.toml"
    config.write_text('provider = "groq"\n', encoding="utf-8")
    monkeypatch.setattr(app_module, "_USER_CONFIG", config)
    assert _needs_setup(_args(setup=True)) is True


def test_the_choice_is_written_to_user_config(monkeypatch, tmp_path):
    import tui.app as app_module

    target = tmp_path / "cfg" / "config.toml"
    monkeypatch.setattr(app_module, "_USER_CONFIG", target)
    _write_user_config("groq", "gpt-oss-120b")

    body = target.read_text(encoding="utf-8")
    assert 'provider = "groq"' in body
    assert 'model = "gpt-oss-120b"' in body


def test_writing_config_preserves_unrelated_settings(monkeypatch, tmp_path):
    """A hand-set max_cost_usd must survive re-running the wizard."""
    import tui.app as app_module

    target = tmp_path / "config.toml"
    target.write_text('max_cost_usd = 5.0\nprovider = "openai"\n', encoding="utf-8")
    monkeypatch.setattr(app_module, "_USER_CONFIG", target)
    _write_user_config("groq", "m")

    body = target.read_text(encoding="utf-8")
    assert "max_cost_usd = 5.0" in body
    assert 'provider = "groq"' in body


# --------------------------------------------------------------------------
# Error explanation — the failure this whole feature replaces
# --------------------------------------------------------------------------


def test_an_auth_error_names_the_environment_variable():
    message = _explain(
        TypeError("Could not resolve authentication method. Expected api_key")
    )
    assert "API_KEY" in message
    # ctrl+s rather than /setup: when this fires at startup there is no working
    # prompt to type a command into.
    assert "ctrl+s" in message


def test_a_401_is_treated_as_an_auth_error():
    assert "ctrl+s" in _explain(RuntimeError("401 Unauthorized"))


def test_an_unrelated_error_is_reported_verbatim():
    """Don't blame auth for a bug that isn't auth."""
    message = _explain(ValueError("something else entirely"))
    assert "ValueError" in message
    assert "something else entirely" in message


def test_the_openai_sdk_missing_credentials_error_is_recognised():
    """The SDK raises this from its *constructor*, before any request. Its own
    wording names SDK parameters, not the variable the user has to export."""
    message = _explain(
        RuntimeError(
            "Missing credentials. Please pass an `api_key`, or set the "
            "`OPENAI_API_KEY` environment variable."
        ),
        provider="groq",
    )
    assert "GROQ_API_KEY" in message


def test_the_named_provider_wins_over_the_config_file(monkeypatch, tmp_path):
    """`--provider groq` with no config would otherwise be explained as a
    missing ANTHROPIC_API_KEY — naming the wrong variable in exactly the
    situation the message exists to fix."""
    import tui.app as app_module

    monkeypatch.setattr(app_module, "_USER_CONFIG", tmp_path / "none.toml")
    monkeypatch.setattr(app_module, "_PROJECT_CONFIG", tmp_path / "none2.toml")
    message = _explain(RuntimeError("Missing credentials"), provider="groq")
    assert "GROQ_API_KEY" in message
    assert "ANTHROPIC" not in message


def test_the_auth_message_tells_wsl_users_why_their_key_is_missing():
    """WSL does not inherit Windows environment variables, which is the most
    common way this happens on this machine."""
    assert "WSL" in _explain(RuntimeError("401 Unauthorized"), provider="groq")


# --------------------------------------------------------------------------
# Startup failure is a message, not a traceback
# --------------------------------------------------------------------------


def _explode(*_args, **_kwargs):
    raise RuntimeError("Missing credentials. Please pass an `api_key`")


def test_a_missing_key_at_construction_does_not_raise(monkeypatch, sessions_dir):
    """Provider SDKs raise from their constructor, so this fires inside
    `build_composition` before a single event exists. Catching only
    CompositionError let it escape as a traceback."""
    monkeypatch.setattr(composition_root, "build_composition", _explode)
    app = ForgeApp(_args(provider="groq"))
    assert app.comp is None
    assert "GROQ_API_KEY" in (app._error or "")


async def test_the_startup_error_is_shown_in_the_transcript(
    monkeypatch, sessions_dir
):
    monkeypatch.setattr(composition_root, "build_composition", _explode)
    app = ForgeApp(_args(provider="groq"))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "GROQ_API_KEY" in app.query_one(TranscriptView).text
        await pilot.press("escape")
        await pilot.pause()


async def test_a_failed_startup_offers_the_wizard(monkeypatch, sessions_dir):
    """With no agent there is no prompt to type /setup into, and picking a
    provider whose key you do have is a real fix."""
    monkeypatch.setattr(composition_root, "build_composition", _explode)
    app = ForgeApp(_args(provider="groq"))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen)
        await pilot.press("escape")
        await pilot.pause()


async def test_a_failed_rebuild_keeps_the_app_alive(sessions_dir, monkeypatch):
    """Switching to a provider whose key is missing must report, not crash."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(composition_root, "build_composition", _explode)
        app._rebuild({"provider": "groq"})
        await pilot.pause()

        assert app.comp is not None, "a failed rebuild dropped the working agent"
        assert "GROQ_API_KEY" in app.query_one(TranscriptView).text


async def test_ctrl_s_opens_the_wizard(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen)
        await pilot.press("escape")
        await pilot.pause()


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------


def test_the_forge_theme_is_active_on_boot(sessions_dir):
    app = ForgeApp(_args())
    assert app.theme == DEFAULT_THEME


async def test_ctrl_t_cycles_within_the_forge_themes(sessions_dir):
    """Cycling into Textual's stock themes would drop $rail-user and friends,
    half-rendering the app."""
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.theme == FORGE_LIGHT.name
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.theme == FORGE_DARK.name


def test_next_theme_recovers_from_an_unknown_theme():
    assert next_theme("some-other-theme") == DEFAULT_THEME


def test_both_themes_define_the_rail_variables():
    """forge.tcss references these; a theme missing one fails at CSS parse."""
    for theme in (FORGE_DARK, FORGE_LIGHT):
        assert "rail-user" in theme.variables
        assert "rail-assistant" in theme.variables


# --------------------------------------------------------------------------
# Branding
# --------------------------------------------------------------------------


def test_the_wordmark_is_five_lines():
    assert len(WORDMARK.splitlines()) == 5


def test_the_banner_carries_version_and_path():
    from rich.console import Console

    console = Console(width=100, no_color=True)
    with console.capture() as captured:
        console.print(banner(version="9.9.9", cwd=Path.home() / "work"))
    text = captured.get()

    assert "9.9.9" in text
    assert "~/work" in text
    for key, _ in TIPS:
        assert key in text


async def test_the_banner_renders_at_the_top_of_the_transcript(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(TranscriptView).query(".banner")


async def test_a_setup_command_reopens_the_wizard(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.text = "/setup"
        prompt.post_message(type(prompt).Submitted("/setup"))
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, SetupScreen)
        await pilot.press("escape")
        await pilot.pause()


async def test_the_status_bar_uses_a_label_not_a_hardcoded_colour(sessions_dir):
    app = ForgeApp(_args())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.query_one("#status-model"), Label)
