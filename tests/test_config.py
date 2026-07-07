import pytest
from pydantic import ValidationError

from config import ForgeConfig, load_config


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_defaults_when_no_files_and_no_cli(tmp_path):
    cfg = load_config(
        {},
        system_path=tmp_path / "sys.toml",
        project_path=tmp_path / "proj.toml",
    )
    assert cfg.provider == "anthropic"
    assert cfg.max_iterations == 25
    assert cfg.allowlist is None


def test_precedence_cli_over_project_over_system_over_default(tmp_path):
    sys_path = tmp_path / "sys.toml"
    proj_path = tmp_path / "proj.toml"
    # system sets provider + model; project overrides model; cli overrides model
    # again. max_iterations is set only by system, so it should survive.
    _write(sys_path, 'provider = "groq"\nmodel = "sys-model"\nmax_iterations = 7\n')
    _write(proj_path, 'model = "proj-model"\n')

    cfg = load_config(
        {"model": "cli-model"},
        system_path=sys_path,
        project_path=proj_path,
    )
    assert cfg.model == "cli-model"        # CLI wins
    assert cfg.provider == "groq"          # from system (nobody else set it)
    assert cfg.max_iterations == 7         # from system, survives
    assert cfg.max_cost_usd == 1.0         # nobody set it -> default


def test_project_overrides_system(tmp_path):
    sys_path = tmp_path / "sys.toml"
    proj_path = tmp_path / "proj.toml"
    _write(sys_path, 'provider = "groq"\n')
    _write(proj_path, 'provider = "openai"\n')
    cfg = load_config({}, system_path=sys_path, project_path=proj_path)
    assert cfg.provider == "openai"


def test_allowlist_list_in_toml_coerces_to_set(tmp_path):
    proj_path = tmp_path / "proj.toml"
    _write(proj_path, 'allowlist = ["read_file", "write_file"]\n')
    cfg = load_config(
        {}, system_path=tmp_path / "none.toml", project_path=proj_path
    )
    assert cfg.allowlist == {"read_file", "write_file"}


def test_unknown_key_is_rejected_loudly(tmp_path):
    proj_path = tmp_path / "proj.toml"
    _write(proj_path, 'max_iteration = 5\n')  # typo: missing trailing "s"
    with pytest.raises(ValidationError):
        load_config({}, system_path=tmp_path / "none.toml", project_path=proj_path)


def test_bad_type_is_rejected():
    with pytest.raises(ValidationError):
        ForgeConfig(max_iterations="not-an-int")
