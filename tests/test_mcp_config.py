"""Phase 6 Task 1 — `.mcp.json` parsing, scope merge, and ${VAR} expansion.

These tests pin the contract in the plan's §4 (Claude Code compatibility) and
§8 Task 1. Two deliberate departures from `tests/test_config.py`'s subject:

  * Unknown keys are IGNORED, not rejected. `ForgeConfig` uses extra="forbid"
    because it is FORGE's own file; `.mcp.json` is a shared-format import and a
    block copy-pasted from a vendor README may carry `headersHelper` or
    `alwaysLoad`. Rejecting those defeats the whole point of "config, not code".
  * `timeout` is MILLISECONDS on the wire (Claude Code's unit) and is surfaced
    as `timeout_ms` so the unit lives in the identifier -- `asyncio.wait_for`
    takes seconds, and that conversion must be written on purpose.
"""

from __future__ import annotations

import json

import pytest

from mcpclient.config import MCPConfigError, load_mcp_config


def _write(path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _load(tmp_path, *, project=None, user=None):
    """Load with both scopes pointed at tmp_path; omit a scope to leave it absent."""
    project_path = tmp_path / ".mcp.json"
    user_path = tmp_path / "user-mcp.json"
    if project is not None:
        _write(project_path, project)
    if user is not None:
        _write(user_path, user)
    return load_mcp_config(project_path=project_path, user_path=user_path)


def _servers(**blocks):
    return {"mcpServers": blocks}


# --------------------------------------------------------------------------
# Discovery and merge
# --------------------------------------------------------------------------


def test_no_config_file_yields_empty_dict(tmp_path):
    assert _load(tmp_path) == {}


def test_two_server_fixture_parses(tmp_path):
    cfg = _load(
        tmp_path,
        project=_servers(
            github={"command": "docker", "args": ["run", "-i", "--rm", "img"]},
            memory={"command": "npx", "args": ["-y", "server-memory"]},
        ),
    )
    assert set(cfg) == {"github", "memory"}
    assert cfg["github"].command == "docker"
    assert list(cfg["github"].args) == ["run", "-i", "--rm", "img"]
    assert cfg["github"].name == "github"


def test_project_overrides_user_per_server_name(tmp_path):
    cfg = _load(
        tmp_path,
        user=_servers(
            shared={"command": "user-cmd"},
            user_only={"command": "keep-me"},
        ),
        project=_servers(shared={"command": "project-cmd"}),
    )
    assert cfg["shared"].command == "project-cmd"   # project wins on the name
    assert cfg["user_only"].command == "keep-me"    # unshadowed user server survives
    assert set(cfg) == {"shared", "user_only"}


def test_scope_is_recorded_per_server(tmp_path):
    """D7 needs the origin scope: project servers are untrusted until approved,
    user servers (`~/.forge/mcp.json`) are trusted because you wrote them."""
    cfg = _load(
        tmp_path,
        user=_servers(mine={"command": "a"}),
        project=_servers(theirs={"command": "b"}),
    )
    assert cfg["mine"].scope == "user"
    assert cfg["theirs"].scope == "project"


def test_server_overridden_by_project_carries_project_scope(tmp_path):
    """The winning block decides trust -- a project file must not inherit the
    user file's trusted scope just by reusing its server name."""
    cfg = _load(
        tmp_path,
        user=_servers(dup={"command": "user-cmd"}),
        project=_servers(dup={"command": "project-cmd"}),
    )
    assert cfg["dup"].scope == "project"


def test_missing_mcpServers_key_yields_empty_dict(tmp_path):
    assert _load(tmp_path, project={}) == {}


def test_malformed_json_is_a_loud_error(tmp_path):
    """A typo'd config must not silently degrade to "no servers configured"."""
    (tmp_path / ".mcp.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(MCPConfigError):
        load_mcp_config(
            project_path=tmp_path / ".mcp.json",
            user_path=tmp_path / "user-mcp.json",
        )


# --------------------------------------------------------------------------
# Transport detection (§4)
# --------------------------------------------------------------------------


def test_absence_of_type_means_stdio(tmp_path):
    cfg = _load(tmp_path, project=_servers(s={"command": "x"}))
    assert cfg["s"].transport == "stdio"


def test_explicit_stdio_type_is_accepted(tmp_path):
    cfg = _load(tmp_path, project=_servers(s={"type": "stdio", "command": "x"}))
    assert cfg["s"].transport == "stdio"


def test_http_type_is_accepted(tmp_path):
    cfg = _load(
        tmp_path, project=_servers(s={"type": "http", "url": "https://e.test/mcp"})
    )
    assert cfg["s"].transport == "http"
    assert cfg["s"].url == "https://e.test/mcp"


def test_streamable_http_is_an_alias_for_http(tmp_path):
    cfg = _load(
        tmp_path,
        project=_servers(s={"type": "streamable-http", "url": "https://e.test/mcp"}),
    )
    assert cfg["s"].transport == "http"


def test_url_without_type_is_an_error_with_claude_codes_message_shape(tmp_path):
    """§4: copy the message shape so a user who hits this can search for it."""
    with pytest.raises(MCPConfigError) as exc:
        _load(tmp_path, project=_servers(gh={"url": "https://e.test/mcp"}))
    msg = str(exc.value)
    assert "gh" in msg
    assert '"url"' in msg
    assert '"type": "http"' in msg


def test_stdio_without_command_is_an_error(tmp_path):
    with pytest.raises(MCPConfigError) as exc:
        _load(tmp_path, project=_servers(s={"type": "stdio", "args": ["x"]}))
    assert "s" in str(exc.value)


def test_http_without_url_is_an_error(tmp_path):
    with pytest.raises(MCPConfigError):
        _load(tmp_path, project=_servers(s={"type": "http"}))


def test_unknown_transport_type_is_an_error(tmp_path):
    """Legacy two-endpoint "sse" is deprecated and deliberately not implemented;
    it must fail loudly rather than fall through to stdio."""
    with pytest.raises(MCPConfigError):
        _load(tmp_path, project=_servers(s={"type": "sse", "url": "https://e.test"}))


# --------------------------------------------------------------------------
# Copy-paste compatibility
# --------------------------------------------------------------------------


def test_unknown_keys_are_ignored_not_rejected(tmp_path):
    cfg = _load(
        tmp_path,
        project=_servers(
            s={
                "command": "x",
                "headersHelper": "./get-token.sh",  # Claude Code key FORGE defers
                "alwaysLoad": True,                 # not a FORGE concept
            }
        ),
    )
    assert cfg["s"].command == "x"


# --------------------------------------------------------------------------
# ${VAR} expansion (§4)
# --------------------------------------------------------------------------


def test_var_expands_from_environ_in_env_block(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_TEST_TOKEN", "sekret")
    cfg = _load(
        tmp_path,
        project=_servers(
            s={"command": "x", "env": {"TOKEN": "${FORGE_TEST_TOKEN}"}}
        ),
    )
    assert cfg["s"].env["TOKEN"] == "sekret"


def test_var_expands_in_command_and_args(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_TEST_BIN", "docker")
    monkeypatch.setenv("FORGE_TEST_IMG", "ghcr.io/x/y")
    cfg = _load(
        tmp_path,
        project=_servers(
            s={"command": "${FORGE_TEST_BIN}", "args": ["run", "${FORGE_TEST_IMG}"]}
        ),
    )
    assert cfg["s"].command == "docker"
    assert list(cfg["s"].args) == ["run", "ghcr.io/x/y"]


def test_var_expands_in_http_url_and_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_TEST_HOST", "e.test")
    monkeypatch.setenv("FORGE_TEST_KEY", "abc123")
    cfg = _load(
        tmp_path,
        project=_servers(
            s={
                "type": "http",
                "url": "https://${FORGE_TEST_HOST}/mcp",
                "headers": {"Authorization": "Bearer ${FORGE_TEST_KEY}"},
            }
        ),
    )
    assert cfg["s"].url == "https://e.test/mcp"
    assert cfg["s"].headers["Authorization"] == "Bearer abc123"


def test_var_expands_when_embedded_in_a_larger_string(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_TEST_REPO", "forge")
    cfg = _load(
        tmp_path,
        project=_servers(
            s={"command": "x", "env": {"R": "org/${FORGE_TEST_REPO}.git"}}
        ),
    )
    assert cfg["s"].env["R"] == "org/forge.git"


def test_default_syntax_falls_back_when_var_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_TEST_ABSENT", raising=False)
    cfg = _load(
        tmp_path,
        project=_servers(
            s={"command": "x", "env": {"T": "${FORGE_TEST_ABSENT:-issues}"}}
        ),
    )
    assert cfg["s"].env["T"] == "issues"


def test_default_syntax_prefers_the_environment_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_TEST_SET", "from-env")
    cfg = _load(
        tmp_path,
        project=_servers(
            s={"command": "x", "env": {"T": "${FORGE_TEST_SET:-fallback}"}}
        ),
    )
    assert cfg["s"].env["T"] == "from-env"


def test_empty_default_is_honoured(tmp_path, monkeypatch):
    """`${VAR:-}` is an explicit "empty is fine" and must not raise."""
    monkeypatch.delenv("FORGE_TEST_ABSENT", raising=False)
    cfg = _load(
        tmp_path,
        project=_servers(s={"command": "x", "env": {"T": "${FORGE_TEST_ABSENT:-}"}}),
    )
    assert cfg["s"].env["T"] == ""


def test_unset_var_with_no_default_is_a_loud_error(tmp_path, monkeypatch):
    """The most valuable test here. Expanding to "" hands the server an empty
    token and turns a config bug into an auth failure ten minutes later."""
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    with pytest.raises(MCPConfigError) as exc:
        _load(
            tmp_path,
            project=_servers(
                gh={
                    "command": "docker",
                    "env": {"T": "${GITHUB_PERSONAL_ACCESS_TOKEN}"},
                }
            ),
        )
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in str(exc.value)


# --------------------------------------------------------------------------
# FORGE extensions and defaults
# --------------------------------------------------------------------------


def test_timeout_is_milliseconds_and_surfaced_as_timeout_ms(tmp_path):
    cfg = _load(tmp_path, project=_servers(s={"command": "x", "timeout": 30000}))
    assert cfg["s"].timeout_ms == 30000


def test_timeout_has_a_default_when_absent(tmp_path):
    cfg = _load(tmp_path, project=_servers(s={"command": "x"}))
    assert cfg["s"].timeout_ms > 0


def test_tools_allowlist_parses_and_defaults_to_none(tmp_path):
    """None means "no filter"; an empty list means "allow nothing" -- these are
    different states and collapsing them to falsy is a real bug."""
    cfg = _load(
        tmp_path,
        project=_servers(
            filtered={"command": "x", "tools": ["get_issue", "list_issues"]},
            unfiltered={"command": "x"},
            none_allowed={"command": "x", "tools": []},
        ),
    )
    assert list(cfg["filtered"].tools) == ["get_issue", "list_issues"]
    assert cfg["unfiltered"].tools is None
    assert cfg["none_allowed"].tools == ()


def test_read_only_tools_parses_and_defaults_to_empty(tmp_path):
    """D2: this local list is the ONLY source of READ classification."""
    cfg = _load(
        tmp_path,
        project=_servers(
            gh={"command": "x", "readOnlyTools": ["get_issue"]},
            plain={"command": "x"},
        ),
    )
    assert "get_issue" in cfg["gh"].read_only_tools
    assert cfg["plain"].read_only_tools == frozenset()


def test_env_and_headers_default_to_empty(tmp_path):
    cfg = _load(tmp_path, project=_servers(s={"command": "x"}))
    assert cfg["s"].env == {}
    assert cfg["s"].headers == {}
