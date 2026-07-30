"""Phase 6 Task 1 — the auth seam.

The point of these tests is not the three-line implementations; it is the
*shape*. The manager must only ever ask "env or headers for this connection?"
and never "which server is this?" -- so both providers answer both questions,
one of them with an empty dict. That symmetry is what keeps server-specific
branching out of `mcpclient/` (anti-pattern #2).
"""

from __future__ import annotations

import pytest

from mcpclient.auth import EnvAuth, HeaderAuth, OAuthAuth


def test_env_auth_supplies_subprocess_env():
    auth = EnvAuth({"GITHUB_PERSONAL_ACCESS_TOKEN": "sekret"})
    assert auth.env() == {"GITHUB_PERSONAL_ACCESS_TOKEN": "sekret"}


def test_env_auth_supplies_no_headers():
    """Answers the question rather than raising -- the caller must not branch."""
    assert EnvAuth({"T": "x"}).headers() == {}


def test_header_auth_supplies_http_headers():
    auth = HeaderAuth({"Authorization": "Bearer abc"})
    assert auth.headers() == {"Authorization": "Bearer abc"}


def test_header_auth_supplies_no_env():
    assert HeaderAuth({"Authorization": "Bearer abc"}).env() == {}


def test_both_providers_satisfy_the_same_two_method_contract():
    for auth in (EnvAuth({}), HeaderAuth({})):
        assert isinstance(auth.env(), dict)
        assert isinstance(auth.headers(), dict)


def test_env_auth_does_not_expose_its_backing_dict():
    """A mutable leak here would let a later server edit an earlier one's secrets."""
    source = {"T": "x"}
    auth = EnvAuth(source)
    auth.env()["T"] = "tampered"
    assert auth.env()["T"] == "x"
    source["T"] = "tampered-at-source"
    assert auth.env()["T"] == "x"


def test_oauth_is_an_honest_unbuilt_seam():
    """§2 Out: the interface exists, the subsystem does not. It must say so
    rather than silently authenticating with nothing."""
    with pytest.raises(NotImplementedError):
        OAuthAuth().headers()
