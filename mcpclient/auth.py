from typing import Protocol


class AuthProvider(Protocol):
    def env(self) -> dict[str, str]:
        pass

    def headers(self) -> dict[str, str]:
        pass


class EnvAuth(AuthProvider):
    def __init__(self, env_vars: dict[str, str]):
        self._env = dict(env_vars)

    def env(self):
        return dict(self._env)

    def headers(self) -> dict[str, str]:
        return {}


class HeaderAuth(AuthProvider):
    def __init__(self, headers: dict[str, str]):
        self._args = dict(headers)

    def headers(self) -> dict[str, str]:
        return dict(self._args)

    def env(self) -> dict[str, str]:
        return {}


class OAuthAuth(AuthProvider):
    def env(self) -> dict[str, str]:
        raise NotImplementedError("OAuth 2.1 is deferred; see plan §2 Out")

    def headers(self) -> dict[str, str]:
        raise NotImplementedError("OAuth 2.1 is deferred; see plan §2 Out")
