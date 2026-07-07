from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


def _noop(*args, **kwargs) -> None:
    pass


@dataclass
class Hooks:
    before_run: Callable[..., None] = _noop
    after_run: Callable[..., None] = _noop
    before_tool: Callable[..., None] = _noop
    after_tool: Callable[..., None] = _noop
    on_error: Callable[..., None] = _noop
