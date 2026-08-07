from __future__ import annotations
import asyncio
from collections import deque


class AgentSession:
    transcript: list[str]
    _seq = 3
    _subs = {}


class Subscriber:
    _replay: deque[dict]
    _queue: asyncio.Queue
    _overflowed: bool
    _overflow_sent: bool
    _closed: bool
