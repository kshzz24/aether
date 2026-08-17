from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import persistence
from approval import Approver
from main import build_composition, connect_mcp
from persistence import SessionMeta
from server.approver import ServerApprover
from server.session import AgentSession
from server.wire import RunParams

logger = logging.getLogger(__name__)


class SessionLimitReached(RuntimeError):
    """All session slots are in use. Becomes HTTP 503 at stage 4."""


class SessionManager:
    def __init__(
        self,
        *,
        sessions_dir: Path,
        make_approver: Callable[[], Approver],
        max_sessions: int = 8,
    ) -> None:
        self._sessions_dir = sessions_dir
        self._make_approver = make_approver
        self._max_sessions = max_sessions
        self._live: dict[str, AgentSession] = {}

    async def create(self, params: RunParams) -> AgentSession:
        if len(self._live) >= self._max_sessions:
            raise SessionLimitReached(...)
        approver = self._make_approver()
        comp = build_composition(params, approver=approver)
        session = AgentSession(comp, approver=approver)
        # The one place that can see approver, composition and session at once, so
        # the one place that can close the cycle between them. Guarded rather than
        # unconditional: the manager's tests inject a `ScriptedApprover`, which has
        # no `bind` and needs none.
        if isinstance(approver, ServerApprover):
            approver.bind(session.publish_frame)
        await connect_mcp(comp)
        self._live[session.id] = session
        return session

    def get(self, session_id: str) -> AgentSession:
        return self._live[session_id]

    @property
    def live(self) -> list[AgentSession]:
        """A snapshot of the in-memory sessions — what the idle reaper sweeps.

        A list, not the dict: the reaper deletes while iterating, and handing out
        the live mapping would make that a RuntimeError. Exists so the reaper can
        stay out of `_live`; `list()` answers a different question (it merges the
        on-disk sessions, which have nothing to reap).
        """
        return list(self._live.values())

    def list(self) -> list[SessionMeta]:
        metas = {
            meta.id: meta for meta in persistence.list_sessions(self._sessions_dir)
        }

        for session in self._live.values():
            metas[session.id] = self._meta_of(session=session)

        return sorted(metas.values(), key=lambda m: m.updated_at, reverse=True)

    async def delete(self, session_id: str) -> None:
        session = self._live.pop(session_id)
        await session.aclose()

    async def aclose_all(self):
        for session in list(self._live.values()):
            try:
                await session.aclose()
            except Exception:
                logger.exception("failed to close session %s", session.id)
        self._live.clear()

    def _meta_of(self, session: AgentSession) -> SessionMeta:
        s = session._comp.session
        return SessionMeta(
            id=s.id,
            goal=s.goal,
            updated_at=s.updated_at,
            total_cost=session.total_cost,
            turns=len(session._comp.agent.messages),
        )
