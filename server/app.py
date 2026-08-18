"""The HTTP layer: the adapter that lets a network drive `AgentSession`.

Nothing here decides anything. Every route is three to eight lines that translate
a request into a method call that already existed and an exception into a status
code — `SessionLimitReached` is 503, `SessionBusy` is 409, `UnknownDecision` is
409, a missing session is 404. If a route grows a branch about *agents*, the logic
belongs one layer down in `server/session.py`.

The two things this module genuinely owns are auth and lifetime: one bearer token
checked on every route including the two odd transports, and a lifespan that
builds the `SessionManager`, sweeps idle sessions, and closes everything down.

Shape follows `gateway/server.py` — `@asynccontextmanager lifespan`, state on
`app.state`, a module-level `app` — so the repo has one server pattern, not two.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

import redis.asyncio
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
)
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import persistence
from gateway import ledger
from gateway.config import load_gateway_config
from gateway.metrics import compute_stats
from main import CompositionError
from server.approver import ServerApprover, UnknownDecision
from server.session import AgentSession, SessionBusy
from server.sessions import SessionLimitReached, SessionManager
from server.transports import sse, ws
from server.wire import RunParams
from tracing import list_trace_ids, read_spans, trace_summary

logger = logging.getLogger(__name__)

TOKEN_ENV = "FORGE_SERVER_TOKEN"

MAX_SESSIONS = 8
IDLE_TIMEOUT_SEC = 900.0
REAP_INTERVAL_SEC = 60.0

STATIC_DIR = Path(__file__).resolve().parent / "static"

_NO_BUNDLE = """<!doctype html>
<title>FORGE - no bundle</title>
<h1>The web bundle has not been built</h1>
<p>Run <code>npm --prefix web run build</code>, then reload.</p>
"""


# --- auth ---------------------------------------------------------------------


def _presented(authorization: str | None, token: str | None) -> str | None:
    """The token the caller offered, from either place it is allowed to be.

    The header is the normal path. `?token=` exists because `EventSource` cannot
    set headers, and it is accepted on *every* route rather than only on SSE — a
    check that varies by route is a check that gets forgotten on the route added
    next. It does put a secret into URLs and access logs.

    TODO(phase-17): tokens in query strings; replace with a short-lived signed
    ticket minted by an authenticated request.
    """
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return token


def _valid(app: FastAPI, presented: str | None) -> bool:
    """`compare_digest`, not `==`: this is a shared secret, so leak no timing."""
    expected: str | None = getattr(app.state, "token", None)
    if not expected or not presented:
        return False
    return secrets.compare_digest(presented, expected)


async def require_token(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    token: Annotated[str | None, Query()] = None,
) -> None:
    if not _valid(request.app, _presented(authorization, token)):
        raise HTTPException(status_code=401, detail="invalid or missing token")


Authed = Depends(require_token)


def get_session(session_id: str, request: Request) -> AgentSession:
    """`{id}` -> the live session, or 404. Written once, not in six routes."""
    try:
        return request.app.state.manager.get(session_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"no live session {session_id!r}"
        ) from None


Session = Annotated[AgentSession, Depends(get_session)]


# --- bodies -------------------------------------------------------------------


class CreateBody(BaseModel):
    goal: str | None = None
    resume: str | None = None
    provider: str | None = None
    model: str | None = None
    approval_mode: str | None = None


class GoalBody(BaseModel):
    text: str


class DecisionBody(BaseModel):
    request_id: str
    approved: bool
    reason: str | None = None
    arguments: dict | None = None
    remember: bool = False


# --- idle reaper --------------------------------------------------------------


async def reap_once(manager: SessionManager, idle_timeout_sec: float) -> list[str]:
    """Evict every session that is idle, unwatched, and not running.

    All three conditions matter, and each one alone would be a bug: reaping a
    session with a subscriber kills a browser that is watching it, reaping a
    running one kills work in progress, and reaping on age alone kills the tab
    you left open over lunch.

    One bad session must not end the sweep — a reaper that dies on the first
    error is worse than no reaper, because it looks like it is working.
    """
    now = time.monotonic()
    reaped: list[str] = []
    for session in manager.live:
        if session.running or session.subscriber_count:
            continue
        if now - session.last_activity < idle_timeout_sec:
            continue
        try:
            await manager.delete(session.id)
        except Exception:
            logger.exception("failed to reap session %s", session.id)
        else:
            reaped.append(session.id)
    if reaped:
        logger.info("reaped %d idle session(s): %s", len(reaped), ", ".join(reaped))
    return reaped


async def _reap_loop(
    manager: SessionManager, *, idle_timeout_sec: float, interval_sec: float
) -> None:
    while True:
        await asyncio.sleep(interval_sec)
        await reap_once(manager, idle_timeout_sec)


# --- the ledger, opened only if asked for -------------------------------------


async def _ledger(state: Any) -> tuple[Any, Any]:
    """Connect to Postgres/Redis on first use, not at startup.

    The chat server has to run with the whole Phase-3 gateway stack down, so a
    connection here cannot be a precondition for serving anything.
    """
    if getattr(state, "pool", None) is None:
        dsn = os.environ.get("FORGE_LEDGER_DSN")
        if not dsn:
            raise RuntimeError("FORGE_LEDGER_DSN is not set")
        state.pool = await ledger.init_pool(dsn)
        state.redis = redis.asyncio.from_url(
            load_gateway_config().redis.url, decode_responses=True
        )
    return state.pool, state.redis


# --- the app ------------------------------------------------------------------


def create_app(
    *,
    root: Path | None = None,
    max_sessions: int = MAX_SESSIONS,
    idle_timeout_sec: float = IDLE_TIMEOUT_SEC,
    reap_interval_sec: float = REAP_INTERVAL_SEC,
) -> FastAPI:
    """Build the ASGI app.

    A factory as well as a module-level instance: `uvicorn server.app:app` needs
    the instance, and tests need an app rooted at a temp directory with a reaper
    they can hurry along.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        token = os.environ.get(TOKEN_ENV, "")
        if not token:
            # Refuse to start rather than serve unauthenticated. This process can
            # read and write files and run shell commands as whoever launched it;
            # there is no safe degraded mode.
            raise RuntimeError(f"{TOKEN_ENV} is not set")
        app.state.token = token
        app.state.root = (root or Path.cwd()).resolve()
        app.state.pool = None
        app.state.redis = None
        app.state.manager = SessionManager(
            sessions_dir=persistence.default_sessions_dir(),
            # A factory, not one shared instance: an approver carries this
            # session's "always allow" grants, and sharing it would leak one
            # tab's grant into every other tab.
            make_approver=ServerApprover,
            max_sessions=max_sessions,
        )
        app.state.reaper = asyncio.create_task(
            _reap_loop(
                app.state.manager,
                idle_timeout_sec=idle_timeout_sec,
                interval_sec=reap_interval_sec,
            )
        )
        yield
        # Cancel *and* await. A task cancelled but never awaited may not have run
        # a single line, so its `finally` never fires — the same bug stage 2b's
        # `test_delete_cancels_an_in_flight_run` pins one level down.
        app.state.reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.reaper
        await app.state.manager.aclose_all()
        if app.state.redis is not None:
            await app.state.redis.aclose()
        if app.state.pool is not None:
            await app.state.pool.close()

    app = FastAPI(title="FORGE", lifespan=lifespan)

    # --- sessions ---

    @app.post("/api/sessions", dependencies=[Authed])
    async def create_session(body: CreateBody, request: Request) -> dict:
        state = request.app.state
        params = RunParams(
            goal=body.goal,
            resume=body.resume,
            provider=body.provider,
            model=body.model,
            approval_mode=body.approval_mode,
            # Per-server, never per-request: a browser cannot be trusted to name
            # a directory on the host. TODO(phase-17): per-user roots.
            project_root=state.root,
        )
        try:
            session = await state.manager.create(params)
        except SessionLimitReached:
            raise HTTPException(
                status_code=503, detail="all session slots are in use"
            ) from None
        except CompositionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # A goal on create is optional: it makes the whole stage testable with one
        # curl, while the web app creates first and posts a goal when the user
        # types one.
        if body.goal:
            session.start(body.goal)
        return {"session_id": session.id}

    @app.get("/api/sessions", dependencies=[Authed])
    async def list_sessions(request: Request) -> list:
        return request.app.state.manager.list()

    @app.post("/api/sessions/{session_id}/goal", status_code=204, dependencies=[Authed])
    async def send_goal(body: GoalBody, session: Session) -> None:
        try:
            session.start(body.text)
        except SessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/sessions/{session_id}/decisions", status_code=204, dependencies=[Authed]
    )
    async def decide(body: DecisionBody, session: Session) -> None:
        approver = session.approver
        if not isinstance(approver, ServerApprover):
            raise HTTPException(
                status_code=400,
                detail="this session does not take decisions over the wire",
            )
        try:
            approver.resolve(
                body.request_id,
                approved=body.approved,
                reason=body.reason,
                arguments=body.arguments,
                remember=body.remember,
            )
        except UnknownDecision as exc:
            # Already answered, or expired. The client's stale confirm modal is a
            # normal thing to happen, not a server error.
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/sessions/{session_id}/interrupt", status_code=204, dependencies=[Authed]
    )
    async def interrupt(session: Session) -> None:
        await session.interrupt()

    @app.delete("/api/sessions/{session_id}", status_code=204, dependencies=[Authed])
    async def delete_session(session_id: str, request: Request) -> None:
        try:
            await request.app.state.manager.delete(session_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"no live session {session_id!r}"
            ) from None

    # --- transports ---

    @app.get("/api/sessions/{session_id}/events", dependencies=[Authed])
    async def events(
        session: Session,
        after_seq: int = -1,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        return sse.stream(session, after_seq=after_seq, last_event_id=last_event_id)

    @app.websocket("/ws/sessions/{session_id}")
    async def websocket_events(
        websocket: WebSocket,
        session_id: str,
        token: str | None = None,
        after_seq: int = -1,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        # Both checks reject *before* `accept()`. There is no HTTP response left
        # to carry a 401 once the handshake is under way, and accepting first has
        # already told the browser it was authorised.
        if not _valid(websocket.app, _presented(authorization, token)):
            await websocket.close(code=1008)
            return
        try:
            session = websocket.app.state.manager.get(session_id)
        except KeyError:
            await websocket.close(code=1008, reason="no such session")
            return

        await websocket.accept()
        await ws.serve(session, websocket, after_seq=after_seq)

    # --- traces ---

    @app.get("/api/traces", dependencies=[Authed])
    async def list_traces() -> list[dict]:
        return [trace_summary(tid) for tid in list_trace_ids()]

    @app.get("/api/traces/{trace_id}", dependencies=[Authed])
    async def get_trace(trace_id: str) -> list[dict]:
        return read_spans(trace_id)

    # --- stats ---

    @app.get("/api/stats", dependencies=[Authed])
    async def stats(request: Request) -> dict:
        """Degrade, never 500. The dashboard renders the offline state itself."""
        try:
            pool, conn = await _ledger(request.app.state)
            return {"available": True, **await compute_stats(pool, conn)}
        except Exception as exc:  # noqa: BLE001 - any connection failure is the same answer
            logger.info("stats unavailable: %s", exc)
            return {"available": False, "detail": str(exc)}

    # --- the bundle ---

    if (STATIC_DIR / "assets").is_dir():
        app.mount(
            "/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets"
        )

    # Unauthenticated on purpose: this is the empty shell, and it has to load
    # before the user can hand it a token via `?token=`. Everything it then asks
    # for is behind `require_token`.
    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def index() -> Any:
        index_html = STATIC_DIR / "index.html"
        if not index_html.is_file():
            # A 404 here reads as "wrong URL" and sends you looking in the wrong
            # place. Say what is actually missing.
            return HTMLResponse(_NO_BUNDLE)
        return FileResponse(index_html)

    return app


app = create_app()
