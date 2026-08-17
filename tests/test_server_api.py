"""Phase 8, stage 4 — the six REST routes.

Each route is a translation and nothing more: a request becomes a method call that
already existed, and an exception becomes a status code. So what is worth testing
is exactly the translation — that `SessionLimitReached` is a 503 and not a 500,
that a stale confirm is a 409 and not a crash, that a browser cannot name a
directory on the host.
"""

from __future__ import annotations

import asyncio

from asgi_client import running

from client import NormalizedResponse, TextBlock, ToolCallBlock


def say(text: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="end_turn",
    )


def call_shell(command: str) -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[
            ToolCallBlock(id="c1", name="run_shell", arguments={"command": command})
        ],
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.01,
        stop_reason="tool_use",
    )


async def wait_for_confirm(approver, timeout: float = 2.0) -> str:
    async with asyncio.timeout(timeout):
        while not approver.pending:
            await asyncio.sleep(0)
    return next(iter(approver.pending))


# --- create -------------------------------------------------------------------


async def test_create_returns_a_session_id_and_lists_it(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        listed = await client.get("/api/sessions", headers=auth)
        assert [meta["id"] for meta in listed.json()] == [session_id]


async def test_a_goal_on_create_starts_the_run(forge_app, auth, stub_responses):
    """The whole stage, reachable with one curl."""
    stub_responses([say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions", json={"goal": "say hello"}, headers=auth
        )
        session = app.state.manager.get(created.json()["session_id"])
        await asyncio.wait_for(session.wait(), 3.0)

    assert session.transcript[-1]["type"] == "terminal"
    assert session.transcript[-1]["reason"] == "completed"


async def test_create_without_a_goal_starts_nothing(forge_app, auth):
    """What the web app does: open a session, then wait for the user to type."""
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session = app.state.manager.get(created.json()["session_id"])
        assert session.running is False
        assert session.transcript == []


async def test_the_client_cannot_choose_the_project_root(forge_app, auth, tmp_path):
    """`project_root` is per-server. A browser is not allowed to name a directory.

    The body model simply has no such field, so a client that sends one is
    ignored rather than obeyed — which is why this asserts on the resulting
    session rather than on a status code.
    """
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions", json={"project_root": "/etc"}, headers=auth
        )
        session = app.state.manager.get(created.json()["session_id"])

    assert session._comp.agent.repo_root == tmp_path.resolve()


async def test_the_cap_is_a_503(forge_app, auth):
    """Capacity, not a bug: the client should back off, not report a failure."""
    app = forge_app(max_sessions=2)
    async with running(app) as client:
        for _ in range(2):
            assert (
                await client.post("/api/sessions", json={}, headers=auth)
            ).status_code == 200
        refused = await client.post("/api/sessions", json={}, headers=auth)

    assert refused.status_code == 503


async def test_an_unreadable_resume_is_a_400(forge_app, auth):
    """`CompositionError` is the caller's fault, so it is not a 500."""
    async with running(forge_app()) as client:
        response = await client.post(
            "/api/sessions", json={"resume": "no-such-session"}, headers=auth
        )
    assert response.status_code == 400


# --- goal ---------------------------------------------------------------------


async def test_the_first_goal_names_the_session(forge_app, auth, stub_responses):
    """Otherwise `GET /api/sessions` lists a column of blanks.

    `build_composition` names the on-disk session from `params.goal`, and the web
    path has no goal at create time — so without naming on first `start` every
    browser-created session would be saved with `goal: ""`.
    """
    stub_responses([say("done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        posted = await client.post(
            f"/api/sessions/{session_id}/goal",
            json={"text": "refactor the parser"},
            headers=auth,
        )
        assert posted.status_code == 204

        session = app.state.manager.get(session_id)
        await asyncio.wait_for(session.wait(), 3.0)

        listed = await client.get("/api/sessions", headers=auth)

    (meta,) = [m for m in listed.json() if m["id"] == session_id]
    assert meta["goal"] == "refactor the parser"


async def test_a_second_goal_while_running_is_a_409(forge_app, auth, stub_responses):
    stub_responses([call_shell("sleep"), say("done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions", json={"goal": "first"}, headers=auth
        )
        session_id = created.json()["session_id"]

        second = await client.post(
            f"/api/sessions/{session_id}/goal", json={"text": "second"}, headers=auth
        )
        assert second.status_code == 409

        await client.post(f"/api/sessions/{session_id}/interrupt", headers=auth)


async def test_a_goal_for_an_unknown_session_is_a_404(forge_app, auth):
    async with running(forge_app()) as client:
        response = await client.post(
            "/api/sessions/missing/goal", json={"text": "hi"}, headers=auth
        )
    assert response.status_code == 404


# --- decisions ----------------------------------------------------------------


async def test_a_decision_resumes_a_parked_run(forge_app, auth, stub_responses):
    stub_responses([call_shell("echo hi"), say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions",
            json={"goal": "run something", "approval_mode": "on-request"},
            headers=auth,
        )
        session_id = created.json()["session_id"]
        session = app.state.manager.get(session_id)

        request_id = await wait_for_confirm(session.approver)
        answered = await client.post(
            f"/api/sessions/{session_id}/decisions",
            json={"request_id": request_id, "approved": True},
            headers=auth,
        )
        assert answered.status_code == 204

        await asyncio.wait_for(session.wait(), 3.0)

    assert session.transcript[-1]["reason"] == "completed"


async def test_answering_twice_is_a_409(forge_app, auth, stub_responses):
    """A stale confirm modal is a normal thing to happen, not a server error."""
    stub_responses([call_shell("echo hi"), say("all done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions",
            json={"goal": "run something", "approval_mode": "on-request"},
            headers=auth,
        )
        session_id = created.json()["session_id"]
        session = app.state.manager.get(session_id)
        request_id = await wait_for_confirm(session.approver)

        body = {"request_id": request_id, "approved": True}
        first = await client.post(
            f"/api/sessions/{session_id}/decisions", json=body, headers=auth
        )
        second = await client.post(
            f"/api/sessions/{session_id}/decisions", json=body, headers=auth
        )
        await asyncio.wait_for(session.wait(), 3.0)

    assert first.status_code == 204
    assert second.status_code == 409


async def test_an_unknown_request_id_is_a_409(forge_app, auth):
    app = forge_app()
    async with running(app) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        response = await client.post(
            f"/api/sessions/{session_id}/decisions",
            json={"request_id": "never-existed", "approved": True},
            headers=auth,
        )
    assert response.status_code == 409


# --- interrupt and delete -----------------------------------------------------


async def test_interrupt_stops_a_run(forge_app, auth, stub_responses):
    stub_responses([call_shell("echo hi"), say("done")])
    app = forge_app()
    async with running(app) as client:
        created = await client.post(
            "/api/sessions",
            json={"goal": "run something", "approval_mode": "on-request"},
            headers=auth,
        )
        session_id = created.json()["session_id"]
        session = app.state.manager.get(session_id)
        await wait_for_confirm(session.approver)

        response = await client.post(
            f"/api/sessions/{session_id}/interrupt", headers=auth
        )

    assert response.status_code == 204
    assert session.running is False
    assert session.transcript[-1] == {
        "type": "status",
        "message": "interrupted",
        "seq": session.transcript[-1]["seq"],
    }


async def test_interrupting_an_idle_session_is_still_204(forge_app, auth):
    async with running(forge_app()) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]
        response = await client.post(
            f"/api/sessions/{session_id}/interrupt", headers=auth
        )
    assert response.status_code == 204


async def test_delete_evicts_and_then_404s(forge_app, auth):
    async with running(forge_app()) as client:
        created = await client.post("/api/sessions", json={}, headers=auth)
        session_id = created.json()["session_id"]

        first = await client.delete(f"/api/sessions/{session_id}", headers=auth)
        second = await client.delete(f"/api/sessions/{session_id}", headers=auth)

        listed = await client.get("/api/sessions", headers=auth)

    assert first.status_code == 204
    assert second.status_code == 404
    # Deleted from memory, still on disk: `DELETE` ends a live session, it does
    # not destroy the checkpoint you might want to resume.
    assert [meta["id"] for meta in listed.json()] == [session_id]
