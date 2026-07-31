"""The `todo` tool and its store.

The store is deliberately a plain object owned by the composition root rather
than module state: that is what lets the TUI panel watch the same list the tool
writes to, and what keeps two runs from sharing a task list.
"""

from __future__ import annotations

import asyncio

from config import ForgeConfig
from tools import build_registry
from tools.base import ToolKind
from tools.todo import TodoStore, build_todo_tool


def drive(coro):
    return asyncio.run(coro)


def _tool(store: TodoStore | None = None):
    return build_todo_tool(store=store or TodoStore())


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def test_add_assigns_incrementing_ids():
    store = TodoStore()
    assert store.add("first").id == 1
    assert store.add("second").id == 2


def test_ids_are_not_reused_after_a_clear():
    """A stale id from before a clear must not silently address a new task."""
    store = TodoStore()
    store.add("first")
    store.clear()
    assert store.add("second").id == 2


def test_set_status_updates_in_place_and_keeps_order():
    store = TodoStore()
    store.add("a")
    store.add("b")
    store.set_status(1, "done")
    assert [t.text for t in store.list()] == ["a", "b"]
    assert store.list()[0].status == "done"


def test_set_status_on_a_missing_id_returns_none():
    assert TodoStore().set_status(99, "done") is None


def test_list_returns_a_copy():
    """Callers must not be able to mutate the store's list by side effect."""
    store = TodoStore()
    store.add("a")
    store.list().clear()
    assert len(store.list()) == 1


def test_render_marks_each_status():
    store = TodoStore()
    store.add("pending one")
    store.add("running one")
    store.set_status(2, "in_progress")
    rendered = store.render()
    assert "[ ] 1. pending one" in rendered
    assert "[~] 2. running one" in rendered


def test_render_handles_an_empty_list():
    assert "no todos" in TodoStore().render()


# --------------------------------------------------------------------------
# Observers — what makes the live panel possible
# --------------------------------------------------------------------------


def test_observers_fire_on_every_mutation():
    store = TodoStore()
    calls = []
    store.subscribe(lambda: calls.append(1))

    store.add("a")
    store.set_status(1, "done")
    store.clear()
    assert len(calls) == 3


def test_a_raising_observer_does_not_break_the_tool():
    """A broken panel is a UI bug; it must not fail the agent's tool call."""
    store = TodoStore()
    store.subscribe(lambda: 1 / 0)
    store.add("still works")
    assert len(store.list()) == 1


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------


def test_tool_is_write_not_read():
    """READ is auto-approved by policy. This mutates user-visible run state, so
    it stays inside the approval gate."""
    assert _tool().kind is ToolKind.WRITE


def test_add_then_list_round_trips():
    store = TodoStore()
    tool = _tool(store)
    drive(tool.run({"action": "add", "text": "write the tests"}))
    assert "write the tests" in drive(tool.run({"action": "list"}))


def test_start_then_complete_moves_status():
    store = TodoStore()
    tool = _tool(store)
    drive(tool.run({"action": "add", "text": "a task"}))
    drive(tool.run({"action": "start", "id": 1}))
    assert store.list()[0].status == "in_progress"
    drive(tool.run({"action": "complete", "id": 1}))
    assert store.list()[0].status == "done"


def test_add_without_text_is_an_error_observation():
    """Invariant 5: a bad call is data the model can correct, not an exception."""
    result = drive(_tool().run({"action": "add"}))
    assert result.startswith("ERROR:")


def test_status_change_without_an_id_is_an_error_observation():
    result = drive(_tool().run({"action": "complete"}))
    assert result.startswith("ERROR:")


def test_unknown_id_is_an_error_observation():
    result = drive(_tool().run({"action": "complete", "id": 42}))
    assert result.startswith("ERROR:")


def test_clear_empties_the_list():
    store = TodoStore()
    tool = _tool(store)
    drive(tool.run({"action": "add", "text": "a"}))
    drive(tool.run({"action": "clear"}))
    assert store.list() == []


# --------------------------------------------------------------------------
# Registry wiring
# --------------------------------------------------------------------------


def test_todo_is_registered_as_a_builtin():
    registry = build_registry(ForgeConfig())
    assert registry.get("todo").kind is ToolKind.WRITE


def test_an_injected_store_is_the_one_the_tool_writes_to():
    """This is what makes the live panel work: the caller holds the store."""
    store = TodoStore()
    registry = build_registry(ForgeConfig(), todo_store=store)
    drive(registry.get("todo").run({"action": "add", "text": "observed"}))
    assert [t.text for t in store.list()] == ["observed"]


def test_two_registries_do_not_share_a_task_list():
    """Per-run isolation — module-level state would leak between runs."""
    first = build_registry(ForgeConfig())
    second = build_registry(ForgeConfig())
    drive(first.get("todo").run({"action": "add", "text": "only mine"}))
    assert "only mine" not in drive(second.get("todo").run({"action": "list"}))
