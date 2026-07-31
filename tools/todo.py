"""The `todo` tool: a scratch task list the agent keeps for itself.

State lives in a `TodoStore` owned by the composition root, not in this module.
Two reasons:

* **Per-run isolation.** Module-level state would leak between runs, and between
  a parent agent and a subagent, in ways nobody asked for.
* **Observability without a backchannel.** A surface (the TUI panel) can watch
  the same store the tool writes to, so the list stays live without the agent
  core learning that a panel exists (Invariant 1).

Deliberately *not* persisted. `persistence.Session` already checkpoints run
state; a second store would be a synchronisation problem with no payoff.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from tools.base import Tool, ToolKind

Status = Literal["pending", "in_progress", "done"]

_STATUS_MARK: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "done": "[x]",
}


@dataclass(frozen=True)
class Todo:
    id: int
    text: str
    status: Status = "pending"


class TodoStore:
    """An ordered task list with change notification."""

    def __init__(self) -> None:
        self._items: list[Todo] = []
        self._next_id = 1
        self._observers: list[Callable[[], None]] = []

    # ------------------------------------------------------------- observers

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Register a change callback. Used by the TUI panel to stay live."""
        self._observers.append(callback)

    def _notify(self) -> None:
        for callback in self._observers:
            # An observer that raises must not break the agent's tool call --
            # a broken panel is a UI bug, not a reason to fail the run.
            try:
                callback()
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- state

    def list(self) -> list[Todo]:
        return list(self._items)

    def add(self, text: str) -> Todo:
        todo = Todo(id=self._next_id, text=text)
        self._next_id += 1
        self._items.append(todo)
        self._notify()
        return todo

    def set_status(self, todo_id: int, status: Status) -> Todo | None:
        for index, todo in enumerate(self._items):
            if todo.id == todo_id:
                updated = Todo(id=todo.id, text=todo.text, status=status)
                self._items[index] = updated
                self._notify()
                return updated
        return None

    def clear(self) -> None:
        self._items.clear()
        self._notify()

    def render(self) -> str:
        if not self._items:
            return "(no todos)"
        return "\n".join(
            f"{_STATUS_MARK[t.status]} {t.id}. {t.text}" for t in self._items
        )


SCHEMA = {
    "name": "todo",
    "description": (
        "Keep a short task list for the current run. Use it to plan multi-step "
        "work and to track what is left: 'add' a task, 'start' it when you begin, "
        "'complete' it when done, 'list' to review. The list is visible to the "
        "user and is discarded when the run ends."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "start", "complete", "list", "clear"],
                "description": "What to do with the task list.",
            },
            "text": {
                "type": "string",
                "description": "Task description. Required for 'add'.",
            },
            "id": {
                "type": "integer",
                "description": "Task id. Required for 'start' and 'complete'.",
            },
        },
        "required": ["action"],
    },
}


def build_todo_tool(*, store: TodoStore) -> Tool:
    """Build the tool over a caller-owned store.

    WRITE rather than READ: it mutates run state the user can see. That keeps it
    inside the approval policy by default instead of quietly exempt.
    """

    async def run(args: dict) -> str:
        action = args["action"]

        if action == "list":
            return store.render()

        if action == "clear":
            store.clear()
            return "cleared"

        if action == "add":
            text = (args.get("text") or "").strip()
            if not text:
                return "ERROR: 'add' needs a non-empty 'text'"
            todo = store.add(text)
            return f"added {todo.id}. {todo.text}"

        todo_id = args.get("id")
        if todo_id is None:
            return f"ERROR: {action!r} needs an 'id'"
        status: Status = "in_progress" if action == "start" else "done"
        updated = store.set_status(todo_id, status)
        if updated is None:
            return f"ERROR: no todo with id {todo_id}"
        return f"{action}ed {updated.id}. {updated.text}"

    return Tool(
        name=SCHEMA["name"],
        description=SCHEMA["description"],
        parameters=SCHEMA["parameters"],
        kind=ToolKind.WRITE,
        run=run,
    )
