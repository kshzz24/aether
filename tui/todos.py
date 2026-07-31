"""The task panel: a live view of the agent's `todo` list.

Subscribes to the same `TodoStore` the tool writes to, so the panel updates the
instant the agent changes a task — no polling, and no backchannel into the agent
core. The store is plain data owned by the composition root; the tool writes it,
the panel reads it, and neither knows the other exists.

Hidden while the list is empty. A permanently-visible empty panel is a standing
tax on transcript height for a feature most short runs never use.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from tools.todo import TodoStore

_MARKS: dict[str, tuple[str, str]] = {
    "pending": ("○", "dim"),
    "in_progress": ("◐", "bold yellow"),
    "done": ("●", "green"),
}


class TodoPanel(Static):
    """Renders a TodoStore; hides itself when there is nothing to show."""

    def __init__(self, store: TodoStore, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._store = store

    def on_mount(self) -> None:
        self._store.subscribe(self.refresh_todos)
        self.refresh_todos()

    def refresh_todos(self) -> None:
        todos = self._store.list()
        self.display = bool(todos)
        if not todos:
            self.update("")
            return

        done = sum(1 for t in todos if t.status == "done")
        body = Text()
        body.append(f"tasks {done}/{len(todos)}\n", style="bold")
        for todo in todos:
            mark, style = _MARKS[todo.status]
            line = Text(f"{mark} {todo.text}", style=style)
            if todo.status == "done":
                line.stylize("strike")
            body.append_text(line)
            body.append("\n")
        self.update(body)
