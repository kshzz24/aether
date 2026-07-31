"""The `ctrl+b` pane: what the agent changed, and the repo it changed it in.

Two questions the transcript answers badly. "What has it touched so far?" is
buried across fifty scrolled-past tool entries. "What else is in here?" requires
leaving the app. Both are nearly free once `UndoStack.touched()` exists — the
snapshotter built for `/undo` pays for this pane too.

The tree *arithmetic* lives in `tui/filetree.py`, widget-free, because `/files`
renders the same structure as text and `tui/commands.py` may not import Textual.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, DirectoryTree, Label, Tree

from tui.filetree import group_paths, sorted_items


class Sidebar(Vertical):
    """Changed files above, the repo below. Hidden until `ctrl+b`."""

    class FileChosen(Message):
        """A file was clicked; the app previews it in the transcript."""

        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path = path

    def __init__(self, root: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root = root

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="sidebar-body"):
            yield Label("changed this session", classes="sidebar-heading")
            yield Tree("changed", id="changed-tree")
            with Collapsible(title="repository", collapsed=True, id="repo-section"):
                yield DirectoryTree(str(self._root), id="repo-tree")

    def on_mount(self) -> None:
        self.refresh_changed([])

    # ------------------------------------------------------------------ state

    def refresh_changed(self, paths: list[Path]) -> None:
        """Rebuild the changed-files tree. Cheap enough to call every turn."""
        tree = self.query_one("#changed-tree", Tree)
        tree.show_root = False
        tree.clear()
        if not paths:
            tree.root.add_leaf("nothing yet")
            return

        def build(node, mapping: dict, prefix: str) -> None:
            for name, child in sorted_items(mapping):
                path = f"{prefix}{name}"
                if child is None:
                    # The resolved path rides on the node, so a click doesn't
                    # have to re-derive it from the label.
                    node.add_leaf(name, data=self._root / path)
                else:
                    build(node.add(name, expand=True), child, f"{path}/")

        build(tree.root, group_paths(paths, self._root), "")
        tree.root.expand_all()

    # --------------------------------------------------------------- handlers

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if isinstance(event.node.data, Path):
            self.post_message(self.FileChosen(event.node.data))

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        event.stop()
        self.post_message(self.FileChosen(Path(event.path)))
