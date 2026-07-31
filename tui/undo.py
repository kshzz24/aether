"""Undo for the agent's file writes, built on the Phase-2 hook points.

`tools/hooks.py` has carried `before_tool`/`after_tool` as no-ops since Phase 2
and nothing has ever used them. This is what they were for: `before_tool` fires
before approval and before dispatch (`agent.py:147`), which is the only moment
the pre-edit bytes still exist, and `after_tool` fires with the result string
(`agent.py:248`), which is how we learn whether the write actually happened.

Two things make this correct rather than approximately correct:

* A snapshot is *pending* until `after_tool` says the tool succeeded. A denied
  or failed write must not become an undo entry, or `/undo` would "restore" a
  file nobody changed and clobber whatever is there now.
* Snapshots are bytes, not text. A file the agent edits may not be UTF-8, and an
  undo that silently transcodes a file is a worse bug than no undo at all.

The surface owns batching: it calls `start_batch()` at the top of each agent
turn, so `/undo` reverts one turn's worth of edits rather than a single write or
the entire session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tools.hooks import Hooks

logger = logging.getLogger(__name__)

# The tools whose effects are undoable. Both take `path`, so one snapshotter
# covers both; anything else (shell commands especially) is out of reach and
# must not pretend otherwise.
WRITE_TOOLS = frozenset({"write_file", "edit_file"})

# Files larger than this are not snapshotted. Holding a 200 MB fixture in memory
# to make undo work is a bad trade; say so rather than doing it.
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024

# Turns of history kept. Undo is a safety net for "that wasn't what I meant",
# not a version-control system — git is right there.
MAX_BATCHES = 20


@dataclass(frozen=True)
class Snapshot:
    """One file as it was before a tool touched it."""

    path: Path
    # None means the file did not exist, so undoing means deleting it.
    before: bytes | None
    # False when the file was too large to capture; recorded so `touched()` is
    # still complete and the user is told why it can't be reverted.
    capturable: bool = True


@dataclass
class UndoResult:
    restored: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.restored) + len(self.deleted)

    def summary(self, empty: str = "nothing to undo") -> str:
        if not (self.changed or self.failed or self.skipped):
            return empty
        parts = []
        if self.restored:
            parts.append(f"restored {len(self.restored)}")
        if self.deleted:
            parts.append(f"removed {len(self.deleted)}")
        if self.skipped:
            parts.append(f"skipped {len(self.skipped)} (too large to snapshot)")
        if self.failed:
            detail = "; ".join(f"{path.name}: {why}" for path, why in self.failed)
            parts.append(f"failed on {len(self.failed)} ({detail})")
        return ", ".join(parts)


class UndoStack:
    """Snapshots file writes per turn and puts them back on request."""

    def __init__(self, *, max_batches: int = MAX_BATCHES) -> None:
        self._max_batches = max_batches
        self._batches: list[list[Snapshot]] = []
        # Undone batches, newest last. Captured *as they were before the undo*,
        # so redo puts the agent's version back.
        self._redo: list[list[Snapshot]] = []
        # Captured by before_tool, promoted into the batch only once after_tool
        # confirms the write landed.
        self._pending: Snapshot | None = None

    # ----------------------------------------------------------------- batches

    def start_batch(self) -> None:
        """Open a new turn. Called by the surface on each StatusEvent."""
        self._batches.append([])
        if len(self._batches) > self._max_batches:
            del self._batches[0]
        # New work invalidates the redo stack, exactly as in an editor: redoing
        # onto a file the agent has since rewritten would silently discard the
        # newer version.
        self._redo.clear()

    # ------------------------------------------------------------------- hooks

    def hooks(self) -> Hooks:
        """A `Hooks` bound to this stack, for `build_composition`."""
        return Hooks(before_tool=self.before_tool, after_tool=self.after_tool)

    def before_tool(self, name: str, arguments: dict) -> None:
        self._pending = None
        if name not in WRITE_TOOLS:
            return
        raw = arguments.get("path")
        if not isinstance(raw, str) or not raw:
            return

        path = Path(raw)
        try:
            if not path.exists():
                self._pending = Snapshot(path, None)
            elif path.stat().st_size > MAX_SNAPSHOT_BYTES:
                self._pending = Snapshot(path, None, capturable=False)
            else:
                self._pending = Snapshot(path, path.read_bytes())
        except OSError as exc:
            # An unreadable file is not a reason to abort the agent's write.
            logger.debug("undo snapshot of %s failed: %s", path, exc)
            self._pending = None

    def after_tool(self, name: str, arguments: dict, result: str) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        # The agent turns every failure into an observation string rather than
        # raising (`agent.py:236`), so the result text is the only signal that
        # the write did not happen.
        if result.startswith("ERROR") or result.startswith("DENIED"):
            return
        if not self._batches:
            self.start_batch()
        self._batches[-1].append(pending)

    # -------------------------------------------------------------- inspection

    def touched(self) -> list[Path]:
        """Every file written this session, most recently written first."""
        seen: dict[Path, None] = {}
        for batch in reversed(self._batches):
            for snapshot in reversed(batch):
                seen.setdefault(snapshot.path, None)
        return list(seen)

    @property
    def depth(self) -> int:
        """How many turns are still undoable."""
        return sum(1 for batch in self._batches if batch)

    @property
    def redo_depth(self) -> int:
        return len(self._redo)

    # ------------------------------------------------------------ undo / redo

    def _capture(self, path: Path) -> Snapshot:
        """The file as it stands right now, for the opposite stack."""
        try:
            if not path.exists():
                return Snapshot(path, None)
            if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                return Snapshot(path, None, capturable=False)
            return Snapshot(path, path.read_bytes())
        except OSError as exc:
            logger.debug("could not capture %s: %s", path, exc)
            return Snapshot(path, None, capturable=False)

    def _apply(self, batch: list[Snapshot]) -> tuple[UndoResult, list[Snapshot]]:
        """Restore `batch`, returning the result and the inverse batch.

        The inverse is what makes redo work: before overwriting a file we record
        what was there, so the operation can be run backwards.
        """
        result = UndoResult()
        inverse: list[Snapshot] = []
        # Reverse order within the turn: two edits to one file must unwind to
        # the state before the first of them.
        for snapshot in reversed(batch):
            if not snapshot.capturable:
                result.skipped.append(snapshot.path)
                continue
            current = self._capture(snapshot.path)
            try:
                if snapshot.before is None:
                    snapshot.path.unlink(missing_ok=True)
                    result.deleted.append(snapshot.path)
                else:
                    snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.path.write_bytes(snapshot.before)
                    result.restored.append(snapshot.path)
            except OSError as exc:
                result.failed.append((snapshot.path, str(exc)))
                continue
            inverse.append(current)
        return result, inverse

    def undo_last(self) -> UndoResult:
        """Revert the newest turn that changed anything."""
        while self._batches and not self._batches[-1]:
            self._batches.pop()
        if not self._batches:
            return UndoResult()

        result, inverse = self._apply(self._batches.pop())
        if inverse:
            self._redo.append(inverse)
        return result

    def redo_last(self) -> UndoResult:
        """Put back what the most recent `/undo` reverted."""
        if not self._redo:
            return UndoResult()
        result, inverse = self._apply(self._redo.pop())
        if inverse:
            self._batches.append(inverse)
        return result
