from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from repomap.tags import Tag


class TagCache:
    """mtime-keyed sqlite cache of per-file tags.

    A file whose stored mtime matches its current mtime reuses its tags; a change
    misses and forces a re-parse. mtime is the OS's free invalidation signal.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tags "
            "(path TEXT PRIMARY KEY, mtime REAL, data TEXT)"
        )
        self._conn.commit()

    def get(self, path: str, mtime: float) -> list[Tag] | None:
        row = self._conn.execute(
            "SELECT mtime, data FROM tags WHERE path = ?", (path,)
        ).fetchone()
        if row is None or row[0] != mtime:
            return None
        return [Tag(**d) for d in json.loads(row[1])]

    def put(self, path: str, mtime: float, tags: list[Tag]) -> None:
        data = json.dumps([asdict(t) for t in tags])
        self._conn.execute(
            "INSERT OR REPLACE INTO tags (path, mtime, data) VALUES (?, ?, ?)",
            (path, mtime, data),
        )
        self._conn.commit()
