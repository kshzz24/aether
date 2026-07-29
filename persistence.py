"""Session checkpoints: snapshot resumable run state to disk and restore it.

This is a composition/data concern. The agent core never imports it (Invariant
1: the agent does no I/O). The driver (main.py) builds a `Session` from the
agent's plain state attributes and calls `save`; on resume it `load`s one and
feeds `session.messages` back as history.

State is stored as JSON, one file per session at ``<dir>/<id>.json``. The block
union (`TextBlock`/`ToolCallBlock`/`ToolResultBlock`) carries a ``kind`` tag so
it round-trips without ambiguity.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from client import Message, TextBlock, ToolCallBlock, ToolResultBlock

_Block = TextBlock | ToolCallBlock | ToolResultBlock


@dataclass
class Session:
    id: str
    goal: str
    provider: str
    model: str
    created_at: str
    updated_at: str
    total_cost: float
    messages: list[Message] = field(default_factory=list)


@dataclass(frozen=True)
class SessionMeta:
    """A cheap header for `list_sessions` — no message bodies."""

    id: str
    goal: str
    updated_at: str
    total_cost: float
    turns: int


def default_sessions_dir() -> Path:
    return Path.home() / ".forge" / "sessions"


def new_session_id() -> str:
    """Sortable timestamp + short random suffix, e.g. ``20260729-143005-a1b2``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


# --- serialization -----------------------------------------------------------

def _block_to_dict(block: _Block) -> dict:
    if isinstance(block, TextBlock):
        return {"kind": "text", "text": block.text}
    if isinstance(block, ToolCallBlock):
        return {
            "kind": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": block.arguments,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "kind": "tool_result",
            "tool_call_id": block.tool_call_id,
            "content": block.content,
        }
    raise TypeError(f"cannot serialize block: {block!r}")


def _block_from_dict(data: dict) -> _Block:
    kind = data["kind"]
    if kind == "text":
        return TextBlock(text=data["text"])
    if kind == "tool_call":
        return ToolCallBlock(
            id=data["id"], name=data["name"], arguments=data["arguments"]
        )
    if kind == "tool_result":
        return ToolResultBlock(
            tool_call_id=data["tool_call_id"], content=data["content"]
        )
    raise ValueError(f"unknown block kind: {kind!r}")


def _message_to_dict(message: Message) -> dict:
    return {
        "role": message.role,
        "blocks": [_block_to_dict(b) for b in message.blocks],
    }


def _message_from_dict(data: dict) -> Message:
    return Message(
        role=data["role"],
        blocks=[_block_from_dict(b) for b in data["blocks"]],
    )


def session_to_dict(session: Session) -> dict:
    return {
        "id": session.id,
        "goal": session.goal,
        "provider": session.provider,
        "model": session.model,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "total_cost": session.total_cost,
        "messages": [_message_to_dict(m) for m in session.messages],
    }


def session_from_dict(data: dict) -> Session:
    return Session(
        id=data["id"],
        goal=data["goal"],
        provider=data["provider"],
        model=data["model"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        total_cost=data["total_cost"],
        messages=[_message_from_dict(m) for m in data["messages"]],
    )


# --- store -------------------------------------------------------------------

def save(session: Session, sessions_dir: Path) -> Path:
    """Atomically write ``<dir>/<id>.json``. Never leaves a partial file."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target = sessions_dir / f"{session.id}.json"
    tmp = sessions_dir / f".{session.id}.json.tmp"
    tmp.write_text(
        json.dumps(session_to_dict(session), indent=2), encoding="utf-8"
    )
    os.replace(tmp, target)  # atomic on the same filesystem
    return target


def load(session_id: str, sessions_dir: Path) -> Session:
    path = sessions_dir / f"{session_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return session_from_dict(data)


def list_sessions(sessions_dir: Path) -> list[SessionMeta]:
    """Return session headers, newest first. Skips unreadable/foreign files."""
    if not sessions_dir.exists():
        return []
    metas: list[SessionMeta] = []
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metas.append(
                SessionMeta(
                    id=data["id"],
                    goal=data["goal"],
                    updated_at=data["updated_at"],
                    total_cost=data["total_cost"],
                    turns=len(data["messages"]),
                )
            )
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logging.warning("skipping unreadable session file %s: %s", path, exc)
    metas.sort(key=lambda m: m.updated_at, reverse=True)
    return metas
