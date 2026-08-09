"""Dump the wire format to JSON so the web UI can be built without a server.

The web UI consumes frames, and stage 4 is what will serve them over HTTP. Rather
than hand-write mock data — which drifts from the encoder the moment either
changes — this runs `sample_events()` through the real `server/wire.py` encoder.
The fixture is therefore provably the wire format, not an approximation of it.

`sample_events()` (tests/conftest.py) holds one instance of every member of the
`Event` union and is already the gate that forces every surface to grow a branch
when a variant is added. Regenerating this file propagates that gate into the
browser, where `tsc` enforces exhaustiveness over the discriminated union.

The four server-minted control frames (`ready`, `confirm`, `overflow`, `error`)
are appended by hand, because they are not `Event`s — they are produced by the
presentation layer itself (`AgentSession`, `ServerApprover`, the transports) and
so have no encoder arm to borrow.

    python scripts/dump_frames.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import sample_events  # noqa: E402

from server import wire  # noqa: E402

OUT = ROOT / "web" / "src" / "mocks" / "frames.json"


def control_frames(start_seq: int) -> list[dict]:
    """The frames no `Event` produces.

    `ready` and `overflow` deliberately carry no `seq`: `seq` addresses a
    position in the transcript for `Last-Event-ID` to resume from, and neither is
    a transcript entry. `confirm` does carry one — a browser reconnecting
    mid-confirm has to be re-shown the pending question.
    """
    return [
        {"type": "ready"},
        {
            "type": "confirm",
            "request_id": "b7c1e2f4a9d04c1e8f3a2b5c6d7e8f90",
            "tool_name": "run_shell",
            "arguments": {"command": "rm -rf build/"},
            "kind": "execute",
            "danger_reasons": ["deletes a directory tree"],
            "diff": None,
            "offers_always": False,
            "seq": start_seq,
        },
        {
            "type": "confirm",
            "request_id": "c8d2f3a5b0e15d2f9a4b3c6d7e8f9012",
            "tool_name": "write_file",
            "arguments": {"path": "src/parser.py", "content": "def parse():\n    ..."},
            "kind": "write",
            "danger_reasons": [],
            "diff": (
                "--- a/src/parser.py\n"
                "+++ b/src/parser.py\n"
                "@@ -1,3 +1,4 @@\n"
                " import re\n"
                "+\n"
                "+def parse():\n"
                "+    ...\n"
            ),
            "offers_always": True,
            "seq": start_seq + 1,
        },
        {"type": "overflow"},
        {"type": "error", "detail": "provider returned 503 after 3 retries"},
    ]


def main() -> int:
    events = sample_events()
    frames = [wire.frame(event, seq) for seq, event in enumerate(events)]
    frames += control_frames(len(frames))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(frames, indent=2) + "\n", encoding="utf-8")

    kinds = sorted({f["type"] for f in frames})
    print(f"wrote {len(frames)} frames to {OUT.relative_to(ROOT)}")
    print(f"{len(kinds)} frame types: {', '.join(kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
