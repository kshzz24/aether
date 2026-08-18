"""JSONL trace storage: one file per trace, one line per span, append-only.

A trace is an append log of many spans, not one final-state snapshot -- JSONL
means a crash mid-run leaves a valid, readable partial trace (read however
many complete lines exist), where a single-JSON write would either need to
happen once at the end (losing everything on crash) or corrupt on a partial
atomic replace. Mirrors the shape of `persistence.py` without reusing its
atomic-single-JSON pattern, which is wrong for this access pattern.
"""

from __future__ import annotations

import json
from pathlib import Path


def default_traces_dir() -> Path:
    return Path.home() / ".forge" / "traces"


def trace_path(trace_id: str, traces_dir: Path | None = None) -> Path:
    return (traces_dir or default_traces_dir()) / f"{trace_id}.jsonl"


def append_span(span_dict: dict, traces_dir: Path | None = None) -> None:
    """Append one span (already a plain dict) as one JSONL line."""
    path = trace_path(span_dict["trace_id"], traces_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(span_dict) + "\n")


def read_spans(trace_id: str, traces_dir: Path | None = None) -> list[dict]:
    """Read every complete span line. A truncated final line (crash mid-write)
    is skipped rather than raising -- the point of JSONL over a single-JSON
    snapshot is that the rest of the file stays readable."""
    path = trace_path(trace_id, traces_dir)
    if not path.exists():
        return []
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return spans


def list_trace_ids(traces_dir: Path | None = None) -> list[str]:
    d = traces_dir or default_traces_dir()
    if not d.exists():
        return []
    return sorted((p.stem for p in d.glob("*.jsonl")), reverse=True)


def trace_summary(trace_id: str, traces_dir: Path | None = None) -> dict:
    """Cheap header for a trace list: id, total cost, span count, duration."""
    spans = read_spans(trace_id, traces_dir)
    total_cost = sum(s.get("cost_usd", 0.0) for s in spans)
    run_spans = [s for s in spans if s.get("kind") == "run"]
    duration = 0.0
    if run_spans:
        run = run_spans[0]
        duration = run.get("ended_at", run["started_at"]) - run["started_at"]
    return {
        "trace_id": trace_id,
        "total_cost_usd": total_cost,
        "span_count": len(spans),
        "duration_sec": duration,
    }
