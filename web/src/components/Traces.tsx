/**
 * Minimal trace table: id, total cost, span count, duration.
 *
 * Deliberately not a flamegraph -- the gate for phase 9 is "tracing shows
 * complete per-request spans", which a table satisfies. A timeline/flamegraph
 * view is real scope, deferred: TODO(phase-14+): span visualization.
 */

import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { TraceSummary } from "../api/frames";

export function Traces() {
  const [traces, setTraces] = useState<TraceSummary[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api
      .listTraces()
      .then((next) => live && setTraces(next))
      .catch((exc: unknown) => {
        if (live) setFailed(exc instanceof Error ? exc.message : String(exc));
      });
    return () => {
      live = false;
    };
  }, []);

  if (failed) {
    return (
      <div className="dash">
        <div className="tile wide">
          <div className="eyebrow">traces unavailable</div>
          <p className="empty" style={{ padding: "0.6rem 0 0" }}>
            The server did not answer <code>/api/traces</code>: {failed}.
          </p>
        </div>
      </div>
    );
  }

  if (!traces) {
    return (
      <div className="dash">
        <div className="tile wide">
          <div className="eyebrow">loading</div>
        </div>
      </div>
    );
  }

  if (traces.length === 0) {
    return (
      <div className="dash">
        <div className="tile wide">
          <div className="eyebrow">no traces yet</div>
          <p className="empty" style={{ padding: "0.6rem 0 0" }}>
            A trace appears here after a run completes.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="dash">
      <div className="tile wide">
        <div className="eyebrow">recent traces</div>
        <table className="trace-table">
          <thead>
            <tr>
              <th>trace id</th>
              <th>total cost</th>
              <th>spans</th>
              <th>duration</th>
            </tr>
          </thead>
          <tbody>
            {traces.map((t) => (
              <tr key={t.trace_id}>
                <td title={t.trace_id}>{t.trace_id}</td>
                <td>${t.total_cost_usd.toFixed(4)}</td>
                <td>{t.span_count}</td>
                <td>{t.duration_sec.toFixed(1)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
