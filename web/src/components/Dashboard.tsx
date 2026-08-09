/**
 * Gateway metrics, straight from `gateway/metrics.py:compute_stats`.
 *
 * Two form decisions, both from the data's job rather than from taste:
 *
 * - **Requests, error rate, p95, cache hits are stat tiles, not charts.** Each is a
 *   single current value with no series behind it; a sparkline would imply history
 *   the endpoint does not return.
 * - **Cost by model is a horizontal bar list in ONE hue.** It is a single measure
 *   compared across categories, so it is one series — identity belongs to the
 *   labels. Giving each model its own colour would be colouring by rank, and the
 *   colours would shuffle the moment the model set changed. Horizontal because model
 *   names are long text.
 *
 * Every bar carries its own value, which makes the chart readable as a table and
 * removes the need for a separate one.
 *
 * The empty state is load-bearing: `/api/stats` degrades to
 * `{available: false, detail}` when Postgres and Redis are absent, and the chat
 * server has to run without the Phase-3 gateway stack up.
 */

import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { Stats } from "../api/frames";

function percent(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

function Tile({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "ok" | "warn" | "danger";
}) {
  // Status colour never travels alone: `sub` names the state in words, so the tile
  // still reads correctly in greyscale, in forced-colors mode, and to a screen
  // reader.
  const color =
    tone === "danger"
      ? "var(--danger)"
      : tone === "warn"
        ? "var(--warn)"
        : tone === "ok"
          ? "var(--ok)"
          : undefined;

  return (
    <div className="tile">
      <div className="eyebrow">{label}</div>
      <div className="tile-figure" style={color ? { color } : undefined}>
        {value}
      </div>
      {sub ? <div className="tile-sub">{sub}</div> : null}
    </div>
  );
}

function CostByModel({ costs }: { costs: Record<string, number> }) {
  const rows = Object.entries(costs).sort((a, b) => b[1] - a[1]);
  const max = rows.length > 0 ? rows[0][1] : 0;
  const total = rows.reduce((sum, [, value]) => sum + value, 0);

  if (rows.length === 0) {
    return (
      <div className="tile wide">
        <div className="eyebrow">spend by model</div>
        <p className="empty" style={{ padding: "0.6rem 0 0" }}>
          No requests have been metered yet. Spend appears here per model as soon
          as the ledger records one.
        </p>
      </div>
    );
  }

  return (
    <div className="tile wide">
      <div className="eyebrow">spend by model · ${total.toFixed(4)} total</div>
      <div className="bars">
        {rows.map(([model, cost]) => (
          <div className="bar-row" key={model}>
            <span className="bar-label" title={model}>
              {model}
            </span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: max > 0 ? `${(cost / max) * 100}%` : "0%",
                  background: "var(--metric)",
                }}
              />
            </div>
            <span className="bar-value">${cost.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const load = () => {
      api
        .stats()
        .then((next) => live && setStats(next))
        .catch((exc: unknown) => {
          if (live) setFailed(exc instanceof Error ? exc.message : String(exc));
        });
    };
    load();
    const timer = window.setInterval(load, 5000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, []);

  if (failed) {
    return (
      <div className="dash">
        <div className="tile wide">
          <div className="eyebrow">metrics unavailable</div>
          <p className="empty" style={{ padding: "0.6rem 0 0" }}>
            The server did not answer <code>/api/stats</code>: {failed}. Start
            FORGE with <code>FORGE_SERVER_TOKEN</code> set and reload.
          </p>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="dash">
        <div className="tile wide">
          <div className="eyebrow">loading</div>
        </div>
      </div>
    );
  }

  if (!stats.available) {
    return (
      <div className="dash">
        <div className="tile wide">
          <div className="eyebrow">ledger offline</div>
          <p className="empty" style={{ padding: "0.6rem 0 0", maxWidth: "56ch" }}>
            Metrics come from the gateway's Postgres ledger and Redis counters, and
            neither is reachable: {stats.detail}. Chat keeps working without them —
            start the gateway stack to see spend, latency, and cache hits here.
          </p>
        </div>
      </div>
    );
  }

  const errorTone =
    stats.error_rate >= 0.05
      ? "danger"
      : stats.error_rate > 0
        ? "warn"
        : "ok";

  return (
    <div className="dash">
      <Tile
        label="requests"
        value={stats.requests_total.toLocaleString()}
        sub="since the ledger was created"
      />
      <Tile
        label="error rate"
        value={percent(stats.error_rate)}
        tone={errorTone}
        sub={
          errorTone === "danger"
            ? "elevated"
            : errorTone === "warn"
              ? "some failures"
              : "no failures"
        }
      />
      <Tile
        label="p95 latency"
        value={`${Math.round(stats.p95_latency_ms)} ms`}
        sub="successful requests only"
      />
      <Tile
        label="cache hits"
        value={percent(stats.cache_hit_rate)}
        sub="exact and semantic combined"
      />
      <CostByModel costs={stats.cost_by_model} />
    </div>
  );
}
