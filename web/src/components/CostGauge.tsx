/**
 * The signature element: a burn-down gauge, not a cost readout.
 *
 * Every FORGE run is bounded by `max_cost_usd` (invariant 6) — the agent stops
 * itself and emits `terminal(max_cost)`. So this is a fuel gauge: it shows how much
 * of the current run's budget is left, and when a run ends on cost, it is the
 * explanation already on screen.
 *
 * It measures the *run*, not the session, because that is what the bound applies
 * to. The session total sits underneath in smaller type, since it is what you are
 * actually spending.
 */

import { money } from "../theme";

export function CostGauge({
  runCost,
  sessionCost,
  budget,
}: {
  runCost: number;
  sessionCost: number;
  budget: number;
}) {
  const used = budget > 0 ? Math.min(runCost / budget, 1) : 0;
  const remaining = Math.max(0, 1 - used);
  const level = used >= 1 ? "over" : used >= 0.75 ? "warn" : "";

  return (
    <div
      className="gauge"
      role="meter"
      aria-valuemin={0}
      aria-valuemax={budget}
      aria-valuenow={runCost}
      aria-label={`run budget: ${money(runCost)} of ${money(budget)} spent`}
    >
      <span className="gauge-cap" aria-hidden="true">
        {Math.round(remaining * 100)}%
      </span>
      <div className="gauge-track">
        {/* The fill percentage rides a custom property rather than an inline
            `height`, because the gauge turns horizontal below 860px and an inline
            dimension cannot be flipped to the other axis by a media query. */}
        <div
          className={`gauge-fill ${level}`}
          style={{ "--fill": `${remaining * 100}%` } as React.CSSProperties}
        />
      </div>
      {/* Value only, no word: the rail is 2.5rem wide and this text is rotated,
          so its length is bounded by the viewport height. The label lives in the
          `aria-label` and the `title` instead of overflowing the track. */}
      <span
        className="gauge-figure"
        title={`${money(sessionCost)} spent across this session`}
      >
        {money(sessionCost)}
      </span>
    </div>
  );
}
