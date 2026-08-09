/**
 * One renderer per frame type, and the exhaustiveness gate.
 *
 * `renderFrame` switches over the full `Frame` union and ends in `assertNever`.
 * Adding a member to the `Event` union in `events.py` therefore breaks `tsc` here
 * until a branch exists — the same discipline `wire.py`'s `assert_never`,
 * `cli/renderer.py`'s match, and `tui/transcript.py` already enforce on the Python
 * side. Five surfaces, one gate each.
 *
 * The visual rule: the model's voice is unboxed serif prose; everything the
 * machine did is a bordered mono card. That is the whole hierarchy, and it means a
 * reader can find the answer in a long trace without reading the trace.
 */

import { Fragment } from "react";

import type { Frame } from "../api/frames";
import { assertNever } from "../api/frames";
import { KIND_COLOR, money } from "../theme";

function Args({ args }: { args: Record<string, unknown> }) {
  const entries = Object.entries(args);
  if (entries.length === 0) {
    return <div className="args-key">no arguments</div>;
  }
  // A keyed Fragment, not `<>`: the two spans are siblings in the parent's grid,
  // so they cannot be wrapped in an element without breaking the two-column
  // layout — and the shorthand syntax cannot take a key.
  return (
    <div className="args">
      {entries.map(([key, value]) => (
        <Fragment key={key}>
          <span className="args-key">{key}</span>
          <span className="args-val">
            {typeof value === "string" ? value : JSON.stringify(value)}
          </span>
        </Fragment>
      ))}
    </div>
  );
}

export function DangerFlags({ reasons }: { reasons: string[] }) {
  if (reasons.length === 0) return null;
  return (
    <ul className="dangers">
      {reasons.map((reason) => (
        <li className="danger-flag" key={reason}>
          {reason}
        </li>
      ))}
    </ul>
  );
}

function Card({
  name,
  tag,
  tagColor,
  flagged,
  children,
}: {
  name: string;
  tag?: string;
  tagColor?: string;
  flagged?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={flagged ? "card flagged" : "card"}>
      <div className="card-head">
        <span className="card-name">{name}</span>
        {tag ? (
          <span className="pill" style={{ color: tagColor ?? "var(--muted)" }}>
            {tag}
          </span>
        ) : null}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export function renderFrame(frame: Frame, streaming: boolean) {
  switch (frame.type) {
    case "ready":
      return <div className="status">stream open</div>;

    case "status":
      return <div className="status">{frame.message}</div>;

    // Deltas are folded into a synthetic `text` row upstream (`rows.ts`), so this
    // arm only ever sees a fully-formed one. The case stays for exhaustiveness.
    case "text_delta":
      return <div className="prose">{frame.text}</div>;

    case "text":
      return (
        <div className="prose">
          {frame.text}
          {streaming ? <span className="caret">&nbsp;</span> : null}
        </div>
      );

    case "tool_call":
      return (
        <Card name={frame.name} tag="call">
          <Args args={frame.arguments} />
        </Card>
      );

    case "tool_result": {
      const denied = frame.result.startsWith("DENIED:");
      const errored = frame.result.startsWith("ERROR:");
      return (
        <Card
          name={frame.name}
          tag={denied ? "denied" : errored ? "error" : "result"}
          tagColor={denied || errored ? "var(--danger)" : undefined}
          flagged={denied || errored}
        >
          <pre className={denied || errored ? "result denied" : "result"}>
            {frame.result}
          </pre>
        </Card>
      );
    }

    case "subagent":
      return (
        <Card name="subagent" tag={frame.phase} tagColor="var(--agent)">
          <div className="args-val">{frame.task}</div>
          {frame.detail ? <div className="args-key">{frame.detail}</div> : null}
        </Card>
      );

    case "cost":
      return (
        <div className="costline">
          {money(frame.cost_usd)} this call · {money(frame.total_cost_usd)} this
          run
        </div>
      );

    case "approval_decision":
      return (
        <Card
          name={frame.tool_name}
          tag={`${frame.approved ? "allowed" : "blocked"} by ${frame.source}`}
          tagColor={
            frame.approved ? KIND_COLOR[frame.kind] : "var(--danger)"
          }
          flagged={!frame.approved || frame.danger_reasons.length > 0}
        >
          <DangerFlags reasons={frame.danger_reasons} />
          <div className="args-key">
            {frame.kind} · verdict {frame.verdict.replace("_", " ")}
          </div>
        </Card>
      );

    // The agent's own heads-up, published just before it parks on the approver.
    // The answerable question is the `confirm` frame that follows.
    case "confirm_request":
      return <div className="status">waiting on you — {frame.reason}</div>;

    case "confirm":
      return (
        <Card
          name={frame.tool_name}
          tag="asked"
          tagColor={
            frame.danger_reasons.length > 0
              ? "var(--danger)"
              : KIND_COLOR[frame.kind]
          }
          flagged={frame.danger_reasons.length > 0}
        >
          <DangerFlags reasons={frame.danger_reasons} />
          <Args args={frame.arguments} />
        </Card>
      );

    case "terminal":
      return (
        <div className="status">
          run ended — {frame.reason.replace(/_/g, " ")}
          {frame.detail ? ` · ${frame.detail}` : ""}
        </div>
      );

    // The connection fell behind and was cut. Say what to do about it, because the
    // recovery is real: reconnecting replays the tail from the last seq held.
    case "overflow":
      return (
        <div className="banner">
          this connection fell behind and was dropped · reconnect to replay what
          it missed
        </div>
      );

    case "error":
      return <div className="banner">{frame.detail}</div>;

    default:
      return assertNever(frame);
  }
}
