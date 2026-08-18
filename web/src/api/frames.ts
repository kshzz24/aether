/**
 * The wire contract, mirroring `server/wire.py`.
 *
 * Ten frames come from the `Event` union via `wire.encode`; four are minted by
 * the presentation layer itself and have no encoder arm. Every frame is
 * discriminated on `type`, so `switch` over `Frame` with `assertNever` in the
 * default makes `tsc` refuse to compile when a variant is added and not handled.
 *
 * That makes this file the fifth gate on the `Event` union, after
 * `tests/conftest.py:sample_events`, `cli/renderer.py`, `tui/transcript.py`, and
 * `server/wire.py`. The mock fixture is generated from the real encoder
 * (`scripts/dump_frames.py`), so a mismatch between these types and the server
 * shows up the moment you regenerate rather than in production.
 *
 * Enums arrive as `.name.lower()`, never `.value` — `ToolKind`, `Verdict` and
 * `TerminalReason` are all `Enum(auto())` in Python, so their values are
 * positional integers and reordering the enum would silently rewrite the wire
 * format. See the note in `server/wire.py:encode`.
 */

export type ToolKindName = "read" | "write" | "execute" | "agent";

export type VerdictName = "auto_approve" | "auto_deny" | "ask";

export type TerminalReasonName =
  | "completed"
  | "max_iterations"
  | "max_cost"
  | "loop_detected"
  | "error";

/** Anything that occupies a numbered position in the transcript. */
interface Sequenced {
  seq: number;
}

// --- the ten Event encodings --------------------------------------------------

export interface StatusFrame extends Sequenced {
  type: "status";
  message: string;
}

export interface TextDeltaFrame extends Sequenced {
  type: "text_delta";
  text: string;
}

export interface TextFrame extends Sequenced {
  type: "text";
  text: string;
}

export interface ToolCallFrame extends Sequenced {
  type: "tool_call";
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultFrame extends Sequenced {
  type: "tool_result";
  name: string;
  /** Untruncated on purpose: trimming for display is the client's job. */
  result: string;
  /** Guardrail reasons (secret redaction, injection warning), if any fired. */
  flags: string[];
}

export interface SubagentFrame extends Sequenced {
  type: "subagent";
  task: string;
  phase: string;
  detail: string;
}

export interface CostFrame extends Sequenced {
  type: "cost";
  /** This call's delta. */
  cost_usd: number;
  /** Cumulative *within the current run* — the agent resets it per run. */
  total_cost_usd: number;
  input_tokens: number;
  output_tokens: number;
}

export interface ApprovalDecisionFrame extends Sequenced {
  type: "approval_decision";
  tool_name: string;
  kind: ToolKindName;
  danger_reasons: string[];
  verdict: VerdictName;
  approved: boolean;
  source: string;
}

/**
 * The agent's own notification that it is about to ask. It carries no
 * `request_id` — the correlated question arrives as a `ConfirmFrame` from
 * `ServerApprover`, always immediately after this one.
 */
export interface ConfirmRequestFrame extends Sequenced {
  type: "confirm_request";
  tool_name: string;
  arguments: Record<string, unknown>;
  reason: string;
}

export interface TerminalFrame extends Sequenced {
  type: "terminal";
  reason: TerminalReasonName;
  detail: string;
}

// --- the four control frames --------------------------------------------------

/** Sent first on every stream, so `onopen` fires and the subscription is proven. */
export interface ReadyFrame {
  type: "ready";
}

/**
 * The answerable question. Carries a `seq` — unlike `overflow` — because a
 * browser reconnecting mid-confirm has to be re-shown what it is being asked.
 * `offers_always` is computed server-side: withholding "always" from a
 * danger-flagged call is a security rule, and a security rule enforced only here
 * is not enforced.
 */
export interface ConfirmFrame extends Sequenced {
  type: "confirm";
  request_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  kind: ToolKindName;
  danger_reasons: string[];
  diff: string | null;
  offers_always: boolean;
}

/**
 * This connection fell too far behind and was dropped. No `seq`: it is not a
 * transcript position, so reconnecting on it would resume from nowhere. Recover
 * by reconnecting with the last real `seq` seen.
 */
export interface OverflowFrame {
  type: "overflow";
}

/** The stream itself failed, as opposed to the run failing (`terminal`). */
export interface ErrorFrame {
  type: "error";
  detail: string;
}

// --- unions -------------------------------------------------------------------

export type EventFrame =
  | StatusFrame
  | TextDeltaFrame
  | TextFrame
  | ToolCallFrame
  | ToolResultFrame
  | SubagentFrame
  | CostFrame
  | ApprovalDecisionFrame
  | ConfirmRequestFrame
  | TerminalFrame;

export type ControlFrame =
  | ReadyFrame
  | ConfirmFrame
  | OverflowFrame
  | ErrorFrame;

export type Frame = EventFrame | ControlFrame;

/**
 * Compile-time exhaustiveness. Reachable at runtime only if a server sends a
 * `type` this build has never heard of, so it degrades rather than throwing.
 */
export function assertNever(frame: never): null {
  console.warn("unhandled frame type", frame);
  return null;
}

export function isSequenced(frame: Frame): frame is Frame & Sequenced {
  return "seq" in frame && typeof (frame as Sequenced).seq === "number";
}

// --- REST payloads ------------------------------------------------------------

/** `persistence.SessionMeta`, as `GET /api/sessions` returns it. */
export interface SessionMeta {
  id: string;
  goal: string;
  updated_at: string;
  total_cost: number;
  turns: number;
}

/** `tracing/store.py:trace_summary`, as `GET /api/traces` lists them. */
export interface TraceSummary {
  trace_id: string;
  total_cost_usd: number;
  span_count: number;
  duration_sec: number;
}

/** `gateway/metrics.py:compute_stats`, plus the degraded shape. */
export type Stats =
  | { available: false; detail: string }
  | {
      available: true;
      requests_total: number;
      error_rate: number;
      p95_latency_ms: number;
      cost_by_model: Record<string, number>;
      cache_hit_rate: number;
    };
