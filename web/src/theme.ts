/**
 * Where colour is allowed.
 *
 * Colour is scarce here on purpose. It marks two things and nothing else: a
 * moment where permission was decided, and the outcome of a run. Everything else
 * — status, prose, tool calls, results, cost — is monochrome. That is what lets a
 * danger flag be visible at a glance in a hundred-row transcript instead of
 * competing with decoration.
 *
 * `tool_call` and `tool_result` stay monochrome even though they name a tool,
 * because `ToolCallEvent` carries no `kind` (`events.py`). Deriving a hue from the
 * tool's *name* would be asserting a fact the wire format does not carry, and it
 * would be wrong the first time someone registers a tool whose name reads like a
 * different kind.
 */

import type { Frame, TerminalReasonName, ToolKindName } from "./api/frames";

export const KIND_COLOR: Record<ToolKindName, string> = {
  read: "var(--read)",
  write: "var(--write)",
  execute: "var(--execute)",
  agent: "var(--agent)",
};

const TERMINAL_COLOR: Record<TerminalReasonName, string> = {
  completed: "var(--ok)",
  max_iterations: "var(--warn)",
  max_cost: "var(--warn)",
  loop_detected: "var(--execute)",
  error: "var(--danger)",
};

/** The hue for a frame's spine mark, or null to leave it monochrome. */
export function accentFor(frame: Frame): string | null {
  switch (frame.type) {
    case "confirm":
      return frame.danger_reasons.length > 0
        ? "var(--danger)"
        : KIND_COLOR[frame.kind];
    case "approval_decision":
      if (frame.danger_reasons.length > 0) return "var(--danger)";
      return frame.approved ? KIND_COLOR[frame.kind] : "var(--muted-2)";
    case "confirm_request":
      return "var(--warn)";
    case "terminal":
      return TERMINAL_COLOR[frame.reason];
    case "overflow":
    case "error":
      return "var(--danger)";
    case "subagent":
      return "var(--agent)";
    default:
      return null;
  }
}

/** Short uppercase label for the spine gutter. Data, not decoration. */
export function labelFor(frame: Frame): string {
  switch (frame.type) {
    case "status":
      return "···";
    case "text":
    case "text_delta":
      return "SAY";
    case "tool_call":
      return "RUN";
    case "tool_result":
      return "OUT";
    case "subagent":
      return "SUB";
    case "cost":
      return "$";
    case "approval_decision":
      return frame.approved ? "OK" : "NO";
    case "confirm_request":
    case "confirm":
      return "ASK";
    case "terminal":
      return "END";
    case "ready":
      return "···";
    case "overflow":
    case "error":
      return "!";
    default:
      return "";
  }
}

export function money(usd: number): string {
  return `$${usd.toFixed(4)}`;
}
