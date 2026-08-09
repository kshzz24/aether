/**
 * Turn a frame log into rows to render. Pure, so it is the easy thing to reason
 * about when the transcript looks wrong.
 *
 * The one transformation: consecutive `text_delta` frames collapse into a single
 * streaming prose row, and the authoritative `text` frame that follows replaces
 * it rather than appending. The agent emits both — deltas so a surface can stream,
 * then `text` as the final word (`agent.py:118-136`) — so a client that renders
 * each frame independently prints every answer twice.
 *
 * `cli/renderer.py` solves this by dropping deltas entirely, which is right for a
 * terminal that only needs the finished answer. A browser streams, so it keeps the
 * deltas and folds them.
 */

import type { Frame, TextFrame } from "./frames";
import { isSequenced } from "./frames";

export interface Row {
  key: string;
  /** null for control frames, which hold no transcript position. */
  seq: number | null;
  frame: Frame;
  /** True while prose is still arriving, so the caret can show. */
  streaming: boolean;
}

export function buildRows(frames: Frame[]): Row[] {
  const rows: Row[] = [];
  let open: { row: Row; text: string } | null = null;

  frames.forEach((frame, index) => {
    const seq = isSequenced(frame) ? frame.seq : null;
    const key = seq === null ? `c${index}` : `s${seq}`;

    if (frame.type === "text_delta") {
      if (open) {
        open.text += frame.text;
        (open.row.frame as TextFrame).text = open.text;
        return;
      }
      const row: Row = {
        key,
        seq,
        frame: { type: "text", text: frame.text, seq: seq ?? 0 },
        streaming: true,
      };
      open = { row, text: frame.text };
      rows.push(row);
      return;
    }

    if (frame.type === "text" && open) {
      // The authoritative version. Replace, never append.
      open.row.frame = frame;
      open.row.streaming = false;
      open = null;
      return;
    }

    open = null;
    rows.push({ key, seq, frame, streaming: false });
  });

  return rows;
}

/** Session cost: the sum of per-call deltas.
 *
 * `total_cost_usd` on a `CostFrame` is cumulative *within one run* — the agent
 * resets it every call (`agent.py:98`) because `max_cost_usd` is a per-run bound.
 * Summing that field across turns would double-count; summing `cost_usd` is the
 * same arithmetic `AgentSession` does server-side.
 */
export function sessionCost(frames: Frame[]): number {
  return frames.reduce(
    (total, frame) => (frame.type === "cost" ? total + frame.cost_usd : total),
    0,
  );
}

/** The cost of the current run, straight off the newest `CostFrame`. */
export function runCost(frames: Frame[]): number {
  for (let i = frames.length - 1; i >= 0; i -= 1) {
    const frame = frames[i];
    if (frame.type === "terminal") break;
    if (frame.type === "cost") return frame.total_cost_usd;
  }
  return 0;
}

export function lastSeq(frames: Frame[]): number {
  for (let i = frames.length - 1; i >= 0; i -= 1) {
    const frame = frames[i];
    if (isSequenced(frame)) return frame.seq;
  }
  return -1;
}
