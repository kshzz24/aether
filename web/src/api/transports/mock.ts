/**
 * Replays a fixture generated from the real encoder — no server needed.
 *
 * `scripts/dump_frames.py` runs `sample_events()` through `server/wire.py`, so
 * this is the actual wire format rather than an approximation of it. That is what
 * lets the UI be built and reviewed before stage 4 exists, and what makes a
 * contract mismatch impossible by construction: regenerate the fixture and `tsc`
 * fails if a frame grew a field the types do not know about.
 */

import fixture from "../../mocks/frames.json";
import type { Frame } from "../frames";
import type { Transport } from "./types";

const FRAMES = fixture as unknown as Frame[];

/** Paced so you can watch the transcript build, not so fast it just appears. */
const INTERVAL_MS = 260;

export const mockTransport: Transport = (_sessionId, afterSeq, handlers) => {
  let cancelled = false;
  const timers: number[] = [];

  const pending = FRAMES.filter(
    (frame) => !("seq" in frame) || (frame as { seq: number }).seq > afterSeq,
  );

  handlers.onState("connecting");

  // A microtask before the first frame, so a caller that subscribes during render
  // does not receive frames before its state setters are wired up.
  timers.push(
    window.setTimeout(() => {
      if (cancelled) return;
      handlers.onState("open");

      pending.forEach((frame, index) => {
        timers.push(
          window.setTimeout(() => {
            if (cancelled) return;
            handlers.onFrame(frame);
            if (index === pending.length - 1) handlers.onState("closed");
          }, index * INTERVAL_MS),
        );
      });
    }, 0),
  );

  return () => {
    cancelled = true;
    timers.forEach(window.clearTimeout);
  };
};
