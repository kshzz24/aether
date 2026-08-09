/**
 * One hook, three transports. The transport toggle in the header is what makes
 * having built both SSE and WS worth its cost: you can A/B reconnect behaviour
 * and confirm round-trips inside a single interaction.
 *
 * The frame log is kept verbatim — every frame the server sent, in arrival order.
 * All shaping (folding text deltas, summing cost) is a pure function over that log
 * (`rows.ts`), so when the transcript looks wrong there is one place to look and it
 * has no timing in it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "./client";
import type { ConfirmFrame, Frame } from "./frames";
import { buildRows, lastSeq, runCost, sessionCost } from "./rows";
import { mockTransport } from "./transports/mock";
import { sseTransport } from "./transports/sse";
import type { ConnectionState, TransportName } from "./transports/types";
import { wsSend, wsTransport } from "./transports/ws";

const TRANSPORTS = {
  mock: mockTransport,
  sse: sseTransport,
  ws: wsTransport,
} as const;

export interface Answer {
  approved: boolean;
  reason?: string;
  arguments?: Record<string, unknown>;
  remember?: boolean;
}

export function useForgeSession(
  sessionId: string | null,
  transport: TransportName,
) {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [state, setState] = useState<ConnectionState>("closed");
  const [answered, setAnswered] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  // The resume offset, held in a ref so reconnecting does not re-run the effect
  // and restart the stream from the beginning.
  const resumeFrom = useRef(-1);

  // A fresh session means a fresh transcript. Without this, switching sessions
  // shows the previous one's frames until the first new frame lands.
  useEffect(() => {
    setFrames([]);
    setAnswered(new Set());
    resumeFrom.current = -1;
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    const connect = TRANSPORTS[transport];

    return connect(sessionId, resumeFrom.current, {
      onFrame: (frame) => {
        setFrames((previous) => [...previous, frame]);
        if ("seq" in frame) resumeFrom.current = frame.seq;
      },
      onState: setState,
    });
  }, [sessionId, transport]);

  const rows = useMemo(() => buildRows(frames), [frames]);

  /** The newest confirm nobody has answered yet.
   *
   * Scanning newest-first rather than tracking a single value, because replay
   * re-delivers confirms that were answered before a reconnect — the server 409s
   * those, so the client is responsible for not re-offering them. */
  const pendingConfirm = useMemo<ConfirmFrame | null>(() => {
    for (let i = frames.length - 1; i >= 0; i -= 1) {
      const frame = frames[i];
      if (frame.type === "terminal") return null;
      if (frame.type === "confirm" && !answered.has(frame.request_id)) {
        return frame;
      }
    }
    return null;
  }, [frames, answered]);

  const running = useMemo(() => {
    for (let i = frames.length - 1; i >= 0; i -= 1) {
      const type = frames[i].type;
      if (type === "terminal") return false;
      if (type === "status" || type === "confirm") return true;
    }
    return false;
  }, [frames]);

  const send = useCallback(
    async (text: string) => {
      if (!sessionId) return;
      setError(null);
      // Over WS the goal rides the open socket; over SSE there is no inbound
      // channel, so it is an ordinary POST. Same server-side path either way.
      if (transport === "ws" && wsSend({ kind: "goal", text })) return;
      if (transport === "mock") return;
      try {
        await api.sendGoal(sessionId, text);
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    },
    [sessionId, transport],
  );

  const answer = useCallback(
    async (requestId: string, decision: Answer) => {
      if (!sessionId) return;
      // Optimistic: close the modal now. A 409 means the question expired or was
      // already answered, which is exactly the case where re-showing it would be
      // wrong, so the dismissal stands and the error is reported instead.
      setAnswered((previous) => new Set(previous).add(requestId));
      setError(null);

      const payload = { request_id: requestId, ...decision };
      if (transport === "ws" && wsSend({ kind: "decision", ...payload })) return;
      if (transport === "mock") return;
      try {
        await api.answerConfirm(sessionId, payload);
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    },
    [sessionId, transport],
  );

  const interrupt = useCallback(async () => {
    if (!sessionId) return;
    if (transport === "ws" && wsSend({ kind: "interrupt" })) return;
    if (transport === "mock") return;
    try {
      await api.interrupt(sessionId);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, [sessionId, transport]);

  return {
    rows,
    frames,
    state,
    running,
    pendingConfirm,
    error,
    clearError: useCallback(() => setError(null), []),
    sessionCost: sessionCost(frames),
    runCost: runCost(frames),
    lastSeq: lastSeq(frames),
    send,
    answer,
    interrupt,
  };
}
