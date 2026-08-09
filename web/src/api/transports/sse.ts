/**
 * Server-Sent Events. Framing and socket handling only.
 *
 * The server writes `id: {seq}\ndata: {json}\n\n`, and the `id:` line is the whole
 * reconnect mechanism: the browser stores it and sends it back as the
 * `Last-Event-ID` header on its own automatic reconnect, with no application code
 * involved. Control frames without a `seq` (`ready`, `overflow`) are written
 * without an `id:` line, so the browser never stores an offset that indexes
 * nothing.
 *
 * `EventSource` cannot set headers, so the token rides in the query string (§4.7,
 * TODO(phase-17)). `afterSeq` is passed explicitly for a *fresh* mount that
 * already holds part of the transcript; the browser's own retry handles the rest.
 */

import { withToken } from "../client";
import type { Frame } from "../frames";
import type { Transport } from "./types";

export const sseTransport: Transport = (sessionId, afterSeq, handlers) => {
  const url = withToken(
    `/api/sessions/${sessionId}/events`,
    afterSeq >= 0 ? { after_seq: String(afterSeq) } : undefined,
  );

  handlers.onState("connecting");
  const source = new EventSource(url);

  source.onopen = () => handlers.onState("open");

  source.onmessage = (event: MessageEvent<string>) => {
    let frame: Frame;
    try {
      frame = JSON.parse(event.data) as Frame;
    } catch {
      handlers.onFrame({
        type: "error",
        detail: "received a malformed frame",
      });
      return;
    }
    if (frame.type === "overflow") handlers.onState("dropped");
    handlers.onFrame(frame);
  };

  // EventSource reports every failure as an opaque `error` and retries by itself.
  // Report the state and let it; closing here would defeat the built-in recovery
  // that `Last-Event-ID` exists to serve.
  source.onerror = () => {
    handlers.onState(
      source.readyState === EventSource.CLOSED ? "closed" : "connecting",
    );
  };

  return () => {
    source.close();
    handlers.onState("closed");
  };
};
