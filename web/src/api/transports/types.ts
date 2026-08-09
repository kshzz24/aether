import type { Frame } from "../frames";

export type TransportName = "mock" | "sse" | "ws";

export type ConnectionState =
  | "connecting"
  | "open"
  /** The server dropped us for falling behind; reconnect from the last seq. */
  | "dropped"
  | "closed";

export interface TransportHandlers {
  onFrame(frame: Frame): void;
  onState(state: ConnectionState): void;
}

/**
 * One shape, three implementations. `afterSeq` is -1 for "send me everything";
 * on a reconnect it is the last `seq` the client actually holds, and the server
 * replays only the tail — which is what makes being dropped for slowness
 * recoverable rather than lossy.
 *
 * Returns its own teardown. Any logic beyond framing and socket handling belongs
 * in `session.py`, not here: the reviewer's check from §10.3 is whether you could
 * delete `ws.ts` and lose nothing but the WS option.
 */
export type Transport = (
  sessionId: string,
  afterSeq: number,
  handlers: TransportHandlers,
) => () => void;
