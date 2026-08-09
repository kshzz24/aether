/**
 * WebSocket. The same frames out; framing and socket handling only.
 *
 * The reviewer's check from §10.3 is whether you could delete this file and lose
 * nothing but the WS option. You can: it holds no session logic, and the frames it
 * receives are byte-identical to SSE's because both read the same
 * `AgentSession.Subscriber`.
 *
 * Inbound frames use `kind` where outbound use `type`, so a frame's direction is
 * obvious from its shape. Sending them is the caller's job (`useForgeSession`);
 * this exposes the socket through `wsSend` for that.
 */

import type { Frame } from "../frames";
import type { Transport } from "./types";

/** The live socket, so the hook can send `goal` / `decision` / `interrupt`. */
let socket: WebSocket | null = null;

export type ClientFrame =
  | { kind: "goal"; text: string }
  | {
      kind: "decision";
      request_id: string;
      approved: boolean;
      reason?: string;
      arguments?: Record<string, unknown>;
      remember?: boolean;
    }
  | { kind: "interrupt" };

/** True when the frame went out. False means fall back to the REST route. */
export function wsSend(frame: ClientFrame): boolean {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(frame));
  return true;
}

export const wsTransport: Transport = (sessionId, afterSeq, handlers) => {
  const url = new URL(
    `/ws/sessions/${sessionId}`,
    window.location.origin.replace(/^http/, "ws"),
  );
  url.searchParams.set(
    "token",
    window.localStorage.getItem("forge.token") ?? "",
  );
  if (afterSeq >= 0) url.searchParams.set("after_seq", String(afterSeq));

  handlers.onState("connecting");
  const ws = new WebSocket(url.toString());
  socket = ws;

  ws.onopen = () => handlers.onState("open");

  ws.onmessage = (event: MessageEvent<string>) => {
    let frame: Frame;
    try {
      frame = JSON.parse(event.data) as Frame;
    } catch {
      handlers.onFrame({ type: "error", detail: "received a malformed frame" });
      return;
    }
    if (frame.type === "overflow") handlers.onState("dropped");
    handlers.onFrame(frame);
  };

  ws.onerror = () =>
    handlers.onFrame({ type: "error", detail: "websocket error" });

  ws.onclose = () => handlers.onState("closed");

  return () => {
    if (socket === ws) socket = null;
    ws.close();
  };
};
