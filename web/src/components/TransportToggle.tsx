/**
 * Switch the event stream between SSE, WebSocket, and the offline fixture.
 *
 * This control is what makes having built both transports worth its cost: the same
 * run, the same frames, and you can A/B reconnect behaviour and confirm latency
 * inside one interaction. `mock` replays a fixture generated from the real encoder,
 * so the interface can be exercised with no server at all.
 */

import type { TransportName } from "../api/transports/types";

const OPTIONS: { value: TransportName; hint: string }[] = [
  { value: "sse", hint: "server-sent events, POST for input" },
  { value: "ws", hint: "duplex websocket" },
  { value: "mock", hint: "offline: replays a captured run" },
];

export function TransportToggle({
  value,
  onChange,
}: {
  value: TransportName;
  onChange(next: TransportName): void;
}) {
  return (
    <div className="toggle" role="group" aria-label="event stream transport">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          className="toggle-option"
          aria-pressed={value === option.value}
          title={option.hint}
          onClick={() => onChange(option.value)}
        >
          {option.value}
        </button>
      ))}
    </div>
  );
}

export function ConnectionDot({ state }: { state: string }) {
  const label =
    state === "dropped" ? "dropped — reconnect to replay" : state;
  return (
    <span className="state">
      <span className={`state-dot ${state}`} aria-hidden="true" />
      {label}
    </span>
  );
}
