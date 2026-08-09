/**
 * REST calls, matching §5 of the phase-8 plan.
 *
 * The token is a single shared bearer secret (§4.7) — this build has no per-user
 * identity, so anyone holding it sees every session. It is sent as a header
 * wherever headers are possible, and as `?token=` only for SSE and WS, because
 * `EventSource` cannot set headers. That puts a secret in URLs and access logs;
 * the debt is named in the plan as TODO(phase-17), not papered over.
 */

import type { SessionMeta, Stats } from "./frames";

const TOKEN_KEY = "forge.token";

export function getToken(): string {
  const fromUrl = new URLSearchParams(window.location.search).get("token");
  if (fromUrl) {
    window.localStorage.setItem(TOKEN_KEY, fromUrl);
    return fromUrl;
  }
  return window.localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

/** For SSE and WS only. Everything else uses the Authorization header. */
export function withToken(path: string, extra?: Record<string, string>): string {
  const url = new URL(path, window.location.origin);
  url.searchParams.set("token", getToken());
  for (const [key, value] of Object.entries(extra ?? {})) {
    url.searchParams.set(key, value);
  }
  return url.toString();
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function call<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (!response.ok) {
    // Surface the server's own wording. A 409 from the decisions route means the
    // question was already answered or has expired, and the UI says exactly that
    // rather than inventing a friendlier lie.
    const detail = await response.text();
    throw new ApiError(response.status, detail || response.statusText);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export interface CreateSessionBody {
  goal?: string;
  resume?: string;
  provider?: string;
  model?: string;
  approval_mode?: string;
}

export const api = {
  createSession: (body: CreateSessionBody = {}) =>
    call<{ session_id: string }>("POST", "/api/sessions", body),

  listSessions: () => call<SessionMeta[]>("GET", "/api/sessions"),

  sendGoal: (id: string, text: string) =>
    call<void>("POST", `/api/sessions/${id}/goal`, { text }),

  answerConfirm: (
    id: string,
    body: {
      request_id: string;
      approved: boolean;
      reason?: string;
      arguments?: Record<string, unknown>;
      remember?: boolean;
    },
  ) => call<void>("POST", `/api/sessions/${id}/decisions`, body),

  interrupt: (id: string) =>
    call<void>("POST", `/api/sessions/${id}/interrupt`),

  deleteSession: (id: string) =>
    call<void>("DELETE", `/api/sessions/${id}`),

  stats: () => call<Stats>("GET", "/api/stats"),
};
