/**
 * Live and resumable sessions, newest first — the same union `SessionManager.list`
 * returns (in-memory ∪ on-disk, live winning on id).
 *
 * `SessionMeta` deliberately omits message bodies, so listing two hundred sessions
 * does not read two hundred transcripts. Everything shown here comes from that
 * header.
 */

import type { SessionMeta } from "../api/frames";
import { money } from "../theme";

function when(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function SessionList({
  sessions,
  currentId,
  offline,
  onPick,
  onCreate,
}: {
  sessions: SessionMeta[];
  currentId: string | null;
  offline: boolean;
  onPick(id: string): void;
  onCreate(): void;
}) {
  return (
    <aside className="aside" aria-label="sessions">
      <div className="aside-head">
        <span className="eyebrow">sessions</span>
        <button className="btn" onClick={onCreate} disabled={offline}>
          New
        </button>
      </div>

      {sessions.length === 0 ? (
        <p className="empty">
          {offline
            ? "No server connected. The mock transport replays a captured run so the interface can be reviewed without one."
            : "No sessions yet. Start one and it appears here, along with any you can resume from disk."}
        </p>
      ) : (
        sessions.map((session) => (
          <button
            key={session.id}
            className="session"
            aria-current={session.id === currentId}
            onClick={() => onPick(session.id)}
          >
            <span className="session-goal">{session.goal || "untitled"}</span>
            <span className="session-meta">
              {when(session.updated_at)} · {money(session.total_cost)} ·{" "}
              {session.turns} turns
            </span>
          </button>
        ))
      )}
    </aside>
  );
}
