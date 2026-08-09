/**
 * The shell: session list, transcript, composer, and the confirm modal over the top.
 *
 * Two views (`chat`, `metrics`) is a `useState`, not a router — the plan says so and
 * a routing dependency for one boolean would be the wrong trade.
 *
 * Session bootstrap is deliberately tolerant. When no server is reachable — which is
 * every run before stage 4 lands — the app falls back to the `mock` transport with a
 * synthetic id and stays fully usable, because the fixture it replays is real
 * encoder output. The interface is reviewable before the HTTP layer exists.
 */

import { useCallback, useEffect, useState } from "react";

import { api } from "./api/client";
import type { SessionMeta } from "./api/frames";
import type { TransportName } from "./api/transports/types";
import { useForgeSession } from "./api/useForgeSession";
import { ConfirmModal } from "./components/ConfirmModal";
import { CostGauge } from "./components/CostGauge";
import { Dashboard } from "./components/Dashboard";
import { GoalInput } from "./components/GoalInput";
import { SessionList } from "./components/SessionList";
import { ConnectionDot, TransportToggle } from "./components/TransportToggle";
import { Transcript } from "./components/Transcript";

/** Mirrors the `max_cost_usd` default in `config.py`; the server will send the
 *  real figure once `/api/sessions` reports it. */
const BUDGET_USD = 5;

const OFFLINE_ID = "offline";

type View = "chat" | "metrics";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [transport, setTransport] = useState<TransportName>("mock");
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(OFFLINE_ID);
  const [offline, setOffline] = useState(true);

  const session = useForgeSession(currentId, transport);

  const refresh = useCallback(async () => {
    try {
      const listed = await api.listSessions();
      setSessions(listed);
      setOffline(false);
      return listed;
    } catch {
      // No server yet. Not an error state to shout about — it is the expected
      // condition until stage 4, and the mock transport covers it.
      setOffline(true);
      return [];
    }
  }, []);

  useEffect(() => {
    void refresh().then((listed) => {
      if (listed.length > 0) {
        setCurrentId(listed[0].id);
        setTransport("sse");
      }
    });
  }, [refresh]);

  const create = async () => {
    try {
      const { session_id } = await api.createSession();
      await refresh();
      setCurrentId(session_id);
      if (transport === "mock") setTransport("sse");
    } catch {
      setOffline(true);
    }
  };

  return (
    <div className="shell">
      <header className="brand">
        <span className="brand-mark">FORGE</span>
        <span className="brand-note">
          {currentId === OFFLINE_ID ? "no session" : currentId}
        </span>
      </header>

      <div className="topbar">
        <div className="toggle" role="group" aria-label="view">
          <button
            className="toggle-option"
            aria-pressed={view === "chat"}
            onClick={() => setView("chat")}
          >
            trace
          </button>
          <button
            className="toggle-option"
            aria-pressed={view === "metrics"}
            onClick={() => setView("metrics")}
          >
            metrics
          </button>
        </div>

        <span className="topbar-spacer" />

        {session.lastSeq >= 0 ? (
          <span className="state">seq {session.lastSeq}</span>
        ) : null}
        <TransportToggle value={transport} onChange={setTransport} />
        <ConnectionDot state={session.state} />
      </div>

      <SessionList
        sessions={sessions}
        currentId={currentId}
        offline={offline}
        onPick={setCurrentId}
        onCreate={create}
      />

      {view === "metrics" ? (
        <Dashboard />
      ) : (
        <main className="main">
          <CostGauge
            runCost={session.runCost}
            sessionCost={session.sessionCost}
            budget={BUDGET_USD}
          />
          <Transcript rows={session.rows} />
        </main>
      )}

      {view === "chat" ? (
        <GoalInput
          running={session.running}
          disabled={offline && transport !== "mock"}
          onSend={(text) => void session.send(text)}
          onInterrupt={() => void session.interrupt()}
        />
      ) : (
        <div className="composer">
          <div className="hint">
            Metrics refresh every 5 seconds from the gateway ledger.
          </div>
        </div>
      )}

      {/* A modal with no way out is a trap. It needs an explicit dismiss because
          the failure it reports is usually recoverable — a 409 means the confirm
          expired, and the right next move is to keep working. */}
      {session.error ? (
        <div className="scrim" role="alert">
          <div className="modal" style={{ width: "min(32rem, 100%)" }}>
            <div className="modal-head">
              <div className="eyebrow">that request failed</div>
            </div>
            <div className="modal-body">
              <div className="banner">{session.error}</div>
            </div>
            <div className="modal-foot">
              <span className="modal-foot-spacer" />
              <button className="btn primary" onClick={session.clearError}>
                Dismiss
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {session.pendingConfirm ? (
        <ConfirmModal
          confirm={session.pendingConfirm}
          onAnswer={(id, answer) => void session.answer(id, answer)}
        />
      ) : null}
    </div>
  );
}
