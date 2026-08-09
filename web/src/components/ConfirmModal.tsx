/**
 * The moment of consent. This is what the whole surface exists for.
 *
 * An agent is suspended mid-run waiting on this answer — `agent.py:208` is parked
 * on a Future that only a decision resolves. So the modal has four jobs, in order:
 * say exactly what is about to happen, flag why it might be wrong, make the four
 * answers reachable in one keystroke, and never let "always" be the easy path out
 * of a warning.
 *
 * Same four affordances and the same keys as the TUI (`tui/approver.py:64-70`), so
 * muscle memory transfers between surfaces.
 *
 * **"Always" is absent, not disabled, when the call is danger-flagged.** A disabled
 * control invites hunting for the unlock; an absent one reads as "not on offer
 * here". The server computes `offers_always` — enforcing this rule only in browser
 * JavaScript would not be enforcing it.
 */

import { useEffect, useRef, useState } from "react";

import type { ConfirmFrame } from "../api/frames";
import type { Answer } from "../api/useForgeSession";
import { KIND_COLOR } from "../theme";
import { DangerFlags } from "./Cards";
import { Diff } from "./Diff";

export function ConfirmModal({
  confirm,
  onAnswer,
}: {
  confirm: ConfirmFrame;
  onAnswer(requestId: string, answer: Answer): void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const allow = useRef<HTMLButtonElement>(null);
  const editor = useRef<HTMLTextAreaElement>(null);

  const flagged = confirm.danger_reasons.length > 0;

  // Reset per question: a stale draft from the previous confirm would be applied
  // to a different call.
  useEffect(() => {
    setEditing(false);
    setDraft(JSON.stringify(confirm.arguments, null, 2));
    setProblem(null);
    allow.current?.focus();
  }, [confirm.request_id, confirm.arguments]);

  useEffect(() => {
    if (editing) editor.current?.focus();
  }, [editing]);

  const answer = (decision: Answer) => onAnswer(confirm.request_id, decision);

  const submitEdit = () => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(draft);
    } catch (exc) {
      setProblem(exc instanceof Error ? exc.message : "that is not valid JSON");
      return;
    }
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      setProblem('Arguments must be a JSON object, e.g. {"path": "a.py"}');
      return;
    }
    // Reject malformed input here rather than sending it on — but note the agent
    // re-validates and re-runs both danger checks on whatever arrives
    // (`agent.py:218-241`), so editing is no more powerful than making the call
    // yourself. That is what keeps this from being a bypass.
    answer({ approved: true, arguments: parsed as Record<string, unknown> });
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      answer({ approved: false, reason: "you declined" });
      return;
    }
    if (editing) return; // let the editor have its keys
    const key = event.key.toLowerCase();
    if (key === "y") answer({ approved: true });
    else if (key === "n") answer({ approved: false, reason: "you declined" });
    else if (key === "e") setEditing(true);
    else if (key === "a" && confirm.offers_always) {
      answer({ approved: true, remember: true });
    }
  };

  return (
    <div className="scrim" onKeyDown={onKeyDown} role="presentation">
      <div
        className={flagged ? "modal flagged" : "modal"}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
      >
        <div className="modal-head">
          <div className="eyebrow">
            {flagged ? "flagged · approval required" : "approval required"}
          </div>
          <h2 className="modal-title" id="confirm-title">
            <span style={{ color: KIND_COLOR[confirm.kind] }}>
              {confirm.tool_name}
            </span>{" "}
            <span className="pill" style={{ color: "var(--muted)" }}>
              {confirm.kind}
            </span>
          </h2>
        </div>

        <div className="modal-body">
          <DangerFlags reasons={confirm.danger_reasons} />

          {editing ? (
            <div>
              <label className="eyebrow" htmlFor="confirm-edit">
                arguments
              </label>
              <textarea
                id="confirm-edit"
                ref={editor}
                className="field-input"
                style={{ width: "100%", minHeight: "8rem" }}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                spellCheck={false}
              />
              {problem ? <div className="banner">{problem}</div> : null}
            </div>
          ) : (
            <pre className="diff scroll-x" aria-label="arguments">
              {JSON.stringify(confirm.arguments, null, 2)}
            </pre>
          )}

          {confirm.diff ? <Diff text={confirm.diff} /> : null}
        </div>

        <div className="modal-foot">
          {editing ? (
            <>
              <button className="btn primary" onClick={submitEdit}>
                Run the edited call
              </button>
              <button className="btn" onClick={() => setEditing(false)}>
                Cancel edit
              </button>
            </>
          ) : (
            <>
              <button
                ref={allow}
                className="btn primary"
                onClick={() => answer({ approved: true })}
              >
                <span className="kbd">y</span>Allow
              </button>

              {/* Absent, not disabled, when flagged — see the module note. */}
              {confirm.offers_always ? (
                <button
                  className="btn"
                  onClick={() => answer({ approved: true, remember: true })}
                >
                  <span className="kbd">a</span>Always allow {confirm.tool_name}
                </button>
              ) : null}

              <button className="btn" onClick={() => setEditing(true)}>
                <span className="kbd">e</span>Edit
              </button>

              <span className="modal-foot-spacer" />

              <button
                className="btn danger"
                onClick={() => answer({ approved: false, reason: "you declined" })}
              >
                <span className="kbd">n</span>Deny
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
