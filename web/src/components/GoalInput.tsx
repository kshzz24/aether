/**
 * Where a turn starts. Enter sends; Shift+Enter adds a line, because goals are
 * often several sentences and losing one to a stray Enter is expensive.
 *
 * While a run is in flight the input is closed and the button becomes Interrupt —
 * the server allows one run per session and answers a second goal with 409
 * (`SessionBusy`), so offering the field would be offering an error.
 */

import { useState } from "react";

export function GoalInput({
  running,
  disabled,
  onSend,
  onInterrupt,
}: {
  running: boolean;
  disabled: boolean;
  onSend(text: string): void;
  onInterrupt(): void;
}) {
  const [text, setText] = useState("");

  const send = () => {
    const goal = text.trim();
    if (!goal) return;
    onSend(goal);
    setText("");
  };

  return (
    <div className="composer">
      <div className="field">
        <textarea
          className="field-input"
          placeholder={
            running ? "FORGE is working…" : "What should FORGE do next?"
          }
          value={text}
          disabled={running || disabled}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
          rows={1}
        />
        {running ? (
          <button className="btn danger" onClick={onInterrupt}>
            Interrupt
          </button>
        ) : (
          <button
            className="btn primary"
            onClick={send}
            disabled={disabled || text.trim() === ""}
          >
            Run
          </button>
        )}
      </div>
      <div className="hint">
        {disabled
          ? "Pick or create a session to start."
          : "Enter to run · Shift+Enter for a new line"}
      </div>
    </div>
  );
}
