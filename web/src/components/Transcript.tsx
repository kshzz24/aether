/**
 * The transcript, and the spine that indexes it.
 *
 * A two-column grid: a gutter carrying each frame's `seq`, a three-letter tag, and
 * a mark whose hue is the frame's accent — then the content. The whole run is
 * scannable as a vertical strip before you read a word, and because colour is
 * reserved for consent moments and outcomes (`theme.ts`), the strip shows you where
 * decisions happened.
 *
 * `seq` in the gutter is not ornament: it is the offset a reconnect resumes from
 * (`Last-Event-ID`). Control frames have none, and show a dash rather than a
 * borrowed number.
 */

import { Fragment, useEffect, useRef } from "react";

import type { Row } from "../api/rows";
import { accentFor, labelFor } from "../theme";
import { renderFrame } from "./Cards";

function Spine({ row }: { row: Row }) {
  const accent = accentFor(row.frame);
  return (
    <div className="spine" style={accent ? { color: accent } : undefined}>
      <span className="spine-tag">{labelFor(row.frame)}</span>
      <span className="spine-seq">
        {row.seq === null ? "—" : String(row.seq).padStart(3, "0")}
      </span>
      <span
        className={accent ? "spine-mark accent" : "spine-mark"}
        style={accent ? { background: accent } : undefined}
      />
    </div>
  );
}

export function Transcript({ rows }: { rows: Row[] }) {
  const end = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // Follow the newest row, but stop following the moment the reader scrolls up —
  // yanking someone back to the bottom while they are reading a tool result is the
  // single most irritating thing a live log can do.
  useEffect(() => {
    const node = scroller.current;
    if (!node) return;
    const onScroll = () => {
      const slack = node.scrollHeight - node.scrollTop - node.clientHeight;
      pinned.current = slack < 80;
    };
    node.addEventListener("scroll", onScroll, { passive: true });
    return () => node.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (pinned.current) end.current?.scrollIntoView({ block: "end" });
  }, [rows.length]);

  if (rows.length === 0) {
    return (
      <div className="scroller" ref={scroller}>
        <p className="empty" style={{ paddingTop: "2rem" }}>
          Nothing has run yet. Give FORGE a goal below and the trace appears here,
          one numbered row per event.
        </p>
      </div>
    );
  }

  return (
    <div className="scroller" ref={scroller}>
      <div className="transcript">
        {rows.map((row) => (
          <Fragment key={row.key}>
            <Spine row={row} />
            <div className="row">{renderFrame(row.frame, row.streaming)}</div>
          </Fragment>
        ))}
      </div>
      <div ref={end} />
    </div>
  );
}
