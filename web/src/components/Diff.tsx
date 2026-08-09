/**
 * A unified diff, coloured by line prefix.
 *
 * Deliberately not a syntax highlighter. What matters at a confirm is *what
 * changes*, and a second colour axis over the language would compete with the one
 * that carries the decision.
 */

const CLASS_BY_PREFIX: Record<string, string> = {
  "+": "diff-add",
  "-": "diff-del",
  "@": "diff-meta",
};

export function Diff({ text }: { text: string }) {
  const lines = text.replace(/\n$/, "").split("\n");

  return (
    <pre className="diff scroll-x" aria-label="proposed change">
      {lines.map((line, index) => {
        // `---`/`+++` are file headers, not content: match them before the
        // single-character prefixes or every header reads as a huge deletion.
        const isHeader = line.startsWith("---") || line.startsWith("+++");
        const className = isHeader
          ? "diff-meta"
          : (CLASS_BY_PREFIX[line[0]] ?? "");
        return (
          <div key={index} className={className}>
            {line === "" ? " " : line}
          </div>
        );
      })}
    </pre>
  );
}
