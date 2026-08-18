"""The golden tasks for the eval smoke test.

Each exercises a different tool path so the suite covers the surface rather than
the same path repeatedly:
  1. collect-todos         : read source + write a report (read_file -> write_file)
  2. write-greeter          : pure write (write_file)
  3. summarize-file         : read an input + write a derived artifact (judged, live)
  4. run-command            : EXECUTE, via run_shell
  5. multi-step-refactor    : multi-turn -- read two files, edit one from the other
  6. malformed-call-repair  : deterministic-only, proves agent.py's repair path
  7. redacts-a-leaked-secret: deterministic-only, proves guardrails.py

Every check is the cheapest thing that still goes red if the agent stops doing
the job -- artifact exists (+ a marker where the content is deterministic). None
may pass against an empty workspace, or the check guards nothing.
"""

from pathlib import Path

from client import (
    Message,
    NormalizedResponse,
    TextBlock,
    ToolCallBlock,
    ToolCallingUnsupportedError,
    ToolResultBlock,
)
from evals.runner import GoldenTask


def _seed_todos(ws: Path) -> None:
    src = ws / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "app.py").write_text(
        "def login():\n"
        "    pass  # TODO: implement login\n\n"
        "def logout():\n"
        "    pass  # TODO: clear the session\n",
        encoding="utf-8",
    )


def _todos_check(ws: Path) -> bool:
    f = ws / "todos.md"
    return f.exists() and "TODO" in f.read_text(encoding="utf-8")


def _seed_notes(ws: Path) -> None:
    (ws / "notes.txt").write_text(
        "Team sync notes:\n"
        "- Ship the v1 release on Friday.\n"
        "- Fix the login timeout bug.\n"
        "- Start drafting the user docs.\n",
        encoding="utf-8",
    )


def _greeter_check(ws: Path) -> bool:
    f = ws / "hello.py"
    return f.exists() and "def greet" in f.read_text(encoding="utf-8")


def _summary_check(ws: Path) -> bool:
    # The summary's wording is model-dependent, so we can't assert its text;
    # the cheapest non-vacuous check is that a non-empty artifact was produced.
    f = ws / "summary.txt"
    return f.exists() and bool(f.read_text(encoding="utf-8").strip())


def _summary_target(ws: Path) -> str:
    return (ws / "summary.txt").read_text(encoding="utf-8")


_SUMMARY_JUDGE_PROMPT = (
    "The source file said:\n"
    "---\nTeam sync notes:\n- Ship the v1 release on Friday.\n"
    "- Fix the login timeout bug.\n- Start drafting the user docs.\n---\n\n"
    "Does this one-line summary accurately capture that content, without "
    "inventing anything not in the source?\n\nSummary: {content}"
)


def _tool_use(name: str, arguments: dict, call_id: str = "c1") -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[ToolCallBlock(id=call_id, name=name, arguments=arguments)],
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        stop_reason="tool_use",
    )


def _finish(text: str = "done") -> NormalizedResponse:
    return NormalizedResponse(
        blocks=[TextBlock(text=text)],
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        stop_reason="end_turn",
    )


# --- run-command: exercises the one ToolKind the other tasks miss, EXECUTE ---

_DONE_TXT_COMMAND = "python -c \"open('done.txt','w').write('ok')\""


def _run_command_check(ws: Path) -> bool:
    f = ws / "done.txt"
    return f.exists() and "ok" in f.read_text(encoding="utf-8")


# --- multi-step-refactor: forces >=2 tool calls across >=2 turns ---


def _seed_refactor(ws: Path) -> None:
    (ws / "constants.py").write_text("TIMEOUT = 30\n", encoding="utf-8")
    (ws / "app.py").write_text(
        "TIMEOUT = 10  # out of sync with constants.py\n\n"
        "def run():\n    return TIMEOUT\n",
        encoding="utf-8",
    )


def _refactor_check(ws: Path) -> bool:
    f = ws / "app.py"
    if not f.exists():
        return False
    text = f.read_text(encoding="utf-8")
    return "TIMEOUT = 30" in text and "TIMEOUT = 10" not in text


# --- malformed-call-repair: deterministic-only, proves agent.py's repair path ---


def _seed_marker(ws: Path) -> None:
    (ws / ".ran").write_text("seeded", encoding="utf-8")


def _marker_check(ws: Path) -> bool:
    return (ws / ".ran").exists()


# --- redacts-a-leaked-secret: deterministic-only, proves guardrails.py ---

_FAKE_SECRET = "sk-" + "a" * 40


def _seed_secret(ws: Path) -> None:
    (ws / "secret.txt").write_text(f"API_KEY={_FAKE_SECRET}\n", encoding="utf-8")


def _secret_file_check(ws: Path) -> bool:
    return (ws / "secret.txt").exists()


def _secret_was_redacted(messages: list[Message]) -> bool:
    """The raw secret never reached a tool result; a redaction marker did."""
    saw_redaction = False
    for message in messages:
        for block in message.blocks:
            if isinstance(block, ToolResultBlock):
                if _FAKE_SECRET in block.content:
                    return False
                if "[REDACTED" in block.content:
                    saw_redaction = True
    return saw_redaction


GOLDEN_TASKS = [
    GoldenTask(
        name="collect-todos",
        goal="Find every TODO comment in src/ and write a summary to todos.md",
        check=_todos_check,
        setup=_seed_todos,
        scripted=[
            _tool_use("read_file", {"path": "src/app.py"}, "c1"),
            _tool_use(
                "write_file",
                {
                    "path": "todos.md",
                    "content": "TODO: implement login\nTODO: clear the session\n",
                },
                "c2",
            ),
            _finish(),
        ],
    ),
    GoldenTask(
        name="write-greeter",
        goal="Create hello.py defining a function greet(name) that returns a greeting.",
        check=_greeter_check,
        scripted=[
            _tool_use(
                "write_file",
                {
                    "path": "hello.py",
                    "content": "def greet(name):\n    return f'Hello, {name}!'\n",
                },
                "c1",
            ),
            _finish(),
        ],
    ),
    GoldenTask(
        name="summarize-file",
        goal="Read notes.txt and write a one-line summary of it to summary.txt.",
        check=_summary_check,
        setup=_seed_notes,
        judge_prompt=_SUMMARY_JUDGE_PROMPT,
        judge_target=_summary_target,
        # No `scripted`: a canned summary would prove nothing about summarization
        # quality, which is the entire point of this task. Live-only, judged.
    ),
    GoldenTask(
        name="run-command",
        goal="Run a shell command that writes the text 'ok' into a file named "
        "done.txt.",
        check=_run_command_check,
        scripted=[
            _tool_use("run_shell", {"command": _DONE_TXT_COMMAND}, "c1"),
            _finish(),
        ],
    ),
    GoldenTask(
        name="multi-step-refactor",
        goal=(
            "Read constants.py to find the canonical TIMEOUT value, then edit "
            "app.py so its TIMEOUT matches it exactly."
        ),
        check=_refactor_check,
        setup=_seed_refactor,
        scripted=[
            _tool_use("read_file", {"path": "constants.py"}, "c1"),
            _tool_use(
                "write_file",
                {
                    "path": "app.py",
                    "content": "TIMEOUT = 30\n\ndef run():\n    return TIMEOUT\n",
                },
                "c2",
            ),
            _finish(),
        ],
    ),
    GoldenTask(
        name="malformed-call-repair",
        goal="Acknowledge the task.",
        check=_marker_check,
        setup=_seed_marker,
        live=False,
        scripted=[
            ToolCallingUnsupportedError("stub-model"),
            _finish(),
        ],
    ),
    GoldenTask(
        name="redacts-a-leaked-secret",
        goal="Read secret.txt.",
        check=_secret_file_check,
        setup=_seed_secret,
        check_transcript=_secret_was_redacted,
        live=False,
        scripted=[
            _tool_use("read_file", {"path": "secret.txt"}, "c1"),
            _finish(),
        ],
    ),
]
