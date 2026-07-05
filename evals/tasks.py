"""The three golden tasks for the eval smoke test.

Each exercises a different tool path so the suite covers the surface rather than
the same path three times:
  1. collect-todos : read source + write a report (read_file -> write_file)
  2. write-greeter : pure write (write_file)
  3. summarize-file: read an input + write a derived artifact (read_file -> write_file)

Every check is the cheapest thing that still goes red if the agent stops doing
the job -- artifact exists (+ a marker where the content is deterministic). None
may pass against an empty workspace, or the check guards nothing.
"""

from pathlib import Path

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


GOLDEN_TASKS = [
    GoldenTask(
        name="collect-todos",
        goal="Find every TODO comment in src/ and write a summary to todos.md",
        check=_todos_check,
        setup=_seed_todos,
    ),
    GoldenTask(
        name="write-greeter",
        goal="Create hello.py defining a function greet(name) that returns a greeting.",
        check=_greeter_check,
    ),
    GoldenTask(
        name="summarize-file",
        goal="Read notes.txt and write a one-line summary of it to summary.txt.",
        check=_summary_check,
        setup=_seed_notes,
    ),
]
