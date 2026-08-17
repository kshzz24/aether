"""`forge serve` / `python -m server` — the process entry point.

Its own argparse rather than another subparser in `main.py`: nothing here
describes *a run*, so none of `RunParams` applies, and grafting `--host` onto the
CLI's parser would put a serving flag next to `--max-cost`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

from server.app import (
    IDLE_TIMEOUT_SEC,
    MAX_SESSIONS,
    TOKEN_ENV,
    create_app,
)

LOOPBACK = ("127.0.0.1", "::1", "localhost")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge serve", description="Serve FORGE over HTTP"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "interface to bind (default 127.0.0.1). Binding 0.0.0.0 exposes an "
            "agent that runs shell commands and in-process user tools as you"
        ),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="project root every session works in (default: the cwd)",
    )
    parser.add_argument("--max-sessions", type=int, default=MAX_SESSIONS)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=IDLE_TIMEOUT_SEC,
        help="seconds an unwatched, idle session survives before it is reaped",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    # Checked here as well as in the lifespan. Both matter: the lifespan is the
    # guarantee, this is the readable error instead of a uvicorn traceback.
    if not os.environ.get(TOKEN_ENV):
        sys.exit(
            f"{TOKEN_ENV} is not set. FORGE refuses to serve without a token — "
            f"it can read, write and run commands as you.\n"
            f'  set it with:  export {TOKEN_ENV}="$(python -c '
            f"'import secrets;print(secrets.token_urlsafe(24))')\""
        )

    # Loopback by default, and a non-loopback bind has to be typed by a human:
    # Phase-2 user tools execute in this process, so the blast radius of this one
    # flag is the whole machine.
    if args.host not in LOOPBACK:
        logging.warning(
            "binding %s — FORGE is reachable from the network and runs tools "
            "with your privileges",
            args.host,
        )

    app = create_app(
        root=args.root,
        max_sessions=args.max_sessions,
        idle_timeout_sec=args.idle_timeout,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
