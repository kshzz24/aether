import asyncio
import subprocess
import sys

SCHEMA = {
    "name": "run_shell",
    "description": "Run a shell command and return its exit code, stdout, and stderr.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number"},
        },
        "required": ["command"],
    },
}


def _coerce_timeout(value: object) -> float:
    """Turn a caller-supplied timeout into a positive float number of seconds.

    Models often emit numbers as strings, so a well-formed "5" is coerced.
    Anything non-numeric or non-positive is a malformed argument: raise a clear
    ValueError, which the agent turns into an observation the model can fix.
    """
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"run_shell timeout must be a number of seconds, got {value!r}"
        ) from None
    if timeout <= 0:
        raise ValueError(f"run_shell timeout must be positive, got {timeout}")
    return timeout


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Force-kill a timed-out command and reap it without hanging the agent.

    A shell doesn't forward the kill to the command it spawned, so killing the
    shell alone leaves the child holding the pipes and ``wait()`` blocks until
    the child exits on its own. Kill the whole tree instead, then reap with a
    bounded wait so cleanup can never re-introduce a hang.
    """
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        await asyncio.to_thread(
            subprocess.run,
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        pass  # tree-kill should have reaped it; never block the agent further


async def run(args: dict) -> str:
    command = args["command"]
    timeout = _coerce_timeout(args.get("timeout", 30))

    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        await _terminate(proc)
        raise TimeoutError(f"command timed out after {timeout}s") from None

    return (
        f"exit_code: {proc.returncode}\n"
        f"--- stdout ---\n{out.decode(errors='replace')}\n"
        f"--- stderr ---\n{err.decode(errors='replace')}"
    )
