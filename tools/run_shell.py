import asyncio

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


async def run(args: dict) -> str:
    command = args["command"]
    timeout = args.get("timeout", 30)

    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            f"exit_code: {proc.returncode}\n"
            f"--- stdout ---\n{out.decode(errors='replace')}\n"
            f"--- stderr ---\n{err.decode(errors='replace')}"
        )
    except TimeoutError:
        proc.kill()
        raise TimeoutError(f"command timed out after {timeout}s") from None
