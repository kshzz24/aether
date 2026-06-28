from tools.fs import atomic_write

SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, creating parent directories. Overwrites.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
}


async def run(args: dict) -> str:
    path_str = args["path"]
    content_str = args["content"]

    atomic_write(path_str, content_str)

    return f"Successfully wrote to {path_str}"
