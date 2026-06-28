from pathlib import Path

SCHEMA = {
    "name": "read_file",
    "description": "Read a UTF-8 text file and return its full contents.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


async def run(args: dict) -> str:
    path_str = args["path"]
    return Path(path_str).read_text(encoding="utf-8")
