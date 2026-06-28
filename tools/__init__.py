from agent import Tool
from tools import edit_file, read_file, run_shell, write_file


def build_tools() -> dict[str, Tool]:
    modules = [read_file, write_file, run_shell, edit_file]
    return {m.SCHEMA["name"]: Tool(schema=m.SCHEMA, run=m.run) for m in modules}
