import re
from pathlib import Path


_DANGEROUS = [
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f"), "recursive force remove (rm -rf)"),
    (re.compile(r"\bsudo\b"), "privilege escalation (sudo)"),
    (re.compile(r"\|\s*sh\b|\|\s*bash\b"), "pipe to shell"),
    (re.compile(r"\bgit\s+push\b.*(--force|\s-f\b)"), "force push"),
    (re.compile(r"\bdd\b"), "raw disk write (dd)"),
    (re.compile(r"\bmkfs\b"), "filesystem format (mkfs)"),
    (re.compile(r":\(\)\s*\{"), "fork bomb"),
    (re.compile(r"\bchmod\s+-R\b"), "recursive chmod"),
]

_PATH_KEYS = ("path", "file", "filename")


def dangerous_command(command) -> list:

    return [reason for pat, reason in _DANGEROUS if pat.search(command)]


def path_escape(args, repo_root):
    repo_root = repo_root.resolve()
    reasons = []

    for key in _PATH_KEYS:
        if key not in args:
            continue
        p = Path(args[key])
        resolved = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
        if not resolved.is_relative_to(repo_root):
            reasons.append(f"{key} escapes repo root: {args[key]}")

    return reasons
