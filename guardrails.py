"""Guardrails on tool output: PII/secret redaction + injection flagging.

Mirrors `safety.py`'s shape exactly -- pure functions returning reason lists,
called inline in the agent loop, no I/O, no new dependency. Secrets are
redacted (the risk is exfiltration to the provider on the next `create()`
call); injection is flagged, not stripped (a heuristic false positive must
not silently delete legitimate content).
"""

import re

from tools.base import ToolKind

_SECRET_PATTERNS = [
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "OpenAI-style secret key"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), "Groq API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key ID"),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "email address",
    ),
    (re.compile(r"\b[A-Za-z0-9_-]{32,}\b"), "high-entropy token"),
]

_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard prior instructions",
    "new system prompt",
    "you are now",
    "system prompt:",
]

_INJECTION_KINDS = (ToolKind.READ, ToolKind.AGENT)


def scan_for_secrets(text: str) -> list[tuple[str, str]]:
    """(reason, matched_span) pairs for substrings that look like credentials."""
    found: list[tuple[str, str]] = []
    for pattern, reason in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            found.append((reason, match.group(0)))
    return found


def scan_for_injection(text: str) -> list[str]:
    """Reasons only, no redaction -- phrase-matching against injection markers."""
    lowered = text.lower()
    return [
        f"possible prompt injection: {phrase!r}"
        for phrase in _INJECTION_PHRASES
        if phrase in lowered
    ]


def apply(result_str: str, *, kind: ToolKind) -> tuple[str, list[str]]:
    """Redact secret spans; prepend an injection banner if flagged.

    Returns (possibly-modified text, all reasons) for the caller to log/trace.
    """
    reasons: list[str] = []

    secrets = scan_for_secrets(result_str)
    redacted = result_str
    for reason, span in secrets:
        reasons.append(reason)
        redacted = redacted.replace(span, f"[REDACTED:{reason}]")

    if kind in _INJECTION_KINDS:
        injection_reasons = scan_for_injection(result_str)
        if injection_reasons:
            reasons.extend(injection_reasons)
            redacted = (
                "[WARNING: possible prompt injection detected in this content]\n"
                + redacted
            )

    return redacted, reasons
