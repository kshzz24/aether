---
name: debugging
description: Use when facing a bug, test failure, or unexpected behavior — debug systematically before proposing a fix.
---
# Systematic debugging

1. **Reproduce** — a reliable, minimal repro. If you can't reproduce it, you can't fix it.
2. **Isolate** — narrow the failing surface: bisect, add assertions, check inputs at the boundary.
3. **Fix** — the root cause, not the symptom.
4. **Verify** — rerun the repro and the surrounding tests. Confirm the fix, no regressions.
