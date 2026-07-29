---
name: writing-tests
description: Use when adding a feature or fixing a bug — write a failing test first, then the code.
---
# Writing tests

1. Write one small test that captures the desired behavior. Run it and watch it FAIL for the right reason.
2. Write the minimal code to make it pass. Run it and watch it PASS.
3. Refactor if needed, keeping the test green. Commit.

Keep tests focused: one behavior per test, assert on outcomes not internals.
