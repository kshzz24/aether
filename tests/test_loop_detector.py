"""Specs for context/loop_detector.py (piece #4).

The detector is a sharper guard than the iteration cap: it aborts a *stuck*
agent -- one whose actions have stopped changing the world -- long before
max_iterations would. "Stuck" is a repeating cycle of (tool call, arguments,
result); the result is part of the fingerprint so a task that is genuinely
making progress (the observation keeps changing) is never flagged.

Design pinned by these tests:
- A step fingerprint = (tool_name, canonicalized args, result).
- Cycle detection: any period p in 1..max_cycle_period that has repeated
  min_repeats times back-to-back at the tail is a loop. Period 1 == the
  same call/result over and over.
- Defaults: max_cycle_period=3, min_repeats=3.
"""


def test_identical_call_and_result_trips_after_threshold():
    from context.loop_detector import LoopDetector

    d = LoopDetector()  # min_repeats=3

    # run_shell("pytest") returning the SAME failure over and over: stuck.
    d.record("run_shell", {"cmd": "pytest"}, "3 failed")
    d.record("run_shell", {"cmd": "pytest"}, "3 failed")
    assert d.is_looping() is False  # only two so far -- not yet conclusive

    d.record("run_shell", {"cmd": "pytest"}, "3 failed")
    assert d.is_looping() is True  # third identical step -> abort


def test_two_step_cycle_trips():
    from context.loop_detector import LoopDetector

    d = LoopDetector()

    # A B A B A B: a period-2 cycle repeated three times.
    for _ in range(3):
        d.record("read_file", {"path": "a"}, "AAA")
        d.record("run_shell", {"cmd": "ls"}, "BBB")

    assert d.is_looping() is True


def test_changing_result_is_not_a_loop():
    from context.loop_detector import LoopDetector

    d = LoopDetector()

    # Identical call and args every iteration, but the observation keeps
    # moving (3 failed -> 2 -> 1 -> 0). That is progress, not a loop -- this
    # is the case the result-in-fingerprint rule exists to protect.
    for result in ("3 failed", "2 failed", "1 failed", "0 failed"):
        d.record("run_shell", {"cmd": "pytest"}, result)

    assert d.is_looping() is False


def test_different_args_is_not_a_loop():
    from context.loop_detector import LoopDetector

    d = LoopDetector()

    # Same tool, same result string, but a different target each time:
    # legitimate iteration over a set of files, not a stuck loop.
    for path in ("a.py", "b.py", "c.py", "d.py"):
        d.record("read_file", {"path": path}, "file body")

    assert d.is_looping() is False


def test_cycle_below_repeat_threshold_is_not_a_loop():
    from context.loop_detector import LoopDetector

    d = LoopDetector()

    # A B A B: the period-2 cycle has only repeated twice. With min_repeats=3
    # this is not yet conclusive -- alternating between two actions can be
    # legitimate for a while.
    d.record("read_file", {"path": "a"}, "AAA")
    d.record("run_shell", {"cmd": "ls"}, "BBB")
    d.record("read_file", {"path": "a"}, "AAA")
    d.record("run_shell", {"cmd": "ls"}, "BBB")

    assert d.is_looping() is False


def test_argument_key_order_does_not_matter():
    from context.loop_detector import LoopDetector

    d = LoopDetector()

    # The same arguments in a different dict order must fingerprint equal,
    # otherwise a real loop slips through. Forces canonicalized args.
    d.record("run_shell", {"cmd": "pytest", "cwd": "/x"}, "fail")
    d.record("run_shell", {"cwd": "/x", "cmd": "pytest"}, "fail")
    d.record("run_shell", {"cmd": "pytest", "cwd": "/x"}, "fail")

    assert d.is_looping() is True
