from tools.hooks import Hooks

HOOK_NAMES = ["before_run", "after_run", "before_tool", "after_tool", "on_error"]


def test_defaults_are_callable_noops():
    h = Hooks()
    for name in HOOK_NAMES:
        hook = getattr(h, name)
        assert callable(hook)
        # a no-op swallows any call shape and returns None
        assert hook("anything", 1, key="v") is None


def test_override_is_stored_and_fires():
    calls = []
    h = Hooks(before_tool=lambda *a: calls.append(a))
    h.before_tool("read_file", {"path": "x"})
    assert calls == [("read_file", {"path": "x"})]
    # untouched hooks remain no-ops
    assert h.after_tool("read_file", {}, "result") is None


def test_hooks_are_independent():
    seen = {"before": 0, "after": 0}
    h = Hooks(
        before_run=lambda *a: seen.__setitem__("before", seen["before"] + 1),
        after_run=lambda *a: seen.__setitem__("after", seen["after"] + 1),
    )
    h.before_run("goal")
    h.after_run("goal")
    assert seen == {"before": 1, "after": 1}
