import inspect

from approval import ApprovalMode, ApprovalRequest, Approver, Decision, Verdict
from tools.base import ToolKind


def test_types_construct_and_expose_expected_members():
    assert {m.name for m in ApprovalMode} == {"AUTO", "ON_REQUEST", "NEVER"}
    assert {m.name for m in Verdict} == {"AUTO_APPROVE", "AUTO_DENY", "ASK"}

    req = ApprovalRequest(
        tool_name="write_file", arguments={"path": "a"}, kind=ToolKind.WRITE,
        danger_reasons=[], diff="--- a\n+++ a\n",
    )
    assert req.diff.startswith("---")
    assert Decision(approved=True).reason is None


def test_approver_is_a_protocol_a_plain_class_can_satisfy():
    class Yes:
        async def decide(self, request):
            return Decision(approved=True)

    # duck-typed conformance: has the async decide method
    assert inspect.iscoroutinefunction(Yes().decide)
    assert hasattr(Approver, "decide")
