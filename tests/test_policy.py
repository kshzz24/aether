import pytest

from approval import ApprovalMode, Verdict
from policy import PolicyEngine
from tools.base import ToolKind

R, W, X = ToolKind.READ, ToolKind.WRITE, ToolKind.EXECUTE


@pytest.mark.parametrize("mode,kind,danger,expected", [
    # AUTO: approve clean; a danger trip forces ASK
    (ApprovalMode.AUTO, R, [], Verdict.AUTO_APPROVE),
    (ApprovalMode.AUTO, W, [], Verdict.AUTO_APPROVE),
    (ApprovalMode.AUTO, X, [], Verdict.AUTO_APPROVE),
    (ApprovalMode.AUTO, X, ["rm -rf"], Verdict.ASK),
    # ON_REQUEST: read auto, write/exec ask
    (ApprovalMode.ON_REQUEST, R, [], Verdict.AUTO_APPROVE),
    (ApprovalMode.ON_REQUEST, W, [], Verdict.ASK),
    (ApprovalMode.ON_REQUEST, X, [], Verdict.ASK),
    # NEVER: read auto, everything else denied (never ASK)
    (ApprovalMode.NEVER, R, [], Verdict.AUTO_APPROVE),
    (ApprovalMode.NEVER, W, [], Verdict.AUTO_DENY),
    (ApprovalMode.NEVER, X, ["sudo"], Verdict.AUTO_DENY),
])
def test_evaluate_table(mode, kind, danger, expected):
    verdict, reason = PolicyEngine(mode).evaluate(kind, danger)
    assert verdict is expected
    if verdict is Verdict.AUTO_DENY:
        assert reason, "AUTO_DENY must carry a reason"
    else:
        assert reason is None


def test_danger_overrides_auto_write():
    verdict, _ = PolicyEngine(ApprovalMode.AUTO).evaluate(ToolKind.WRITE, ["escape"])
    assert verdict is Verdict.ASK
