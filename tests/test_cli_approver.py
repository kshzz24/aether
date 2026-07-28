import asyncio

from approval import ApprovalRequest
from cli.approver import CliApprover
from tools.base import ToolKind


def _req():
    return ApprovalRequest("write_file", {"path": "a"}, ToolKind.WRITE, [], "diff")


def test_cli_approver_approves_on_yes():
    ap = CliApprover(input_fn=lambda: "y")
    d = asyncio.run(ap.decide(_req()))
    assert d.approved is True


def test_cli_approver_denies_on_no():
    ap = CliApprover(input_fn=lambda: "n")
    d = asyncio.run(ap.decide(_req()))
    assert d.approved is False
    assert d.reason
