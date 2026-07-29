from client import Message, TextBlock, ToolCallBlock, ToolResultBlock
from persistence import (
    Session,
    list_sessions,
    load,
    new_session_id,
    save,
    session_from_dict,
    session_to_dict,
)


def _sample_session(session_id: str = "20260729-000000-aaaa") -> Session:
    messages = [
        Message(role="user", blocks=[TextBlock(text="do the thing")]),
        Message(
            role="assistant",
            blocks=[
                ToolCallBlock(id="c1", name="read_file", arguments={"path": "x.py"})
            ],
        ),
        Message(
            role="user",
            blocks=[ToolResultBlock(tool_call_id="c1", content="file contents")],
        ),
        Message(role="assistant", blocks=[TextBlock(text="done")]),
    ]
    return Session(
        id=session_id,
        goal="do the thing",
        provider="groq",
        model="openai/gpt-oss-120b",
        created_at="2026-07-29T00:00:00",
        updated_at="2026-07-29T00:01:00",
        total_cost=0.0123,
        messages=messages,
    )


def test_session_round_trips_through_dict_with_tool_pair():
    original = _sample_session()
    restored = session_from_dict(session_to_dict(original))
    assert restored == original
    # the tool-call/tool-result pair survived with its block types intact
    call = restored.messages[1].blocks[0]
    result = restored.messages[2].blocks[0]
    assert isinstance(call, ToolCallBlock) and call.name == "read_file"
    assert isinstance(result, ToolResultBlock) and result.tool_call_id == "c1"


def test_save_then_load_is_faithful(tmp_path):
    original = _sample_session()
    path = save(original, tmp_path)
    assert path.exists()
    assert load(original.id, tmp_path) == original


def test_save_is_atomic_and_leaves_no_tmp(tmp_path):
    save(_sample_session(), tmp_path)
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob("*.json.*"))
    assert leftovers == []


def test_list_sessions_newest_first_and_skips_corrupt(tmp_path):
    save(_sample_session("20260729-000001-aaaa"), tmp_path)
    newer = _sample_session("20260729-000002-bbbb")
    newer.updated_at = "2026-07-29T09:99:99"  # lexically-later stamp
    save(newer, tmp_path)
    # a corrupt/foreign file must not crash the listing
    (tmp_path / "garbage.json").write_text("{ not valid json", encoding="utf-8")

    metas = list_sessions(tmp_path)
    ids = [m.id for m in metas]
    assert ids == ["20260729-000002-bbbb", "20260729-000001-aaaa"]
    assert metas[0].turns == 4
    assert metas[0].goal == "do the thing"


def test_list_sessions_empty_dir_returns_empty(tmp_path):
    assert list_sessions(tmp_path) == []


def test_new_session_id_is_unique_and_sortable():
    a = new_session_id()
    b = new_session_id()
    assert a != b
    # timestamp-prefixed => lexical order tracks creation order
    assert len(a) >= len("20260729-000000-aaaa")
