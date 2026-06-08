import json

from agent import planning


def test_write_todos_replaces_list():
    todos: list[dict] = [{"content": "old", "status": "pending"}]
    out = planning.dispatch(
        "write_todos",
        {"todos": [
            {"content": "step one", "status": "in_progress"},
            {"content": "step two", "status": "pending"},
        ]},
        todos,
        {},
    )
    res = json.loads(out)
    assert res["ok"] is True
    assert res["count"] == 2
    # replaced, not appended — old item gone
    assert [t["content"] for t in todos] == ["step one", "step two"]


def test_write_todos_drops_blank_and_coerces_bad_status():
    todos: list[dict] = []
    planning.dispatch(
        "write_todos",
        {"todos": [
            {"content": "", "status": "pending"},          # blank → dropped
            {"content": "real", "status": "bogus"},          # bad status → pending
        ]},
        todos,
        {},
    )
    assert todos == [{"content": "real", "status": "pending"}]


def test_write_todos_identical_is_noop():
    todos = [{"content": "clone", "status": "in_progress"}]
    same = [{"content": "clone", "status": "in_progress"}]
    out = json.loads(planning.dispatch("write_todos", {"todos": same}, todos, {}))
    assert out["noop"] is True
    assert "do NOT call write_todos again" in out["note"]


def test_write_todos_change_is_not_noop():
    todos = [{"content": "clone", "status": "in_progress"}]
    out = json.loads(planning.dispatch(
        "write_todos",
        {"todos": [{"content": "clone", "status": "completed"}]},
        todos, {},
    ))
    assert "noop" not in out
    assert out["completed"] == 1


def test_scratchpad_write_then_read_key():
    pad: dict[str, str] = {}
    planning.dispatch("scratchpad_write", {"key": "flow_id", "content": "abc-123"}, [], pad)
    assert pad["flow_id"] == "abc-123"

    out = planning.dispatch("scratchpad_read", {"key": "flow_id"}, [], pad)
    assert json.loads(out) == {"key": "flow_id", "content": "abc-123"}


def test_scratchpad_read_all_and_missing_key():
    pad = {"a": "1", "b": "2"}
    assert json.loads(planning.dispatch("scratchpad_read", {}, [], pad))["notes"] == pad
    miss = json.loads(planning.dispatch("scratchpad_read", {"key": "zzz"}, [], pad))
    assert "error" in miss and set(miss["keys"]) == {"a", "b"}


def test_scratchpad_write_requires_key():
    pad: dict[str, str] = {}
    out = planning.dispatch("scratchpad_write", {"content": "x"}, [], pad)
    assert "error" in json.loads(out)
    assert pad == {}


def test_remember_helper_captures_determinant():
    pad: dict[str, str] = {}
    planning.remember(pad, "flow:rag", "uuid-1")
    planning.remember(pad, "", "ignored")          # empty key → no-op
    planning.remember(pad, "k", "")                # empty content → no-op
    assert pad == {"flow:rag": "uuid-1"}


def test_render_state_empty_is_blank():
    assert planning.render_state([], {}) == ""


def test_render_state_includes_todos_and_pad():
    block = planning.render_state(
        [{"content": "build flow", "status": "in_progress"}],
        {"flow:rag": "uuid-1"},
    )
    assert "build flow" in block
    assert "[~]" in block               # in_progress marker
    assert "flow:rag" in block and "uuid-1" in block


def test_planning_tool_names_match_schemas():
    schema_names = {s["function"]["name"] for s in planning.PLANNING_TOOL_SCHEMAS}
    assert schema_names == set(planning.PLANNING_TOOL_NAMES)
    assert schema_names == {"write_todos", "scratchpad_write", "scratchpad_read"}
