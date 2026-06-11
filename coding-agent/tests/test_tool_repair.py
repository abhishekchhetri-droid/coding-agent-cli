"""Regression tests for _repair_tool_results — the guard that keeps an interrupted turn
from poisoning the persisted message history with a dangling tool_call (Anthropic 400:
"tool_use ids were found without tool_result blocks")."""
from agent.agent import _repair_tool_results


def _assistant(*ids):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "x", "arguments": "{}"}}
            for i in ids
        ],
    }


def _result(tid):
    return {"role": "tool", "tool_call_id": tid, "content": "ok"}


def _resulted(msgs):
    return {m["tool_call_id"] for m in msgs if m.get("role") == "tool"}


def test_inserts_missing_result_mid_batch():
    # assistant fired two tools, only the first got a result before the turn died
    msgs = [{"role": "user", "content": "go"}, _assistant("t1", "t2"), _result("t1")]
    _repair_tool_results(msgs)
    assert _resulted(msgs) == {"t1", "t2"}
    # repaired result sits inside the consecutive tool block after the call
    assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == "t2"


def test_dangling_call_at_end_with_no_results():
    msgs = [_assistant("x1")]
    _repair_tool_results(msgs)
    assert len(msgs) == 2 and msgs[1]["tool_call_id"] == "x1"


def test_healthy_history_is_noop():
    msgs = [_assistant("h1"), _result("h1"), {"role": "assistant", "content": "done"}]
    before = [dict(m) for m in msgs]
    _repair_tool_results(msgs)
    assert msgs == before


def test_multiple_independent_calls():
    msgs = [_assistant("a1"), _result("a1"), _assistant("b1", "b2"), _result("b1")]
    _repair_tool_results(msgs)
    assert _resulted(msgs) == {"a1", "b1", "b2"}
