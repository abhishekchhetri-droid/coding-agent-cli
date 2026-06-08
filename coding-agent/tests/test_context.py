import json

import pytest

from agent import context


class _FakeLLM:
    """Records the summarize call and returns a canned summary."""

    def __init__(self, summary="- flow_id abc-123 built and verified"):
        self.summary = summary
        self.calls: list[dict] = []

    async def complete(self, messages, tools, system=""):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return {"content": self.summary, "tool_calls": [], "usage": None}


class _Settings:
    summarize_threshold_messages = 6


def _msgs(n):
    """n user/assistant pairs as a flat message list."""
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"u{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


@pytest.mark.asyncio
async def test_below_threshold_keeps_history_verbatim():
    msgs = _msgs(2)  # 4 messages, threshold 6
    llm = _FakeLLM()
    out = await context.summarize_history(llm, msgs, _Settings())
    assert out == msgs
    assert llm.calls == []  # no summarizer call


@pytest.mark.asyncio
async def test_above_threshold_summarizes_prefix_keeps_tail():
    msgs = _msgs(10)  # 20 messages
    llm = _FakeLLM()
    out = await context.summarize_history(llm, msgs, _Settings())
    # one summarizer call happened
    assert len(llm.calls) == 1
    # result is shorter than input and begins with the summary memo
    assert len(out) < len(msgs)
    assert out[0]["role"] == "user"
    assert "[Earlier conversation summary]" in out[0]["content"]
    assert "abc-123" in out[0]["content"]
    # the most recent message is preserved verbatim
    assert out[-1] == msgs[-1]


@pytest.mark.asyncio
async def test_tail_never_starts_on_orphan_tool_message():
    # Build history ending in an assistant tool_calls + tool result group, then a final
    # user/assistant. The kept tail must start at a `user`, never a bare tool result.
    msgs = _msgs(8)
    msgs += [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "build_flow", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "done"},
    ]
    out = await context.summarize_history(_FakeLLM(), msgs, _Settings())
    # first non-summary message must be a user message (no orphaned tool result)
    body = out[1:] if out and "[Earlier conversation summary]" in out[0].get("content", "") else out
    assert body[0]["role"] == "user"
    assert all(
        not (m["role"] == "tool" and i == 0) for i, m in enumerate(body)
    )


def _flow_result(verdict: str = "") -> str:
    flow = {
        "id": "abc-123",
        "name": "RAG",
        "data": {
            "nodes": [
                {"id": "FAISS-1", "data": {"type": "FAISS", "node": {"template": {"x": "y" * 4000}}}},
                {"id": "ChatInput-1", "type": "ChatInput", "data": {"type": "ChatInput"}},
            ],
            "edges": [{"source": "ChatInput-1", "target": "FAISS-1"}],
        },
    }
    out = json.dumps(flow)
    return out + ("\n\n" + verdict if verdict else "")


def test_compact_flow_snapshots_shrinks_node_json():
    big = _flow_result()
    msgs = [{"role": "tool", "tool_call_id": "t1", "content": big}]
    before = len(big)
    context.compact_flow_snapshots(msgs)
    after = json.loads(msgs[0]["content"])
    assert after["node_count"] == 2
    assert after["edge_count"] == 1
    assert {n["type"] for n in after["nodes"]} == {"FAISS", "ChatInput"}
    assert len(msgs[0]["content"]) < before  # heavy template JSON dropped


def test_compact_flow_snapshots_preserves_verdict():
    msgs = [{"role": "tool", "tool_call_id": "t1",
             "content": _flow_result("✅ VERIFIED: Flow is wired and ran a smoke test.")}]
    context.compact_flow_snapshots(msgs)
    assert "✅ VERIFIED" in msgs[0]["content"]
    # the JSON portion still parses
    json.loads(msgs[0]["content"].split("\n\n")[0])


def test_compact_flow_snapshots_ignores_non_flow_results():
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": '{"ok": true, "count": 3}'},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "done"},
    ]
    snapshot = [dict(m) for m in msgs]
    context.compact_flow_snapshots(msgs)
    assert msgs == snapshot  # untouched


@pytest.mark.asyncio
async def test_summarizer_failure_falls_back_to_tail():
    class _BoomLLM:
        async def complete(self, *a, **k):
            raise RuntimeError("provider down")

    msgs = _msgs(10)
    out = await context.summarize_history(_BoomLLM(), msgs, _Settings())
    # no summary memo prepended; tail preserved; tool pairing intact (starts at user)
    assert all("[Earlier conversation summary]" not in (m.get("content") or "") for m in out)
    assert len(out) < len(msgs)
    assert out[0]["role"] == "user"
