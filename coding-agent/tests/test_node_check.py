"""Tests for _inject_node_check schema-driven required-input verification."""
import json
from unittest.mock import MagicMock

from agent.agent import _inject_node_check


def _node(node_id, ntype, template, outputs=None):
    return {
        "id": node_id,
        "type": ntype,
        "data": {
            "type": ntype,
            "id": node_id,
            "node": {"template": template, "outputs": outputs or []},
        },
    }


def _flow(nodes, edges):
    return json.dumps({"id": "f1", "name": "F", "data": {"nodes": nodes, "edges": edges}})


def _oe(d):
    """Encode a handle dict the way React Flow / Langflow serializes it: a string
    with double-quotes replaced by U+0153 (œ), stored at the top level."""
    return json.dumps(d).replace('"', "œ")


def _edge(source, target, field, src_name="out", field_type="str", input_types=None):
    """Real Langflow edge shape: œ-encoded strings at top level, dicts under edge['data']."""
    input_types = input_types if input_types is not None else ["Message"]
    sh = {"id": source, "name": src_name, "output_types": ["Message"]}
    th = {"fieldName": field, "id": target, "inputTypes": input_types, "type": field_type}
    return {
        "source": source,
        "target": target,
        "sourceHandle": _oe(sh),
        "targetHandle": _oe(th),
        "data": {"sourceHandle": sh, "targetHandle": th},
    }


def _intended(source, target, field, src_name="out", field_type="str"):
    """An edge as the agent constructs it pre-build: clean dict handles."""
    return {
        "source": source,
        "target": target,
        "sourceHandle": {"id": source, "name": src_name, "output_types": ["Message"]},
        "targetHandle": {"fieldName": field, "id": target, "inputTypes": ["Message"], "type": field_type},
    }


def test_wiring_gap_blocks_success():
    """A required component-output field (Message) with no incoming edge must block success."""
    agent = _node(
        "SQLAgent-1",
        "SQLAgent",
        {"input_value": {"required": True, "type": "other", "input_types": ["Message"], "value": ""}},
    )
    chat = _node("ChatInput-1", "ChatInput", {})
    # input_value is NOT wired
    flow = _flow([chat, agent], [])
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "x", "error": ""}

    out = _inject_node_check(flow, mcp, "f1")

    assert "WIRING INCOMPLETE" in out
    assert "SQLAgent-1.input_value" in out
    assert "VERIFIED" not in out


def test_empty_model_field_is_soft_note_not_block():
    """Model edges are stripped platform-wide, so an empty required model field is a
    soft MODEL note (configure provider), never a hard wiring block / false positive."""
    agent = _node(
        "SQLAgent-1",
        "SQLAgent",
        {
            "model": {"required": True, "type": "model", "input_types": ["LanguageModel"], "value": ""},
            "input_value": {"required": True, "type": "other", "input_types": ["Message"], "value": ""},
        },
    )
    chat = _node("ChatInput-1", "ChatInput", {})
    flow = _flow([chat, agent], [_edge("ChatInput-1", "SQLAgent-1", "input_value")])
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "x", "error": ""}

    out = _inject_node_check(flow, mcp, "f1")

    assert "WIRING INCOMPLETE" not in out
    assert "MODEL NOT CONFIGURED" in out
    assert "SQLAgent-1.model" in out


def test_credential_gap_does_not_block():
    """A required literal/credential field (no input_types, empty value) is a user-fill note, not a block."""
    db = _node(
        "SQLDatabase-1",
        "SQLDatabase",
        {"uri": {"required": True, "type": "str", "input_types": [], "value": "", "password": True}},
        outputs=[{"name": "out", "types": ["Message"]}],
    )
    chat_out = _node("ChatOutput-1", "ChatOutput", {})
    flow = _flow([db, chat_out], [_edge("SQLDatabase-1", "ChatOutput-1", "input_value")])
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "x", "error": ""}

    out = _inject_node_check(flow, mcp, "f1")

    assert "WIRING INCOMPLETE" not in out
    assert "NEEDS CREDENTIALS" in out
    assert "SQLDatabase-1.uri" in out


def test_all_required_satisfied_verifies():
    """When every required field is satisfied and test-run passes, report VERIFIED."""
    chat = _node("ChatInput-1", "ChatInput", {})
    out_node = _node(
        "ChatOutput-1",
        "ChatOutput",
        {"input_value": {"required": True, "type": "str", "input_types": ["Message"], "value": ""}},
    )
    flow = _flow([chat, out_node], [_edge("ChatInput-1", "ChatOutput-1", "input_value")])
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "4", "error": ""}

    out = _inject_node_check(flow, mcp, "f1")

    assert "VERIFIED" in out
    assert "WIRING INCOMPLETE" not in out


def test_zero_nodes_still_flags():
    """0 nodes after build still returns the existing failure marker."""
    flow = _flow([], [])
    mcp = MagicMock()
    out = _inject_node_check(flow, mcp, "f1")
    assert "0 nodes" in out


def test_real_langflow_handle_strings_are_parsed():
    """Edges from a live flow carry œ-encoded handle strings at top level (dict under
    edge['data']). The verifier must read them, not crash and silently skip checks."""
    chat = _node("ChatInput-1", "ChatInput", {})
    out_node = _node(
        "ChatOutput-1",
        "ChatOutput",
        {"input_value": {"required": True, "type": "other", "input_types": ["Message"], "value": ""}},
    )
    flow = _flow([chat, out_node], [_edge("ChatInput-1", "ChatOutput-1", "input_value", field_type="other")])
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "4", "error": ""}

    out = _inject_node_check(flow, mcp, "f1")

    # Edge was read → input_value counts as satisfied → VERIFIED, no false wiring gap.
    assert "VERIFIED" in out
    assert "WIRING INCOMPLETE" not in out


def test_str_message_field_is_credential_not_wiring():
    """A required str field that merely *accepts* Message input (input_types=['Message'])
    is user-fillable (e.g. a DB URI), so an empty one is a credential note, not a block."""
    db = _node(
        "SQLAgent-1",
        "SQLAgent",
        {"database_uri": {"required": True, "type": "str", "input_types": ["Message"], "value": ""}},
        outputs=[{"name": "response", "types": ["Message"]}],
    )
    out_node = _node("ChatOutput-1", "ChatOutput", {})
    flow = _flow([db, out_node], [_edge("SQLAgent-1", "ChatOutput-1", "input_value", field_type="other")])
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "x", "error": ""}

    out = _inject_node_check(flow, mcp, "f1")

    assert "WIRING INCOMPLETE" not in out
    assert "NEEDS CREDENTIALS" in out
    assert "SQLAgent-1.database_uri" in out


def test_stripped_edge_detected_via_intended_diff():
    """An intended edge to a field that vanished after build (Langflow rejected it as
    invalid) must be reported, even though the target field no longer exists to audit."""
    chat = _node("ChatInput-1", "ChatInput", {})
    # Prompt Template with NO metadata field (template string was never set).
    prompt = _node("PromptTemplate-1", "Prompt Template", {"template": {"type": "prompt", "value": ""}})
    flow = _flow([chat, prompt], [])  # nothing survived
    intended = [_intended("ChatInput-1", "PromptTemplate-1", "metadata")]
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "x", "error": ""}

    out = _inject_node_check(flow, mcp, "f1", intended_edges=intended)

    assert "EDGES REJECTED" in out
    assert "PromptTemplate-1.metadata" in out
    assert "VERIFIED" not in out


def test_stripped_model_edge_is_soft_not_block():
    """A stripped model edge is the known platform limitation, not a build bug — soft note."""
    chat = _node("ChatInput-1", "ChatInput", {})
    agent = _node(
        "Agent-1",
        "Agent",
        {"model": {"required": True, "type": "model", "input_types": ["LanguageModel"], "value": ""}},
    )
    flow = _flow([chat, agent], [])
    intended = [_intended("AzureOpenAIModel-1", "Agent-1", "model", field_type="model")]
    mcp = MagicMock()
    mcp.test_run_flow.return_value = {"ok": True, "answer": "x", "error": ""}

    out = _inject_node_check(flow, mcp, "f1", intended_edges=intended)

    assert "EDGES REJECTED" not in out
    assert "MODEL NOT CONFIGURED" in out
