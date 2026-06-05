"""Unit tests for server.canvas.apply_ops — the pure in-memory flow mutation that powers
direct canvas edits (drag / field edit / add+delete edge / delete node)."""
import copy

from server.canvas import apply_ops


def _flow_data() -> dict:
    return {
        "nodes": [
            {
                "id": "A",
                "position": {"x": 0, "y": 0},
                "data": {
                    "type": "ChatInput",
                    "node": {
                        "template": {"input_value": {"value": "hi", "type": "str"}},
                        "outputs": [{"name": "message", "types": ["Message"]}],
                    },
                },
            },
            {
                "id": "B",
                "position": {"x": 100, "y": 0},
                "data": {
                    "type": "Agent",
                    "node": {
                        "template": {
                            "input_value": {"value": "", "type": "str", "input_types": ["Message"]},
                            "tools": {"value": "", "type": "other", "input_types": ["Tool"]},
                            "max_tokens": {"value": "100", "type": "int"},
                            "plain": {"value": "x", "type": "str"},  # no input_types → not connectable
                            # ModelInput: has input_types but dropdown-only, no edge port
                            "model": {
                                "value": "gpt-4",
                                "type": "model",
                                "input_types": ["LanguageModel"],
                                "_input_type": "ModelInput",
                            },
                        },
                        "outputs": [{"name": "response", "types": ["Message"]}],
                    },
                },
            },
        ],
        "edges": [{"id": "e1", "source": "A", "target": "B"}],
    }


# enrich stub: mark that enrichment ran without needing an MCP session/schema fetch.
def _fake_enrich(edges, nodes):
    return [{**e, "sourceHandle": "œenrichedœ", "targetHandle": "œenrichedœ"} for e in edges]


def test_move_updates_position():
    data = apply_ops(_flow_data(), [{"op": "move", "id": "A", "position": {"x": 42, "y": 7}}], _fake_enrich)
    a = next(n for n in data["nodes"] if n["id"] == "A")
    assert a["position"] == {"x": 42, "y": 7}


def test_edit_field_sets_template_value():
    data = apply_ops(
        _flow_data(),
        [{"op": "edit_field", "id": "B", "key": "input_value", "value": "hello"}],
        _fake_enrich,
    )
    b = next(n for n in data["nodes"] if n["id"] == "B")
    assert b["data"]["node"]["template"]["input_value"]["value"] == "hello"


def test_edit_field_coerces_int():
    data = apply_ops(
        _flow_data(),
        [{"op": "edit_field", "id": "B", "key": "max_tokens", "value": "256"}],
        _fake_enrich,
    )
    b = next(n for n in data["nodes"] if n["id"] == "B")
    assert b["data"]["node"]["template"]["max_tokens"]["value"] == 256


def test_delete_edge():
    data = apply_ops(_flow_data(), [{"op": "delete_edge", "id": "e1"}], _fake_enrich)
    assert data["edges"] == []


def test_delete_node_drops_incident_edges():
    data = apply_ops(_flow_data(), [{"op": "delete_node", "id": "B"}], _fake_enrich)
    assert [n["id"] for n in data["nodes"]] == ["A"]
    assert data["edges"] == []  # e1 touched B


def test_add_edge_valid_target_enriched():
    data = apply_ops(
        _flow_data(),
        [{"op": "add_edge", "source": "A", "target": "B", "output": "message", "field": "input_value"}],
        _fake_enrich,
    )
    added = [e for e in data["edges"] if e["id"] != "e1"]
    assert len(added) == 1
    assert added[0]["source"] == "A" and added[0]["target"] == "B"
    assert added[0]["sourceHandle"] == "œenrichedœ"  # went through enrich


def test_add_edge_rejects_non_connectable_field():
    # 'plain' has no input_types → not a real drop target; edge must be dropped.
    data = apply_ops(
        _flow_data(),
        [{"op": "add_edge", "source": "A", "target": "B", "output": "message", "field": "plain"}],
        _fake_enrich,
    )
    assert len(data["edges"]) == 1  # only the original e1


def test_add_edge_rejects_modelinput_field():
    # 'model' is ModelInput (dropdown-only, no port) → edge must be dropped even though it
    # carries input_types — matches Langflow stripping model edges.
    data = apply_ops(
        _flow_data(),
        [{"op": "add_edge", "source": "A", "target": "B", "output": "message", "field": "model"}],
        _fake_enrich,
    )
    assert len(data["edges"]) == 1  # only the original e1


def test_does_not_mutate_input():
    original = _flow_data()
    snapshot = copy.deepcopy(original)
    apply_ops(original, [{"op": "delete_node", "id": "B"}], _fake_enrich)
    # apply_ops returns new lists; original list identity may differ but original content
    # for edges/nodes lists should be untouched at the top level.
    assert original["edges"] == snapshot["edges"]
