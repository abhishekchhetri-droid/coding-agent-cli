import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_client_converts_mcp_tools_to_openai_format():
    """MCP tool schemas must be wrapped in {type: function, function: {...}} for OpenAI."""
    from mcpbridge.client import LangflowMCPClient

    mock_tool = MagicMock()
    mock_tool.name = "list_flows"
    mock_tool.description = "List all flows"
    mock_tool.inputSchema = {"type": "object", "properties": {"page": {"type": "integer"}}}

    client = LangflowMCPClient.__new__(LangflowMCPClient)
    client._tools_cache = [mock_tool]

    schemas = client.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "list_flows"
    assert "properties" in schemas[0]["function"]["parameters"]


@pytest.mark.asyncio
async def test_client_raises_on_call_before_connect():
    from mcpbridge.client import LangflowMCPClient
    client = LangflowMCPClient.__new__(LangflowMCPClient)
    client._session = None
    client._tools_cache = []

    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("list_flows", {})


# ─────────────────────────────────────────────────────────────────────
# Update-flow merge helpers (LangflowMCPClient static methods)
# ─────────────────────────────────────────────────────────────────────


def _langmodel_node(node_id: str, type_name: str = "AzureOpenAIModel") -> dict:
    return {
        "id": node_id,
        "data": {
            "type": type_name,
            "node": {"outputs": [{"name": "model_output", "types": ["LanguageModel"]}]},
        },
    }


def _agent_node(node_id: str) -> dict:
    return {
        "id": node_id,
        "data": {
            "type": "Agent",
            "node": {"template": {"tools": {}, "input_value": {}}},
        },
    }


def test_find_node_by_type_returns_match_or_none():
    from mcpbridge.client import LangflowMCPClient
    nodes = [{"id": "a", "data": {"type": "ChatInput"}}, {"id": "b", "data": {"type": "Agent"}}]
    assert LangflowMCPClient.find_node_by_type(nodes, "Agent")["id"] == "b"
    assert LangflowMCPClient.find_node_by_type(nodes, "Missing") is None


def test_find_llm_node_matches_by_output_type_not_name():
    from mcpbridge.client import LangflowMCPClient
    nodes = [
        {"id": "chat", "data": {"type": "ChatInput", "node": {"outputs": [{"name": "message", "types": ["Message"]}]}}},
        _langmodel_node("llm-x", type_name="LanguageModelComponent"),
    ]
    found = LangflowMCPClient.find_llm_node(nodes)
    assert found["id"] == "llm-x"


def test_find_llm_node_prefers_azure_when_multiple():
    from mcpbridge.client import LangflowMCPClient
    nodes = [
        _langmodel_node("openai-1", type_name="OpenAIModel"),
        _langmodel_node("azure-7", type_name="AzureOpenAIModel"),
    ]
    assert LangflowMCPClient.find_llm_node(nodes)["id"] == "azure-7"


def test_find_llm_node_returns_none_when_absent():
    from mcpbridge.client import LangflowMCPClient
    nodes = [{"id": "chat", "data": {"type": "ChatInput", "node": {"outputs": []}}}]
    assert LangflowMCPClient.find_llm_node(nodes) is None


def test_find_agent_node_matches_by_tools_template_field():
    from mcpbridge.client import LangflowMCPClient
    custom_agent = {
        "id": "ta-1",
        "data": {"type": "ToolCallingAgent", "node": {"template": {"tools": {}, "model": {}}}},
    }
    nodes = [{"id": "x", "data": {"type": "ChatInput", "node": {"template": {"input_value": {}}}}}, custom_agent]
    assert LangflowMCPClient.find_agent_node(nodes)["id"] == "ta-1"


def test_find_agent_node_returns_none_when_no_tools_field():
    from mcpbridge.client import LangflowMCPClient
    nodes = [{"id": "x", "data": {"type": "ChatInput", "node": {"template": {"input_value": {}}}}}]
    assert LangflowMCPClient.find_agent_node(nodes) is None


def test_offset_new_positions_places_additions_below_existing():
    from mcpbridge.client import LangflowMCPClient
    existing = [
        {"id": "a", "position": {"x": 100, "y": 100}},
        {"id": "b", "position": {"x": 200, "y": 400}},
    ]
    additions = [{"id": "n1"}, {"id": "n2"}]
    LangflowMCPClient.offset_new_positions(existing, additions, y_gap=200)
    assert additions[0]["position"]["y"] == 600
    assert additions[1]["position"]["y"] == 800


def test_offset_new_positions_preserves_explicit_positions():
    from mcpbridge.client import LangflowMCPClient
    existing = [{"id": "a", "position": {"x": 0, "y": 0}}]
    additions = [{"id": "n1", "position": {"x": 999, "y": 999}}, {"id": "n2"}]
    LangflowMCPClient.offset_new_positions(existing, additions)
    assert additions[0]["position"] == {"x": 999, "y": 999}
    assert "position" in additions[1]


def test_classify_update_payload_patch_meta_when_no_nodes():
    from mcpbridge.client import LangflowMCPClient
    assert LangflowMCPClient.classify_update_payload(None, {"a"}) == "patch_meta"
    assert LangflowMCPClient.classify_update_payload({"description": "x"}, {"a"}) == "patch_meta"


def test_classify_update_payload_full_replace_when_all_existing_ids_present():
    from mcpbridge.client import LangflowMCPClient
    payload = {"nodes": [{"id": "a"}, {"id": "b"}, {"id": "c-new"}]}
    assert LangflowMCPClient.classify_update_payload(payload, {"a", "b"}) == "full_replace"


def test_classify_update_payload_merge_when_delta_only():
    from mcpbridge.client import LangflowMCPClient
    payload = {"nodes": [{"id": "c-new"}]}
    assert LangflowMCPClient.classify_update_payload(payload, {"a", "b"}) == "merge"


def test_merge_flow_data_appends_only_new_entries():
    from mcpbridge.client import LangflowMCPClient
    existing = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"id": "e1", "source": "a", "target": "b"}],
    }
    payload = {
        "nodes": [{"id": "b"}, {"id": "c"}],
        "edges": [{"id": "e1"}, {"id": "e2", "source": "b", "target": "c"}],
    }
    merged = LangflowMCPClient.merge_flow_data(existing, payload)
    assert [n["id"] for n in merged["nodes"]] == ["a", "b", "c"]
    assert [e["id"] for e in merged["edges"]] == ["e1", "e2"]
