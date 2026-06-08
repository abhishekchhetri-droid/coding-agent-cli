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

    client._discovered = []
    schemas = client.get_tool_schemas()
    # baseline-matched cached tool + always-on virtual tools (search_tools, etc.)
    converted = next(s for s in schemas if s["function"]["name"] == "list_flows")
    assert converted["type"] == "function"
    assert "properties" in converted["function"]["parameters"]
    assert any(s["function"]["name"] == "search_tools" for s in schemas)


def _mk_tool(name, desc):
    t = MagicMock()
    t.name = name
    t.description = desc
    t.inputSchema = {"type": "object", "properties": {}}
    return t


@pytest.mark.asyncio
async def test_search_tools_ranks_name_match_above_description_and_activates():
    from mcpbridge.client import LangflowMCPClient
    client = LangflowMCPClient.__new__(LangflowMCPClient)
    client._session = object()
    client._discovered = []
    client._tools_cache = [
        _mk_tool("create_variable", "Create a global variable"),
        _mk_tool("list_flows", "List all flows; mentions variable in passing"),
        _mk_tool("delete_variable", "Remove a variable by name"),
    ]
    import json as _json
    out = _json.loads(await client.call_tool("search_tools", {"query": "variable"}))
    names = [m["name"] for m in out["matches"]]
    # name-matches rank ahead of description-only match (list_flows)
    assert names[0] in ("create_variable", "delete_variable")
    assert names[-1] == "list_flows"
    # matched non-baseline tools become active for next turn
    assert "create_variable" in client._discovered
    schemas = client.get_tool_schemas()
    assert any(s["function"]["name"] == "create_variable" for s in schemas)


@pytest.mark.asyncio
async def test_search_tools_empty_match_returns_hint():
    from mcpbridge.client import LangflowMCPClient
    client = LangflowMCPClient.__new__(LangflowMCPClient)
    client._session = object()
    client._discovered = []
    client._tools_cache = [_mk_tool("list_flows", "List all flows")]
    import json as _json
    out = _json.loads(await client.call_tool("search_tools", {"query": "zzznope"}))
    assert out["matches"] == []
    assert "note" in out


@pytest.mark.asyncio
async def test_search_tools_fifo_cap_bounds_active_set():
    from mcpbridge.client import LangflowMCPClient
    client = LangflowMCPClient.__new__(LangflowMCPClient)
    client._session = object()
    client._discovered = []
    client._tools_cache = [_mk_tool(f"tool_kb_{i}", "knowledge base op") for i in range(40)]
    import json as _json
    await client.call_tool("search_tools", {"query": "knowledge", "limit": 25})
    assert len(client._discovered) <= LangflowMCPClient._DISCOVERY_CAP


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


def test_offset_new_positions_places_additions_right_of_existing():
    from mcpbridge.client import LangflowMCPClient
    existing = [
        {"id": "a", "position": {"x": 100, "y": 100}},
        {"id": "b", "position": {"x": 200, "y": 400}},
    ]
    additions = [{"id": "n1"}, {"id": "n2"}]
    LangflowMCPClient.offset_new_positions(existing, additions, x_gap=350, y_gap=200)
    # rightmost x + 350, vertical stack starting at topmost existing y
    assert additions[0]["position"] == {"x": 550, "y": 100}
    assert additions[1]["position"] == {"x": 550, "y": 300}


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


def _client_with_schemas(schemas):
    """Build a bare client whose component-schema cache is pre-seeded (no HTTP)."""
    from mcpbridge.client import LangflowMCPClient
    c = LangflowMCPClient.__new__(LangflowMCPClient)
    c._component_schema_cache = schemas
    return c


def test_get_component_schema_surfaces_legacy_flag():
    """get_component_schema flags legacy components (schema-driven) and omits the key when modern."""
    schemas = {
        "SQLGenerator": {"legacy": True, "template": {}, "outputs": []},
        "SQLComponent": {"legacy": False, "template": {}, "outputs": []},
        "BetaThing": {"beta": True, "template": {}, "outputs": []},
    }
    c = _client_with_schemas(schemas)
    assert c.get_component_schema("SQLGenerator")["legacy"] is True
    assert "legacy" not in c.get_component_schema("SQLComponent")  # modern → key omitted, stays compact
    assert c.get_component_schema("BetaThing").get("beta") is True


def test_is_legacy_drives_hard_block():
    """is_legacy reads the live legacy flag; unknown types are treated as non-legacy."""
    c = _client_with_schemas({"SQLGenerator": {"legacy": True}, "SQLComponent": {"legacy": False}})
    assert c.is_legacy("SQLGenerator") is True
    assert c.is_legacy("SQLComponent") is False
    assert c.is_legacy("DoesNotExist") is False


def _prompt_node(node_id="Prompt-1", value=""):
    """Freshly-enriched Prompt node: bare schema, prompt field present but no var handles."""
    return {
        "id": node_id,
        "type": "genericNode",
        "data": {
            "type": "Prompt",
            "id": node_id,
            "node": {
                "template": {
                    "_type": "Component",
                    "template": {"type": "prompt", "value": value, "name": "template"},
                },
                "custom_fields": {},
                "outputs": [],
            },
        },
    }


def _endpoint_frontend_node(template_str, vars_):
    """Mimic /api/v1/validate/prompt: returns var fields + custom_fields, value left blank."""
    tmpl = {"_type": "Component", "template": {"type": "prompt", "value": "", "name": "template"}}
    for v in vars_:
        tmpl[v] = {"name": v, "type": "str", "input_types": ["Message"], "value": "", "show": True}
    return {"template": tmpl, "custom_fields": {"template": list(vars_)}}


def test_apply_prompt_fields_materializes_var_handles_via_endpoint():
    """Endpoint path: var fields + custom_fields spliced in, prompt value restored.
    Without these the {var} input handles don't exist and inbound edges get stripped."""
    from mcpbridge.client import LangflowMCPClient
    c = LangflowMCPClient.__new__(LangflowMCPClient)
    tpl = "Q {question} M {metadata}"
    c._validate_prompt = lambda fn, ts, node: _endpoint_frontend_node(ts, ["question", "metadata"])

    node = _prompt_node()
    c.apply_prompt_fields([node], tpl)

    schema = node["data"]["node"]
    assert schema["template"]["template"]["value"] == tpl  # value restored (endpoint blanks it)
    assert "question" in schema["template"] and "metadata" in schema["template"]
    assert schema["template"]["question"]["input_types"] == ["Message"]
    assert schema["custom_fields"]["template"] == ["question", "metadata"]


def test_apply_prompt_fields_falls_back_to_local_injection_when_endpoint_down():
    """Offline fallback: vars parsed from the template and DefaultPromptField entries injected."""
    from mcpbridge.client import LangflowMCPClient
    c = LangflowMCPClient.__new__(LangflowMCPClient)
    c._validate_prompt = lambda fn, ts, node: None  # endpoint unreachable

    node = _prompt_node(value="Hello {name}, ask {topic}")
    c.apply_prompt_fields([node])  # no design template — uses node's own value

    schema = node["data"]["node"]
    assert schema["template"]["template"]["value"] == "Hello {name}, ask {topic}"
    assert schema["template"]["name"]["_input_type"] == "DefaultPromptField"
    assert schema["custom_fields"]["template"] == ["name", "topic"]


def test_apply_prompt_fields_prefers_node_value_over_design_template():
    """A value already on the node (LLM/clone authored) wins over the design-level template."""
    from mcpbridge.client import LangflowMCPClient
    c = LangflowMCPClient.__new__(LangflowMCPClient)
    captured = {}
    def _vp(fn, ts, node):
        captured["template_str"] = ts
        return _endpoint_frontend_node(ts, ["own"])
    c._validate_prompt = _vp

    node = _prompt_node(value="node {own}")
    c.apply_prompt_fields([node], "design {other}")
    assert captured["template_str"] == "node {own}"


def test_apply_prompt_fields_noops_without_prompt_field_or_template():
    """Non-prompt nodes and empty prompts are left untouched (no spurious var fields)."""
    from mcpbridge.client import LangflowMCPClient
    c = LangflowMCPClient.__new__(LangflowMCPClient)
    c._validate_prompt = lambda *a: (_ for _ in ()).throw(AssertionError("should not be called"))

    plain = {"id": "T1", "data": {"type": "TextInput", "node": {"template": {"input_value": {"type": "str"}}}}}
    empty_prompt = _prompt_node(value="")  # no value, no design template
    c.apply_prompt_fields([plain, empty_prompt])  # must not call endpoint or raise
    assert "custom_fields" not in plain["data"]["node"] or not plain["data"]["node"]["custom_fields"]
