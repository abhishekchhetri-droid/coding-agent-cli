"""Schema-driven validate_design / find_bridges tests.

Proves GENERALITY, not one flow: the same validator passes/flags correctly across differently
shaped graphs, and every fix-suggestion is READ FROM the fixture catalog (discovery, not
hardcoding). Pattern mirrors test_mcp_client.py — bare client + pre-seeded schema cache, no HTTP.
"""

from mcpbridge.client import LangflowMCPClient


# A deliberately small but varied catalog: vector store (Data out), dynamic-field prompt,
# chat in/out pair, a Data→Message bridge, a legacy component, a value-typed (empty input_types)
# field, and a tool-mode component + an Agent (tools consumer).
CATALOG = {
    "ChatInput": {
        "display_name": "Chat Input", "template": {},
        "outputs": [{"name": "message", "types": ["Message"]}],
    },
    "ChatOutput": {
        "display_name": "Chat Output",
        "template": {"input_value": {"type": "str", "required": True, "input_types": ["Message"]}},
        "outputs": [],
    },
    "Prompt": {
        "display_name": "Prompt", "template": {"template": {"type": "prompt"}},
        "outputs": [{"name": "prompt", "types": ["Message"]}],
    },
    "VectorStore": {
        "display_name": "Vector Store",
        "template": {"search_query": {"type": "str", "input_types": ["Message"]}},
        "outputs": [{"name": "search_results", "types": ["Data"]}],
    },
    "Parser": {
        "display_name": "Parser",
        "template": {"input_data": {"type": "other", "input_types": ["Data"]}},
        "outputs": [{"name": "parsed", "types": ["Message"]}],
    },
    "LegacyThing": {
        "display_name": "Legacy Thing", "legacy": True,
        "template": {"q": {"type": "str", "input_types": ["Message"]}},
        "outputs": [{"name": "out", "types": ["Message"]}],
    },
    "ValueField": {
        "display_name": "Value Field",
        "template": {
            "name": {"type": "str", "input_types": None},  # value-typed — not handle-typed
            "input_value": {"type": "str", "input_types": ["Message"]},
        },
        "outputs": [{"name": "out", "types": ["Message"]}],
    },
    "ToolComp": {
        "display_name": "Tool Comp",
        "template": {"expr": {"type": "str", "input_types": ["Message"]}},
        "outputs": [{"name": "component_as_tool", "types": ["Tool"], "tool_mode": True}],
    },
    "Agent": {
        "display_name": "Agent",
        "template": {
            "tools": {"type": "other", "input_types": ["Tool"]},
            "input_value": {"type": "str", "input_types": ["Message"]},
        },
        "outputs": [{"name": "response", "types": ["Message"]}],
    },
    "FileLoader": {
        "display_name": "File", "template": {},
        "outputs": [{"name": "data", "types": ["Data"]}],
    },
    # Required PURE-handle input (type "other"): genuinely must be wired.
    "Summarizer": {
        "display_name": "Summarizer",
        "template": {"data": {"type": "other", "required": True, "input_types": ["Data"]}},
        "outputs": [{"name": "summary", "types": ["Message"]}],
    },
    # Required CONFIG fields: value-capable str fields that also expose a Message handle
    # (the azure_endpoint / azure_deployment / api_key shape). Filled in the UI or injected
    # from settings — never a forced wire, never a feeder node.
    "CloudLLM": {
        "display_name": "Cloud LLM",
        "template": {
            "endpoint": {"type": "str", "required": True, "input_types": ["Message"]},
            "api_key": {"type": "str", "required": True, "password": True, "input_types": ["Message"]},
            "input_value": {"type": "str", "input_types": ["Message"]},
        },
        "outputs": [{"name": "text_output", "types": ["Message"]}],
    },
    # Stateful vector store: BOTH a data-ingest input and a query input (the FAISS/Chroma/Qdrant
    # signature). Splitting ingest and search across two instances → separate indexes → broken.
    "VStore": {
        "display_name": "Vector Store",
        "template": {
            "ingest_data": {"type": "other", "input_types": ["Data", "DataFrame", "Table"]},
            "search_query": {"type": "query", "input_types": ["Message"]},
            "embedding": {"type": "other", "input_types": ["Embeddings"]},
            "index_name": {"type": "str"},  # value-typed persistence key
        },
        "outputs": [{"name": "search_results", "types": ["Data"]}],
    },
}


def _client():
    c = LangflowMCPClient.__new__(LangflowMCPClient)
    c._component_schema_cache = dict(CATALOG)
    return c


def _node(nid, typ, template=None):
    n = {"id": nid, "type": typ}
    if template is not None:
        n["template"] = template
    return n


def _edge(src, out, tgt, field, out_types, in_types):
    return {
        "source": src, "target": tgt,
        "sourceHandle": {"id": src, "name": out, "output_types": out_types},
        "targetHandle": {"id": tgt, "fieldName": field, "inputTypes": in_types},
    }


# ---------------------------------------------------------------- regression: discovery
def test_incompatible_data_to_prompt_var_suggests_catalog_bridge():
    """The original bug: VectorStore.search_results [Data] → Prompt.{context} [Message] is
    incompatible. Exactly one violation, whose bridge suggestion is READ FROM the fixture
    catalog (Parser) — proving discovery, not a hardcoded name."""
    c = _client()
    nodes = [_node("VS-1", "VectorStore"), _node("P-1", "Prompt", template="ctx: {context}")]
    edges = [_edge("VS-1", "search_results", "P-1", "context", ["Data"], ["Message"])]
    v = c.validate_design(nodes, edges, node_templates={"P-1": "ctx: {context}"})
    assert len(v) == 1
    assert "Parser" in v[0] and "incompatible" in v[0]
    # The suggestion is the fixture's bridge — change the catalog and it would change too.
    assert "(Data→Message)" in v[0]


# ---------------------------------------------------------------- generality matrix
def test_valid_retrieval_flow_passes():
    c = _client()
    nodes = [
        _node("CI-1", "ChatInput"),
        _node("VS-1", "VectorStore"),
        _node("PR-1", "Parser"),
        _node("P-1", "Prompt", template="answer using {context}"),
        _node("CO-1", "ChatOutput"),
    ]
    edges = [
        _edge("CI-1", "message", "VS-1", "search_query", ["Message"], ["Message"]),
        _edge("VS-1", "search_results", "PR-1", "input_data", ["Data"], ["Data"]),
        _edge("PR-1", "parsed", "P-1", "context", ["Message"], ["Message"]),
        _edge("P-1", "prompt", "CO-1", "input_value", ["Message"], ["Message"]),
    ]
    assert c.validate_design(nodes, edges, node_templates={"P-1": "answer using {context}"}) == []


def test_nl_to_sql_style_text_chain_passes():
    """Message text flowing ChatInput → Prompt → ChatOutput — a different shape, same checker."""
    c = _client()
    nodes = [_node("CI-1", "ChatInput"), _node("P-1", "Prompt", template="q: {user}"),
             _node("CO-1", "ChatOutput")]
    edges = [
        _edge("CI-1", "message", "P-1", "user", ["Message"], ["Message"]),
        _edge("P-1", "prompt", "CO-1", "input_value", ["Message"], ["Message"]),
    ]
    assert c.validate_design(nodes, edges, node_templates={"P-1": "q: {user}"}) == []


def test_agent_tools_edge_is_skipped():
    """tools edges (rewritten schema-driven by enrich_edges) skip output/compat checks."""
    c = _client()
    nodes = [_node("T-1", "ToolComp"), _node("A-1", "Agent"), _node("CI-1", "ChatInput")]
    edges = [
        # deliberately "wrong" source output name on the tool edge — must NOT be flagged
        _edge("T-1", "bogus_name", "A-1", "tools", ["Tool"], ["Tool"]),
        _edge("CI-1", "message", "A-1", "input_value", ["Message"], ["Message"]),
    ]
    assert c.validate_design(nodes, edges) == []


# ---------------------------------------------------------------- edge cases
def test_var_field_accepted():
    c = _client()
    nodes = [_node("CI-1", "ChatInput"), _node("P-1", "Prompt", template="{q}")]
    edges = [_edge("CI-1", "message", "P-1", "q", ["Message"], ["Message"])]
    assert c.validate_design(nodes, edges, node_templates={"P-1": "{q}"}) == []


def test_empty_input_types_field_skips_compat():
    """A value-typed field (input_types None) is not handle-typed — compat is skipped even
    when the source output type would not match."""
    c = _client()
    nodes = [_node("VS-1", "VectorStore"), _node("VF-1", "ValueField")]
    edges = [_edge("VS-1", "search_results", "VF-1", "name", ["Data"], [])]
    assert c.validate_design(nodes, edges) == []


def test_unknown_output_lists_real_outputs():
    c = _client()
    nodes = [_node("VS-1", "VectorStore"), _node("P-1", "Prompt", template="{c}")]
    edges = [_edge("VS-1", "no_such_output", "P-1", "c", ["Data"], ["Message"])]
    v = c.validate_design(nodes, edges, node_templates={"P-1": "{c}"})
    assert len(v) == 1 and "no_such_output" in v[0] and "search_results" in v[0]


def test_legacy_node_flagged():
    c = _client()
    v = c.validate_design([_node("L-1", "LegacyThing")], [])
    assert any("legacy" in x for x in v)


def test_unknown_component_flagged():
    c = _client()
    v = c.validate_design([_node("X-1", "NotARealComponent")], [])
    assert any("unknown component type" in x for x in v)


def test_nonexistent_node_id_flagged():
    c = _client()
    nodes = [_node("CI-1", "ChatInput")]
    edges = [_edge("CI-1", "message", "Ghost-1", "input_value", ["Message"], ["Message"])]
    v = c.validate_design(nodes, edges)
    assert any("Ghost-1" in x and "unknown target" in x for x in v)


def test_required_pure_handle_unwired_flagged():
    """A required type-"other" input is a pure handle — it MUST be wired; with no incoming
    edge it's flagged (generalizes the dead-end/severed post-build check to design time)."""
    c = _client()
    v = c.validate_design([_node("S-1", "Summarizer")], [])
    assert any("data" in x and "no incoming edge" in x for x in v)


def test_required_input_satisfied_by_edge():
    c = _client()
    nodes = [_node("F-1", "FileLoader"), _node("S-1", "Summarizer")]
    edges = [_edge("F-1", "data", "S-1", "data", ["Data"], ["Data"])]
    assert c.validate_design(nodes, edges) == []


def test_required_config_fields_left_empty_pass():
    """The TextInput-feeder regression: required value-capable config fields (endpoint,
    api_key — str fields that also expose a Message handle) are NOT forced wires. They stay
    empty at design time (user fills them in the UI / settings inject them), so an unwired
    CloudLLM produces no violations — and the designer has no reason to invent feeder nodes.
    Mirrors the post-build audit's pure-handle vs credential/literal distinction."""
    c = _client()
    nodes = [_node("CI-1", "ChatInput"), _node("LLM-1", "CloudLLM")]
    edges = [_edge("CI-1", "message", "LLM-1", "input_value", ["Message"], ["Message"])]
    assert c.validate_design(nodes, edges) == []


def test_notenode_skipped():
    c = _client()
    nodes = [_node("CI-1", "ChatInput"), {"id": "note-1", "type": "noteNode"},
             _node("CO-1", "ChatOutput")]
    edges = [_edge("CI-1", "message", "CO-1", "input_value", ["Message"], ["Message"])]
    assert c.validate_design(nodes, edges) == []


# ---------------------------------------------------------------- duplicate stateful stores
def _store_node(nid, index_name=None):
    """A VStore node; when index_name is given, shaped like an enriched node carrying a
    value-typed config (the persistence key)."""
    n = _node(nid, "VStore")
    if index_name is not None:
        n["data"] = {"type": "VStore", "node": {"template": {
            "index_name": {"type": "str", "value": index_name},
        }}}
    return n


def test_duplicate_stateful_store_split_flagged():
    """The over-decomposition bug: designer splits ingest and search across TWO instances of
    the same store. Each keeps its own index, so ingested data is invisible to the search
    node. One violation naming both ids, the merge fix, and the schema-discovered field roles."""
    c = _client()
    nodes = [
        _node("CI-1", "ChatInput"),
        _store_node("VS-ingest"), _store_node("VS-search"),
        _node("F-1", "FileLoader"),
    ]
    edges = [
        # ingest side wired into one instance, query side into the other
        _edge("F-1", "data", "VS-ingest", "ingest_data", ["Data"], ["Data"]),
        _edge("CI-1", "message", "VS-search", "search_query", ["Message"], ["Message"]),
    ]
    v = c.validate_design(nodes, edges)
    dup = [x for x in v if "VS-ingest" in x and "VS-search" in x]
    assert len(dup) == 1
    assert "merge" in dup[0].lower()
    # field roles are read from the schema, not hardcoded
    assert "ingest_data" in dup[0] and "search_query" in dup[0]


def test_single_store_wiring_both_passes():
    """The always-valid fix: ONE store node carrying both ingest and query edges."""
    c = _client()
    nodes = [_node("CI-1", "ChatInput"), _store_node("VS-1"), _node("F-1", "FileLoader")]
    edges = [
        _edge("F-1", "data", "VS-1", "ingest_data", ["Data"], ["Data"]),
        _edge("CI-1", "message", "VS-1", "search_query", ["Message"], ["Message"]),
    ]
    assert [x for x in c.validate_design(nodes, edges) if "VS-1" in x] == []


def test_duplicate_stores_with_distinct_persistence_pass():
    """Two instances pointing at DIFFERENT collections (distinct value-typed config) are a
    legitimate multi-store flow — not flagged."""
    c = _client()
    nodes = [_store_node("VS-a", index_name="policies"), _store_node("VS-b", index_name="faqs")]
    assert [x for x in c.validate_design(nodes, []) if "duplicate" in x.lower()] == []


def test_duplicate_stateless_components_pass():
    """Duplicating a stateless component (Parser) is normal — never flagged."""
    c = _client()
    nodes = [_node("PR-1", "Parser"), _node("PR-2", "Parser")]
    assert [x for x in c.validate_design(nodes, []) if "duplicate" in x.lower()] == []


def test_duplicate_agents_pass():
    """Shape-predicate guard: Agent has a Message input + Message output + a Tool input, but is
    NOT a stateful store — multi-agent flows must not be flagged."""
    c = _client()
    nodes = [_node("A-1", "Agent"), _node("A-2", "Agent")]
    v = c.validate_design(nodes, [])
    assert [x for x in v if "duplicate" in x.lower()] == []


# ---------------------------------------------------------------- find_bridges
def test_find_bridges_returns_only_nonlegacy_matches():
    c = _client()
    bridges = c.find_bridges({"Data"}, {"Message"})
    types = {b["type"] for b in bridges}
    assert "Parser" in types
    assert "LegacyThing" not in types  # legacy excluded


def test_find_bridges_empty_when_no_match():
    c = _client()
    # No component bridges Tool → Data in the fixture.
    assert c.find_bridges({"Tool"}, {"Data"}) == []


def test_find_bridges_respects_limit():
    c = _client()
    # Message → Message: many components qualify; cap at 1.
    assert len(c.find_bridges({"Message"}, {"Message"}, limit=1)) == 1
