import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import designer


def _spec_tool_call(nodes, edges, summary="pipeline", template="", varlist=None):
    return {
        "id": "tc1",
        "name": "submit_design",
        "arguments": {
            "summary": summary,
            "nodes": nodes,
            "edges": edges,
            "prompt_template": template,
            "vars": varlist or [],
        },
    }


@pytest.mark.asyncio
async def test_design_flow_returns_compact_spec_contract():
    """The sub-agent returns ONLY the spec contract — no full schemas leak to main thread."""
    llm = MagicMock()
    nodes = [{"id": "Prompt-1", "type": "Prompt"}, {"id": "ChatOutput-1", "type": "ChatOutput"}]
    edges = [{"source": "Prompt-1", "target": "ChatOutput-1",
              "sourceHandle": {"name": "prompt"}, "targetHandle": {"fieldName": "input_value"}}]
    llm.complete = AsyncMock(return_value={
        "content": "", "tool_calls": [_spec_tool_call(nodes, edges, template="{q}", varlist=["q"])],
    })
    mcp = MagicMock()

    spec = await designer.design_flow("build X", mcp, llm)

    assert set(spec.keys()) == {"summary", "nodes", "edges", "prompt_template", "vars"}
    assert spec["nodes"] == nodes and spec["edges"] == edges
    assert spec["vars"] == ["q"]
    llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_design_flow_fetches_schema_then_submits():
    """First turn fetches a schema (legacy-aware), second turn submits — schema stays in sub-context."""
    llm = MagicMock()
    nodes = [{"id": "SQLComponent-1", "type": "SQLComponent"}]
    llm.complete = AsyncMock(side_effect=[
        {"content": "", "tool_calls": [
            {"id": "s1", "name": "get_component_schema", "arguments": {"type_name": "SQLComponent"}}]},
        {"content": "", "tool_calls": [_spec_tool_call(nodes, [])]},
    ])
    mcp = MagicMock()
    mcp.get_component_schema = MagicMock(return_value={"type": "SQLComponent", "legacy": False, "inputs": []})

    spec = await designer.design_flow("run sql", mcp, llm)

    mcp.get_component_schema.assert_called_once_with("SQLComponent")
    assert spec["nodes"] == nodes
    assert llm.complete.await_count == 2


@pytest.mark.asyncio
async def test_design_flow_gives_up_after_budget():
    """If the sub-agent never submits, it returns an error rather than looping forever."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": "thinking", "tool_calls": []})
    spec = await designer.design_flow("x", MagicMock(), llm)
    assert "error" in spec


@pytest.mark.asyncio
async def test_design_flow_rejects_then_accepts_on_resubmit():
    """An invalid submit yields a design_invalid tool-result; the corrected resubmit is accepted."""
    llm = MagicMock()
    nodes = [{"id": "P-1", "type": "Prompt"}]
    llm.complete = AsyncMock(side_effect=[
        {"content": "", "tool_calls": [_spec_tool_call(nodes, [])]},
        {"content": "", "tool_calls": [_spec_tool_call(nodes, [])]},
    ])
    mcp = MagicMock()
    mcp.validate_design = MagicMock(side_effect=[["edge X: incompatible"], []])

    spec = await designer.design_flow("build", mcp, llm)

    assert "violations" not in spec  # final accepted spec is clean
    assert spec["nodes"] == nodes
    assert llm.complete.await_count == 2
    assert mcp.validate_design.call_count == 2
    # The rejection was fed back as a design_invalid tool result (pairing intact).


@pytest.mark.asyncio
async def test_design_flow_attaches_violations_when_retries_exhausted():
    """After _MAX_VALIDATION_RETRIES, return the spec WITH violations (human gate decides)."""
    llm = MagicMock()
    nodes = [{"id": "P-1", "type": "Prompt"}]
    llm.complete = AsyncMock(return_value={"content": "", "tool_calls": [_spec_tool_call(nodes, [])]})
    mcp = MagicMock()
    mcp.validate_design = MagicMock(return_value=["bad edge"])

    spec = await designer.design_flow("x", mcp, llm)

    assert spec["violations"] == ["bad edge"]
    # 1 initial submit + _MAX_VALIDATION_RETRIES rejections before giving up.
    assert mcp.validate_design.call_count == designer._MAX_VALIDATION_RETRIES + 1


@pytest.mark.asyncio
async def test_design_flow_magicmock_validate_treated_as_valid():
    """Existing tests pass an un-stubbed MagicMock mcp — non-list validate result → treated []."""
    llm = MagicMock()
    nodes = [{"id": "P-1", "type": "Prompt"}]
    llm.complete = AsyncMock(return_value={"content": "", "tool_calls": [_spec_tool_call(nodes, [])]})
    spec = await designer.design_flow("x", MagicMock(), llm)
    assert spec["nodes"] == nodes and "violations" not in spec


def test_render_design_shows_unresolved_violations():
    spec = {
        "summary": "s", "nodes": [{"type": "Prompt"}], "edges": [],
        "violations": ["edge A→B: incompatible — compatible bridges in catalog: Parser (Data→Message)"],
    }
    out = designer.render_design(spec)
    assert "Unresolved validation warnings" in out and "Parser" in out


def test_render_design_shows_stages_and_vars():
    spec = {
        "summary": "Prompt to SQL to output",
        "nodes": [{"type": "Prompt"}, {"type": "AzureOpenAIModel"}, {"type": "SQLComponent"}],
        "edges": [{"source": "Prompt-1", "target": "AzureOpenAIModel-1",
                   "sourceHandle": {"name": "prompt"}, "targetHandle": {"fieldName": "input_value"}}],
        "prompt_template": "{metadata} {question}",
        "vars": ["metadata", "question"],
    }
    out = designer.render_design(spec)
    assert "Prompt" in out and "SQLComponent" in out
    assert "Prompt-1`.prompt → `AzureOpenAIModel-1`.input_value" in out
    assert "{metadata}" in out
