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
