import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_cmd_flow_list_calls_mcp():
    from agent.cmd import run_cmd

    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='[{"id": "abc", "name": "test"}]')

    await run_cmd(["flow", "list"], mock_mcp, pretty=False)
    mock_mcp.call_tool.assert_called_once_with("list_flows", {"page": 1, "limit": 20})


@pytest.mark.asyncio
async def test_cmd_flow_get_calls_mcp():
    from agent.cmd import run_cmd

    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"id": "abc"}')

    await run_cmd(["flow", "get", "abc-123"], mock_mcp, pretty=False)
    mock_mcp.call_tool.assert_called_once_with("get_flow", {"flow_id": "abc-123"})


@pytest.mark.asyncio
async def test_cmd_health_calls_mcp():
    from agent.cmd import run_cmd

    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"status": "ok"}')

    await run_cmd(["health"], mock_mcp, pretty=False)
    mock_mcp.call_tool.assert_called_once_with("health_check", {})


@pytest.mark.asyncio
async def test_cmd_unknown_raises():
    from agent.cmd import run_cmd, CmdError

    mock_mcp = MagicMock()
    with pytest.raises(CmdError):
        await run_cmd(["flow", "explode"], mock_mcp, pretty=False)
