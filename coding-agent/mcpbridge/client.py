import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class LangflowMCPClient:
    def __init__(self, mcp_path: str, langflow_api_key: str, langflow_base_url: str) -> None:
        self._mcp_path = mcp_path
        self._env = {
            **os.environ,
            "LANGFLOW_API_KEY": langflow_api_key,
            "LANGFLOW_BASE_URL": langflow_base_url,
        }
        self._session: ClientSession | None = None
        self._tools_cache: list[Any] = []
        self._exit_stack = None

    async def connect(self) -> None:
        from contextlib import AsyncExitStack
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command="node",
            args=[self._mcp_path],
            env=self._env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tools_cache = result.tools

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in self._tools_cache
        ]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if self._session is None:
            raise RuntimeError("MCP client not connected. Call connect() first.")
        result = await self._session.call_tool(name, arguments)
        if result.content:
            return result.content[0].text if hasattr(result.content[0], "text") else result.content
        return None

    async def close(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
