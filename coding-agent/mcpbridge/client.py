import os
from contextlib import AsyncExitStack
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
            "DOTENV_CONFIG_QUIET": "true",  # suppress dotenvx stdout banner (JSONRPC channel noise)
            "LOG_LEVEL": "error",  # suppress langflow-mcp startup info logs from stdout
        }
        self._session: ClientSession | None = None
        self._tools_cache: list[Any] = []
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command="node",
            args=[self._mcp_path],
            env=self._env,
        )
        try:
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
        except FileNotFoundError:
            raise RuntimeError("'node' binary not found. Install Node.js to run langflow-mcp.") from None
        except Exception as e:
            raise RuntimeError(f"Failed to start langflow-mcp server: {e}") from e

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
        if not result.content:
            return None
        item = result.content[0]
        if hasattr(item, "text"):
            return item.text
        return str(item)

    async def close(self) -> None:
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass  # suppress ExceptionGroup from JSONRPC parse errors on shutdown
            finally:
                self._exit_stack = None
                self._session = None
