import gzip
import json
import os
from contextlib import AsyncExitStack
from typing import Any

import urllib.request

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class LangflowMCPClient:
    def __init__(self, mcp_path: str, langflow_api_key: str, langflow_base_url: str) -> None:
        self._mcp_path = mcp_path
        self._langflow_api_key = langflow_api_key
        self._langflow_base_url = langflow_base_url.rstrip("/")
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
        self._component_schema_cache: dict[str, Any] = {}  # type_name → full schema

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

    def _fetch_component_schemas(self) -> dict[str, Any]:
        """Fetch all component schemas from /api/v1/all. Cached after first call."""
        if self._component_schema_cache:
            return self._component_schema_cache
        url = f"{self._langflow_base_url}/api/v1/all"
        req = urllib.request.Request(url, headers={"x-api-key": self._langflow_api_key, "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            data = json.loads(raw)
        # Flatten category → {type_name: schema} into {type_name: schema}
        flat: dict[str, Any] = {}
        for _category, components in data.items():
            if isinstance(components, dict):
                flat.update(components)
        self._component_schema_cache = flat
        return flat

    def enrich_nodes(self, nodes: list[dict]) -> list[dict]:
        """Inject data.node (full Langflow component schema) for nodes that are missing it."""
        schemas = self._fetch_component_schemas()
        enriched = []
        for node in nodes:
            node = dict(node)
            node_type = node.get("type") or node.get("data", {}).get("type", "")
            data = dict(node.get("data", {}))
            if "node" not in data and node_type in schemas:
                data["node"] = schemas[node_type]
            # Ensure data.type and data.id are always set
            data.setdefault("type", node_type)
            data.setdefault("id", node.get("id", ""))
            node["data"] = data
            enriched.append(node)
        return enriched

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
