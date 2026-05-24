import gzip
import json
import os
from contextlib import AsyncExitStack
from typing import Any

import urllib.request

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _extract_text(obj: Any, depth: int = 0) -> str:
    """Walk Langflow run response tree to find the first non-trivial text value."""
    if depth > 10:
        return ""
    if isinstance(obj, dict):
        for k in ("text", "message"):
            v = obj.get(k)
            if isinstance(v, str) and len(v) > 2:
                return v
        for v in obj.values():
            r = _extract_text(v, depth + 1)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _extract_text(item, depth + 1)
            if r:
                return r
    return ""


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

    def enrich_nodes(
        self,
        nodes: list[dict],
        credential_overrides: dict[str, dict[str, str]] | None = None,
    ) -> list[dict]:
        """Inject data.node (full Langflow component schema) and apply credential overrides."""
        schemas = self._fetch_component_schemas()
        enriched = []
        for node in nodes:
            node = dict(node)
            node_type = node.get("type") or node.get("data", {}).get("type", "")
            data = dict(node.get("data", {}))
            if "node" not in data and node_type in schemas:
                data["node"] = json.loads(json.dumps(schemas[node_type]))  # deep copy
            # Inject credentials into template fields
            if credential_overrides and node_type in credential_overrides and "node" in data:
                tmpl = data["node"].get("template", {})
                for field_name, value in credential_overrides[node_type].items():
                    if field_name in tmpl:
                        tmpl[field_name]["value"] = value
            # Ensure data.type and data.id are always set
            data.setdefault("type", node_type)
            data.setdefault("id", node.get("id", ""))
            node["data"] = data
            # React-flow requires type="genericNode" for Langflow's custom node renderer.
            # The component type lives in data.type; top-level type is only for the canvas.
            node["type"] = "genericNode"
            enriched.append(node)
        return enriched

    def enrich_edges(self, edges: list[dict], nodes: list[dict]) -> list[dict]:
        """Serialize edge handles as JSON strings, add IDs, and fix targetHandle.type
        by looking up the actual field type from the component schema.
        React-flow requires handles to be JSON strings; wrong type causes edge rejection.
        """
        schemas = self._fetch_component_schemas()
        # Build node_id → component_type map from the (already enriched) nodes
        node_type_map: dict[str, str] = {}
        for node in nodes:
            nid = node.get("id", "")
            comp_type = node.get("data", {}).get("type") or node.get("type", "")
            if nid and comp_type and comp_type != "genericNode":
                node_type_map[nid] = comp_type

        result = []
        for i, edge in enumerate(edges):
            edge = dict(edge)
            sh = edge.get("sourceHandle", {})
            th = edge.get("targetHandle", {})

            # Fix targetHandle.type from schema
            if isinstance(th, dict):
                th = dict(th)
                tgt_node_id = edge.get("target", "")
                tgt_comp_type = node_type_map.get(tgt_node_id, "")
                field_name = th.get("fieldName", "")
                if tgt_comp_type and field_name and tgt_comp_type in schemas:
                    tmpl_field = schemas[tgt_comp_type].get("template", {}).get(field_name, {})
                    actual_type = tmpl_field.get("type")
                    if actual_type:
                        th["type"] = actual_type

            # Serialize handles using Langflow's œ-encoding (frontend np() function:
            # JSON.stringify(obj).replace(/"/g, "œ")) — sun() validation requires this format.
            if isinstance(sh, dict):
                edge["sourceHandle"] = json.dumps(sh).replace('"', 'œ')
            if isinstance(th, dict):
                edge["targetHandle"] = json.dumps(th).replace('"', 'œ')

            # Ensure unique edge ID
            if "id" not in edge:
                edge["id"] = f"{edge.get('source', 'src')}-{edge.get('target', 'tgt')}-{i}"

            # Keep parsed handles in data for Langflow backend
            data = dict(edge.get("data", {}))
            if isinstance(sh, dict):
                data.setdefault("sourceHandle", sh)
            if isinstance(th, dict):
                data.setdefault("targetHandle", th)
            edge["data"] = data

            result.append(edge)
        return result

    def test_run_flow(self, flow_id: str, input_value: str = "2+2") -> dict:
        """POST to /api/v1/run/{flow_id} with dummy input. Returns {ok, answer, error}."""
        url = f"{self._langflow_base_url}/api/v1/run/{flow_id}"
        payload = json.dumps({
            "input_value": input_value,
            "input_type": "chat",
            "output_type": "chat",
        }).encode()
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"x-api-key": self._langflow_api_key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            # An empty outputs list means flow has no ChatOutput node — treat as failure
            if not result.get("outputs"):
                return {"ok": False, "answer": "", "error": "Flow returned empty outputs — missing ChatOutput node or flow is empty"}
            text = _extract_text(result)
            return {"ok": True, "answer": text or "(no text output)", "error": ""}
        except Exception as e:
            # Try to extract error detail from HTTP response body
            error_str = str(e)
            if hasattr(e, "read"):
                try:
                    body = json.loads(e.read())
                    error_str = body.get("detail", error_str)
                    if isinstance(error_str, str) and len(error_str) > 300:
                        error_str = error_str[:300]
                except Exception:
                    pass
            return {"ok": False, "answer": "", "error": error_str}

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
