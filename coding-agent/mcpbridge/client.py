import asyncio
import gzip
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

import urllib.request

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcpbridge.redis_cache import RedisEntityCache

logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        mcp_path: str,
        langflow_api_key: str,
        langflow_base_url: str,
        redis_cache: RedisEntityCache | None = None,
    ) -> None:
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
        self._redis_cache = redis_cache
        self._sync_task: asyncio.Task | None = None

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

        if self._redis_cache and await self._redis_cache.connect():
            try:
                flows_raw = await self._session_call_json("list_flows", {})
                starters_raw = await self._session_call_json("list_starter_projects", {})
                if isinstance(flows_raw, list):
                    await self._redis_cache.sync_flows(flows_raw)
                if isinstance(starters_raw, list):
                    await self._redis_cache.sync_starters(starters_raw)
                self._sync_task = asyncio.create_task(self._background_sync())
            except Exception as e:
                logger.warning("Redis cold sync failed: %s", e)

    async def _session_call_json(self, tool_name: str, args: dict) -> Any:
        result = await self._session.call_tool(tool_name, args)
        if not result.content:
            return None
        item = result.content[0]
        text = item.text if hasattr(item, "text") else str(item)
        try:
            return json.loads(text)
        except Exception:
            return text

    async def _background_sync(self) -> None:
        while True:
            await asyncio.sleep(self._redis_cache._sync_interval)
            try:
                flows_raw = await self._session_call_json("list_flows", {})
                starters_raw = await self._session_call_json("list_starter_projects", {})
                if isinstance(flows_raw, list):
                    await self._redis_cache.sync_flows(flows_raw)
                if isinstance(starters_raw, list):
                    await self._redis_cache.sync_starters(starters_raw)
            except Exception as e:
                logger.warning("Redis background sync error: %s", e)

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
        # Invalidate dynamic resolution indexes so they rebuild against new schemas
        self._schema_lower_index = {}
        self._schema_display_index = {}
        return flat

    def _resolve_type(self, raw_type: str, schemas: dict) -> str:
        """Dynamically resolve a type string to the canonical schema key.
        Tries exact → case-insensitive → display_name match → best prefix match.
        All resolution is against the live /api/v1/all schema — no static aliases."""
        if raw_type in schemas:
            return raw_type
        lower = raw_type.lower()
        # Build case-insensitive index and display_name index lazily on first mismatch
        if not hasattr(self, "_schema_lower_index") or len(self._schema_lower_index) != len(schemas):
            self._schema_lower_index: dict[str, str] = {}
            self._schema_display_index: dict[str, str] = {}
            for key, schema in schemas.items():
                self._schema_lower_index[key.lower()] = key
                dn = schema.get("display_name", "")
                if dn:
                    self._schema_display_index[dn.lower().replace(" ", "")] = key
        # 1. Case-insensitive exact match
        if lower in self._schema_lower_index:
            return self._schema_lower_index[lower]
        # 2. Match against display_name (e.g. "Prompt Template" → "PromptTemplate")
        normalized = lower.replace(" ", "").replace("_", "").replace("-", "")
        if normalized in self._schema_display_index:
            return self._schema_display_index[normalized]
        # 3. Prefix match — find all schema keys that start with or contain the raw_type string
        candidates = [k for lk, k in self._schema_lower_index.items() if lk.startswith(lower) or lower.startswith(lk)]
        if len(candidates) == 1:
            return candidates[0]
        return raw_type  # unknown — let caller raise

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
            # Note nodes are UI-only annotations; graph builder skips type=="noteNode".
            # Pass them through unchanged — overwriting their type breaks graph skip logic.
            if node.get("type") == "noteNode" or node.get("data", {}).get("type") in ("note", "noteNode"):
                enriched.append(node)
                continue
            # Prefer data.type — top-level type is "genericNode" for canvas-rendered nodes
            raw_type = node.get("data", {}).get("type") or node.get("type", "")
            # node_type resolves to the /api/v1/all schema key (used for schema lookup only)
            node_type = self._resolve_type(raw_type, schemas)
            data = dict(node.get("data", {}))
            if node_type not in schemas:
                raise ValueError(
                    f"Unknown component type: {node_type!r}. "
                    f"Call list_components to find the exact 'type' string, then retry."
                )
            # Always use the live schema — template-cloned nodes can carry stale data.node
            # (missing display_name, empty outputs) which causes the Langflow frontend to show
            # "undefined" for node names and drop all edges as invalid.
            # data.type is preserved separately so the canvas renderer stays correct.
            data["node"] = json.loads(json.dumps(schemas[node_type]))  # deep copy of live schema
            # Only patch data.type for fresh nodes — for template nodes data.type is already correct
            if "node" not in node.get("data", {}):
                if node_type != raw_type:
                    data["type"] = node_type
            # Inject credentials — uses node_type (schema key) to find the right overrides
            if credential_overrides and node_type in credential_overrides and "node" in data:
                tmpl = data["node"].get("template", {})
                for field_name, value in credential_overrides[node_type].items():
                    # /api/v1/all omits SecretStrInput fields from schema — create entry if missing
                    if field_name not in tmpl:
                        tmpl[field_name] = {"name": field_name, "type": "str", "value": ""}
                    tmpl[field_name]["value"] = value
            # Ensure data.type and data.id are always set
            data.setdefault("type", raw_type)  # preserve original type if not already set
            data.setdefault("id", node.get("id", ""))
            # For nodes with multiple outputs, pin the correct active output so the
            # canvas connects the right handle (e.g. model_output not text_output).
            _SELECTED_OUTPUTS = {"AzureOpenAIModel": "model_output", "AnthropicModel": "model_output"}
            if node_type in _SELECTED_OUTPUTS:
                data.setdefault("selected_output", _SELECTED_OUTPUTS[node_type])
            node["data"] = data
            # React-flow requires type="genericNode" for Langflow's custom node renderer.
            # The component type lives in data.type; top-level type is only for the canvas.
            node["type"] = "genericNode"
            enriched.append(node)
        # Auto-enable tool_mode for non-native-tool nodes feeding Agent.tools
        self._auto_tool_mode(enriched)
        return enriched

    def _auto_tool_mode(self, nodes: list[dict]) -> None:
        """Enable tool_mode on non-native-tool nodes that expose a tool-eligible output.
        Skips tool CONSUMERS (nodes with a 'tools' input, e.g. Agent) — they must never
        have tool_mode enabled or their Toolset handle renders instead of the tools input."""
        for node in nodes:
            d = node.get("data", {})
            schema = d.get("node", {})
            if not schema:
                continue
            # Skip tool consumers — enabling tool_mode on them breaks their tools input handle
            if "tools" in schema.get("template", {}):
                continue
            outputs = schema.get("outputs", [])
            has_native_tool = any(o.get("name") == "api_build_tool" for o in outputs)
            has_tool_eligible = any(o.get("tool_mode") for o in outputs)
            if not has_native_tool and has_tool_eligible:
                schema["tool_mode"] = True

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

        # Build node_id → enriched node (so we can read its outputs for tool-mode rewriting)
        node_by_id = {n.get("id", ""): n for n in nodes}

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

            # Rewrite sourceHandle for tool-mode connections to Agent.tools.
            # Always rewrite when target is tools input — regardless of sh["name"].
            # The LLM may pass the original (non-tool-mode) output name from the template.
            if isinstance(sh, dict) and isinstance(th, dict) and th.get("fieldName") == "tools":
                src_node = node_by_id.get(edge.get("source", ""))
                if src_node:
                    src_schema = src_node.get("data", {}).get("node", {})
                    if src_schema.get("tool_mode"):
                        outputs = src_schema.get("outputs", [])
                        tool_out = next(
                            (o for o in outputs if o.get("tool_mode")),
                            next((o for o in outputs if o.get("name") == "component_as_tool"), None),
                        )
                        if tool_out:
                            sh = dict(sh)
                            sh["name"] = tool_out.get("name", "component_as_tool")
                            sh["output_types"] = ["Tool"]

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

    def ensure_tool_edges(self, nodes: list[dict], edges: list[dict]) -> list[dict]:
        """Auto-add missing Tool→Agent.tools edges when tool-capable nodes have no connection.

        Called after enrich_nodes (which sets tool_mode) and before enrich_edges (which
        serializes handles). Detects any enriched node with tool_mode=True that lacks an
        edge to an Agent's tools input, and synthesizes the missing connections.
        """
        agent_nodes = [
            n for n in nodes
            if "tools" in n.get("data", {}).get("node", {}).get("template", {})
        ]
        tool_nodes = [
            n for n in nodes
            if n.get("data", {}).get("node", {}).get("tool_mode")
        ]
        if not agent_nodes or not tool_nodes:
            return edges

        # Track existing tool-edge pairs (source_id, target_id)
        existing: set[tuple[str, str]] = set()
        for e in edges:
            th_raw = e.get("targetHandle", {})
            if isinstance(th_raw, str):
                try:
                    th = json.loads(th_raw.replace("œ", '"'))
                except Exception:
                    th = {}
            else:
                th = th_raw
            if isinstance(th, dict) and th.get("fieldName") == "tools":
                existing.add((e.get("source", ""), e.get("target", "")))

        new_edges = list(edges)
        for agent_node in agent_nodes:
            agent_id = agent_node.get("id", "")
            agent_type = agent_node.get("data", {}).get("type", "Agent")
            for tool_node in tool_nodes:
                tool_id = tool_node.get("id", "")
                if (tool_id, agent_id) in existing:
                    continue
                tool_comp_type = tool_node.get("data", {}).get("type", "")
                outputs = tool_node.get("data", {}).get("node", {}).get("outputs", [])
                tool_out = next(
                    (o for o in outputs if o.get("tool_mode")),
                    next((o for o in outputs if o.get("name") == "component_as_tool"), None),
                )
                if not tool_out:
                    continue
                new_edges.append({
                    "source": tool_id,
                    "target": agent_id,
                    "sourceHandle": {
                        "dataType": tool_comp_type,
                        "id": tool_id,
                        "name": tool_out.get("name", "component_as_tool"),
                        "output_types": ["Tool"],
                    },
                    "targetHandle": {
                        "fieldName": "tools",
                        "id": agent_id,
                        "inputTypes": ["Tool"],
                        "type": "other",
                    },
                })
                existing.add((tool_id, agent_id))
        return new_edges

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

    _CORE_TOOLS = {
        "create_flow", "update_flow", "get_flow", "list_flows", "delete_flow",
        "build_flow", "run_flow",
        "list_components", "list_variables", "list_folders",
        "get_basic_examples", "list_starter_projects", "health_check",
    }

    _VIRTUAL_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_flows",
                "description": (
                    "Search user flows by keyword. Returns [{id, name, description}] for matching flows. "
                    "Use instead of list_flows when looking for a specific flow by name or topic."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keyword to search flow names and descriptions"},
                        "limit": {"type": "integer", "default": 15},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_component_schema",
                "description": (
                    "Get exact input field names and output handle names for a specific component type. "
                    "Call this for ANY component not listed in the system prompt's Component Reference table "
                    "before building edges to/from it. Prevents invalid connections."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type_name": {
                            "type": "string",
                            "description": "Exact component type string (e.g. 'SplitText', 'Chroma', 'AzureOpenAIEmbeddings')"
                        }
                    },
                    "required": ["type_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_starter_template",
                "description": (
                    "Get full nodes[] and edges[] for ONE specific starter template by name or id. "
                    "Call this AFTER scoring templates from list_starter_projects index to fetch the winning template's full data. "
                    "Much cheaper than re-calling list_starter_projects — returns only the one template you need."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name_or_id": {
                            "type": "string",
                            "description": "Template name (e.g. 'Hybrid RAG Agent') or id from list_starter_projects index"
                        }
                    },
                    "required": ["name_or_id"],
                },
            },
        },
    ]

    def get_component_schema(self, type_name: str) -> dict:
        """Return compact schema (inputs + outputs) for one component. Uses cached /api/v1/all data."""
        schemas = self._fetch_component_schemas()
        if type_name not in schemas:
            return {"error": f"Unknown type: {type_name!r}. Call list_components to find the exact type string."}
        schema = schemas[type_name]
        tmpl = schema.get("template", {})
        outputs = schema.get("outputs", [])
        inputs = [
            {
                "field": k,
                "type": v.get("type", ""),
                "display": v.get("display_name", k),
                "required": v.get("required", False),
                "input_types": v.get("input_types", []),
            }
            for k, v in tmpl.items()
            if isinstance(v, dict)
            and not v.get("advanced", False) and v.get("show", True)
            and v.get("type") not in ("code", "prompt")
        ]
        outs = [
            {
                "name": o.get("name"),
                "types": o.get("output_types", []),
                "tool_mode": o.get("tool_mode", False),
            }
            for o in outputs
        ]
        return {"type": type_name, "inputs": inputs, "outputs": outs}

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
            if t.name in self._CORE_TOOLS
        ] + self._VIRTUAL_TOOLS

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if self._session is None:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        # search_flows virtual tool
        if name == "search_flows":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 15)
            if self._redis_cache and await self._redis_cache.is_warm():
                results = await self._redis_cache.search_flows(query, limit)
                return json.dumps(results)
            # Fallback: MCP list_flows then filter
            all_flows = await self._session_call_json("list_flows", {})
            if isinstance(all_flows, list):
                q = query.lower()
                filtered = [
                    {"id": f.get("id"), "name": f.get("name", ""), "description": f.get("description", "")}
                    for f in all_flows
                    if q in f.get("name", "").lower() or q in (f.get("description") or "").lower()
                ]
                return json.dumps(filtered[:limit])
            return json.dumps([])

        # Redis-cached list_flows
        if name == "list_flows" and self._redis_cache:
            if await self._redis_cache.is_warm():
                flows = await self._redis_cache.list_all_flows()
                return json.dumps(flows)

        # Redis-cached list_starter_projects
        if name == "list_starter_projects" and self._redis_cache:
            if await self._redis_cache.is_warm():
                starters = await self._redis_cache.list_all_starters()
                return json.dumps(starters)

        result = await self._session.call_tool(name, arguments)

        # Lazy populate Redis on first MCP call when cold
        if name in ("list_flows", "list_starter_projects") and self._redis_cache and result.content:
            item0 = result.content[0]
            raw_text = item0.text if hasattr(item0, "text") else str(item0)
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    if name == "list_flows":
                        await self._redis_cache.sync_flows(parsed)
                    else:
                        await self._redis_cache.sync_starters(parsed)
            except Exception:
                pass

        if not result.content:
            return None
        item = result.content[0]
        if hasattr(item, "text"):
            return item.text
        return str(item)

    async def close(self) -> None:
        if self._sync_task:
            self._sync_task.cancel()
            self._sync_task = None
        if self._redis_cache:
            await self._redis_cache.close()
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass  # suppress ExceptionGroup from JSONRPC parse errors on shutdown
            finally:
                self._exit_stack = None
                self._session = None
