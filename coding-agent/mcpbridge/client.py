import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any

import urllib.request

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcpbridge.enrichment import FlowEnrichmentMixin
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


class LangflowMCPClient(FlowEnrichmentMixin):
    """Transport layer (Layer B): MCP session lifecycle, Redis entity cache, tool discovery,
    and tool dispatch. Flow-correctness transforms live in ``FlowEnrichmentMixin``
    (``mcpbridge.enrichment``) and are inherited so the public method surface is unchanged."""

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
        self._discovered: list[str] = []  # FIFO, capped at _DISCOVERY_CAP; session-activated long-tail tools
        # When set, the 4 ported composite tools (search_flows, delete_node, get_component_schema,
        # get_starter_template) are routed to langflow-mcp instead of being served in-process by
        # Python. Lets you A/B the migrated server path against the Python path. Default: Python.
        self._use_server_composite = os.getenv("USE_SERVER_COMPOSITE_TOOLS", "").strip().lower() in ("1", "true", "yes", "on")

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
                # get_basic_examples returns full template data (nodes+edges) — always prefer it.
                # list_starter_projects often returns metadata-only stubs without data.nodes.
                starters_raw = await self._session_call_json("get_basic_examples", {})
                if not starters_raw:
                    starters_raw = await self._session_call_json("list_starter_projects", {})
                if isinstance(flows_raw, list):
                    await self._redis_cache.sync_flows(flows_raw)
                if isinstance(starters_raw, list) and starters_raw:
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
                starters_raw = await self._session_call_json("get_basic_examples", {})
                if not starters_raw:
                    starters_raw = await self._session_call_json("list_starter_projects", {})
                if isinstance(flows_raw, list):
                    await self._redis_cache.sync_flows(flows_raw)
                if isinstance(starters_raw, list) and starters_raw:
                    await self._redis_cache.sync_starters(starters_raw)
            except Exception as e:
                logger.warning("Redis background sync error: %s", e)

    async def fetch_starter(self, name_or_id: str) -> dict | None:
        """Look up a starter template by name or id. Checks Redis first, then HTTP."""
        key = name_or_id.strip()
        if self._redis_cache:
            data = await self._redis_cache.get_starter_data(key)
            if data:
                return data
        # HTTP fallback: GET /api/v1/flows/basic_examples/
        url = f"{self._langflow_base_url}/api/v1/flows/basic_examples/"
        req = urllib.request.Request(url, headers={"x-api-key": self._langflow_api_key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                starters = json.loads(resp.read())
            q = key.lower()
            match = next(
                (s for s in starters if s.get("id") == key or q in s.get("name", "").lower()),
                None,
            )
            if match and self._redis_cache:
                await self._redis_cache.sync_starters(starters)
            return match
        except Exception as e:
            logger.warning("fetch_starter HTTP fallback failed: %s", e)
            return None

    def _create_flow_direct(self, name: str, description: str, data: dict) -> dict:
        """POST a complete flow directly to /api/v1/flows/. Bypasses MCP."""
        url = f"{self._langflow_base_url}/api/v1/flows/"
        payload = json.dumps({"name": name, "description": description, "data": data}).encode()
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"x-api-key": self._langflow_api_key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

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

    # Names here are required by Redis caching (this file) and result-stripping (agent.py) — single source of hot-path coupling.
    _BASELINE_TOOLS = {
        "create_flow", "update_flow", "get_flow", "list_flows", "delete_flow",
        "build_flow", "run_flow",
        "list_components", "get_basic_examples", "list_starter_projects",
    }

    _DISCOVERY_CAP = 12  # max simultaneously-active long-tail tools

    _VIRTUAL_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "clone_starter_template",
                "description": (
                    "Clone a starter template into a new flow server-side — "
                    "no need to call get_basic_examples, get_starter_template, or create_flow. "
                    "Use for Score ≥ 8.5 direct clones. "
                    "Returns {flow_id, name, node_count, edge_count}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name_or_id": {"type": "string", "description": "Template name (e.g. 'Simple Agent') or id"},
                        "name": {"type": "string", "description": "Name for the new flow (defaults to template name)"},
                        "description": {"type": "string", "description": "Description for the new flow"},
                    },
                    "required": ["name_or_id"],
                },
            },
        },
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
                "name": "delete_node",
                "description": (
                    "Remove one or more nodes from a flow in a single round-trip. "
                    "Fetches the flow, drops matched nodes + any edges referencing them, "
                    "and PATCHes the result. Use this for 'remove X' / 'delete X' user requests "
                    "instead of update_flow — update_flow's merge semantics can only ADD, "
                    "never remove, so a delete via update_flow silently no-ops. "
                    "Pass node_ids (exact IDs from get_flow) OR types (e.g. 'CalculatorComponent', "
                    "'URLComponent'); type→ID resolution happens server-side. "
                    "Returns {flow_id, removed_node_ids, removed_edge_count}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flow_id": {"type": "string", "description": "Flow UUID"},
                        "node_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact node IDs to delete (preferred when known)",
                        },
                        "types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Component types — every node of these types is removed",
                        },
                    },
                    "required": ["flow_id"],
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
        {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": (
                    "Discover tools not currently visible. Call this when you need a capability "
                    "that isn't in your active tool array (e.g. variables, folders, knowledge base, "
                    "files, store, monitoring, health). "
                    "Returns {matches: [{name, summary}], note}. "
                    "Matched tools become available on the NEXT step — then call them directly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Capability keyword, e.g. 'variable', 'folder', 'health'"},
                        "limit": {"type": "integer", "default": 8, "description": "Max results (default 8)"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    # Names the agent serves itself via _VIRTUAL_TOOLS. langflow-mcp now also exposes the
    # composite ones server-side, so they appear in _tools_cache — exclude them from discovery
    # so search_tools/get_tool_schemas never emit a duplicate of a virtual tool. Derived from
    # _VIRTUAL_TOOLS so adding/removing a virtual tool needs no change here.
    _VIRTUAL_TOOL_NAMES = {t["function"]["name"] for t in _VIRTUAL_TOOLS}

    def get_tool_schemas(self) -> list[dict]:
        # Stable block (baseline + virtual) is emitted first so it can be prompt-cached as a
        # unit. Session-discovered tools change as search_tools runs, so they go LAST and are
        # tagged `_volatile`: the provider keeps the tool cache breakpoint on the stable block,
        # so a changed discovered set only re-caches the volatile tail — not baseline/virtual.
        discovered = [
            n for n in getattr(self, "_discovered", ())
            if n not in self._BASELINE_TOOLS and n not in self._VIRTUAL_TOOL_NAMES
        ]

        def _dict(t, volatile=False):
            d = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            if volatile:
                d["_volatile"] = True
            return d

        baseline = [_dict(t) for t in self._tools_cache if t.name in self._BASELINE_TOOLS]
        discovered_tools = [_dict(t, volatile=True) for t in self._tools_cache if t.name in discovered]
        return baseline + self._VIRTUAL_TOOLS + discovered_tools

    async def _handle_delete_node(self, args: dict) -> str:
        """Server-side node removal: get_flow → drop matched nodes + dangling edges → PATCH → build.
        Bypasses agent merge logic (which is union-only and cannot remove)."""
        from rich.console import Console as _Console
        _con = _Console()

        flow_id = args.get("flow_id", "")
        node_ids: set[str] = set(args.get("node_ids") or [])
        types: set[str] = set(args.get("types") or [])
        if not flow_id or (not node_ids and not types):
            return json.dumps({"error": "flow_id and (node_ids or types) required"})

        flow = await self._session_call_json("get_flow", {"flow_id": flow_id})
        if not isinstance(flow, dict):
            return json.dumps({"error": "get_flow returned non-dict payload"})
        data = flow.get("data") or {}
        nodes = list(data.get("nodes") or [])
        edges = list(data.get("edges") or [])

        if types:
            for n in nodes:
                t = (n.get("data") or {}).get("type") or n.get("type", "")
                nid = n.get("id", "")
                if t in types and nid:
                    node_ids.add(nid)

        if not node_ids:
            _con.print(f"[yellow]↳ delete_node: no nodes matched types={sorted(types)} in flow {flow_id}[/yellow]")
            return json.dumps({
                "flow_id": flow_id,
                "removed_node_ids": [],
                "removed_edge_count": 0,
                "note": "no matching nodes",
            })

        kept_nodes = [n for n in nodes if n.get("id") not in node_ids]
        kept_edges = [
            e for e in edges
            if e.get("source") not in node_ids and e.get("target") not in node_ids
        ]
        removed_edge_count = len(edges) - len(kept_edges)
        _con.print(f"[dim]↳ delete_node: removing {sorted(node_ids)}, dropping {removed_edge_count} edge(s)[/dim]")

        new_data = {**data, "nodes": kept_nodes, "edges": kept_edges}
        await self._session.call_tool(
            "update_flow",
            {"flow_id": flow_id, "data": new_data},
        )
        # Trigger a build so Langflow invalidates its canvas cache and the UI
        # reflects the removal immediately without needing a browser refresh.
        await self._session.call_tool("build_flow", {"flow_id": flow_id})
        return json.dumps({
            "flow_id": flow_id,
            "removed_node_ids": sorted(node_ids),
            "removed_edge_count": removed_edge_count,
        })

    def _tool_summary(self, tool) -> str:
        desc = tool.description or ""
        first_line = desc.split("\n")[0].strip()
        return first_line[:120] if first_line else tool.name

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if self._session is None:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        # USE_SERVER_COMPOSITE_TOOLS: route ported tools to langflow-mcp by letting them fall
        # through to self._session.call_tool below (the server now exposes them) instead of the
        # Python interceptions. search_tools/enrich_* are never routed — they stay agent-side.
        use_server = getattr(self, "_use_server_composite", False)
        if use_server and name in ("search_flows", "delete_node"):
            # Visible A/B marker (the agent prints its own for the schema/starter tools).
            try:
                from rich.console import Console as _Console
                _Console().print(f"[dim]↳ {name} via langflow-mcp[/dim]")
            except Exception:
                pass

        if name == "delete_node" and not use_server:
            return await self._handle_delete_node(arguments)

        if name == "search_tools":
            query = (arguments.get("query") or "").strip().lower()
            if not query:
                return json.dumps({"matches": [], "note": "Empty query — pass a capability keyword, e.g. 'variable'."})
            try:
                limit = int(arguments.get("limit") or 8)
            except (TypeError, ValueError):
                limit = 8
            limit = max(1, min(limit, 25))
            # Token-scored ranking: phrase + per-token, name weighted over description.
            # Beats naive substring — multi-word queries ("delete variable") still match.
            tokens = [w for w in re.split(r"\W+", query) if len(w) >= 2]
            scored = []
            for t in self._tools_cache:
                # Skip tools the agent already serves itself (virtual) — discovering the
                # server copy would duplicate a virtual tool in the advertised schema.
                if t.name in self._VIRTUAL_TOOL_NAMES:
                    continue
                nm = t.name.lower()
                desc = (t.description or "").lower()
                score = 0
                if query in nm:
                    score += 5
                elif query in desc:
                    score += 2
                for tok in tokens:
                    if tok in nm:
                        score += 3
                    elif tok in desc:
                        score += 1
                if score > 0:
                    scored.append((score, t))
            scored.sort(key=lambda st: st[0], reverse=True)  # stable: ties keep cache order
            matches = [t for _, t in scored[:limit]]
            if not matches:
                return json.dumps({
                    "matches": [],
                    "note": f"No tools matched '{query}'. Retry with a broader/different keyword.",
                })
            # Activate matched tools: append to _discovered (dedupe), FIFO-evict past cap
            for t in matches:
                if t.name not in self._BASELINE_TOOLS and t.name not in self._discovered:
                    self._discovered.append(t.name)
            while len(self._discovered) > self._DISCOVERY_CAP:
                self._discovered.pop(0)
            return json.dumps({
                "matches": [{"name": t.name, "summary": self._tool_summary(t)} for t in matches],
                "note": "These tools are now active — call them directly on your next step.",
            })

        # search_flows virtual tool
        if name == "search_flows" and not use_server:
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
