"""Flow-correctness layer (Layer A).

Pure Langflow-payload transforms that turn LLM-generated / template-cloned flow JSON into
Langflow-*valid* JSON: live-schema injection, display-name resolution, tool_mode inference,
œ-encoded handle serialization, prompt-var materialization, dual-output LLM selection, and the
structural merge/find helpers.

These are split out of the transport client (``mcpbridge.client``) so the correctness logic is
isolated from MCP/Redis/discovery concerns and can be ported behind the MCP boundary later
without dragging agent-side orchestration along. The mixin is consumed by
``LangflowMCPClient``; it relies on the host class providing ``_langflow_base_url``,
``_langflow_api_key`` and ``_component_schema_cache`` (set in the client ``__init__``).
"""

import gzip
import json
import logging
import re
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class FlowEnrichmentMixin:
    """Schema-driven flow-correctness transforms. See module docstring."""

    # Structural/base nodes that must never be used as tools.
    # These are architectural invariants, not use-case hacks.
    _NEVER_TOOL: set[str] = {"AzureOpenAIModel", "AnthropicModel", "ChatInput", "ChatOutput", "Agent"}

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

    def is_legacy(self, type_name: str) -> bool:
        """True if the component is flagged legacy in the live /api/v1/all schema.
        Resolves display-name/casing variants first so the hard-block can't be bypassed by
        spelling (e.g. "Natural Language to SQL" → SQLGenerator). Schema-driven, no blocklist."""
        schemas = self._fetch_component_schemas()
        schema = schemas.get(self._resolve_type(type_name, schemas))
        return bool(schema.get("legacy")) if isinstance(schema, dict) else False

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
            self._schema_display_exact: dict[str, str] = {}   # display_name.lower() (spaces kept)
            self._schema_display_index: dict[str, str] = {}   # normalized (spaces/_/- stripped)

            def _prefer(index: dict[str, str], k: str, key: str) -> None:
                # Deterministic on collision (two components, same (normalized) display name —
                # e.g. "SQL Database"/SQLComponent vs "SQLDatabase"/SQLDatabase normalize alike):
                # keep the non-legacy twin, else first seen. General, not tied to any pair.
                prev = index.get(k)
                if prev is None or (bool(schemas.get(prev, {}).get("legacy")) and not bool(schema.get("legacy"))):
                    index[k] = key

            for key, schema in schemas.items():
                self._schema_lower_index[key.lower()] = key
                dn = schema.get("display_name", "")
                if dn:
                    _prefer(self._schema_display_exact, dn.lower(), key)
                    _prefer(self._schema_display_index, dn.lower().replace(" ", ""), key)
        # 1. Case-insensitive exact match
        if lower in self._schema_lower_index:
            return self._schema_lower_index[lower]
        # 2. Exact display_name match (spaces preserved) — disambiguates twins whose displays
        # differ only by spacing ("SQL Database" vs "SQLDatabase") before normalization erases it.
        if lower in self._schema_display_exact:
            return self._schema_display_exact[lower]
        # 3. Normalized display_name match (e.g. "prompt_template" → "PromptTemplate")
        normalized = lower.replace(" ", "").replace("_", "").replace("-", "")
        if normalized in self._schema_display_index:
            return self._schema_display_index[normalized]
        # 4. Prefix match — find all schema keys that start with or contain the raw_type string
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
            # Capture show=True fields from original template before clobbering with live schema.
            # Live schema defaults many fields to show=False (e.g. lexical_terms on AstraDB);
            # template-saved nodes may have show=True with edges wired — we must preserve that.
            _orig_tmpl = (node.get("data") or {}).get("node", {}).get("template", {})
            _original_shows: dict[str, bool] = {
                _fn: True
                for _fn, _fd in (_orig_tmpl.items() if isinstance(_orig_tmpl, dict) else [])
                if isinstance(_fd, dict) and _fd.get("show") is True
            }
            # Always use the live schema — template-cloned nodes can carry stale data.node
            # (missing display_name, empty outputs) which causes the Langflow frontend to show
            # "undefined" for node names and drop all edges as invalid.
            # data.type is preserved separately so the canvas renderer stays correct.
            data["node"] = json.loads(json.dumps(schemas[node_type]))  # deep copy of live schema
            # Restore show=True from original template (live schema may default to show=False)
            if _original_shows:
                _live_tmpl = data["node"].get("template", {})
                for _fn, _ in _original_shows.items():
                    if _fn in _live_tmpl and isinstance(_live_tmpl[_fn], dict):
                        _live_tmpl[_fn]["show"] = True
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
            # Default selected_output for dual-output LLM nodes — overridden later by
            # fix_selected_outputs() once edges are known and wiring is clear.
            outputs = data.get("node", {}).get("outputs", [])
            output_names = {o.get("name") for o in outputs}
            if "model_output" in output_names and "text_output" in output_names:
                data.setdefault("selected_output", "model_output")
            node["data"] = data
            # React-flow requires type="genericNode" for Langflow's custom node renderer.
            # The component type lives in data.type; top-level type is only for the canvas.
            node["type"] = "genericNode"
            enriched.append(node)
        # Auto-enable tool_mode for non-native-tool nodes feeding Agent.tools
        self._auto_tool_mode(enriched)
        return enriched

    def apply_prompt_fields(self, nodes: list[dict], prompt_template: str = "") -> None:
        """Materialize dynamic ``{var}`` input handles on Prompt-like nodes.

        Langflow creates one input handle per ``{var}`` in a prompt-type field's value,
        but only if the SAVED node carries both (a) the field value and (b) a template
        entry + ``custom_fields`` listing for each var. A freshly enriched node has the
        bare component schema (empty prompt value, no var fields), so any edge wired into
        a var handle is silently dropped on load — the NL→SQL "connections were removed
        because they were invalid" symptom.

        Fully schema-driven: targets ANY field whose schema ``type == "prompt"`` (Prompt
        Template, Cleanlab, future components), never a hardcoded component or field name.
        The template string is taken from the node's own value if set (LLM/clone authored),
        otherwise from the design-level ``prompt_template``. Materialization defers to
        Langflow's own ``/api/v1/validate/prompt`` so the emitted field shapes always match
        the running version; a local regex-based injection is the offline fallback.
        """
        for node in nodes:
            schema = node.get("data", {}).get("node", {})
            tmpl = schema.get("template", {})
            if not isinstance(tmpl, dict):
                continue
            for field_name, field in list(tmpl.items()):
                if not (isinstance(field, dict) and field.get("type") == "prompt"):
                    continue
                template_str = field.get("value") or prompt_template
                if not template_str:
                    continue
                self._materialize_prompt_field(schema, field_name, template_str)

    def _materialize_prompt_field(self, schema: dict, field_name: str, template_str: str) -> None:
        """Inject var fields + custom_fields for one prompt-type field, then set its value."""
        updated = self._validate_prompt(field_name, template_str, schema)
        if updated is not None and isinstance(updated.get("template"), dict):
            # Endpoint returns the var fields + custom_fields but leaves the prompt value
            # blank — splice them in (keeping the rest of the enriched schema) and restore
            # the value ourselves.
            schema["template"] = updated["template"]
            if "custom_fields" in updated:
                schema["custom_fields"] = updated["custom_fields"]
            schema["template"].setdefault(field_name, {})["value"] = template_str
            return
        self._inject_prompt_vars_local(schema, field_name, template_str)

    def _validate_prompt(self, field_name: str, template_str: str, frontend_node: dict) -> dict | None:
        """POST /api/v1/validate/prompt → updated frontend_node, or None if unreachable."""
        url = f"{self._langflow_base_url}/api/v1/validate/prompt"
        body = json.dumps(
            {"name": field_name, "template": template_str, "frontend_node": frontend_node}
        ).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"x-api-key": self._langflow_api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                out = json.loads(resp.read())
            return out.get("frontend_node")
        except Exception as e:  # network/422/down — fall back to local injection
            logger.warning("validate/prompt failed (%s); using local var injection", e)
            return None

    @staticmethod
    def _inject_prompt_vars_local(schema: dict, field_name: str, template_str: str) -> None:
        """Offline fallback mirroring Langflow's DefaultPromptField shape for each {var}."""
        tmpl = schema.setdefault("template", {})
        tmpl.setdefault(field_name, {})["value"] = template_str
        seen: list[str] = []
        for v in re.findall(r"\{\s*([a-zA-Z_]\w*)\s*\}", template_str):
            if v not in seen and v != field_name:
                seen.append(v)
        for v in seen:
            tmpl[v] = {
                "field_type": "str", "required": False, "placeholder": "", "list": False,
                "show": True, "multiline": True, "value": "", "fileTypes": [], "file_path": "",
                "name": v, "display_name": v, "advanced": False, "input_types": ["Message"],
                "dynamic": False, "info": "", "load_from_db": False, "title_case": False,
                "type": "str", "_input_type": "DefaultPromptField",
            }
        schema.setdefault("custom_fields", {})[field_name] = seen

    def _auto_tool_mode(self, nodes: list[dict]) -> None:
        """Enable tool_mode on components that declare tool-capable inputs.

        In Langflow, tool capability is declared on INPUT fields (tool_mode=True on
        MessageTextInput, etc.), not on output fields. Components like CalculatorComponent
        and URLComponent declare tool_mode on their inputs — that is the canonical signal
        that the component supports being wrapped as a StructuredTool.

        Only runs when an Agent node is present — without an Agent there is no consumer
        for tool outputs, so enabling tool_mode would corrupt data-pipeline nodes (e.g.
        AstraDB's search_results output gets overridden with component_as_tool).

        Skips tool CONSUMERS (nodes with a 'tools' input, e.g. Agent) and native tool
        nodes (already have api_build_tool output)."""
        has_agent = any(
            "tools" in n.get("data", {}).get("node", {}).get("template", {})
            for n in nodes
        )
        if not has_agent:
            return
        for node in nodes:
            d = node.get("data", {})
            schema = d.get("node", {})
            if not schema:
                continue
            # Skip tool consumers — enabling tool_mode on them breaks their tools input handle
            if "tools" in schema.get("template", {}):
                continue
            outputs = schema.get("outputs", [])
            # Both api_build_tool (older Langflow) and component_as_tool (current) signal native tool
            has_native_tool = any(o.get("name") in ("api_build_tool", "component_as_tool") for o in outputs)
            if has_native_tool:
                continue  # already a native tool, no wrapping needed
            # Detect tool capability from INPUT fields — this is where Langflow components
            # declare tool_mode support (e.g., expression/urls inputs with tool_mode=True)
            tmpl = schema.get("template", {})
            has_tool_input = any(
                isinstance(v, dict) and v.get("tool_mode")
                for v in tmpl.values()
            )
            if has_tool_input:
                schema["tool_mode"] = True
                # Inject component_as_tool output — the exact schema Langflow uses for
                # tool-wrapped components (to_toolkit method). Without this, enrich_edges
                # and ensure_tool_edges can't find the tool output after enrich_nodes
                # overwrites data.node with the pre-tool_mode schema from /api/v1/all.
                outputs = schema.setdefault("outputs", [])
                if not any(o.get("name") == "component_as_tool" for o in outputs):
                    outputs.append({
                        "allows_loop": False,
                        "cache": True,
                        "display_name": "Toolset",
                        "group_outputs": False,
                        "hidden": None,
                        "loop_types": None,
                        "method": "to_toolkit",
                        "name": "component_as_tool",
                        "options": None,
                        "required_inputs": None,
                        "selected": "Tool",
                        "tool_mode": True,
                        "types": ["Tool"],
                        "value": "__UNDEFINED__",
                    })
                # Set selected_output so Langflow's frontend uses component_as_tool handle
                d["selected_output"] = "component_as_tool"
                # Inject tools_metadata into template — Langflow's backend requires this
                # field to validate tool-mode edges for non-native tool components.
                # Without it, Langflow strips component_as_tool→Agent.tools edges on save/load.
                # CalculatorComponent has this natively; URLComponent and others don't.
                if "tools_metadata" not in tmpl:
                    tool_inputs = [
                        (k, v) for k, v in tmpl.items()
                        if isinstance(v, dict) and v.get("tool_mode") and k != "_type"
                    ]
                    args: dict = {}
                    for input_name, input_cfg in tool_inputs:
                        if input_cfg.get("list"):
                            args[input_name] = {
                                "default": "",
                                "description": input_cfg.get("info", ""),
                                "items": {"type": "string"},
                                "title": (input_cfg.get("display_name") or input_name).title(),
                                "type": "array",
                            }
                        else:
                            args[input_name] = {
                                "default": input_cfg.get("value", ""),
                                "description": input_cfg.get("info", ""),
                                "title": (input_cfg.get("display_name") or input_name).title(),
                                "type": "string",
                            }
                    tool_method = next(
                        (o.get("method", "") for o in outputs if o.get("method") and o.get("name") not in ("component_as_tool", "api_build_tool")),
                        (schema.get("display_name") or "tool").lower().replace(" ", "_"),
                    )
                    tmpl["tools_metadata"] = {
                        "_input_type": "ToolsInput",
                        "advanced": False,
                        "display_name": "Actions",
                        "dynamic": False,
                        "info": "Modify tool names and descriptions to help agents understand when to use each tool.",
                        "is_list": True,
                        "list_add_label": "Add More",
                        "name": "tools_metadata",
                        "override_skip": False,
                        "placeholder": "",
                        "real_time_refresh": True,
                        "required": False,
                        "show": True,
                        "title_case": False,
                        "tool_mode": False,
                        "trace_as_metadata": True,
                        "track_in_telemetry": False,
                        "type": "tools",
                        "value": [{
                            "args": args,
                            "description": schema.get("description", ""),
                            "display_description": schema.get("description", ""),
                            "display_name": tool_method,
                            "name": tool_method,
                            "readonly": False,
                            "status": True,
                            "tags": [tool_method],
                        }],
                    }

    @staticmethod
    def _parse_handle(handle: Any) -> dict:
        """Normalize edge handles to dict.
        Template-cloned edges store handles as JSON strings or Å/œ-encoded strings.
        LLM-generated edges use plain dicts. Normalize all to dict for uniform processing."""
        if isinstance(handle, dict):
            return handle
        if isinstance(handle, str):
            for char in ('œ', 'Å'):
                try:
                    return json.loads(handle.replace(char, '"'))
                except Exception:
                    continue
        return {}

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
            # Normalize handles to dicts regardless of source format.
            # Template edges arrive as JSON strings; LLM edges as dicts.
            sh = self._parse_handle(edge.get("sourceHandle", {}))
            th = self._parse_handle(edge.get("targetHandle", {}))

            # Sync handle ids with edge source/target — Langflow validates handle.id == node id.
            # _remap_edge fixes edge.source/target but cannot fix handle.id when handle is a string;
            # enrich_edges always has the final remapped ids, so we enforce them here.
            sh = dict(sh)
            th = dict(th)
            if edge.get("source"):
                sh["id"] = edge["source"]
            if edge.get("target"):
                th["id"] = edge["target"]

            # Fix targetHandle.type AND inputTypes from schema. The frontend only
            # renders an edge when its serialized targetHandle matches the string the
            # node computes from its template, so stale inputTypes (from LLM guesses or
            # legacy templates) silently drop the edge.
            tgt_node_id = edge.get("target", "")
            tgt_comp_type = node_type_map.get(tgt_node_id, "")
            field_name = th.get("fieldName", "")
            if tgt_comp_type and field_name and tgt_comp_type in schemas:
                tmpl_field = schemas[tgt_comp_type].get("template", {}).get(field_name, {})
                actual_type = tmpl_field.get("type")
                if actual_type:
                    th["type"] = actual_type
                input_types = tmpl_field.get("input_types")
                if input_types is not None:
                    th["inputTypes"] = input_types

            # Rewrite sourceHandle for ALL tool edges (fieldName=="tools").
            # Runs regardless of tool_mode state — template clones, LLM-created edges, native tools.
            # Detection is fully schema-driven (no component name hardcoding):
            #   priority 1: native api_build_tool output (CalculatorTool, SearchAPI, WikipediaAPI, ...)
            #   priority 2: output with tool_mode:true in schema (URLComponent.page_results, ...)
            #   priority 3: fallback — enable tool_mode, Langflow framework wraps primary output
            # _NEVER_TOOL: structural nodes that must never be treated as tools; wrong edges from
            # LLM hallucination are left unrewritten so Langflow drops them cleanly.
            if th.get("fieldName") == "tools":
                src_node = node_by_id.get(edge.get("source", ""))
                if src_node:
                    src_type = src_node.get("data", {}).get("type", "")
                    src_schema = src_node.get("data", {}).get("node", {})
                    outputs = src_schema.get("outputs", [])
                    if src_type not in self._NEVER_TOOL:
                        # Priority 1: native tool output (api_build_tool or component_as_tool)
                        native_out = next(
                            (o for o in outputs if o.get("name") in ("api_build_tool", "component_as_tool")),
                            None,
                        )
                        if native_out:
                            sh = dict(sh)
                            sh["name"] = native_out["name"]
                            sh["output_types"] = ["Tool"]
                        else:
                            tool_out = next((o for o in outputs if o.get("tool_mode")), None)
                            if tool_out:
                                # Priority 2: any output with tool_mode=True in schema
                                sh = dict(sh)
                                sh["name"] = tool_out.get("name", "component_as_tool")
                                sh["output_types"] = ["Tool"]
                            else:
                                # Priority 3: fallback — enable tool_mode, use component_as_tool
                                src_schema["tool_mode"] = True
                                sh = dict(sh)
                                sh["name"] = "component_as_tool"
                                sh["output_types"] = ["Tool"]

            # Reconcile sourceHandle against the live schema for non-tool edges.
            # (Tool edges already had their source rewritten above.) The LLM and legacy
            # templates frequently supply a stale output name or *_types array; since the
            # frontend renders an edge only when its serialized sourceHandle matches the
            # string the node computes from schema, any stale field drops the edge — this
            # is why ingestion edges (Directory/SplitText) silently disappeared while the
            # main path survived. Resolve the real output: match by name → else the output
            # whose types intersect the (already schema-corrected) target inputTypes →
            # else the first output. Fully schema-driven, no per-component hardcoding.
            if th.get("fieldName") != "tools":
                src_node_id = edge.get("source", "")
                src_comp_type = node_type_map.get(src_node_id, "")
                if src_comp_type and src_comp_type in schemas:
                    src_outputs = schemas[src_comp_type].get("outputs", []) or []
                    if src_outputs:
                        tgt_input_types = set(th.get("inputTypes") or [])
                        chosen = next((o for o in src_outputs if o.get("name") == sh.get("name")), None)
                        if chosen is None:
                            chosen = next(
                                (o for o in src_outputs if tgt_input_types & set(o.get("types") or [])),
                                None,
                            ) or src_outputs[0]
                        sh["name"] = chosen.get("name", sh.get("name"))
                        sh["output_types"] = chosen.get("types", sh.get("output_types"))
                        sh["dataType"] = src_comp_type

            # Serialize handles using Langflow's œ-encoding (frontend np() function:
            # JSON.stringify(obj).replace(/"/g, "œ")) — sun() validation requires this format.
            # JS JSON.stringify is compact (no spaces); Python's default json.dumps emits
            # ", " / ": " separators, which makes the handle string fail to match the id the
            # frontend computes and silently drops every edge. Force compact separators.
            edge["sourceHandle"] = json.dumps(sh, separators=(",", ":")).replace('"', 'œ')
            edge["targetHandle"] = json.dumps(th, separators=(",", ":")).replace('"', 'œ')

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

    def fix_selected_outputs(self, nodes: list[dict], edges: list[dict]) -> None:
        """Set selected_output on dual-output LLM nodes from actual wiring.

        Nodes with both model_output (LanguageModel) and text_output (Message) must
        select the right active handle based on what they connect to:
          - Any outgoing edge whose targetHandle.type == 'model' → model_output
            (consumed by Agent.model or any future LanguageModel-consuming field)
          - Otherwise → text_output (Message — for ChatOutput, prompts, answer gen)

        Called after enrich_edges so targetHandle.type is already resolved from schema.
        Works for any LLM component regardless of name (AzureOpenAI, Anthropic, OpenAI…).
        """
        node_by_id = {n.get("id", ""): n for n in nodes}
        for node in nodes:
            outputs = node.get("data", {}).get("node", {}).get("outputs", [])
            output_names = {o.get("name") for o in outputs}
            if "model_output" not in output_names or "text_output" not in output_names:
                continue
            node_id = node.get("id", "")
            feeds_model_input = any(
                self._parse_handle(e.get("targetHandle") or {}).get("type") == "model"
                for e in edges if e.get("source") == node_id
            )
            node["data"]["selected_output"] = "model_output" if feeds_model_input else "text_output"

    @staticmethod
    def find_node_by_type(nodes: list[dict], type_name: str) -> dict | None:
        """Return first node whose data.type matches type_name, or None."""
        for n in nodes:
            if n.get("data", {}).get("type") == type_name:
                return n
        return None

    @staticmethod
    def find_llm_node(nodes: list[dict]) -> dict | None:
        """Return first node whose outputs include LanguageModel, prioritising AzureOpenAIModel.

        Detection is output-type driven, not name-driven, so any LLM component
        (LanguageModelComponent, OpenAI, Anthropic, etc.) is matched.
        """
        def outputs_langmodel(n: dict) -> bool:
            return any(
                "LanguageModel" in (o.get("types") or o.get("output_types") or [])
                for o in n.get("data", {}).get("node", {}).get("outputs", [])
            )

        llm_nodes = [n for n in nodes if outputs_langmodel(n)]
        if not llm_nodes:
            return None
        azure = next(
            (n for n in llm_nodes if n.get("data", {}).get("type") == "AzureOpenAIModel"),
            None,
        )
        return azure or llm_nodes[0]

    @staticmethod
    def find_agent_node(nodes: list[dict]) -> dict | None:
        """Return first node that has a `tools` template field (i.e., any Agent variant).

        Match by template structure, not type name, so ToolCallingAgent / custom agents work.
        """
        for n in nodes:
            tmpl = n.get("data", {}).get("node", {}).get("template", {})
            if "tools" in tmpl:
                return n
        return None

    @staticmethod
    def offset_new_positions(
        existing: list[dict],
        additions: list[dict],
        x_gap: int = 350,
        y_gap: int = 200,
    ) -> None:
        """Mutate additions in place: assign positions to nodes lacking explicit ones.

        New nodes are stacked vertically to the RIGHT of the existing canvas so they
        sit alongside existing nodes inside the same viewport (rather than far below
        where users have to scroll to find them). Nodes that already carry a complete
        position dict are left alone.
        """
        existing_positions = [
            n.get("position", {})
            for n in existing
            if isinstance(n.get("position"), dict)
        ]
        if existing_positions:
            base_x = max(p.get("x", 0) for p in existing_positions) + x_gap
            base_y = min(p.get("y", 0) for p in existing_positions)
        else:
            base_x = 250
            base_y = 200
        next_y = base_y
        for n in additions:
            pos = n.get("position")
            if isinstance(pos, dict) and "x" in pos and "y" in pos:
                continue
            n["position"] = {"x": base_x, "y": next_y}
            next_y += y_gap

    @staticmethod
    def classify_update_payload(
        payload_data: dict | None,
        existing_node_ids: set[str],
    ) -> str:
        """Return 'patch_meta', 'full_replace', or 'merge'.

        - patch_meta:  payload_data has no `nodes` key (rename / folder move only).
        - full_replace: every existing node id appears in payload (LLM resent whole flow).
        - merge:       payload contains a delta (some new ids, some/no existing ids).
        """
        if not isinstance(payload_data, dict) or "nodes" not in payload_data:
            return "patch_meta"
        payload_ids = {n.get("id", "") for n in payload_data.get("nodes", []) if n.get("id")}
        if existing_node_ids and existing_node_ids.issubset(payload_ids):
            return "full_replace"
        return "merge"

    @staticmethod
    def merge_flow_data(existing_data: dict, payload_data: dict) -> dict:
        """Merge payload's new nodes/edges into existing_data; existing entries take precedence by id."""
        existing_nodes = list(existing_data.get("nodes", []))
        existing_edges = list(existing_data.get("edges", []))
        existing_node_ids = {n.get("id", "") for n in existing_nodes}
        existing_edge_ids = {e.get("id", "") for e in existing_edges if e.get("id")}

        addition_nodes = [
            n for n in payload_data.get("nodes", [])
            if n.get("id") and n.get("id") not in existing_node_ids
        ]
        addition_edges = [
            e for e in payload_data.get("edges", [])
            if (not e.get("id")) or e.get("id") not in existing_edge_ids
        ]
        merged = dict(existing_data)
        merged["nodes"] = existing_nodes + addition_nodes
        merged["edges"] = existing_edges + addition_edges
        return merged

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
        # Detect tool-capable nodes: tool_mode=True OR has component_as_tool/api_build_tool output
        tool_nodes = [
            n for n in nodes
            if n.get("data", {}).get("node", {}).get("tool_mode")
            or any(
                o.get("name") in ("api_build_tool", "component_as_tool")
                for o in n.get("data", {}).get("node", {}).get("outputs", [])
            )
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
                # Priority: component_as_tool/api_build_tool (native) → tool_mode output → fallback
                tool_out = (
                    next((o for o in outputs if o.get("name") in ("component_as_tool", "api_build_tool")), None)
                    or next((o for o in outputs if o.get("tool_mode")), None)
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

    def get_component_schema(self, type_name: str) -> dict:
        """Return compact schema (inputs + outputs) for one component. Uses cached /api/v1/all data."""
        schemas = self._fetch_component_schemas()
        # Resolve display names / casing variants ("SQL Database" → "SQLComponent",
        # "Prompt Template" → "Prompt") so callers don't have to guess the exact key.
        resolved = self._resolve_type(type_name, schemas)
        if resolved not in schemas:
            return {"error": f"Unknown type: {type_name!r}. Call list_components to find the exact type string."}
        type_name = resolved
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
        result = {"type": type_name, "inputs": inputs, "outputs": outs}
        # Component quality (schema-driven). legacy is hard-blocked at build time — the agent
        # must avoid these and decompose into non-legacy primitives. Flag only when set, to
        # keep the compact schema small.
        if schema.get("legacy"):
            result["legacy"] = True
        if schema.get("beta"):
            result["beta"] = True
        return result
