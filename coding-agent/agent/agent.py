import asyncio
import json
import random
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from llm.base import LLMProvider
from mcpbridge.client import LangflowMCPClient
from config.settings import Settings
from agent.prompts import SYSTEM_PROMPT
from agent.events import ConsoleSink, slim_graph

console = Console()


def _canonical_handle(handle):
    """Normalize a react-flow handle to Langflow's canonical string form.

    Langflow's frontend matches edges to handles via ``scapedJSONStringfy``,
    which is ``JSON.stringify(obj)`` — compact, NO spaces after ':' or ','.
    Python's default ``json.dumps`` emits ``", "`` / ``": "`` separators, so an
    object-based handle serialized that way never matches the handle id the
    frontend computes from the node template, and the edge is silently dropped
    on render. Always emit the compact form; also re-normalize handles that
    arrive already-stringified (possibly space-polluted, possibly œ-escaped) so
    re-saving repairs previously-broken flows.
    """
    if isinstance(handle, dict):
        return json.dumps(handle, separators=(",", ":"))
    if isinstance(handle, str):
        try:
            obj = json.loads(handle.replace("œ", '"'))
        except (ValueError, TypeError):
            return handle
        canonical = json.dumps(obj, separators=(",", ":"))
        # Preserve œ-escaping if the source used it (Langflow's wire format).
        return canonical.replace('"', "œ") if "œ" in handle else canonical
    return handle


def _serialize_edge_handles(edges: list[dict]) -> list[dict]:
    """Serialize sourceHandle/targetHandle to Langflow's canonical handle form
    and ensure each edge has an id. React-flow requires handle identifiers to be
    compact JSON strings, not objects, and not space-padded.
    """
    result = []
    for i, edge in enumerate(edges):
        edge = dict(edge)
        if edge.get("sourceHandle") is not None:
            edge["sourceHandle"] = _canonical_handle(edge["sourceHandle"])
        if edge.get("targetHandle") is not None:
            edge["targetHandle"] = _canonical_handle(edge["targetHandle"])
        if "id" not in edge:
            edge["id"] = f"{edge.get('source', 'src')}-{edge.get('target', 'tgt')}-{i}"
        result.append(edge)
    return result


def _inject_node_check(
    get_flow_result: str | None,
    mcp: "LangflowMCPClient",
    flow_id: str,
) -> str:
    """Check node/edge count and test-run the flow. Append VERIFIED or FAILED to the tool result."""
    if not get_flow_result:
        return get_flow_result or "null"
    try:
        data = json.loads(get_flow_result)
        flow_data = data.get("data", {}) if isinstance(data, dict) else {}
        nodes = flow_data.get("nodes", [])
        edges = flow_data.get("edges", [])
        n_nodes = len(nodes)
        n_edges = len(edges)

        if n_nodes == 0:
            return (
                get_flow_result
                + "\n\n⚠ VERIFICATION FAILED: 0 nodes in flow after build. "
                "All your node type strings were rejected by Langflow. "
                "Call list_components, read exact 'type' field values, retry update_flow. "
                "Do NOT report success until node count > 0."
            )

        # Detect silently-dropped edges (Langflow removes invalid connections without error)
        edge_warning = ""
        min_expected = n_nodes - 1
        if n_nodes > 4 and n_edges < min_expected:
            edge_warning = (
                f"\n\n⚠ EDGE WARNING: {n_nodes} nodes but only {n_edges} edges "
                f"(expected ≥{min_expected}). Langflow removed invalid connections. "
                "Call get_component_schema for each non-core component, verify fieldNames "
                "and output names, then update_flow with corrected edges. Do NOT report "
                "success — the flow is partially wired."
            )

        # Test-run to confirm execution
        test = mcp.test_run_flow(flow_id)
        if test["ok"]:
            suffix = (
                f"\n\n✅ VERIFIED: Flow executed successfully. "
                f"Test input '2+2' → '{test['answer']}'. "
                f"Flow has {n_nodes} nodes, {n_edges} edges."
            )
            if edge_warning:
                suffix += " However, edge count is low — fix wiring before reporting success."
            else:
                suffix += " You may report success."
            return get_flow_result + edge_warning + suffix
        else:
            return (
                get_flow_result
                + edge_warning
                + f"\n\n⚠ EXECUTION FAILED: Flow has {n_nodes} nodes but failed to run: "
                f"{test['error']}. "
                "Do NOT report success. Fix credentials/wiring/config and retry."
            )
    except (json.JSONDecodeError, AttributeError):
        pass
    return get_flow_result


def _tool_result_message(tool_call_id: str, result: str | None) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result or "null",
    }


def _compact_tool_args(tc: dict) -> dict:
    """Strip large data payloads from create/update tool calls before storing in history.
    These calls carry full node schemas (~5K tokens each) that repeat in every LLM call."""
    if tc["name"] in ("create_flow", "update_flow"):
        args = dict(tc["arguments"])
        if isinstance(args.get("data"), dict):
            d = args["data"]
            n = len(d.get("nodes", []))
            e = len(d.get("edges", []))
            args["data"] = f"<{n} nodes, {e} edges — payload stripped>"
        return args
    return tc["arguments"]


def _assistant_tool_call_message(tool_calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(_compact_tool_args(tc)),
                },
            }
            for tc in tool_calls
        ],
    }



def _is_langmodel_node(n: dict) -> bool:
    return any(
        "LanguageModel" in (o.get("types") or o.get("output_types") or [])
        for o in n.get("data", {}).get("node", {}).get("outputs", [])
    )

def _find_model_field(n: dict) -> str | None:
    """Return template field name of the first ModelInput that accepts LanguageModel, or None."""
    tmpl = n.get("data", {}).get("node", {}).get("template", {})
    for k, v in tmpl.items():
        if isinstance(v, dict) and v.get("type") == "model" and "LanguageModel" in (v.get("input_types") or []):
            return k
    return None

def _has_only_tool_outputs(n: dict) -> bool:
    """True if a node's only outputs are tool handles (component_as_tool/api_build_tool).
    Such nodes MUST be used as Agent tools — they have no data output for pipeline use."""
    outputs = n.get("data", {}).get("node", {}).get("outputs", [])
    return bool(outputs) and all(o.get("name") in ("component_as_tool", "api_build_tool") for o in outputs)

_THINKING_WORDS = [
    "Thinking", "Pondering", "Cogitating", "Marinating", "Ruminating",
    "Deliberating", "Analyzing", "Reasoning", "Contemplating", "Brewing",
    "Calculating", "Reflecting", "Theorizing", "Synthesizing", "Formulating",
    "Extrapolating", "Inferring", "Deducing", "Hallucinating", "Vibing",
    "Cooking", "Simmering", "Baking", "Percolating", "Noodling",
]


async def _cycle_status(status_obj: object) -> None:
    while True:
        await asyncio.sleep(2)
        status_obj.update(f"[dim]{random.choice(_THINKING_WORDS)}…[/dim]")


async def run_turn(llm, mcp, settings, tools, messages, _starter_cache, sink):
    iterations = 0
    prompt_tokens = 0
    completion_tokens = 0
    turn_start = time.perf_counter()

    while iterations < settings.max_tool_iterations:
        tools = mcp.get_tool_schemas()  # rebuilt each iteration so discovered tools appear next turn
        try:
            t0 = time.perf_counter()
            with console.status(f"[dim]{random.choice(_THINKING_WORDS)}…[/dim]", spinner="dots") as _status:
                _cycle = asyncio.create_task(_cycle_status(_status))
                try:
                    response = await llm.complete(messages, tools, system=SYSTEM_PROMPT)
                finally:
                    _cycle.cancel()
            elapsed = time.perf_counter() - t0
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[dim]Interrupted.[/dim]")
            raise

        iter_prompt = 0
        iter_completion = 0
        cache_read = 0
        cache_write = 0
        if response["usage"]:
            iter_prompt = response["usage"]["prompt_tokens"]
            iter_completion = response["usage"]["completion_tokens"]
            cache_read = response["usage"].get("cache_read_tokens", 0) or 0
            cache_write = response["usage"].get("cache_creation_tokens", 0) or 0
            prompt_tokens += iter_prompt
            completion_tokens += iter_completion
        cache_str = f" · 📦 r={cache_read:,} w={cache_write:,}" if cache_read or cache_write else ""
        console.print(
            f"[dim]⏱ {elapsed:.1f}s · ↑{iter_prompt:,} ↓{iter_completion:,} tokens{cache_str}[/dim]"
        )

        if response["tool_calls"]:
            for tc in response["tool_calls"]:
                args_str = json.dumps(tc['arguments'], indent=None)
                if len(args_str) > 300:
                    args_str = args_str[:300] + "…"
                console.print(f"[dim]→ {tc['name']}({args_str})[/dim]")
                sink.tool_call(tc['name'], tc['arguments'])

            messages.append(_assistant_tool_call_message(response["tool_calls"]))

            # Build credential overrides once per turn — reused by create_flow, update_flow, clone_starter_template
            _credential_overrides: dict = {}
            if settings.azure_anthropic_api_key:
                _credential_overrides["AnthropicModel"] = {
                    "api_key": settings.azure_anthropic_api_key,
                    "base_url": settings.azure_anthropic_endpoint,
                    "model_name": settings.azure_anthropic_deployment,
                }
                _credential_overrides["Agent"] = {
                    "api_key": settings.azure_anthropic_api_key,
                }
            if settings.azure_openai_endpoint:
                _credential_overrides["AzureOpenAIModel"] = {
                    "azure_endpoint": settings.azure_openai_endpoint,
                    "api_key": settings.azure_openai_api_key,
                    "azure_deployment": settings.azure_openai_deployment,
                    "api_version": settings.azure_openai_api_version,
                }

            _last_build_flow_id: str | None = None
            _starter_template_msg_ids: set[str] = set()
            for tc in response["tool_calls"]:
                args = tc["arguments"]
                _canvas_graph: dict | None = None  # slim graph for the live canvas, set when a tool returns flow data

                # Auto-enrich nodes with full component schemas + credentials before sending to Langflow.
                # create_flow and update_flow have distinct semantics:
                #   create: build-from-scratch, dedup structural singletons, inject hardcoded stub IDs
                #   update: fetch existing, merge delta, preserve existing IDs (no destructive dedup)
                enrich_error: str | None = None

                def _node_outputs_langmodel(n: dict) -> bool:
                    return any(
                        "LanguageModel" in (o.get("types") or o.get("output_types") or [])
                        for o in n.get("data", {}).get("node", {}).get("outputs", [])
                    )

                def _enrich_create_data(data: dict) -> dict:
                    # Deduplicate structural singletons (non-LLM) by type name
                    _STRUCTURAL_SINGLETONS = {"ChatInput", "ChatOutput", "Agent"}
                    seen_singletons: set[str] = set()
                    deduped: list[dict] = []
                    removed_ids: set[str] = set()
                    for _n in data["nodes"]:
                        _nt = _n.get("data", {}).get("type") or _n.get("type", "")
                        if _nt in _STRUCTURAL_SINGLETONS and _nt in seen_singletons:
                            removed_ids.add(_n.get("id", ""))
                            console.print(f"[yellow]↳ dedup: removed extra '{_nt}' node[/yellow]")
                            continue
                        seen_singletons.add(_nt)
                        deduped.append(_n)
                    if removed_ids:
                        data["nodes"] = deduped
                    # Drop orphan edges before enrichment
                    valid_node_ids = {n.get("id", "") for n in data["nodes"]}
                    orphans_before = len(data.get("edges", []))
                    data["edges"] = [
                        e for e in data.get("edges", [])
                        if e.get("source") in valid_node_ids and e.get("target") in valid_node_ids
                    ]
                    orphans_dropped = orphans_before - len(data["edges"])
                    if orphans_dropped:
                        console.print(f"[yellow]↳ dropped {orphans_dropped} orphan edge(s) with missing node references[/yellow]")
                    data["nodes"] = mcp.enrich_nodes(data["nodes"], credential_overrides=_credential_overrides)
                    # Dynamic LLM dedup (post-enrichment): always keeps AzureOpenAIModel.
                    llm_nodes = [n for n in data["nodes"] if _node_outputs_langmodel(n)]
                    azure_llm = next((n for n in llm_nodes if n.get("data", {}).get("type") == "AzureOpenAIModel"), None)
                    if not azure_llm:
                        if llm_nodes:
                            remove_ids = {n.get("id", "") for n in llm_nodes}
                            data["nodes"] = [n for n in data["nodes"] if n.get("id", "") not in remove_ids]
                            data["edges"] = [
                                e for e in data.get("edges", [])
                                if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                            ]
                            for rid in remove_ids:
                                console.print(f"[yellow]↳ replaced non-Azure LLM '{rid}' with AzureOpenAIModel[/yellow]")
                        removed_pos = next(
                            (n.get("position", {"x": 250, "y": 200}) for n in llm_nodes),
                            {"x": 250, "y": 200},
                        )
                        azure_stub = [{"id": "AzureOpenAIModel-1", "type": "AzureOpenAIModel", "position": removed_pos, "data": {"type": "AzureOpenAIModel", "id": "AzureOpenAIModel-1"}}]
                        data["nodes"].extend(mcp.enrich_nodes(azure_stub, credential_overrides=_credential_overrides))
                    # Wire AzureOpenAI to ALL nodes with ModelInput fields
                    for _n in data["nodes"]:
                        if _n.get("data", {}).get("type") == "AzureOpenAIModel":
                            continue
                        _mf = _find_model_field(_n)
                        if _mf:
                            data["edges"].append({
                                "source": "AzureOpenAIModel-1",
                                "target": _n.get("id"),
                                "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                                "targetHandle": {"fieldName": _mf, "id": _n.get("id"), "inputTypes": ["LanguageModel"], "type": "model"},
                            })
                    console.print("[dim]↳ wired AzureOpenAIModel → all ModelInput fields[/dim]")
                    # Inject Agent when tool-only components exist without one
                    _cf_has_agent = any(n.get("data", {}).get("type") == "Agent" for n in data["nodes"])
                    _cf_has_tool_only = any(_has_only_tool_outputs(n) for n in data["nodes"])
                    if _cf_has_tool_only and not _cf_has_agent:
                        _agent_stub = [{"id": "Agent-1", "type": "Agent", "position": {"x": 670, "y": 540}, "data": {"type": "Agent", "id": "Agent-1"}}]
                        data["nodes"].extend(mcp.enrich_nodes(_agent_stub, credential_overrides=_credential_overrides))
                        _ci = next((n for n in data["nodes"] if n.get("data", {}).get("type") == "ChatInput"), None)
                        if _ci:
                            data["edges"].append({
                                "source": _ci.get("id"), "target": "Agent-1",
                                "sourceHandle": {"dataType": "ChatInput", "id": _ci.get("id"), "name": "message", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": "Agent-1", "inputTypes": ["Message"], "type": "str"},
                            })
                        data["edges"].append({
                            "source": "AzureOpenAIModel-1", "target": "Agent-1",
                            "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                            "targetHandle": {"fieldName": "model", "id": "Agent-1", "inputTypes": ["LanguageModel"], "type": "model"},
                        })
                        _co = next((n for n in data["nodes"] if n.get("data", {}).get("type") == "ChatOutput"), None)
                        if _co:
                            data["edges"].append({
                                "source": "Agent-1", "target": _co.get("id"),
                                "sourceHandle": {"dataType": "Agent", "id": "Agent-1", "name": "response", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": _co.get("id"), "inputTypes": ["Data", "JSON", "DataFrame", "Table", "Message"], "type": "other"},
                            })
                        console.print("[dim]↳ injected Agent + wired ChatInput→Agent→ChatOutput[/dim]")
                    if azure_llm and len(llm_nodes) > 1:
                        extra_llm_ids = {n.get("id", "") for n in llm_nodes if n is not azure_llm}
                        data["nodes"] = [n for n in data["nodes"] if n.get("id", "") not in extra_llm_ids]
                        data["edges"] = [
                            e for e in data.get("edges", [])
                            if e.get("source") not in extra_llm_ids and e.get("target") not in extra_llm_ids
                        ]
                        for eid in extra_llm_ids:
                            console.print(f"[yellow]↳ dedup: removed extra LLM node '{eid}'[/yellow]")
                    data["edges"] = mcp.ensure_tool_edges(data["nodes"], data.get("edges", []))
                    if "edges" in data:
                        data["edges"] = mcp.enrich_edges(data["edges"], data["nodes"])
                    mcp.fix_selected_outputs(data["nodes"], data.get("edges", []))
                    return data

                def _enrich_update_merge(existing_data: dict, payload_data: dict) -> dict:
                    existing_nodes = list(existing_data.get("nodes", []) or [])
                    existing_edges = list(existing_data.get("edges", []) or [])

                    # Honor explicit removal lists in the payload. Lets LLM combine
                    # "add X / remove Y" in one update_flow call without falling into
                    # the union-only merge trap.
                    remove_ids: set[str] = set(payload_data.get("remove_node_ids") or [])
                    remove_types: set[str] = set(payload_data.get("remove_types") or [])
                    if remove_types:
                        for n in existing_nodes:
                            t = (n.get("data") or {}).get("type") or n.get("type", "")
                            if t in remove_types and n.get("id"):
                                remove_ids.add(n["id"])
                    if remove_ids:
                        existing_nodes = [n for n in existing_nodes if n.get("id") not in remove_ids]
                        existing_edges = [
                            e for e in existing_edges
                            if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                        ]
                        console.print(f"[dim]↳ removing {len(remove_ids)} node(s): {sorted(remove_ids)}[/dim]")
                        # Strip control fields so they don't leak into PATCH body
                        payload_data = {k: v for k, v in payload_data.items() if k not in ("remove_node_ids", "remove_types")}

                    existing_node_ids = {n.get("id", "") for n in existing_nodes if n.get("id")}
                    existing_edge_ids = {e.get("id", "") for e in existing_edges if e.get("id")}

                    def _edge_fingerprint(e: dict) -> tuple:
                        """Structural identity: same source node + target node + target field = same edge.
                        Used to dedup edges the LLM resends without IDs (ID-based dedup misses these)."""
                        th = e.get("targetHandle") or {}
                        if isinstance(th, str):
                            try:
                                th = json.loads(th.replace('œ', '"'))
                            except Exception:
                                th = {}
                        return (e.get("source", ""), e.get("target", ""), th.get("fieldName", ""))

                    existing_edge_fingerprints = {_edge_fingerprint(e) for e in existing_edges}

                    # Type → existing-node-id map. LLMs routinely invent fresh IDs
                    # ("ChatInput-1") for components that already exist in the flow under
                    # UUIDs. Without remapping we'd duplicate every structural node.
                    existing_by_type: dict[str, str] = {}
                    for _n in existing_nodes:
                        _t = _n.get("data", {}).get("type") or _n.get("type", "")
                        if _t and _t not in existing_by_type:
                            existing_by_type[_t] = _n.get("id", "")

                    id_map: dict[str, str] = {}
                    for _n in payload_data.get("nodes", []):
                        _nid = _n.get("id", "")
                        if not _nid or _nid in existing_node_ids:
                            continue
                        _t = _n.get("data", {}).get("type") or _n.get("type", "")
                        if _t in existing_by_type:
                            id_map[_nid] = existing_by_type[_t]

                    if id_map:
                        for _src, _dst in id_map.items():
                            console.print(f"[dim]↳ remap payload id '{_src}' → existing '{_dst}' (same type)[/dim]")

                    def _remap_id(node_id: str) -> str:
                        return id_map.get(node_id, node_id)

                    def _remap_edge(e: dict) -> dict:
                        e2 = dict(e)
                        if e2.get("source") in id_map:
                            e2["source"] = id_map[e2["source"]]
                        if e2.get("target") in id_map:
                            e2["target"] = id_map[e2["target"]]
                        for hk in ("sourceHandle", "targetHandle"):
                            h = e2.get(hk)
                            if isinstance(h, dict) and h.get("id") in id_map:
                                h = dict(h)
                                h["id"] = id_map[h["id"]]
                                e2[hk] = h
                        return e2

                    addition_nodes = [
                        n for n in payload_data.get("nodes", [])
                        if n.get("id")
                        and n.get("id") not in existing_node_ids
                        and n.get("id") not in id_map
                    ]
                    addition_edges = [
                        _remap_edge(e) for e in payload_data.get("edges", [])
                        if ((not e.get("id")) or e.get("id") not in existing_edge_ids)
                        and _edge_fingerprint(_remap_edge(e)) not in existing_edge_fingerprints
                    ]
                    console.print(f"[dim]↳ merging {len(addition_nodes)} new node(s), {len(addition_edges)} new edge(s) into flow[/dim]")

                    existing_llm = mcp.find_llm_node(existing_nodes)
                    existing_agent = mcp.find_agent_node(existing_nodes)

                    mcp.offset_new_positions(existing_nodes, addition_nodes)
                    enriched_additions = (
                        mcp.enrich_nodes(addition_nodes, credential_overrides=_credential_overrides)
                        if addition_nodes else []
                    )

                    if existing_llm is None:
                        added_llm = next((n for n in enriched_additions if _node_outputs_langmodel(n)), None)
                        if added_llm and added_llm.get("data", {}).get("type") == "AzureOpenAIModel":
                            llm_id = added_llm.get("id")
                        elif added_llm:
                            # Non-Azure LLM in additions → replace with AzureOpenAIModel stub
                            remove_id = added_llm.get("id", "")
                            pos = added_llm.get("position", {"x": 250, "y": 200})
                            enriched_additions = [n for n in enriched_additions if n.get("id", "") != remove_id]
                            addition_edges = [
                                e for e in addition_edges
                                if e.get("source") != remove_id and e.get("target") != remove_id
                            ]
                            console.print(f"[yellow]↳ replaced non-Azure LLM '{remove_id}' with AzureOpenAIModel[/yellow]")
                            stub_id = f"AzureOpenAIModel-{int(time.time() * 1000) % 1000000}"
                            stub = [{"id": stub_id, "type": "AzureOpenAIModel", "position": pos, "data": {"type": "AzureOpenAIModel", "id": stub_id}}]
                            enriched_additions.extend(mcp.enrich_nodes(stub, credential_overrides=_credential_overrides))
                            llm_id = stub_id
                        else:
                            needs_model = any(_find_model_field(n) for n in enriched_additions)
                            if needs_model:
                                stub_id = f"AzureOpenAIModel-{int(time.time() * 1000) % 1000000}"
                                stub = [{"id": stub_id, "type": "AzureOpenAIModel", "position": {"x": 250, "y": 200}, "data": {"type": "AzureOpenAIModel", "id": stub_id}}]
                                enriched_additions.extend(mcp.enrich_nodes(stub, credential_overrides=_credential_overrides))
                                llm_id = stub_id
                                console.print(f"[dim]↳ injected AzureOpenAIModel '{stub_id}' for new ModelInput fields[/dim]")
                            else:
                                llm_id = None
                    else:
                        llm_id = existing_llm.get("id")
                        # Drop duplicate LLMs from additions (existing wins)
                        dup_llm_ids = {n.get("id", "") for n in enriched_additions if _node_outputs_langmodel(n)}
                        if dup_llm_ids:
                            enriched_additions = [n for n in enriched_additions if n.get("id", "") not in dup_llm_ids]
                            addition_edges = [
                                e for e in addition_edges
                                if e.get("source") not in dup_llm_ids and e.get("target") not in dup_llm_ids
                            ]
                            for rid in dup_llm_ids:
                                console.print(f"[yellow]↳ dropped duplicate LLM '{rid}' from additions (existing LLM kept)[/yellow]")

                    # Wire any new ModelInput fields → discovered llm_id (not hardcoded)
                    if llm_id:
                        llm_node_lookup = mcp.find_node_by_type(existing_nodes + enriched_additions, "AzureOpenAIModel")
                        llm_type = (llm_node_lookup or existing_llm or {}).get("data", {}).get("type", "AzureOpenAIModel")
                        for n in enriched_additions:
                            if n.get("id") == llm_id:
                                continue
                            mf = _find_model_field(n)
                            if mf:
                                addition_edges.append({
                                    "source": llm_id, "target": n.get("id"),
                                    "sourceHandle": {"dataType": llm_type, "id": llm_id, "name": "model_output", "output_types": ["LanguageModel"]},
                                    "targetHandle": {"fieldName": mf, "id": n.get("id"), "inputTypes": ["LanguageModel"], "type": "model"},
                                })

                    # Inject Agent only when additions need one AND none exists anywhere
                    has_tool_only = any(_has_only_tool_outputs(n) for n in enriched_additions)
                    has_agent_anywhere = existing_agent is not None or any(
                        n.get("data", {}).get("type") == "Agent" for n in enriched_additions
                    )
                    if has_tool_only and not has_agent_anywhere:
                        stub_id = f"Agent-{int(time.time() * 1000) % 1000000}"
                        stub = [{"id": stub_id, "type": "Agent", "position": {"x": 670, "y": 540}, "data": {"type": "Agent", "id": stub_id}}]
                        enriched_additions.extend(mcp.enrich_nodes(stub, credential_overrides=_credential_overrides))
                        ci = mcp.find_node_by_type(existing_nodes + enriched_additions, "ChatInput")
                        if ci:
                            addition_edges.append({
                                "source": ci.get("id"), "target": stub_id,
                                "sourceHandle": {"dataType": "ChatInput", "id": ci.get("id"), "name": "message", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": stub_id, "inputTypes": ["Message"], "type": "str"},
                            })
                        if llm_id:
                            addition_edges.append({
                                "source": llm_id, "target": stub_id,
                                "sourceHandle": {"dataType": "AzureOpenAIModel", "id": llm_id, "name": "model_output", "output_types": ["LanguageModel"]},
                                "targetHandle": {"fieldName": "model", "id": stub_id, "inputTypes": ["LanguageModel"], "type": "model"},
                            })
                        co = mcp.find_node_by_type(existing_nodes + enriched_additions, "ChatOutput")
                        if co:
                            addition_edges.append({
                                "source": stub_id, "target": co.get("id"),
                                "sourceHandle": {"dataType": "Agent", "id": stub_id, "name": "response", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": co.get("id"), "inputTypes": ["Data", "JSON", "DataFrame", "Table", "Message"], "type": "other"},
                            })
                        console.print(f"[dim]↳ injected Agent '{stub_id}' for tool-only additions[/dim]")

                    merged_nodes = existing_nodes + enriched_additions
                    merged_edges = existing_edges + addition_edges
                    merged_edges = mcp.ensure_tool_edges(merged_nodes, merged_edges)
                    merged_edges = mcp.enrich_edges(merged_edges, merged_nodes)
                    mcp.fix_selected_outputs(merged_nodes, merged_edges)

                    # Regression guard: every existing id must survive merge
                    final_ids = {n.get("id", "") for n in merged_nodes}
                    dropped = existing_node_ids - final_ids
                    if dropped:
                        raise RuntimeError(f"merge would drop existing nodes: {sorted(dropped)}")

                    return {**payload_data, "nodes": merged_nodes, "edges": merged_edges}

                if tc["name"] == "create_flow":
                    data = args.get("data", {})
                    if isinstance(data, dict) and "nodes" in data:
                        try:
                            data = _enrich_create_data(data)
                            args = {**args, "data": data}
                            console.print("[dim]↳ enriched nodes with component schemas + credentials[/dim]")
                        except Exception as enrich_err:
                            enrich_error = str(enrich_err)
                            console.print(f"[red]✗ schema enrichment failed: {enrich_err}[/red]")
                            console.print("[dim]↳ skipping call to prevent broken flow[/dim]")

                elif tc["name"] == "update_flow":
                    payload_data = args.get("data")
                    flow_id = args.get("flow_id")
                    if isinstance(payload_data, dict) and "nodes" in payload_data and flow_id:
                        try:
                            existing_raw = await mcp.call_tool("get_flow", {"flow_id": flow_id})
                            existing = json.loads(existing_raw) if isinstance(existing_raw, str) else existing_raw
                            existing_data = existing.get("data", {}) if isinstance(existing, dict) else {}
                            existing_node_ids = {n.get("id", "") for n in existing_data.get("nodes", []) if n.get("id")}

                            mode = mcp.classify_update_payload(payload_data, existing_node_ids)
                            console.print(f"[dim]↳ update_flow mode: {mode}[/dim]")

                            if mode == "full_replace":
                                payload_data = _enrich_create_data(payload_data)
                            else:
                                payload_data = _enrich_update_merge(existing_data, payload_data)
                            args = {**args, "data": payload_data}
                        except Exception as enrich_err:
                            enrich_error = str(enrich_err)
                            console.print(f"[red]✗ update_flow enrichment failed: {enrich_err}[/red]")
                            console.print("[dim]↳ skipping call to prevent broken flow[/dim]")

                if enrich_error:
                    # Refuse to send broken nodes to Langflow. Return error as tool result so LLM retries.
                    result = f"ERROR: {enrich_error} Do NOT retry blindly — call list_components first and find the exact 'type' string."
                elif tc["name"] == "get_component_schema":
                    result = json.dumps(mcp.get_component_schema(args.get("type_name", "")))
                elif tc["name"] == "clone_starter_template":
                    # Server-side clone: fetch template → enrich → POST directly, zero LLM token cost
                    name_or_id = (args.get("name_or_id") or "").strip()
                    flow_name = (args.get("name") or name_or_id).strip()
                    flow_desc = args.get("description", "")
                    key = name_or_id.lower()
                    # Lookup chain: in-memory cache → Redis/HTTP via client
                    template = _starter_cache.get(key)
                    if not template:
                        for k, v in _starter_cache.items():
                            if key in k or k in key:
                                template = v
                                break
                    if not template:
                        template = await mcp.fetch_starter(name_or_id)
                    if not template:
                        result = json.dumps({"error": f"Template '{name_or_id}' not found. Call get_basic_examples to populate cache first."})
                    else:
                        try:
                            tdata = json.loads(json.dumps(template.get("data", {})))  # deep copy
                            if not tdata.get("nodes"):
                                result = json.dumps({"error": "Template has no nodes."})
                            else:
                                tdata["nodes"] = mcp.enrich_nodes(tdata["nodes"], credential_overrides=_credential_overrides)
                                # Always ensure AzureOpenAIModel is the LLM.
                                # Simple Agent uses built-in ModelInput (no separate LLM node) — must inject.
                                # Other templates may have a non-Azure LLM node — must replace.
                                llm_nodes = [n for n in tdata["nodes"] if _is_langmodel_node(n)]
                                azure_llm = next((n for n in llm_nodes if n.get("data", {}).get("type") == "AzureOpenAIModel"), None)
                                if not azure_llm:
                                    # Remove any non-Azure LLM nodes that exist
                                    if llm_nodes:
                                        remove_ids = {n.get("id", "") for n in llm_nodes}
                                        tdata["nodes"] = [n for n in tdata["nodes"] if n.get("id", "") not in remove_ids]
                                        tdata["edges"] = [
                                            e for e in tdata.get("edges", [])
                                            if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                                        ]
                                    # Position: near removed LLM or sensible default
                                    removed_pos = next(
                                        (n.get("position", {"x": 250, "y": 200}) for n in llm_nodes),
                                        {"x": 250, "y": 200},
                                    )
                                    azure_stub = [{"id": "AzureOpenAIModel-1", "type": "AzureOpenAIModel", "position": removed_pos, "data": {"type": "AzureOpenAIModel", "id": "AzureOpenAIModel-1"}}]
                                    tdata["nodes"].extend(mcp.enrich_nodes(azure_stub, credential_overrides=_credential_overrides))
                                # Wire AzureOpenAI to ALL nodes with ModelInput fields (Agent, StructuredOutput, etc.)
                                for _n in tdata["nodes"]:
                                    if _n.get("data", {}).get("type") == "AzureOpenAIModel":
                                        continue
                                    _mf = _find_model_field(_n)
                                    if _mf:
                                        tdata["edges"].append({
                                            "source": "AzureOpenAIModel-1",
                                            "target": _n.get("id"),
                                            "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                                            "targetHandle": {"fieldName": _mf, "id": _n.get("id"), "inputTypes": ["LanguageModel"], "type": "model"},
                                        })
                                console.print("[dim]↳ wired AzureOpenAIModel → all ModelInput fields[/dim]")
                                # Inject Agent when tool-only components exist without one.
                                # Nodes whose only outputs are component_as_tool/api_build_tool cannot
                                # connect via data edges — an Agent is mandatory to use them.
                                _has_agent = any(n.get("data", {}).get("type") == "Agent" for n in tdata["nodes"])
                                _has_tool_only = any(_has_only_tool_outputs(n) for n in tdata["nodes"])
                                if _has_tool_only and not _has_agent:
                                    _agent_stub = [{"id": "Agent-1", "type": "Agent", "position": {"x": 670, "y": 540}, "data": {"type": "Agent", "id": "Agent-1"}}]
                                    tdata["nodes"].extend(mcp.enrich_nodes(_agent_stub, credential_overrides=_credential_overrides))
                                    _ci = next((n for n in tdata["nodes"] if n.get("data", {}).get("type") == "ChatInput"), None)
                                    if _ci:
                                        tdata["edges"].append({
                                            "source": _ci.get("id"), "target": "Agent-1",
                                            "sourceHandle": {"dataType": "ChatInput", "id": _ci.get("id"), "name": "message", "output_types": ["Message"]},
                                            "targetHandle": {"fieldName": "input_value", "id": "Agent-1", "inputTypes": ["Message"], "type": "str"},
                                        })
                                    tdata["edges"].append({
                                        "source": "AzureOpenAIModel-1", "target": "Agent-1",
                                        "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                                        "targetHandle": {"fieldName": "model", "id": "Agent-1", "inputTypes": ["LanguageModel"], "type": "model"},
                                    })
                                    _co = next((n for n in tdata["nodes"] if n.get("data", {}).get("type") == "ChatOutput"), None)
                                    if _co:
                                        tdata["edges"].append({
                                            "source": "Agent-1", "target": _co.get("id"),
                                            "sourceHandle": {"dataType": "Agent", "id": "Agent-1", "name": "response", "output_types": ["Message"]},
                                            "targetHandle": {"fieldName": "input_value", "id": _co.get("id"), "inputTypes": ["Data", "JSON", "DataFrame", "Table", "Message"], "type": "other"},
                                        })
                                    console.print("[dim]↳ injected Agent + wired ChatInput→Agent→ChatOutput[/dim]")
                                tdata["edges"] = mcp.ensure_tool_edges(tdata["nodes"], tdata.get("edges", []))
                                tdata["edges"] = mcp.enrich_edges(tdata["edges"], tdata["nodes"])
                                mcp.fix_selected_outputs(tdata["nodes"], tdata["edges"])
                                created = mcp._create_flow_direct(
                                    name=flow_name or template.get("name", "Cloned Flow"),
                                    description=flow_desc or template.get("description", ""),
                                    data=tdata,
                                )
                                _last_build_flow_id = None  # reset; build_flow will set it
                                result = json.dumps({
                                    "flow_id": created.get("id"),
                                    "name": created.get("name"),
                                    "node_count": len(tdata["nodes"]),
                                    "edge_count": len(tdata["edges"]),
                                })
                                console.print(f"[dim]↳ cloned '{template.get('name')}' → {created.get('id')} ({len(tdata['nodes'])} nodes, {len(tdata['edges'])} edges)[/dim]")
                                sink.flow_built(created.get('id'), graph=slim_graph({"data": tdata}))
                                if mcp._redis_cache and created.get("id") and created.get("name"):
                                    try:
                                        await mcp._redis_cache.upsert_flow(
                                            flow_id=created["id"],
                                            name=created["name"],
                                            description=created.get("description") or "",
                                        )
                                    except Exception:
                                        pass
                        except Exception as clone_err:
                            result = f"ERROR cloning template: {clone_err}"
                elif tc["name"] == "get_starter_template":
                    # Virtual tool — look up cached starter data, return full template for ONE winner
                    key = (args.get("name_or_id") or "").strip().lower()
                    match = _starter_cache.get(key)
                    if not match:
                        # Try partial name match
                        for k, v in _starter_cache.items():
                            if key in k or k in key:
                                match = v
                                break
                    if match:
                        result = json.dumps({"id": match.get("id"), "name": match.get("name"), "data": match.get("data", {})})
                        _starter_template_msg_ids.add(tc["id"])
                        console.print(f"[dim]↳ starter template '{match.get('name')}' served from cache[/dim]")
                    else:
                        result = json.dumps({"error": f"Template '{args.get('name_or_id')}' not found in cache. Call get_basic_examples first."})
                else:
                    try:
                        result = await mcp.call_tool(tc["name"], args)
                    except Exception as e:
                        result = f"ERROR: {e}"

                # Immediately strip template list results — cache full data, send index only to LLM
                if tc["name"] in ("list_starter_projects", "get_basic_examples"):
                    try:
                        parsed = json.loads(result) if isinstance(result, str) else result
                        if isinstance(parsed, list):
                            for t in parsed:
                                tid = str(t.get("id", "")).lower()
                                tname = str(t.get("name", "")).lower()
                                if tid:
                                    _starter_cache[tid] = t
                                if tname:
                                    _starter_cache[tname] = t
                            result = json.dumps([
                                {"id": t.get("id"), "name": t.get("name"), "description": t.get("description", "")}
                                for t in parsed
                            ])
                            console.print(f"[dim]↳ template index: {len(parsed)} templates cached, full data stripped from context[/dim]")
                    except Exception:
                        pass

                if tc["name"] == "build_flow":
                    _last_build_flow_id = tc["arguments"].get("flow_id")
                    sink.flow_built(_last_build_flow_id)
                    # build_flow result is large job metadata — LLM only needs the job_id
                    try:
                        bdata = json.loads(result) if isinstance(result, str) else result
                        if isinstance(bdata, dict) and "id" in bdata:
                            result = json.dumps({"id": bdata["id"], "status": bdata.get("status", "")})
                    except Exception:
                        pass

                # Strip list_components to type+display_name only — full schema is ~1M tokens
                if tc["name"] == "list_components":
                    try:
                        components = json.loads(result) if isinstance(result, str) else result
                        if isinstance(components, list):
                            result = json.dumps([
                                {"type": c.get("type"), "display_name": c.get("display_name", c.get("type"))}
                                for c in components
                            ])
                    except Exception:
                        pass

                # After create/update: LLM has consumed template data — strip get_starter_template
                # results from message history to reclaim the ~63KB they occupy in every subsequent call
                if tc["name"] in ("create_flow", "update_flow") and _starter_template_msg_ids:
                    for msg in messages:
                        if msg.get("role") == "tool" and msg.get("tool_call_id") in _starter_template_msg_ids:
                            try:
                                parsed = json.loads(msg["content"])
                                if isinstance(parsed, dict):
                                    msg["content"] = json.dumps({
                                        "id": parsed.get("id"),
                                        "name": parsed.get("name"),
                                        "_note": "full data stripped after use",
                                    })
                            except Exception:
                                pass
                    _starter_template_msg_ids.clear()

                # Strip data.node schemas from any tool that returns flow JSON.
                # create_flow/update_flow return the full updated flow — same schema bloat as get_flow.
                # Must run before _inject_node_check (which appends text and breaks JSON parsing).
                if tc["name"] in ("get_flow", "create_flow", "update_flow"):
                    try:
                        flow = json.loads(result) if isinstance(result, str) else result
                        if isinstance(flow, dict) and "data" in flow:
                            # Capture the canvas graph BEFORE stripping data.node (labels live there).
                            _canvas_graph = slim_graph(flow)
                            for node in flow.get("data", {}).get("nodes", []):
                                # Preserve data.node for noteNodes: their visible text is
                                # stored in data.node.description, not a component schema.
                                # Stripping it blanks the note permanently on full_replace.
                                if node.get("type") == "noteNode" or node.get("data", {}).get("type") in ("note", "noteNode"):
                                    continue
                                node.get("data", {}).pop("node", None)
                            result = json.dumps({
                                "id": flow.get("id"),
                                "name": flow.get("name"),
                                "data": flow.get("data"),
                            })
                        # Keep Redis cache in sync so search_flows/list_flows reflect
                        # renames and new flows without waiting for background sync.
                        if isinstance(flow, dict) and flow.get("id") and flow.get("name") and mcp._redis_cache:
                            try:
                                await mcp._redis_cache.upsert_flow(
                                    flow_id=flow["id"],
                                    name=flow["name"],
                                    description=flow.get("description") or "",
                                    folder_id=str(flow.get("folder_id") or ""),
                                    updated_at=str(flow.get("updated_at") or ""),
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass

                # After get_flow following a build: check nodes + test-run (runs after truncation)
                if tc["name"] == "get_flow" and _last_build_flow_id:
                    console.print("[dim]↳ verifying flow execution…[/dim]")
                    result = _inject_node_check(result, mcp, _last_build_flow_id)
                    _last_build_flow_id = None

                messages.append(_tool_result_message(tc["id"], str(result)))

                # Canvas sync: after create_flow (else path) capture new flow id;
                # after any write op on the current flow, bust the iframe cache.
                _tc_name = tc["name"]
                _result_str = str(result)
                if _tc_name == "create_flow":
                    try:
                        _r = json.loads(_result_str) if isinstance(_result_str, str) else _result_str
                        if isinstance(_r, dict) and _r.get("id"):
                            sink.flow_built(_r["id"], graph=_canvas_graph)
                    except Exception:
                        pass
                elif sink.flow_id and not _result_str.startswith("ERROR"):
                    _is_write = any(_tc_name.startswith(p) for p in ("delete_", "add_", "update_", "remove_", "patch_"))
                    # Only push to the canvas when the op targets the flow on the canvas —
                    # a get_flow/edit on some other flow must not overwrite what's shown.
                    _target = args.get("flow_id") if isinstance(args, dict) else None
                    _on_canvas = (not _target) or (_target == sink.flow_id)
                    if _on_canvas and (_is_write or _tc_name == "get_flow"):
                        graph = _canvas_graph
                        # delete_node etc. return only a status — refetch the graph so the
                        # canvas reflects the change.
                        if graph is None and _is_write:
                            try:
                                _raw = await mcp.call_tool("get_flow", {"flow_id": sink.flow_id})
                                _f = json.loads(_raw) if isinstance(_raw, str) else _raw
                                graph = slim_graph(_f) if isinstance(_f, dict) else None
                            except Exception:
                                graph = None
                        sink.flow_modified(graph=graph)

            iterations += 1
        else:
            if response["content"]:
                console.print(Markdown(response["content"]))
            messages.append({"role": "assistant", "content": response["content"]})
            sink.final(response["content"])
            if prompt_tokens or completion_tokens:
                total_elapsed = time.perf_counter() - turn_start
                console.print(
                    f"[dim]total: {total_elapsed:.1f}s · ↑{prompt_tokens:,} ↓{completion_tokens:,} tokens[/dim]"
                )
            # Trim history to last 2 user turns + this assistant response.
            # Drops tool call/result messages from prior turns to prevent token explosion.
            prior_user = [m for m in messages if m["role"] == "user"]
            messages = prior_user[-2:] + [{"role": "assistant", "content": response["content"]}]
            break
    else:
        console.print("[yellow]⚠ Max tool iterations reached.[/yellow]")

    return messages


async def run_chat(llm: LLMProvider, mcp: LangflowMCPClient, settings: Settings) -> None:
    tools = mcp.get_tool_schemas()
    messages: list[dict] = []
    _starter_cache: dict[str, dict] = {}  # name_lower/id → full template dict
    sink = ConsoleSink()

    console.print(Panel(
        "[bold green]Langflow Coding Agent[/bold green]\n"
        "Type your request. Ctrl+C or 'exit' to quit.",
        border_style="green",
    ))

    while True:
        try:
            user_input = console.input("[bold cyan]nokia>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            messages = await run_turn(llm, mcp, settings, tools, messages, _starter_cache, sink)
        except (KeyboardInterrupt, asyncio.CancelledError):
            break
