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

console = Console()


def _serialize_edge_handles(edges: list[dict]) -> list[dict]:
    """Serialize sourceHandle/targetHandle as JSON strings and ensure each edge has an id.
    React-flow requires handle identifiers to be strings, not objects.
    """
    result = []
    for i, edge in enumerate(edges):
        edge = dict(edge)
        if isinstance(edge.get("sourceHandle"), dict):
            edge["sourceHandle"] = json.dumps(edge["sourceHandle"])
        if isinstance(edge.get("targetHandle"), dict):
            edge["targetHandle"] = json.dumps(edge["targetHandle"])
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


async def run_chat(llm: LLMProvider, mcp: LangflowMCPClient, settings: Settings) -> None:
    tools = mcp.get_tool_schemas()
    messages: list[dict] = []
    _starter_cache: dict[str, dict] = {}  # name_lower/id → full template dict

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

        iterations = 0
        prompt_tokens = 0
        completion_tokens = 0
        turn_start = time.perf_counter()

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

        while iterations < settings.max_tool_iterations:
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
                return

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

                    # Auto-enrich nodes with full component schemas + credentials before sending to Langflow
                    enrich_error: str | None = None
                    if tc["name"] in ("update_flow", "create_flow"):
                        data = args.get("data", {})
                        if isinstance(data, dict) and "nodes" in data:
                            try:
                                credential_overrides = _credential_overrides
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
                                data["nodes"] = mcp.enrich_nodes(data["nodes"], credential_overrides=credential_overrides)
                                # Dynamic LLM dedup (post-enrichment): detect by LanguageModel output type.
                                # Catches ANY LLM component regardless of name (LanguageModelComponent,
                                # AzureOpenAIModel, OpenAI, etc.) — always keeps AzureOpenAIModel.
                                def _node_outputs_langmodel(n: dict) -> bool:
                                    return any(
                                        "LanguageModel" in (o.get("types") or o.get("output_types") or [])
                                        for o in n.get("data", {}).get("node", {}).get("outputs", [])
                                    )
                                llm_nodes = [n for n in data["nodes"] if _node_outputs_langmodel(n)]
                                if len(llm_nodes) > 1:
                                    preferred_llm = next(
                                        (n for n in llm_nodes if n.get("data", {}).get("type") == "AzureOpenAIModel"),
                                        llm_nodes[0],
                                    )
                                    extra_llm_ids = {n.get("id", "") for n in llm_nodes if n is not preferred_llm}
                                    data["nodes"] = [n for n in data["nodes"] if n.get("id", "") not in extra_llm_ids]
                                    data["edges"] = [
                                        e for e in data.get("edges", [])
                                        if e.get("source") not in extra_llm_ids and e.get("target") not in extra_llm_ids
                                    ]
                                    for eid in extra_llm_ids:
                                        console.print(f"[yellow]↳ dedup: removed extra LLM node '{eid}'[/yellow]")
                                # Auto-add missing tool→Agent edges (LLM often omits them)
                                data["edges"] = mcp.ensure_tool_edges(data["nodes"], data.get("edges", []))
                                if "edges" in data:
                                    data["edges"] = mcp.enrich_edges(data["edges"], data["nodes"])
                                args = {**args, "data": data}
                                console.print("[dim]↳ enriched nodes with component schemas + credentials[/dim]")
                            except Exception as enrich_err:
                                enrich_error = str(enrich_err)
                                console.print(f"[red]✗ schema enrichment failed: {enrich_err}[/red]")
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
                                    tdata["edges"] = mcp.ensure_tool_edges(tdata["nodes"], tdata.get("edges", []))
                                    tdata["edges"] = mcp.enrich_edges(tdata["edges"], tdata["nodes"])
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
                                for node in flow.get("data", {}).get("nodes", []):
                                    node.get("data", {}).pop("node", None)
                                result = json.dumps({
                                    "id": flow.get("id"),
                                    "name": flow.get("name"),
                                    "data": flow.get("data"),
                                })
                        except Exception:
                            pass

                    # After get_flow following a build: check nodes + test-run (runs after truncation)
                    if tc["name"] == "get_flow" and _last_build_flow_id:
                        console.print("[dim]↳ verifying flow execution…[/dim]")
                        result = _inject_node_check(result, mcp, _last_build_flow_id)
                        _last_build_flow_id = None

                    messages.append(_tool_result_message(tc["id"], str(result)))

                iterations += 1
            else:
                if response["content"]:
                    console.print(Markdown(response["content"]))
                messages.append({"role": "assistant", "content": response["content"]})
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
