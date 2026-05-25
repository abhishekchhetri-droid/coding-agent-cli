import asyncio
import json
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

        while iterations < settings.max_tool_iterations:
            try:
                with console.status("[dim]thinking…[/dim]", spinner="dots"):
                    response = await llm.complete(messages, tools, system=SYSTEM_PROMPT)
            except (KeyboardInterrupt, asyncio.CancelledError):
                console.print("\n[dim]Interrupted.[/dim]")
                return

            if response["usage"]:
                prompt_tokens += response["usage"]["prompt_tokens"]
                completion_tokens += response["usage"]["completion_tokens"]

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    console.print(
                        f"[dim]→ {tc['name']}({json.dumps(tc['arguments'], indent=None)})[/dim]"
                    )

                messages.append(_assistant_tool_call_message(response["tool_calls"]))

                _last_build_flow_id: str | None = None
                for tc in response["tool_calls"]:
                    args = tc["arguments"]

                    # Auto-enrich nodes with full component schemas + credentials before sending to Langflow
                    enrich_error: str | None = None
                    if tc["name"] in ("update_flow", "create_flow"):
                        data = args.get("data", {})
                        if isinstance(data, dict) and "nodes" in data:
                            try:
                                credential_overrides: dict = {}
                                if settings.azure_anthropic_api_key:
                                    credential_overrides["AnthropicModel"] = {
                                        "api_key": settings.azure_anthropic_api_key,
                                        "base_url": settings.azure_anthropic_endpoint,
                                        "model_name": settings.azure_anthropic_deployment,
                                    }
                                    credential_overrides["Agent"] = {
                                        "api_key": settings.azure_anthropic_api_key,
                                    }
                                if settings.azure_openai_endpoint:
                                    credential_overrides["AzureOpenAIModel"] = {
                                        "azure_endpoint": settings.azure_openai_endpoint,
                                        "api_key": settings.azure_openai_api_key,
                                        "azure_deployment": settings.azure_openai_deployment,
                                        "api_version": settings.azure_openai_api_version,
                                    }
                                data["nodes"] = mcp.enrich_nodes(data["nodes"], credential_overrides=credential_overrides)
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
                        # Virtual tool — handled locally, no MCP call needed
                        result = json.dumps(mcp.get_component_schema(args.get("type_name", "")))
                    else:
                        try:
                            result = await mcp.call_tool(tc["name"], args)
                        except Exception as e:
                            result = f"ERROR: {e}"

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
                    console.print(
                        f"[dim]↑{prompt_tokens:,} ↓{completion_tokens:,} tokens[/dim]"
                    )
                # Trim history to last 2 user turns + this assistant response.
                # Drops tool call/result messages from prior turns to prevent token explosion.
                prior_user = [m for m in messages if m["role"] == "user"]
                messages = prior_user[-2:] + [{"role": "assistant", "content": response["content"]}]
                break
        else:
            console.print("[yellow]⚠ Max tool iterations reached.[/yellow]")
