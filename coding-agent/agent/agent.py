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
    """Check node count and test-run the flow. Append VERIFIED or FAILED to the tool result."""
    if not get_flow_result:
        return get_flow_result or "null"
    try:
        data = json.loads(get_flow_result)
        nodes = data.get("data", {}).get("nodes", []) if isinstance(data, dict) else []
        if len(nodes) == 0:
            return (
                get_flow_result
                + "\n\n⚠ VERIFICATION FAILED: 0 nodes in flow after build. "
                "All your node type strings were rejected by Langflow. "
                "Call list_components again, read the exact 'type' field values from the response, "
                "write them out explicitly, then retry update_flow with ONLY those type strings. "
                "Do NOT report success until node count > 0."
            )
        # Nodes exist — test-run the flow to confirm it actually executes
        test = mcp.test_run_flow(flow_id)
        if test["ok"]:
            return (
                get_flow_result
                + f"\n\n✅ VERIFIED: Flow executed successfully. "
                f"Test input '2+2' → '{test['answer']}'. "
                f"Flow has {len(nodes)} nodes and is working. You may report success."
            )
        else:
            return (
                get_flow_result
                + f"\n\n⚠ EXECUTION FAILED: Flow has {len(nodes)} nodes but failed to run: "
                f"{test['error']}. "
                "Do NOT report success. Fix the flow (check credentials, edge wiring, component config) and retry."
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
                    "arguments": json.dumps(tc["arguments"]),
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
                    if tc["name"] in ("update_flow", "create_flow"):
                        data = args.get("data", {})
                        if isinstance(data, dict) and "nodes" in data:
                            try:
                                credential_overrides: dict = {}
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
                                console.print(f"[yellow]⚠ schema enrichment failed: {enrich_err}[/yellow]")

                    try:
                        result = await mcp.call_tool(tc["name"], args)
                    except Exception as e:
                        result = f"ERROR: {e}"

                    if tc["name"] == "build_flow":
                        _last_build_flow_id = tc["arguments"].get("flow_id")

                    # After get_flow following a build: check nodes + test-run
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
                break
        else:
            console.print("[yellow]⚠ Max tool iterations reached.[/yellow]")
