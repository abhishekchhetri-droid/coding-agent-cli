import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from llm.base import LLMProvider
from mcpbridge.client import LangflowMCPClient
from config.settings import Settings
from agent.prompts import SYSTEM_PROMPT

console = Console()


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
            with console.status("[dim]thinking…[/dim]", spinner="dots"):
                response = await llm.complete(messages, tools, system=SYSTEM_PROMPT)

            if response["usage"]:
                prompt_tokens += response["usage"]["prompt_tokens"]
                completion_tokens += response["usage"]["completion_tokens"]

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    console.print(
                        f"[dim]→ {tc['name']}({json.dumps(tc['arguments'], indent=None)})[/dim]"
                    )

                messages.append(_assistant_tool_call_message(response["tool_calls"]))

                for tc in response["tool_calls"]:
                    try:
                        result = await mcp.call_tool(tc["name"], tc["arguments"])
                    except Exception as e:
                        result = f"ERROR: {e}"
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
