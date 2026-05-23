import json
from typing import Any
from rich.console import Console
from rich.table import Table
from mcpbridge.client import LangflowMCPClient

console = Console()


class CmdError(Exception):
    pass


async def run_cmd(args: list[str], mcp: LangflowMCPClient, pretty: bool = True) -> None:
    if not args:
        raise CmdError("No subcommand given. Try: flow list, flow get <id>, health")

    entity = args[0]

    if entity == "health":
        result = await mcp.call_tool("health_check", {})
        _print(result, pretty)

    elif entity == "flow":
        if len(args) < 2:
            raise CmdError("Usage: flow <list|get|create|update|delete|run>")
        action = args[1]

        if action == "list":
            page = _flag(args, "--page", default=1, cast=int)
            size = _flag(args, "--size", default=20, cast=int)
            result = await mcp.call_tool("list_flows", {"page": page, "size": size})
            _print(result, pretty)

        elif action == "get":
            _require_arg(args, 2, "flow get <flow_id>")
            result = await mcp.call_tool("get_flow", {"flow_id": args[2]})
            _print(result, pretty)

        elif action == "create":
            name = _flag(args, "--name", required=True)
            desc = _flag(args, "--description", default="")
            result = await mcp.call_tool("create_flow", {"name": name, "description": desc})
            _print(result, pretty)

        elif action == "update":
            _require_arg(args, 2, "flow update <flow_id> [--name X]")
            flow_id = args[2]
            name = _flag(args, "--name")
            payload: dict[str, Any] = {"flow_id": flow_id}
            if name:
                payload["name"] = name
            result = await mcp.call_tool("update_flow", payload)
            _print(result, pretty)

        elif action == "delete":
            _require_arg(args, 2, "flow delete <flow_id>")
            result = await mcp.call_tool("delete_flow", {"flow_id": args[2]})
            _print(result, pretty)

        elif action == "run":
            _require_arg(args, 2, "flow run <flow_id> --input '...'")
            flow_id = args[2]
            input_val = _flag(args, "--input", required=True)
            result = await mcp.call_tool("run_flow", {"flow_id": flow_id, "input_value": input_val})
            _print(result, pretty)

        else:
            raise CmdError(f"Unknown flow action '{action}'. Available: list, get, create, update, delete, run")

    elif entity == "folder":
        if len(args) < 2:
            raise CmdError("Usage: folder <list>")
        action = args[1]
        if action == "list":
            page = _flag(args, "--page", default=1, cast=int)
            size = _flag(args, "--size", default=20, cast=int)
            result = await mcp.call_tool("list_folders", {"page": page, "size": size})
            _print(result, pretty)
        else:
            raise CmdError(f"Unknown folder action '{action}'. Available: list")

    else:
        raise CmdError(f"Unknown command '{entity}'. Available: flow, folder, health")


def _require_arg(args: list[str], idx: int, usage: str) -> None:
    if len(args) <= idx:
        raise CmdError(f"Missing argument. Usage: {usage}")


def _flag(args: list[str], flag: str, default: Any = None, cast: type = str, required: bool = False) -> Any:
    try:
        idx = args.index(flag)
    except ValueError:
        if required:
            raise CmdError(f"Required flag {flag} not provided.")
        return default

    try:
        return cast(args[idx + 1])
    except IndexError:
        raise CmdError(f"Flag {flag} requires a value.")
    except (ValueError, TypeError):
        raise CmdError(f"Invalid value for {flag}: expected {cast.__name__}.")


def _print(result: Any, pretty: bool) -> None:
    if not pretty:
        print(result)
        return

    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        console.print(result)
        return

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            table = Table(show_header=True, header_style="bold cyan")
            keys = list(data[0].keys())
            for k in keys:
                table.add_column(k)
            for row in data:
                table.add_row(*[str(row.get(k, "")) for k in keys])
            console.print(table)
        else:
            console.print_json(json.dumps(data))
    elif isinstance(data, dict):
        console.print_json(json.dumps(data))
    else:
        console.print(result)
