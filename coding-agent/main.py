import asyncio
import logging
import sys
from pathlib import Path
from config.settings import Settings
from mcpbridge.client import LangflowMCPClient
from agent.cmd import run_cmd, CmdError


def main() -> None:
    settings = Settings()
    asyncio.run(_main(settings))


async def _main(settings: Settings) -> None:
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "agent.log"),
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    mcp = LangflowMCPClient(
        mcp_path=settings.langflow_mcp_path,
        langflow_api_key=settings.langflow_api_key,
        langflow_base_url=settings.langflow_base_url,
    )

    await mcp.connect()

    try:
        args = sys.argv[1:]

        if args and args[0] == "--cmd":
            # Direct command mode — no LLM
            pretty = "--pretty" in args
            cmd_args = [a for a in args[1:] if a != "--pretty"]
            try:
                await run_cmd(cmd_args, mcp, pretty=pretty)
            except CmdError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args:
            # Single-shot command without --cmd prefix (convenience)
            pretty = "--pretty" in args
            cmd_args = [a for a in args if a != "--pretty"]
            try:
                await run_cmd(cmd_args, mcp, pretty=pretty)
            except CmdError:
                # Not a direct command — treat as chat input
                from agent.agent import run_chat
                from llm.registry import get_provider
                llm = get_provider(settings)
                await run_chat(llm, mcp, settings)

        else:
            # Interactive chat REPL
            from agent.agent import run_chat
            from llm.registry import get_provider
            llm = get_provider(settings)
            await run_chat(llm, mcp, settings)

    except Exception as e:
        logging.exception("Unhandled error in agent")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await mcp.close()


if __name__ == "__main__":
    main()
