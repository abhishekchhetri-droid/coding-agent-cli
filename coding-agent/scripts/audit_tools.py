"""Audit langflow-mcp tools the agent can see.

Usage:
  .venv/bin/python scripts/audit_tools.py            # list + counts
  .venv/bin/python scripts/audit_tools.py --smoke    # also call read-only tools

Connects via the real LangflowMCPClient (same path the agent uses), so it
reflects exactly what the running server advertises (163 on v4.0.2).
"""
import asyncio
import sys
from config.settings import Settings
from mcpbridge.client import LangflowMCPClient

# Read-only tools safe to invoke with no/zero args for a smoke test.
SMOKE = [
    "health_check", "get_version", "list_flows", "list_components",
    "list_variables", "list_folders", "list_projects", "get_basic_examples",
    "list_starter_projects", "get_default_model", "get_provider_variable_mapping",
]


async def main(smoke: bool) -> None:
    s = Settings()
    mcp = LangflowMCPClient(
        mcp_path=s.langflow_mcp_path,
        langflow_api_key=s.langflow_api_key,
        langflow_base_url=s.langflow_base_url,
    )
    await mcp.connect()
    try:
        tools = mcp._tools_cache
        names = sorted(t.name for t in tools)
        print(f"TOTAL tools advertised by server: {len(names)}")
        # group by prefix verb
        from collections import Counter
        verbs = Counter(n.split("_")[0] for n in names)
        print("by verb:", dict(sorted(verbs.items(), key=lambda x: -x[1])))
        print("\nall names:")
        for i, n in enumerate(names, 1):
            print(f"{i:3} {n}")

        if smoke:
            print("\n=== smoke test (read-only) ===")
            present = {t.name for t in tools}
            for name in SMOKE:
                if name not in present:
                    print(f"  -  {name}: NOT in server")
                    continue
                try:
                    res = await mcp.call_tool(name, {})
                    head = str(res)[:80].replace("\n", " ")
                    print(f"  ok {name}: {head}")
                except Exception as e:
                    print(f"  ERR {name}: {str(e)[:80]}")
    finally:
        await mcp.close()


if __name__ == "__main__":
    asyncio.run(main("--smoke" in sys.argv))
