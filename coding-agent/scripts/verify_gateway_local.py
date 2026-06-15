"""Local, gateway-free verification of the openai_gateway provider.

Runs the REAL OpenAIGatewayProvider + real openai SDK against a throwaway in-process HTTP server
that pretends to be the corporate gateway. This exercises the actual HTTP serialization (unlike
the mocked unit tests), so it catches anything that would only fail on the wire — e.g. a private
`_ephemeral` / `_volatile` key leaking into the request body.

What it PROVES (no gateway needed):
  • auth `api-key` + `workspacename` headers reach the server
  • `prompt_cache_key` / `prompt_cache_retention` are forwarded in the body
  • `_ephemeral` (messages) and `_volatile` (tools) are stripped before sending
  • a server `cached_tokens` value is parsed back into usage["cache_read_tokens"] → the 📦 line

What it does NOT prove: a *real* cache hit. The server FAKES cached_tokens (0 on call 1, >0 on
call 2) to show the readback path. A genuine hit can only come from a caching-capable endpoint
(the real gateway, or — see README notes — Azure OpenAI / OpenAI platform with a ≥1024-token prefix).

Run:  uv run python scripts/verify_gateway_local.py
"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config.settings import Settings
from llm.openai_gateway import OpenAIGatewayProvider

# Shared state between the server thread and the test: every request the server saw, and a
# call counter so it can fake "no cache on first call, cache hit on second".
_seen: list[dict] = []


class _FakeGateway(BaseHTTPRequestHandler):
    def log_message(self, *_):  # silence default stderr access log
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        _seen.append({"path": self.path, "headers": dict(self.headers), "body": body})

        # Fake the cache: first request is a miss (cached=0), later ones hit (most of the prompt).
        call_n = len(_seen)
        prompt_tokens = 1500
        cached = 0 if call_n == 1 else 1408

        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"reply #{call_n}"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 10,
                "total_tokens": prompt_tokens + 10,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _check(label: str, ok: bool) -> bool:
    print(f"  {'✓' if ok else '✗'} {label}")
    return ok


async def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGateway)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    settings = Settings(
        llmgw_api_key="test-token",
        llmgw_api_base=f"http://127.0.0.1:{port}/v1",
        llmgw_model="gpt-4o",
        llmgw_workspace="team-x",
        llmgw_workspace_header="workspacename",
        llmgw_prompt_cache_key="sess-verify",
        llmgw_prompt_cache_retention="24h",
    )
    provider = OpenAIGatewayProvider(settings)

    # Messages/tools carry the private markers the agent attaches — they must NOT reach the wire.
    messages = [{"role": "user", "content": "build me a chatbot", "_ephemeral": True}]
    tools = [{"type": "function", "function": {"name": "list_flows"}, "_volatile": True}]

    print("\nCall 1 (cold — expect cache_read=0):")
    r1 = await provider.complete(messages, tools, system="You are a Langflow builder.")
    print(f"    usage = {r1['usage']}")

    print("Call 2 (warm — server fakes a hit, expect cache_read>0):")
    r2 = await provider.complete(messages, tools, system="You are a Langflow builder.")
    print(f"    usage = {r2['usage']}")

    server.shutdown()

    req = _seen[0]
    body = req["body"]
    print("\nChecks:")
    ok = True
    ok &= _check("api-key header reached server", req["headers"].get("api-key") == "test-token")
    ok &= _check("workspacename header reached server", req["headers"].get("workspacename") == "team-x")
    ok &= _check("prompt_cache_key forwarded", body.get("prompt_cache_key") == "sess-verify")
    ok &= _check("prompt_cache_retention forwarded", body.get("prompt_cache_retention") == "24h")
    ok &= _check("system message present", body["messages"][0]["role"] == "system")
    ok &= _check("_ephemeral stripped from message", "_ephemeral" not in body["messages"][1])
    ok &= _check("_volatile stripped from tool", "_volatile" not in body["tools"][0])
    ok &= _check("call 1 parsed cache_read_tokens == 0", r1["usage"]["cache_read_tokens"] == 0)
    ok &= _check("call 2 parsed cache_read_tokens == 1408", r2["usage"]["cache_read_tokens"] == 1408)
    ok &= _check("cache_creation_tokens always 0 (OpenAI has no write metric)",
                 r2["usage"]["cache_creation_tokens"] == 0)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
