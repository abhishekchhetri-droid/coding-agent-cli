import json
import httpx
from anthropic import AsyncAnthropicFoundry
from .base import LLMProvider, LLMResponse, ToolCall, Usage
from config.settings import Settings


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool schemas to Anthropic format."""
    result = []
    for t in tools:
        fn = t.get("function", t)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _apply_cache_control(msg: dict) -> None:
    """Attach an ephemeral cache breakpoint to an anthropic message's last content block.

    String content is normalized to a single text block so cache_control can be attached.
    Blocks are copied before mutation to avoid touching shared dicts.
    """
    content = msg["content"]
    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(content, list) and content:
        last = dict(content[-1])
        last["cache_control"] = {"type": "ephemeral"}
        content[-1] = last


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format message history to Anthropic format.

    Batches consecutive tool-result messages into a single user turn,
    which Anthropic requires.
    """
    out: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]

        if role == "user":
            out.append({"role": "user", "content": msg["content"]})
            i += 1

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in tool_calls:
                    fn = tc["function"]
                    try:
                        inp = json.loads(fn["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        inp = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": inp,
                    })
                out.append({"role": "assistant", "content": content_blocks})
            else:
                text = msg.get("content") or ""
                out.append({"role": "assistant", "content": text})
            i += 1

        elif role == "tool":
            # Batch all consecutive tool results into one user message
            tool_results = []
            while i < len(messages) and messages[i]["role"] == "tool":
                m = messages[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m.get("content") or "",
                })
                i += 1
            out.append({"role": "user", "content": tool_results})

        else:
            i += 1

    # Coalesce consecutive user turns into one (Anthropic requires alternating roles). This
    # occurs when a turn is aborted mid-flight — e.g. a transient API error leaves an unanswered
    # user message and the user re-sends — which would otherwise produce two user turns in a row
    # and 400. Merge their content (normalizing str → a text block) so retries just work.
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == "user" and m["role"] == "user":
            def _blocks(c):
                return list(c) if isinstance(c, list) else [{"type": "text", "text": c}]
            merged[-1]["content"] = _blocks(merged[-1]["content"]) + _blocks(m["content"])
        else:
            merged.append(dict(m))
    return merged


class AzureAnthropicProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncAnthropicFoundry(
            api_key=settings.azure_anthropic_api_key,
            base_url=settings.azure_anthropic_endpoint,
            # SDK default connect timeout is 5s — too tight for a VPN'd corporate Foundry
            # endpoint, where a brief connect stall surfaced as APITimeoutError and killed the
            # session. Generous connect; the read timeout is PER-CHUNK while streaming, so 120s
            # is plenty for first-token latency yet fails a true stall in ~2 min instead of
            # hanging ~10 min. Keep retries at 2 (the SDK default): each retry can wait up to the
            # full read timeout, so more retries MULTIPLY worst-case latency on a stalling
            # endpoint (a 4-retry stall ≈ 8 min). 2 bounds it while still absorbing brief blips.
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=2,
        )
        self._model = settings.azure_anthropic_deployment

    @property
    def supports_reasoning(self) -> bool:
        return False

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        anthropic_messages = _to_anthropic_messages(messages)
        anthropic_tools = _to_anthropic_tools(tools) if tools else []

        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if anthropic_tools:
            # Breakpoint at the end of the full tool list: on turns where the discovered set
            # is unchanged, the whole tool block is a cache hit.
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            # Second breakpoint at the end of the stable block (last non-volatile tool). When
            # search_tools changes the volatile discovered tail, only that tail is re-cached;
            # baseline/virtual/planning stay a cache hit. tools <-> anthropic_tools are 1:1.
            last_stable = -1
            for idx, t in enumerate(tools):
                if not t.get("_volatile"):
                    last_stable = idx
            if 0 <= last_stable < len(anthropic_tools) - 1:
                anthropic_tools[last_stable]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = anthropic_tools

        # Cache the conversation-history prefix. Without this, only system+tools are cached
        # and the entire (growing) message history — fat flow JSON, component schemas — is
        # re-billed as fresh input every iteration. The longest stable prefix is everything
        # up to (but not including) the trailing live-state message, which mutates each step
        # and would bust the cache if cached. agent.py marks it with `_ephemeral`.
        if anthropic_messages:
            cache_idx = len(anthropic_messages) - 1
            if messages and messages[-1].get("_ephemeral"):
                cache_idx -= 1
            if cache_idx >= 0:
                _apply_cache_control(anthropic_messages[cache_idx])

        # Stream the response. A non-streaming messages.create() holds one socket open for the
        # whole generation and can STALL silently on a flaky/VPN'd endpoint — the request just
        # hangs until the read timeout (~10 min) with no output. Streaming keeps the connection
        # active with incremental events (resetting the read clock per chunk) and is Anthropic's
        # recommended path for reliable / long requests. get_final_message() reassembles the same
        # Message object, so the parsing below is unchanged.
        async with self._client.messages.stream(**kwargs) as stream:
            response = await stream.get_final_message()

        text_parts = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        usage: Usage | None = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            }

        return {
            "content": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "usage": usage,
        }
