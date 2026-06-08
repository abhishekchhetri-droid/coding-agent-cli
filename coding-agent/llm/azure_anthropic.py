import json
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

    return out


class AzureAnthropicProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncAnthropicFoundry(
            api_key=settings.azure_anthropic_api_key,
            base_url=settings.azure_anthropic_endpoint,
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

        response = await self._client.messages.create(**kwargs)

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
