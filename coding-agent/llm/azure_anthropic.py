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
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = anthropic_tools

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
