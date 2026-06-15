import json
import httpx
from openai import AsyncOpenAI

from .base import LLMProvider, LLMResponse, ToolCall, Usage
from .openai_usage import usage_dict_from_openai_usage
from config.settings import Settings

# Default workspace header name for Nokia-style gateways (override with LLMGW_WORKSPACE_HEADER).
_DEFAULT_WORKSPACE_HEADER = "workspacename"


def _strip_private(d: dict) -> dict:
    """Return a shallow copy of ``d`` without our private "_"-prefixed marker keys.

    The agent tags messages with ``_ephemeral`` and tools with ``_volatile`` so the *Anthropic*
    provider can place cache breakpoints. AsyncOpenAI serializes these dicts verbatim into the
    JSON request body, and the gateway would reject the unknown ``_ephemeral`` / ``_volatile``
    fields with a 400. They are also meaningless here — OpenAI caching is automatic — so drop
    them. Copy rather than mutate: agent.py reuses the same message dicts across iterations.
    """
    return {k: v for k, v in d.items() if not k.startswith("_")}


class OpenAIGatewayProvider(LLMProvider):
    """OpenAI-compatible API via a corporate LLM gateway (api-key + workspacename headers).

    Caching note: unlike azure_anthropic, there is no cache_control breakpoint to set. The
    gateway automatically caches the longest common prefix of requests (typically ≥1024 tokens);
    we just (a) keep the prefix stable — the agent already tails volatile tools and appends live
    state as a trailing message — and (b) pass ``prompt_cache_key`` to route same-prefix requests
    to the same cache node. Hits come back via usage.prompt_tokens_details.cached_tokens.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.llmgw_api_key.strip():
            raise ValueError("LLMGW_API_KEY is required when LLM_PROVIDER=openai_gateway")
        if not settings.llmgw_api_base.strip():
            raise ValueError("LLMGW_API_BASE is required when LLM_PROVIDER=openai_gateway")
        if not settings.llmgw_model.strip():
            raise ValueError("LLMGW_MODEL is required when LLM_PROVIDER=openai_gateway")

        token = settings.llmgw_api_key.strip()
        base_url = settings.llmgw_api_base.strip().rstrip("/")

        headers: dict[str, str] = {"api-key": token}
        ws = settings.llmgw_workspace.strip()
        if ws:
            header_name = settings.llmgw_workspace_header.strip() or _DEFAULT_WORKSPACE_HEADER
            headers[header_name] = ws

        self._client = AsyncOpenAI(
            # Auth rides on the api-key header, not the bearer token — but the SDK requires a
            # non-empty api_key, so pass a placeholder.
            api_key="NONE",
            base_url=base_url,
            default_headers=headers,
            # Mirror azure_anthropic's hardening: the gateway sits on the same VPN'd corporate
            # network, where a bare client's 5s connect timeout surfaces brief stalls as fatal
            # errors. Generous connect; 120s read; 2 retries bounds worst-case latency.
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=2,
        )
        self._model = settings.llmgw_model.strip()
        self._reasoning = settings.llm_supports_reasoning
        self._prompt_cache_key = settings.llmgw_prompt_cache_key.strip()
        self._prompt_cache_retention = settings.llmgw_prompt_cache_retention.strip()

    @property
    def supports_reasoning(self) -> bool:
        return self._reasoning

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        all_messages: list[dict] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(_strip_private(m) for m in messages)

        kwargs: dict = {
            "model": self._model,
            "messages": all_messages,
        }

        if tools and not self._reasoning:
            kwargs["tools"] = [_strip_private(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        # Prompt-cache controls. Both are no-ops if the gateway ignores them, so they are safe to
        # always send; the cached_tokens readback below tells us whether they took effect.
        if self._prompt_cache_key:
            kwargs["prompt_cache_key"] = self._prompt_cache_key
        if self._prompt_cache_retention in ("in_memory", "24h"):
            kwargs["prompt_cache_retention"] = self._prompt_cache_retention

        response = await self._client.chat.completions.create(**kwargs)
        if not response.choices:
            raise ValueError("No completion choices returned from OpenAI-compatible gateway")
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse tool arguments for {tc.function.name}: {e}") from e
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": arguments,
                })

        usage: Usage | None = None
        if response.usage:
            usage = usage_dict_from_openai_usage(response.usage)

        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "usage": usage,
        }
