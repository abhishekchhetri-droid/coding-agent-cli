import json
from openai import AsyncAzureOpenAI
from .base import LLMProvider, LLMResponse, ToolCall, Usage
from config.settings import Settings


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_deployment
        self._reasoning = settings.llm_supports_reasoning

    @property
    def supports_reasoning(self) -> bool:
        return self._reasoning

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        kwargs: dict = {
            "model": self._deployment,
            "messages": all_messages,
        }

        if tools and not self._reasoning:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        if not response.choices:
            raise ValueError("No completion choices returned from Azure OpenAI")
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
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "usage": usage,
        }
