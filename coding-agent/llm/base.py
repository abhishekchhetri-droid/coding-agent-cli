from abc import ABC, abstractmethod
from typing import TypedDict


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict


class _UsageBase(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Usage(_UsageBase, total=False):
    cache_creation_tokens: int
    cache_read_tokens: int


class LLMResponse(TypedDict):
    content: str
    tool_calls: list[ToolCall]
    usage: Usage | None


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse: ...

    @property
    @abstractmethod
    def supports_reasoning(self) -> bool: ...
