from abc import ABC, abstractmethod
from typing import TypedDict


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict


class LLMResponse(TypedDict):
    content: str
    tool_calls: list[ToolCall]


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
