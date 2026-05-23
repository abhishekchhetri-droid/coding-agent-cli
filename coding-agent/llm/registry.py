from .base import LLMProvider
from .azure_openai import AzureOpenAIProvider
from config.settings import Settings

PROVIDERS: dict[str, type[LLMProvider]] = {
    "azure_openai": AzureOpenAIProvider,
    # "anthropic": AnthropicProvider,
    # "openai": OpenAIProvider,
}


def get_provider(settings: Settings) -> LLMProvider:
    cls = PROVIDERS.get(settings.llm_provider)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{settings.llm_provider}'. "
            f"Available: {list(PROVIDERS)}"
        )
    return cls(settings)
