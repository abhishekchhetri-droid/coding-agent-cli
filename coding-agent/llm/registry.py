from .base import LLMProvider
from .azure_openai import AzureOpenAIProvider
from .azure_anthropic import AzureAnthropicProvider
from .openai_gateway import OpenAIGatewayProvider
from config.settings import Settings

PROVIDERS: dict[str, type[LLMProvider]] = {
    "azure_openai": AzureOpenAIProvider,
    "azure_anthropic": AzureAnthropicProvider,
    "openai_gateway": OpenAIGatewayProvider,
}


def get_provider(settings: Settings) -> LLMProvider:
    cls = PROVIDERS.get(settings.llm_provider)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{settings.llm_provider}'. "
            f"Available: {list(PROVIDERS)}"
        )
    return cls(settings)
