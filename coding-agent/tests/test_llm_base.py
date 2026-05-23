import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from llm.base import LLMProvider, LLMResponse, ToolCall


def test_llm_response_shape():
    r: LLMResponse = {"content": "hello", "tool_calls": []}
    assert r["content"] == "hello"
    assert r["tool_calls"] == []


def test_tool_call_shape():
    tc: ToolCall = {"id": "call_1", "name": "list_flows", "arguments": {"page": 1}}
    assert tc["name"] == "list_flows"


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # cannot instantiate abstract class


@pytest.mark.asyncio
async def test_azure_provider_complete_no_tools():
    """Provider returns content when no tool calls in response."""
    from llm.azure_openai import AzureOpenAIProvider
    from config.settings import Settings

    settings = MagicMock(spec=Settings)
    settings.azure_openai_endpoint = "https://test.openai.azure.com"
    settings.azure_openai_api_key = "key"
    settings.azure_openai_api_version = "2024-12-01-preview"
    settings.azure_openai_deployment = "gpt-4.1"
    settings.llm_supports_reasoning = False

    mock_msg = MagicMock()
    mock_msg.content = "Hello!"
    mock_msg.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("llm.azure_openai.AsyncAzureOpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = AzureOpenAIProvider(settings)
        result = await provider.complete([{"role": "user", "content": "hi"}], [])

    assert result["content"] == "Hello!"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_azure_provider_complete_with_tool_calls():
    """Provider parses tool calls from response."""
    from llm.azure_openai import AzureOpenAIProvider
    from config.settings import Settings

    settings = MagicMock(spec=Settings)
    settings.azure_openai_endpoint = "https://test.openai.azure.com"
    settings.azure_openai_api_key = "key"
    settings.azure_openai_api_version = "2024-12-01-preview"
    settings.azure_openai_deployment = "gpt-4.1"
    settings.llm_supports_reasoning = False

    mock_tc = MagicMock()
    mock_tc.id = "call_abc"
    mock_tc.function.name = "list_flows"
    mock_tc.function.arguments = json.dumps({"page": 1})

    mock_msg = MagicMock()
    mock_msg.content = None
    mock_msg.tool_calls = [mock_tc]
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("llm.azure_openai.AsyncAzureOpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = AzureOpenAIProvider(settings)
        result = await provider.complete([{"role": "user", "content": "list flows"}], [{"type": "function"}])

    assert result["content"] == ""
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "list_flows"
    assert result["tool_calls"][0]["arguments"] == {"page": 1}


@pytest.mark.asyncio
async def test_azure_provider_raises_on_empty_choices():
    """Provider raises ValueError when no choices returned."""
    from llm.azure_openai import AzureOpenAIProvider
    from config.settings import Settings

    settings = MagicMock(spec=Settings)
    settings.azure_openai_endpoint = "https://test.openai.azure.com"
    settings.azure_openai_api_key = "key"
    settings.azure_openai_api_version = "2024-12-01-preview"
    settings.azure_openai_deployment = "gpt-4.1"
    settings.llm_supports_reasoning = False

    mock_response = MagicMock()
    mock_response.choices = []

    with patch("llm.azure_openai.AsyncAzureOpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        provider = AzureOpenAIProvider(settings)
        with pytest.raises(ValueError, match="No completion choices"):
            await provider.complete([{"role": "user", "content": "hi"}], [])


def test_registry_get_provider_unknown_raises():
    """Registry raises ValueError for unknown provider."""
    from llm.registry import get_provider
    from config.settings import Settings

    settings = MagicMock(spec=Settings)
    settings.llm_provider = "nonexistent_provider"

    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider(settings)
