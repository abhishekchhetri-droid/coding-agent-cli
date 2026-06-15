import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings
from llm.openai_usage import usage_dict_from_openai_usage


def _settings(**overrides):
    s = MagicMock(spec=Settings)
    s.llmgw_api_key = "tok"
    s.llmgw_api_base = "https://gw.example.com/v1/"
    s.llmgw_model = "gpt-4.1"
    s.llmgw_workspace = ""
    s.llmgw_workspace_header = "workspacename"
    s.llmgw_prompt_cache_key = ""
    s.llmgw_prompt_cache_retention = ""
    s.llm_supports_reasoning = False
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _usage(prompt=100, completion=20, cached=None):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=details,
    )


def _response(content="ok", tool_calls=None, usage=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# usage helper — the bit the colleague's sample imported but never wrote
# ---------------------------------------------------------------------------

def test_usage_maps_cached_tokens_to_cache_read():
    """A cache HIT (cached_tokens) becomes cache_read_tokens; there is no cache-creation metric."""
    u = usage_dict_from_openai_usage(_usage(prompt=1000, completion=50, cached=896))
    assert u["prompt_tokens"] == 1000
    assert u["completion_tokens"] == 50
    assert u["total_tokens"] == 1050
    assert u["cache_read_tokens"] == 896
    assert u["cache_creation_tokens"] == 0


def test_usage_handles_missing_details():
    """First turn / gateway that omits details → no cache read, no crash."""
    u = usage_dict_from_openai_usage(_usage(cached=None))
    assert u["cache_read_tokens"] == 0
    assert u["cache_creation_tokens"] == 0


def test_usage_handles_none_cached_tokens():
    u = usage_dict_from_openai_usage(_usage(cached=0))
    assert u["cache_read_tokens"] == 0


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------

def test_provider_requires_key_base_model():
    from llm.openai_gateway import OpenAIGatewayProvider
    with pytest.raises(ValueError, match="LLMGW_API_KEY"):
        OpenAIGatewayProvider(_settings(llmgw_api_key=""))
    with pytest.raises(ValueError, match="LLMGW_API_BASE"):
        OpenAIGatewayProvider(_settings(llmgw_api_base=""))
    with pytest.raises(ValueError, match="LLMGW_MODEL"):
        OpenAIGatewayProvider(_settings(llmgw_model=""))


def test_provider_sets_auth_and_workspace_headers():
    from llm.openai_gateway import OpenAIGatewayProvider
    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        OpenAIGatewayProvider(_settings(llmgw_workspace="team-x", llmgw_workspace_header="workspacename"))
        _, kwargs = cls.call_args
        assert kwargs["default_headers"] == {"api-key": "tok", "workspacename": "team-x"}
        # trailing slash on the base url is trimmed
        assert kwargs["base_url"] == "https://gw.example.com/v1"


@pytest.mark.asyncio
async def test_complete_strips_private_keys_and_preserves_caller_dicts():
    """`_ephemeral` (messages) and `_volatile` (tools) must never reach the gateway, and the
    caller's dicts — reused across agent iterations — must not be mutated."""
    from llm.openai_gateway import OpenAIGatewayProvider

    messages = [{"role": "user", "content": "hi", "_ephemeral": True}]
    tools = [{"type": "function", "function": {"name": "x"}, "_volatile": True}]

    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_response(usage=_usage()))
        cls.return_value = client
        provider = OpenAIGatewayProvider(_settings())
        await provider.complete(messages, tools, system="sys")

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}  # no _ephemeral
    assert kwargs["tools"][0] == {"type": "function", "function": {"name": "x"}}  # no _volatile
    # caller objects untouched
    assert messages[0]["_ephemeral"] is True
    assert tools[0]["_volatile"] is True


@pytest.mark.asyncio
async def test_complete_forwards_cache_params_when_set():
    from llm.openai_gateway import OpenAIGatewayProvider
    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_response(usage=_usage()))
        cls.return_value = client
        provider = OpenAIGatewayProvider(
            _settings(llmgw_prompt_cache_key="sess-42", llmgw_prompt_cache_retention="24h")
        )
        await provider.complete([{"role": "user", "content": "hi"}], [])

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["prompt_cache_key"] == "sess-42"
    assert kwargs["prompt_cache_retention"] == "24h"


@pytest.mark.asyncio
async def test_complete_omits_cache_params_and_invalid_retention():
    from llm.openai_gateway import OpenAIGatewayProvider
    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_response(usage=_usage()))
        cls.return_value = client
        provider = OpenAIGatewayProvider(_settings(llmgw_prompt_cache_retention="bogus"))
        await provider.complete([{"role": "user", "content": "hi"}], [])

    _, kwargs = client.chat.completions.create.call_args
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_retention" not in kwargs  # invalid value dropped


@pytest.mark.asyncio
async def test_complete_surfaces_cache_read_in_usage():
    """End-to-end: a gateway cache hit propagates to the usage agent.py reads for the 📦 line."""
    from llm.openai_gateway import OpenAIGatewayProvider
    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_response(usage=_usage(prompt=2000, completion=30, cached=1792))
        )
        cls.return_value = client
        provider = OpenAIGatewayProvider(_settings())
        result = await provider.complete([{"role": "user", "content": "hi"}], [])

    assert result["usage"]["cache_read_tokens"] == 1792
    assert result["usage"]["cache_creation_tokens"] == 0


@pytest.mark.asyncio
async def test_complete_parses_tool_calls():
    from llm.openai_gateway import OpenAIGatewayProvider
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "list_flows"
    tc.function.arguments = json.dumps({"page": 1})

    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_response(content=None, tool_calls=[tc], usage=_usage())
        )
        cls.return_value = client
        provider = OpenAIGatewayProvider(_settings())
        result = await provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}])

    assert result["content"] == ""
    assert result["tool_calls"][0] == {"id": "call_1", "name": "list_flows", "arguments": {"page": 1}}


@pytest.mark.asyncio
async def test_complete_omits_tools_when_reasoning():
    from llm.openai_gateway import OpenAIGatewayProvider
    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_response(usage=_usage()))
        cls.return_value = client
        provider = OpenAIGatewayProvider(_settings(llm_supports_reasoning=True))
        await provider.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])

    _, kwargs = client.chat.completions.create.call_args
    assert "tools" not in kwargs


@pytest.mark.asyncio
async def test_complete_raises_on_empty_choices():
    from llm.openai_gateway import OpenAIGatewayProvider
    resp = MagicMock()
    resp.choices = []
    with patch("llm.openai_gateway.AsyncOpenAI") as cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        cls.return_value = client
        provider = OpenAIGatewayProvider(_settings())
        with pytest.raises(ValueError, match="No completion choices"):
            await provider.complete([{"role": "user", "content": "hi"}], [])


def test_registry_resolves_openai_gateway():
    from llm.registry import get_provider
    with patch("llm.openai_gateway.AsyncOpenAI"):
        s = _settings()
        s.llm_provider = "openai_gateway"
        provider = get_provider(s)
    from llm.openai_gateway import OpenAIGatewayProvider
    assert isinstance(provider, OpenAIGatewayProvider)
