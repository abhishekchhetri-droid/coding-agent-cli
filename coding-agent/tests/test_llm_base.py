import pytest
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
