import os
import pytest
from unittest.mock import patch


def test_settings_loads_azure_vars():
    env = {
        "LANGFLOW_API": "sk-test",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "key123",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4.1",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
        "LANGFLOW_MCP_PATH": "/some/path/index.js",
    }
    with patch.dict(os.environ, env, clear=False):
        from config.settings import Settings
        s = Settings()
        assert s.langflow_api_key == "sk-test"
        assert s.azure_openai_deployment == "gpt-4.1"
        assert s.max_tool_iterations == 25


def test_settings_defaults():
    env = {
        "LANGFLOW_API": "sk-test",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "key123",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4.1",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
        "LANGFLOW_MCP_PATH": "/some/path/index.js",
    }
    with patch.dict(os.environ, env, clear=False):
        from config.settings import Settings
        s = Settings()
        assert s.langflow_base_url == "http://localhost:7860"
        assert s.llm_provider == "azure_openai"
        assert s.llm_supports_reasoning is False
