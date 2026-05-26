from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Langflow — env var is LANGFLOW_API (not LANGFLOW_API_KEY)
    langflow_api_key: str = Field(default="", validation_alias="LANGFLOW_API")
    langflow_base_url: str = "http://localhost:7860"
    langflow_mcp_path: str = Field(
        default="/home/abhishekks1369/ai/nokia/langflow-mcp/dist/mcp/index.js",
        validation_alias="LANGFLOW_MCP_PATH",
    )

    # LLM provider
    llm_provider: str = "azure_openai"
    llm_supports_reasoning: bool = False

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"

    # Azure Anthropic (AnthropicFoundry)
    azure_anthropic_endpoint: str = ""
    azure_anthropic_api_key: str = ""
    azure_anthropic_deployment: str = "claude-sonnet-4-6"

    # Redis entity cache
    redis_url: str = Field(default="", validation_alias="REDIS_URL")
    redis_sync_interval: int = Field(default=60, validation_alias="REDIS_SYNC_INTERVAL")
    entity_top_k: int = Field(default=15, validation_alias="ENTITY_TOP_K")

    # Agent behaviour
    max_tool_iterations: int = 10
