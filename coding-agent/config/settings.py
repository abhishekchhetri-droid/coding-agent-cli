from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _REPO_ROOT / ".env"


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
        default=str(_REPO_ROOT / "langflow-mcp" / "dist" / "mcp" / "index.js"),
        validation_alias="LANGFLOW_MCP_PATH",
    )

    # LLM provider
    llm_provider: str = "azure_anthropic"
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

    # OpenAI-compatible corporate gateway (LLM_PROVIDER=openai_gateway). Auth via an `api-key`
    # header (not bearer) plus an optional workspace header. Prompt caching is automatic on the
    # gateway side; the cache key/retention just route + extend it (see llm/openai_gateway.py).
    llmgw_api_key: str = ""
    llmgw_api_base: str = ""
    llmgw_model: str = ""
    llmgw_workspace: str = ""
    llmgw_workspace_header: str = "workspacename"
    llmgw_prompt_cache_key: str = ""
    llmgw_prompt_cache_retention: str = ""  # "" (off) | "in_memory" | "24h"

    # Redis entity cache
    redis_url: str = Field(default="", validation_alias="REDIS_URL")
    redis_sync_interval: int = Field(default=60, validation_alias="REDIS_SYNC_INTERVAL")
    entity_top_k: int = Field(default=15, validation_alias="ENTITY_TOP_K")

    # Agent behaviour. Headroom for large multi-node flows; the ceiling is a
    # safety stop, not the normal path — topology-first planning + batched
    # schema fetch keep real builds well under it.
    max_tool_iterations: int = 25

    # Context management. Below this message count the full history is kept verbatim;
    # above it, the older prefix is summarized at turn end (see agent/context.py).
    summarize_threshold_messages: int = 30
