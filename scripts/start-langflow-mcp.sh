#!/bin/bash
# Loads .env and starts langflow-mcp-server. Secrets stay in .env, never hardcoded.
set -a
source "$(dirname "$0")/../.env"
set +a

LANGFLOW_API_KEY="${LANGFLOW_API_KEY:-$LANGFLOW_API}"
LANGFLOW_BASE_URL="${LANGFLOW_BASE_URL:-http://localhost:7860}"
export LANGFLOW_API_KEY LANGFLOW_BASE_URL

exec node "$(dirname "$0")/../langflow-mcp/dist/mcp/index.js"
