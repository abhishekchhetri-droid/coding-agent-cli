#!/usr/bin/env bash
# Launch the CopilotKit web stack: AG-UI agent server (:8000) + Next.js UI (:3000).
# Assumes Langflow + Redis are already running (docker compose up -d).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "→ starting AG-UI agent server on :8000"
( cd "$ROOT/coding-agent" && uv run uvicorn server.app:app --port 8000 ) &

echo "→ starting Next.js UI on :3000"
( cd "$ROOT/web" && npm run dev ) &

wait
