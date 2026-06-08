# Nokia AI — Langflow Agent Platform

Natural language interface for building and managing [Langflow](https://github.com/langflow-ai/langflow) AI workflows. Describe a flow in plain English; the agent builds, wires, and deploys it on a live Langflow instance.

---

## Components

```
nokia/
├── coding-agent/        — Python agent: chat REPL, direct commands, + AG-UI server (server/)
├── langflow-mcp/        — MCP stdio bridge (Node.js, talks to Langflow REST API)
├── web/                 — CopilotKit web UI (Next.js): chat + live Langflow canvas
├── docker-compose.yml   — Langflow + Redis stack
├── scripts/
│   ├── start-langflow-mcp.sh  — starts MCP bridge with .env credentials
│   └── start-web.sh           — starts AG-UI server (:8000) + Next.js UI (:3000)
└── .env                 — all secrets live here
```

---

## Quick Start

### 1. Start Langflow + Redis

```bash
docker compose up -d
```

Langflow UI available at `http://localhost:7860`.

### 2. Configure credentials

```bash
cp .env.example .env   # then fill in values
```

```env
LANGFLOW_API=your-langflow-api-key

# Azure OpenAI (default LLM provider)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Anthropic (alternative — set LLM_PROVIDER=azure_anthropic)
AZURE_ANTHROPIC_ENDPOINT=https://your-resource.services.ai.azure.com
AZURE_ANTHROPIC_API_KEY=your-key
AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-4-6

# Redis cache (optional but recommended)
REDIS_URL=redis://localhost:6379
```

### 3. Run the agent

```bash
cd coding-agent
uv sync
python main.py
```

```
nokia> build me a research agent with web search
nokia> create a document Q&A flow using hybrid search
nokia> build a memory chatbot
```

---

## Deploy to a VM

1. Clone the repo on the VM:

```bash
git clone  ~/nokia
cd ~/nokia
```

2. Create `.env` in `~/nokia` using `.env.example` and fill in your secrets.

3. Start Langflow and Redis:

```bash
docker compose up -d
```

4. Build the MCP bridge:

```bash
cd langflow-mcp
npm ci
npm run build
```

5. Install Python dependencies and run the agent:

```bash
cd ../coding-agent
uv sync
uv run python main.py
```

6. Verify with:

```bash
curl -s localhost:7860/health
uv run python main.py health
```

---

## How it works

```
User prompt
    │
    ▼
coding-agent (Python)
    │  scores prompt against 32 templates
    │  selects: clone / cherry-pick / scratch
    ▼
langflow-mcp (Node.js stdio bridge)
    │  enriches nodes with live schemas + Azure credentials
    │  auto-wires tool edges, injects AzureOpenAIModel
    ▼
Langflow REST API (localhost:7860)
    │  creates / updates / builds flow
    ▼
Flow ready — ID returned
```

---

## Web UI (CopilotKit)

A browser experience with a chat panel and the Langflow flow canvas side-by-side —
describe a flow, watch it render live. Same agent loop as the REPL, exposed over the
AG-UI protocol.

```bash
docker compose up -d                 # Langflow + Redis
cd web && npm install                # first time only
scripts/start-web.sh                 # agent server :8000 + Next.js UI :3000
```

Open http://localhost:3000. See [web/README.md](web/README.md) for architecture.

---

## Direct CLI commands

Skip the LLM for simple operations:

```bash
python main.py flow list
python main.py flow get <flow_id>
python main.py flow run <flow_id> --input "your message"
python main.py flow delete <flow_id>
python main.py folder list
python main.py health
```

---

## MCP bridge

The `langflow-mcp` server exposes Langflow's REST API as MCP tools over stdio. The coding agent spawns it as a subprocess.

Start it standalone (for debugging):

```bash
scripts/start-langflow-mcp.sh
```

---

## Docs

- [coding-agent/README.md](coding-agent/README.md) — agent setup, architecture, all options
- [docs/langflow-mcp-reference.md](docs/langflow-mcp-reference.md) — MCP tool reference
- [docs/langflow-mcp-azure-openai-setup.md](docs/langflow-mcp-azure-openai-setup.md) — Azure OpenAI setup guide

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.12+ |
| Node.js | 18+ |
| Docker | for Langflow + Redis |
| Azure OpenAI or Azure Anthropic | — |
