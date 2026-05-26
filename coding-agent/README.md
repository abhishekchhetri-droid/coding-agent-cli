# Langflow Coding Agent

CLI agent that builds and manages [Langflow](https://github.com/langflow-ai/langflow) flows via natural language. Talks to a live Langflow instance through the MCP stdio bridge, auto-injects Azure credentials, and caches flow/template metadata in Redis.

---

## What it does

- **Chat REPL** — describe the flow you want, agent builds it
- **Direct CLI commands** — `flow list`, `flow run`, `health`, etc. without LLM
- **Template library** — 32 starter templates scored against your intent; best match cloned instantly
- **Auto-wiring** — injects AzureOpenAIModel, tool handles, and credential fields automatically
- **Redis cache** — flow list and starter templates cached, synced in background every 60 s

---

## Requirements

- Python 3.12+
- Node.js (for `langflow-mcp` stdio bridge)
- Langflow running at `localhost:7860` (or set `LANGFLOW_BASE_URL`)
- Azure OpenAI **or** Azure Anthropic credentials
- Redis (optional, for flow/template cache)

---

## Setup

```bash
# 1. Start Langflow + Redis
docker compose up -d

# 2. Install Python deps
cd coding-agent
uv sync           # or: pip install -e .

# 3. Copy and fill env
cp ../.env.example ../.env
```

### `.env` reference

```env
# Langflow
LANGFLOW_API=your-langflow-api-key
LANGFLOW_BASE_URL=http://localhost:7860        # default
LANGFLOW_MCP_PATH=/path/to/langflow-mcp/dist/mcp/index.js

# LLM — pick one
LLM_PROVIDER=azure_openai                     # or azure_anthropic

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Azure Anthropic (alternative)
AZURE_ANTHROPIC_ENDPOINT=https://your-resource.services.ai.azure.com
AZURE_ANTHROPIC_API_KEY=your-key
AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-4-6

# Redis (optional)
REDIS_URL=redis://localhost:6379
REDIS_SYNC_INTERVAL=60                        # seconds between background syncs
ENTITY_TOP_K=15                               # max flows returned in search
```

---

## Usage

### Interactive chat (default)

```bash
cd coding-agent
python main.py
```

```
nokia> build me a research agent with web search
nokia> create a document Q&A flow
nokia> build a simple chatbot
```

The agent scores your request against 32 templates, clones the best match or builds from scratch, wires credentials, and returns the flow ID.

### Direct CLI commands (no LLM)

```bash
python main.py flow list
python main.py flow list --page 2 --size 10
python main.py flow get <flow_id>
python main.py flow run <flow_id> --input "hello"
python main.py flow delete <flow_id>
python main.py folder list
python main.py health
```

Add `--pretty` for table/JSON output (default: on).

---

## Architecture

```
main.py
├── agent/
│   ├── agent.py      — chat REPL, tool loop, flow building logic
│   ├── cmd.py        — direct CLI commands
│   └── prompts.py    — system prompt with template index and build protocol
├── mcpbridge/
│   ├── client.py     — MCP stdio client, flow CRUD, node enrichment, edge wiring
│   └── redis_cache.py — flow/starter cache (search, list, sync)
├── llm/
│   ├── azure_openai.py
│   └── azure_anthropic.py
├── config/
│   └── settings.py   — pydantic-settings, reads .env
└── templates/
    ├── base_flow.json — default 4-node scaffold (ChatInput→AzureOpenAI→Agent→ChatOutput)
    └── starter-pack.md — indexed list of 32 cloneable templates
```

### Flow building rules (agent)

| Template score | Strategy |
|---|---|
| ≥ 8.5 | Clone starter directly — fastest path |
| 6–8.4 | Cherry-pick domain nodes onto base scaffold |
| < 6 | Build from scratch using base_flow.json |

### Node enrichment (client)

Every node stub `{type, id, position}` gets enriched before POSTing:
1. Full schema fetched from `/api/v1/all`
2. Azure credentials injected into matching fields
3. Tool-mode components get `tools_metadata` + `component_as_tool` output injected
4. Non-Azure LLM nodes replaced with `AzureOpenAIModel`
5. Missing `Tool→Agent.tools` edges auto-added

---

## Available templates

```
Custom Component Generator    Instagram Copywriter         Image Sentiment Analysis
Financial Report Parser       Search Agent                 Text Sentiment Analysis
SaaS Pricing                  Knowledge Base               Sequential Tasks Agents
Research Agent                NVIDIA RTX Remix             Twitter Thread Generator
Market Research               Social Media Agent           Simple Agent
Memory Chatbot                News Aggregator              Hybrid Search RAG
SEO Keyword Generator         Basic Prompt Chaining        Price Deal Finder
YouTube Analysis              Meeting Summary              Pokédex Agent
Document Q&A                  Research Translation Loop    Basic Prompting
Portfolio Website Generator   Travel Planning Agents       ...and more
```

---

## Tests

```bash
cd coding-agent
pytest
```

---

## Logs

Errors written to `coding-agent/logs/agent.log`. Console output is clean by design.
