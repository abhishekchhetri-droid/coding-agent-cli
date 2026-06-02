# Nokia Flow Builder — Web UI

CopilotKit web frontend for the Langflow flow-builder agent. Chat on the left, the
live Langflow flow canvas on the right.

```
Browser (:3000)
  ├─ CopilotChat ──/api/copilotkit──► CopilotRuntime (HttpAgent)
  │                                        └──► Python AG-UI agent (:8000 /agent)
  └─ <iframe> Langflow editor (:7860/flow/<id>)  ◄── shared agent state (flow_url)
```

The chat drives the *same* agent loop as the terminal REPL (`coding-agent/agent.run_turn`),
exposed over the AG-UI protocol by `coding-agent/server/app.py`. When the agent builds a
flow it emits a `STATE_SNAPSHOT` carrying `flow_url`; `FlowCanvas` swaps the iframe to it.

## Run

Prereqs: Langflow + Redis (`docker compose up -d` from repo root) and the agent server.

```bash
# 1. agent server (from repo root)
cd coding-agent && uv run uvicorn server.app:app --port 8000

# 2. web UI
cd web && npm install && npm run dev
```

Or run both with `scripts/start-web.sh` from the repo root. Open http://localhost:3000.

`AGENT_URL` (default `http://localhost:8000/agent`) overrides the agent endpoint.

## Key files

- `app/page.tsx` — two-pane layout; `FlowCanvas` reads agent state via `useCoAgent("langflow")`.
- `app/api/copilotkit/route.ts` — CopilotRuntime endpoint wiring the AG-UI `HttpAgent`.
- `app/layout.tsx` — imports CopilotKit styles.
