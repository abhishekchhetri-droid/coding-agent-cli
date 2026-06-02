"""AG-UI HTTP server wrapping the Langflow flow-builder agent.

Exposes POST /agent speaking the AG-UI protocol so a CopilotKit frontend can drive
the *same* agent loop the terminal REPL uses (agent.run_turn). The agent's turn loop
emits three structured signals through an EventSink (agent/events.py); AGUISink maps
them to AG-UI events:

  tool_call(name, args) -> STEP_STARTED / STEP_FINISHED  (tool activity in the chat)
  flow_built(flow_id)   -> STATE_SNAPSHOT {flow_id, flow_url}  (drives the canvas iframe)
  final(text)           -> TEXT_MESSAGE_START/CONTENT/END  (the assistant's answer)

Conversation history is kept server-side per thread_id; run_turn trims it itself.
All credentials stay server-side — the browser never sees Azure/Langflow keys.
"""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    StateSnapshotEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
)
from ag_ui.encoder import EventEncoder

from config.settings import Settings
from mcpbridge.client import LangflowMCPClient
from mcpbridge.redis_cache import RedisEntityCache
from llm.registry import get_provider
from agent.agent import run_turn

logger = logging.getLogger("agent.server")

# Per-thread conversation history (internal OpenAI-style message dicts) and last flow.
_THREADS: dict[str, list[dict]] = {}
_THREAD_FLOW: dict[str, str] = {}


class AGUISink:
    """Translates agent run_turn signals into AG-UI events on an asyncio.Queue."""

    def __init__(self, queue: "asyncio.Queue", langflow_base_url: str):
        self._q = queue
        self._base = langflow_base_url.rstrip("/")
        self.flow_id: str | None = None

    def tool_call(self, name: str, arguments: dict) -> None:
        self._q.put_nowait(StepStartedEvent(type=EventType.STEP_STARTED, step_name=name))
        self._q.put_nowait(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=name))

    def flow_built(self, flow_id: str | None) -> None:
        if not flow_id:
            return
        self.flow_id = flow_id
        self._q.put_nowait(StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot={"flow_id": flow_id, "flow_url": f"{self._base}/flow/{flow_id}"},
        ))

    def final(self, text: str | None) -> None:
        if not text:
            return
        mid = uuid.uuid4().hex
        self._q.put_nowait(TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START, message_id=mid, role="assistant",
        ))
        self._q.put_nowait(TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=text or "",
        ))
        self._q.put_nowait(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END, message_id=mid,
        ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    redis_cache = None
    if settings.redis_url:
        redis_cache = RedisEntityCache(
            redis_url=settings.redis_url,
            sync_interval=settings.redis_sync_interval,
            top_k=settings.entity_top_k,
        )
    mcp = LangflowMCPClient(
        mcp_path=settings.langflow_mcp_path,
        langflow_api_key=settings.langflow_api_key,
        langflow_base_url=settings.langflow_base_url,
        redis_cache=redis_cache,
    )
    await mcp.connect()
    app.state.settings = settings
    app.state.mcp = mcp
    app.state.llm = get_provider(settings)
    app.state.tools = mcp.get_tool_schemas()
    logger.info("agent server ready")
    try:
        yield
    finally:
        await mcp.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _latest_user_text(messages: list) -> str:
    """Extract the newest user message text from the AG-UI input message list."""
    for m in reversed(messages or []):
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role == "user":
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multimodal parts -> join text parts
                return "".join(
                    getattr(p, "text", "") or (p.get("text", "") if isinstance(p, dict) else "")
                    for p in content
                )
    return ""


@app.post("/agent")
async def agent_endpoint(request: Request):
    raw = await request.json()
    run_input = RunAgentInput.model_validate(raw)
    thread_id = run_input.thread_id or uuid.uuid4().hex
    run_id = run_input.run_id or uuid.uuid4().hex

    settings: Settings = request.app.state.settings
    mcp = request.app.state.mcp
    llm = request.app.state.llm
    tools = request.app.state.tools

    messages = _THREADS.setdefault(thread_id, [])
    user_text = _latest_user_text(run_input.messages)
    if user_text:
        messages.append({"role": "user", "content": user_text})

    encoder = EventEncoder()
    queue: "asyncio.Queue" = asyncio.Queue()
    sink = AGUISink(queue, settings.langflow_base_url)
    _DONE = object()

    async def runner():
        try:
            updated = await run_turn(llm, mcp, settings, tools, messages, {}, sink)
            _THREADS[thread_id] = updated
            if sink.flow_id:
                _THREAD_FLOW[thread_id] = sink.flow_id
        except Exception as e:  # surface as RUN_ERROR
            logger.exception("run_turn failed")
            queue.put_nowait(("__error__", str(e)))
        finally:
            queue.put_nowait(_DONE)

    async def event_stream():
        yield encoder.encode(RunStartedEvent(
            type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id,
        ))
        # Replay the thread's current flow so a reconnecting client keeps its canvas.
        prior_flow = _THREAD_FLOW.get(thread_id)
        if prior_flow:
            base = settings.langflow_base_url.rstrip("/")
            yield encoder.encode(StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot={"flow_id": prior_flow, "flow_url": f"{base}/flow/{prior_flow}"},
            ))

        task = asyncio.create_task(runner())
        error_msg: str | None = None
        while True:
            ev = await queue.get()
            if ev is _DONE:
                break
            if isinstance(ev, tuple) and ev and ev[0] == "__error__":
                error_msg = ev[1]
                continue
            yield encoder.encode(ev)
        await task

        if error_msg is not None:
            yield encoder.encode(RunErrorEvent(
                type=EventType.RUN_ERROR, message=error_msg,
            ))
        else:
            yield encoder.encode(RunFinishedEvent(
                type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id,
            ))

    return StreamingResponse(event_stream(), media_type=encoder.get_content_type())


@app.get("/health")
async def health():
    return {"status": "ok"}
