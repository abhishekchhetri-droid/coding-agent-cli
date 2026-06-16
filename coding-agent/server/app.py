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
import json
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
from agent.events import slim_graph
from server.canvas import apply_canvas_ops

logger = logging.getLogger("agent.server")

# Per-thread conversation history (internal OpenAI-style message dicts), last flow,
# and cumulative session token total (sum of every turn's tokens on the thread).
_THREADS: dict[str, list[dict]] = {}
_THREAD_FLOW: dict[str, str] = {}
_THREAD_TOKENS: dict[str, int] = {}
# Persistent planning/design state per thread so a web session behaves like the CLI:
# todos + scratchpad survive summarization, approved designs + intended edges survive
# across turns (design in one POST, build in the next). run_turn mutates these in place.
_THREAD_STATE: dict[str, dict] = {}


def _thread_state(thread_id: str) -> dict:
    return _THREAD_STATE.setdefault(
        thread_id,
        {"todos": [], "scratchpad": {}, "intended": {}, "approved": {}, "pending": None},
    )


# Words that count as "approve" when the user replies to a paused confirm gate. Anything
# else is treated as feedback → the model revises instead of building.
_AFFIRMATIVE = {
    "y", "yes", "yeah", "yep", "yup", "ok", "okay", "approve", "approved", "proceed",
    "build", "build it", "go", "go ahead", "confirm", "confirmed", "sure", "lgtm", "do it",
}


def _is_affirmative(text: str) -> bool:
    """True if the user's reply to a pending gate is an approval (vs. feedback to revise)."""
    if not text:
        return False
    # Strip the injected [Context: …] flow prefix the handler prepends to user messages.
    cleaned = text.split("]\n", 1)[-1] if text.startswith("[Context:") else text
    cleaned = cleaned.strip().lower().rstrip(".!")
    return cleaned in _AFFIRMATIVE


class AGUISink:
    """Translates agent run_turn signals into AG-UI events on an asyncio.Queue."""

    interactive = False  # no stdin in the web server — confirm gates must auto-approve

    def __init__(self, queue: "asyncio.Queue", langflow_base_url: str, session_tokens: int = 0):
        self._q = queue
        self._base = langflow_base_url.rstrip("/")
        self.flow_id: str | None = None
        self.graph: dict | None = None  # last known graph, so graph-less snapshots don't blank the canvas
        self.usage_metrics: dict | None = None  # last turn's token/timing totals
        self.session_tokens = session_tokens  # cumulative tokens across this thread's turns
        self.awaiting: dict | None = None  # {"kind","ref"} when a gate is paused awaiting approval

    def tool_call(self, name: str, arguments: dict) -> None:
        self._q.put_nowait(StepStartedEvent(type=EventType.STEP_STARTED, step_name=name))
        self._q.put_nowait(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=name))

    def _full_state(self) -> dict:
        # STATE_SNAPSHOT replaces the WHOLE client state, so every snapshot must carry all
        # known fields — flow + last graph + usage — or an event that knows only one of them
        # (e.g. a usage tick on a pure-chat turn, or a graph-less build_flow) would erase the
        # rest. Each field is included only once it has a value.
        snap: dict = {}
        if self.flow_id:
            snap["flow_id"] = self.flow_id
            snap["flow_url"] = f"{self._base}/flow/{self.flow_id}"
            if self.graph is not None:
                snap["graph"] = self.graph
        if self.usage_metrics is not None:
            snap["usage"] = self.usage_metrics
        if self.awaiting is not None:
            snap["awaiting_confirmation"] = self.awaiting
        return snap

    def _emit_state(self) -> None:
        self._q.put_nowait(StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=self._full_state(),
        ))

    def flow_built(self, flow_id: str | None, graph: dict | None = None) -> None:
        if not flow_id:
            return
        self.flow_id = flow_id
        if graph is not None:
            self.graph = graph
        self._emit_state()

    def flow_modified(self, graph: dict | None = None) -> None:
        """Re-emit STATE_SNAPSHOT with the latest graph after in-place modifications.

        The self-rendered canvas diffs by node id, so streaming the graph updates it in
        place — no iframe reload, viewport preserved.
        """
        if not self.flow_id:
            return
        if graph is not None:
            self.graph = graph
        self._emit_state()

    def usage(self, metrics: dict) -> None:
        """End-of-turn token/timing totals → token meter. Tracks a running session total."""
        self.session_tokens += int(metrics.get("total", 0) or 0)
        self.usage_metrics = {**metrics, "session_total": self.session_tokens}
        self._emit_state()

    def _emit_message(self, text: str) -> None:
        """Stream one complete assistant chat message (START/CONTENT/END)."""
        mid = uuid.uuid4().hex
        self._q.put_nowait(TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START, message_id=mid, role="assistant",
        ))
        self._q.put_nowait(TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=text,
        ))
        self._q.put_nowait(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END, message_id=mid,
        ))

    def notice(self, markdown: str | None) -> None:
        # Intermediate artifacts (proposed design, plan) the CLI shows as a Rich Panel —
        # surface them in the web chat as their own assistant message. Display only: no
        # turn-stop, no state, the build still proceeds in the same turn.
        if markdown:
            self._emit_message(markdown)

    def confirm_request(self, kind: str, markdown: str, ref: str | None) -> None:
        # A web gate paused: flag the client to show the Approve / Request-changes bar. The
        # proposal markdown already streamed via notice() just above; this only sets the
        # awaiting flag on the next STATE_SNAPSHOT so the confirm bar renders under it.
        self.awaiting = {"kind": kind, "ref": ref}
        self._emit_state()

    def final(self, text: str | None) -> None:
        if not text:
            return
        self._emit_message(text)


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

    # The dock sends the canvas's open flow id so the agent edits THAT flow rather than
    # creating a new one. AG-UI carries arbitrary client data in forwarded_props; also accept
    # a top-level key for convenience.
    fwd = getattr(run_input, "forwarded_props", None) or {}
    open_flow_id = (fwd.get("flow_id") if isinstance(fwd, dict) else None) or raw.get("flow_id")

    state = _thread_state(thread_id)
    if open_flow_id:
        _THREAD_FLOW[thread_id] = open_flow_id
        # Make "this/that flow" resolve to the open canvas flow for the agent.
        state["scratchpad"]["flow:current"] = open_flow_id

    messages = _THREADS.setdefault(thread_id, [])
    raw_user_text = _latest_user_text(run_input.messages)
    user_text = raw_user_text
    if user_text:
        if open_flow_id:
            user_text = (
                f"[Context: the user is viewing flow_id={open_flow_id} on the canvas. "
                f"Edit THIS flow (update_flow/get_flow with that id) unless they explicitly "
                f"ask to create a new flow.]\n{user_text}"
            )
        messages.append({"role": "user", "content": user_text})

    # Resolve a paused confirm gate (turn-boundary gating): if a design/plan was awaiting
    # approval, classify this reply. Affirmative → confirm (and, for a design, inject a
    # deterministic build directive); anything else → feedback, the model revises.
    gate_confirm = False
    pending = state.get("pending")
    if pending:
        state["pending"] = None
        if _is_affirmative(raw_user_text):
            gate_confirm = True
            if pending.get("kind") == "design" and pending.get("ref"):
                messages.append({
                    "role": "user",
                    "content": (
                        f'User approved the design. Build it now: call create_flow with '
                        f'{{"data":{{"_design_ref":"{pending["ref"]}"}}}}.'
                    ),
                })

    encoder = EventEncoder()
    queue: "asyncio.Queue" = asyncio.Queue()
    sink = AGUISink(queue, settings.langflow_base_url, session_tokens=_THREAD_TOKENS.get(thread_id, 0))
    # Seed the flow id from the thread so edits to an EXISTING flow (e.g. "remove the
    # url tool") emit canvas snapshots. Without this, a turn that never calls create_flow
    # leaves sink.flow_id=None and flow_modified() is suppressed.
    sink.flow_id = _THREAD_FLOW.get(thread_id)
    _DONE = object()

    async def runner():
        try:
            updated = await run_turn(
                llm, mcp, settings, tools, messages, {}, sink,
                todos=state["todos"], scratchpad=state["scratchpad"],
                _intended_edges_by_flow=state["intended"], _approved_designs=state["approved"],
                gate_state=state, gate_confirm=gate_confirm,
            )
            _THREADS[thread_id] = updated
            if sink.flow_id:
                _THREAD_FLOW[thread_id] = sink.flow_id
            _THREAD_TOKENS[thread_id] = sink.session_tokens
        except Exception as e:  # surface as RUN_ERROR
            logger.exception("run_turn failed")
            queue.put_nowait(("__error__", str(e)))
        finally:
            queue.put_nowait(_DONE)

    async def event_stream():
        yield encoder.encode(RunStartedEvent(
            type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id,
        ))
        # Replay the thread's current flow + session token total so a reconnecting client
        # repaints its canvas and meter. Fetch the graph too — a reloaded browser has no
        # prior state to diff against. Routed through the sink so the replayed snapshot
        # carries every known field (flow + graph + usage) and nothing gets erased.
        prior_flow = _THREAD_FLOW.get(thread_id)
        if prior_flow:
            try:
                raw = await mcp.call_tool("get_flow", {"flow_id": prior_flow})
                flow = json.loads(raw) if isinstance(raw, str) else raw
                graph = slim_graph(flow) if isinstance(flow, dict) else None
                if graph is not None:
                    sink.graph = graph  # seed so a graph-less build_flow this turn won't blank it
            except Exception:
                logger.warning("replay get_flow failed for %s", prior_flow, exc_info=True)
        if sink.session_tokens:
            # Show the carried-over session total immediately (no per-field turn detail yet).
            sink.usage_metrics = {"session_total": sink.session_tokens}
        if prior_flow or sink.session_tokens:
            yield encoder.encode(StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=sink._full_state(),
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


@app.post("/canvas/mutate")
async def canvas_mutate(request: Request):
    """Persist direct canvas edits (drag/field/edge/node) for a thread's flow.

    Body: {thread_id, ops:[{op, ...}], flow_id?}. Resolves flow_id from the thread (the
    same map the chat agent populates) so the browser need only send the thread id.
    Returns {graph: <slim_graph>} for the canvas to reconcile against.
    """
    body = await request.json()
    thread_id = body.get("thread_id") or ""
    ops = body.get("ops") or []
    flow_id = _THREAD_FLOW.get(thread_id) or body.get("flow_id")
    if not flow_id:
        return {"error": "no flow for thread"}
    if not isinstance(ops, list) or not ops:
        return {"error": "no ops"}

    mcp = request.app.state.mcp
    try:
        graph = await apply_canvas_ops(mcp, flow_id, ops)
    except Exception as e:
        logger.exception("canvas mutate failed")
        return {"error": str(e)}
    return {"graph": graph}


@app.get("/health")
async def health():
    return {"status": "ok"}
