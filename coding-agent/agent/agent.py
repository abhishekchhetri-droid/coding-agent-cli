import asyncio
import json
import random
import re
import time
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from llm.base import LLMProvider
from mcpbridge.client import LangflowMCPClient
from config.settings import Settings
from agent.prompts import SYSTEM_PROMPT
from agent.events import ConsoleSink, slim_graph
from agent import planning
from agent import designer
from agent import pipeline
from agent.context import summarize_history, compact_flow_snapshots

console = Console()


def _canonical_handle(handle):
    """Normalize a react-flow handle to Langflow's canonical string form.

    Langflow's frontend matches edges to handles via ``scapedJSONStringfy``,
    which is ``JSON.stringify(obj)`` — compact, NO spaces after ':' or ','.
    Python's default ``json.dumps`` emits ``", "`` / ``": "`` separators, so an
    object-based handle serialized that way never matches the handle id the
    frontend computes from the node template, and the edge is silently dropped
    on render. Always emit the compact form; also re-normalize handles that
    arrive already-stringified (possibly space-polluted, possibly œ-escaped) so
    re-saving repairs previously-broken flows.
    """
    if isinstance(handle, dict):
        return json.dumps(handle, separators=(",", ":"))
    if isinstance(handle, str):
        try:
            obj = json.loads(handle.replace("œ", '"'))
        except (ValueError, TypeError):
            return handle
        canonical = json.dumps(obj, separators=(",", ":"))
        # Preserve œ-escaping if the source used it (Langflow's wire format).
        return canonical.replace('"', "œ") if "œ" in handle else canonical
    return handle


def _serialize_edge_handles(edges: list[dict]) -> list[dict]:
    """Serialize sourceHandle/targetHandle to Langflow's canonical handle form
    and ensure each edge has an id. React-flow requires handle identifiers to be
    compact JSON strings, not objects, and not space-padded.
    """
    result = []
    for i, edge in enumerate(edges):
        edge = dict(edge)
        if edge.get("sourceHandle") is not None:
            edge["sourceHandle"] = _canonical_handle(edge["sourceHandle"])
        if edge.get("targetHandle") is not None:
            edge["targetHandle"] = _canonical_handle(edge["targetHandle"])
        if "id" not in edge:
            edge["id"] = f"{edge.get('source', 'src')}-{edge.get('target', 'tgt')}-{i}"
        result.append(edge)
    return result


def _handle_dict(edge: dict, side: str) -> dict:
    """Return an edge's source/targetHandle as a dict, tolerating Langflow's two
    serializations: a parsed dict under edge['data'][side], or a top-level string
    where double-quotes are encoded as U+0153 (œ). Live get_flow returns the latter;
    agent-built edges use plain dicts."""
    d = edge.get("data")
    if isinstance(d, dict) and isinstance(d.get(side), dict):
        return d[side]
    h = edge.get(side)
    if isinstance(h, dict):
        return h
    if isinstance(h, str) and h:
        try:
            return json.loads(h.replace("œ", '"'))
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _edge_key(edge: dict) -> tuple | None:
    """Stable identity of an edge as (source, target, target_field), handle-format agnostic."""
    th = _handle_dict(edge, "targetHandle")
    src, tgt, fld = edge.get("source"), edge.get("target"), th.get("fieldName")
    if src and tgt and fld:
        return (src, tgt, fld)
    return None


def _node_type(n: dict) -> str:
    return n.get("data", {}).get("type") or n.get("type", "")


def _severed_nodes(nodes: list[dict], edges: list[dict], extra_edges: list[dict] | None = None) -> list[str]:
    """Node ids NOT weakly-connected to the input chain — orphan islands no input can reach.

    Edges are treated as UNDIRECTED: a stage is "live" if anything links it (in either
    direction) back toward the entry, so forward producers (Embeddings → vector store) and
    model providers (LLM → Agent.model) are not false-flagged. ``extra_edges`` (the intended
    edges) are merged in so that model-input edges — which Langflow STRIPS from every saved
    flow — still count for connectivity; otherwise an LLM whose only link is a model handle
    would look severed on every agent flow. Seeds = ChatInput nodes; if a flow has none, every
    node with no incoming edge. Catches a severed pipeline whose feeder was dropped (a same-type
    stage removed by dedup, or an edge wired to a wrong/non-existent field): the downstream tail
    (e.g. executor → ChatOutput) lands in its own component. A built pipeline should be one piece.
    """
    ids = [n.get("id", "") for n in nodes if n.get("id")]
    if not ids:
        return []
    adj: dict[str, set] = {i: set() for i in ids}
    has_incoming: set = set()
    for e in list(edges) + list(extra_edges or []):
        s, t = e.get("source"), e.get("target")
        if s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)
            has_incoming.add(t)
    seeds = [n.get("id", "") for n in nodes if _node_type(n) == "ChatInput" and n.get("id")]
    if not seeds:
        seeds = [i for i in ids if i not in has_incoming]
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for nb in adj.get(cur, ()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return [i for i in ids if i not in seen]


def _dead_end_producers(nodes: list[dict], edges: list[dict], extra_edges: list[dict] | None = None) -> list[str]:
    """Producer nodes whose DATA output is consumed by nothing — a computed-and-discarded branch
    (e.g. an intent-classification LLM whose result feeds no downstream stage).

    Schema-driven: a node is a dead-end iff it emits at least one non-tool, non-model output yet
    appears as the source of no edge (surviving ∪ intended). ChatOutput (the terminal sink) is
    excluded. Model providers are NOT flagged on their model output — Langflow strips model
    edges, so connectivity there comes from ``extra_edges`` (intended); their data output being
    idle is normal. General — no component or field names hardcoded.
    """
    out_sources = {e.get("source") for e in list(edges) + list(extra_edges or []) if e.get("source")}
    dead: list[str] = []
    for n in nodes:
        nid = n.get("id", "")
        if not nid or _node_type(n) == "ChatOutput":
            continue
        outs = n.get("data", {}).get("node", {}).get("outputs", []) or []
        if not outs:
            continue  # pure sink — nothing to consume
        if all(o.get("name") in ("component_as_tool", "api_build_tool") for o in outs):
            continue  # tool-only outputs are consumed via Agent.tools, not data edges
        has_data_output = any(
            any(t != "LanguageModel" for t in (o.get("types") or o.get("output_types") or []))
            for o in outs
        )
        if has_data_output and nid not in out_sources:
            dead.append(nid)
    return dead


def _inject_node_check(
    get_flow_result: str | None,
    mcp: "LangflowMCPClient",
    flow_id: str,
    intended_edges: list[dict] | None = None,
) -> str:
    """Verify a built flow: detect edges Langflow rejected (diff vs intended),
    audit required inputs, then smoke test-run. Append a verdict to the result."""
    if not get_flow_result:
        return get_flow_result or "null"
    try:
        data = json.loads(get_flow_result)
        flow_data = data.get("data", {}) if isinstance(data, dict) else {}
        nodes = flow_data.get("nodes", [])
        edges = flow_data.get("edges", [])
        n_nodes = len(nodes)
        n_edges = len(edges)

        if n_nodes == 0:
            return (
                get_flow_result
                + "\n\n⚠ VERIFICATION FAILED: 0 nodes in flow after build. "
                "All your node type strings were rejected by Langflow. "
                "Call list_components, read exact 'type' field values, retry update_flow. "
                "Do NOT report success until node count > 0."
            )

        # Map each node's satisfied target fields from the SURVIVING edges.
        incoming: dict[str, set[str]] = {}
        for e in edges:
            key = _edge_key(e)
            if key:
                incoming.setdefault(key[1], set()).add(key[2])

        def _is_model_field(node_id: str, fname: str, intended_handle: dict) -> bool:
            """A field is a model input if the node's template says type 'model', or the
            intended edge declared a LanguageModel/model handle (field may have vanished)."""
            tmpl = next(
                (x.get("data", {}).get("node", {}).get("template", {})
                 for x in nodes if x.get("id") == node_id),
                {},
            )
            f = tmpl.get(fname)
            if isinstance(f, dict) and (f.get("type") == "model" or "LanguageModel" in (f.get("input_types") or [])):
                return True
            return intended_handle.get("type") == "model" or "LanguageModel" in (intended_handle.get("inputTypes") or [])

        # Detect edges Langflow rejected as invalid: intended − surviving. This is
        # the robust catch — it works even when the rejection deleted the target
        # field entirely (e.g. wiring to a Prompt variable never declared in the
        # template). Model-handle rejections are the known platform limitation, not
        # a build bug, so they only feed the soft MODEL note.
        stripped_real: list[str] = []
        model_gaps: list[str] = []
        if intended_edges:
            surviving_keys = {k for k in (_edge_key(e) for e in edges) if k}
            for ie in intended_edges:
                key = _edge_key(ie)
                if not key or key in surviving_keys:
                    continue
                th = _handle_dict(ie, "targetHandle")
                label = f"{key[1]}.{key[2]}"
                if _is_model_field(key[1], key[2], th):
                    model_gaps.append(label)
                else:
                    stripped_real.append(f"{key[0]} → {label}")

        # Schema-driven required-input audit over the SURVIVING graph.
        wiring_gaps: list[str] = []   # required pure-handle field (must be wired), no edge
        cred_gaps: list[str] = []     # required literal/credential field left empty
        for n in nodes:
            nid = n.get("id", "")
            tmpl = n.get("data", {}).get("node", {}).get("template", {})
            satisfied = incoming.get(nid, set())
            for fname, f in tmpl.items():
                if not isinstance(f, dict) or not f.get("required") or fname in satisfied:
                    continue
                itypes = f.get("input_types") or []
                has_value = f.get("value") not in (None, "", [], {})
                is_model = f.get("type") == "model" or "LanguageModel" in itypes
                # Only 'other'-typed fields are pure handles that MUST be wired.
                # A str field exposing input_types=['Message'] is also user-fillable,
                # so treat it as a credential/literal, not a forced wire.
                is_pure_handle = (not is_model) and f.get("type") == "other"
                if is_model:
                    if not has_value:
                        label = f"{nid}.{fname}"
                        if label not in model_gaps:
                            model_gaps.append(label)
                elif is_pure_handle:
                    wiring_gaps.append(f"{nid}.{fname}")
                elif not has_value:
                    cred_gaps.append(f"{nid}.{fname}")

        # Rejected edges = real structural failure; block first and loudest.
        if stripped_real:
            return (
                get_flow_result
                + f"\n\n⚠ EDGES REJECTED: Langflow dropped these connections as invalid: "
                f"{'; '.join(stripped_real)}. Usual cause: the target field does not exist "
                "on the component (e.g. wiring to a Prompt variable before setting the "
                "component's `template` value with that {variable}; or a source/target type "
                "mismatch). Set the template value (so the variable handles are created) or "
                "fix the field name/handle types, then update_flow + rebuild. "
                "Do NOT report success until these edges survive a rebuild."
            )

        # Pure-handle wiring gaps mean the graph is structurally broken — block.
        if wiring_gaps:
            return (
                get_flow_result
                + f"\n\n⚠ WIRING INCOMPLETE: required input(s) with no incoming edge: "
                f"{', '.join(wiring_gaps)}. "
                "Call get_component_schema for each affected component, then update_flow "
                "with edges feeding these fields. "
                "Do NOT report success until every listed field is wired."
            )

        # Severed pipeline: a node island unreachable from the input chain. Catches damage the
        # intended-vs-surviving diff misses (a feeder removed BEFORE build, or an edge wired to
        # the wrong field leaving the real port empty) — both rewrite "intent" so the diff sees
        # nothing, but connectivity does not lie. Skipped when model gaps exist: stripped model
        # edges are the dominant cause of apparent islands and a known platform limitation, so
        # defer to the MODEL note (re-checked on the next rebuild once the model is configured).
        if not model_gaps:
            severed = _severed_nodes(nodes, edges, extra_edges=intended_edges)
            if severed:
                return (
                    get_flow_result
                    + f"\n\n⚠ PIPELINE SEVERED: these nodes are not connected to the ChatInput "
                    f"chain: {', '.join(severed)}. A stage's feeder is missing — usually a "
                    "same-type stage was dropped, or an edge was wired to a wrong/non-existent "
                    "field (so the real input port is empty). Call get_component_schema for the "
                    "affected components, confirm each edge's target field EXISTS and the source "
                    "output type is in the target field's input_types, then update_flow to "
                    "reconnect. Do NOT report success until every node is reachable from ChatInput."
                )

            # Dead branch: a producer whose data output feeds nothing — computed then discarded
            # (e.g. an intent-classification LLM with no consumer). Same model-gap gate so a
            # stripped-model provider isn't false-flagged.
            dead = _dead_end_producers(nodes, edges, extra_edges=intended_edges)
            if dead:
                return (
                    get_flow_result
                    + f"\n\n⚠ DEAD BRANCH: these stages produce output that nothing consumes: "
                    f"{', '.join(dead)}. Each is computed then discarded. Either wire its output "
                    "into a downstream stage that uses it (e.g. route on a classification result), "
                    "or remove the stage if it is not needed. Then update_flow + rebuild. "
                    "Do NOT report success while a stage's output goes nowhere."
                )

        model_note = ""
        if model_gaps:
            model_note = (
                f"\n\n⚠ MODEL NOT CONFIGURED: {', '.join(model_gaps)}. Langflow strips "
                "model-input edges from saved flows, so connect the model in the UI "
                "(or have the user pick a provider). This is a known platform limitation, "
                "not a wiring bug — report it as a setup step, not a failure."
            )

        cred_note = ""
        if cred_gaps:
            cred_note = (
                f"\n\n⚠ NEEDS CREDENTIALS: structurally complete, but these required "
                f"fields are empty and must be filled by the user before the flow runs: "
                f"{', '.join(cred_gaps)}."
            )
        cred_note = model_note + cred_note

        # Smoke test-run to confirm execution of a structurally-complete flow.
        test = mcp.test_run_flow(flow_id)
        if test["ok"]:
            suffix = (
                f"\n\n✅ VERIFIED: Flow is wired and ran a smoke test. "
                f"Flow has {n_nodes} nodes, {n_edges} edges."
            )
            suffix += " Report success (note setup items above)." if (cred_gaps or model_gaps) else " You may report success."
            return get_flow_result + cred_note + suffix
        else:
            # A run failure when the only gaps are unfilled credentials / an
            # unconfigured model is the user's to resolve, not a structural build
            # error — don't hard-fail.
            if cred_gaps or model_gaps:
                return (
                    get_flow_result
                    + cred_note
                    + f"\n\n✅ VERIFIED (structure): Flow is wired correctly with "
                    f"{n_nodes} nodes, {n_edges} edges. Smoke run could not complete "
                    "because the credential fields above are empty — expected until the "
                    "user fills them. Report success and list the fields to fill."
                )
            return (
                get_flow_result
                + f"\n\n⚠ EXECUTION FAILED: Flow has {n_nodes} nodes but failed to run: "
                f"{test['error']}. "
                "Do NOT report success. Fix credentials/wiring/config and retry."
            )
    except (json.JSONDecodeError, AttributeError):
        pass
    return get_flow_result


def _tool_result_message(tool_call_id: str, result: str | None) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result or "null",
    }


def _compact_tool_args(tc: dict) -> dict:
    """Strip large data payloads from create/update tool calls before storing in history.
    These calls carry full node schemas (~5K tokens each) that repeat in every LLM call."""
    if tc["name"] in ("create_flow", "update_flow"):
        args = dict(tc["arguments"])
        if isinstance(args.get("data"), dict):
            d = args["data"]
            n = len(d.get("nodes", []))
            e = len(d.get("edges", []))
            args["data"] = f"<{n} nodes, {e} edges — payload stripped>"
        return args
    # The live todo list is re-injected as working state every iteration, so the full
    # list in each historical write_todos call is pure duplication — strip it.
    if tc["name"] == "write_todos":
        return {"todos": "<updated — see live working state>"}
    return tc["arguments"]


def _assistant_tool_call_message(tool_calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(_compact_tool_args(tc)),
                },
            }
            for tc in tool_calls
        ],
    }


def _is_langmodel_node(n: dict) -> bool:
    return any(
        "LanguageModel" in (o.get("types") or o.get("output_types") or [])
        for o in n.get("data", {}).get("node", {}).get("outputs", [])
    )

def _find_model_field(n: dict) -> str | None:
    """Return template field name of the first ModelInput that accepts LanguageModel, or None."""
    tmpl = n.get("data", {}).get("node", {}).get("template", {})
    for k, v in tmpl.items():
        if isinstance(v, dict) and v.get("type") == "model" and "LanguageModel" in (v.get("input_types") or []):
            return k
    return None

def _has_only_tool_outputs(n: dict) -> bool:
    """True if a node's only outputs are tool handles (component_as_tool/api_build_tool).
    Such nodes MUST be used as Agent tools — they have no data output for pipeline use."""
    outputs = n.get("data", {}).get("node", {}).get("outputs", [])
    return bool(outputs) and all(o.get("name") in ("component_as_tool", "api_build_tool") for o in outputs)

_THINKING_WORDS = [
    "Thinking", "Pondering", "Cogitating", "Marinating", "Ruminating",
    "Deliberating", "Analyzing", "Reasoning", "Contemplating", "Brewing",
    "Calculating", "Reflecting", "Theorizing", "Synthesizing", "Formulating",
    "Extrapolating", "Inferring", "Deducing", "Hallucinating", "Vibing",
    "Cooking", "Simmering", "Baking", "Percolating", "Noodling",
]


async def _cycle_status(status_obj: object) -> None:
    while True:
        await asyncio.sleep(2)
        status_obj.update(f"[dim]{random.choice(_THINKING_WORDS)}…[/dim]")


async def run_turn(
    llm, mcp, settings, tools, messages, _starter_cache, sink,
    todos=None, scratchpad=None, _intended_edges_by_flow=None, _approved_designs=None,
):
    iterations = 0
    prompt_tokens = 0
    completion_tokens = 0
    turn_start = time.perf_counter()
    tool_time = 0.0  # cumulative wall-clock spent executing tools this turn (pure Python)
    # Persistent agent state. Owned by run_chat (CLI) so it survives across turns; the
    # web path (server/app.py) passes nothing and gets fresh state per turn.
    #   todos/scratchpad        — plan + saved determinants, rendered each iteration
    #   _intended_edges_by_flow — edges last sent per flow_id, for the post-build verifier
    #   _approved_designs       — designer sub-agent outputs, resolved by create_flow ref
    if todos is None:
        todos = []
    if scratchpad is None:
        scratchpad = {}
    if _intended_edges_by_flow is None:
        _intended_edges_by_flow = {}
    if _approved_designs is None:
        _approved_designs = {}
    # Plan confirm-gate fires once per user request: the first multi-step plan is shown and
    # approved, then the agent auto-locks through execution without re-prompting.
    _plan_confirmed = False
    # True once propose_pipeline has surfaced open questions to the user this turn. When a
    # later propose_pipeline call comes back with no open questions, the user has actually
    # aligned the interpretation → we can auto-confirm the downstream design graph (no
    # redundant second y/n). Without an earlier ask (all stages 'ok' first try), the design
    # gate stays active as the sole human checkpoint.
    _pipeline_asked = False
    # tool_call_ids of flow-JSON snapshots (get_flow/create/update) this turn. Only the
    # latest snapshot is relevant, so older ones are stubbed to reclaim ~5K each.
    _flow_snapshot_ids: list[str] = []
    # Guard against a planning loop: consecutive iterations that only touched planning
    # tools (no real work). Nudge at 3, abort at 6.
    _stall = 0
    # tool_call_ids of get_component_schema results this turn. Schemas are only needed
    # while wiring edges; once a create/update_flow consumes them they're stubbed.
    _schema_msg_ids: set[str] = set()

    while iterations < settings.max_tool_iterations:
        # planning tools are agent-side (loop state closures), appended to the MCP set.
        # Order: [stable baseline+virtual] + [stable planning] + [volatile discovered].
        # Planning tools are stable, so they belong in the cached region ahead of the
        # volatile discovered tail (tagged `_volatile` by get_tool_schemas).
        _tool_schemas = mcp.get_tool_schemas()  # rebuilt each iteration so discovered tools appear next turn
        _volatile = [t for t in _tool_schemas if t.get("_volatile")]
        _stable = [t for t in _tool_schemas if not t.get("_volatile")]
        _planning_tools = planning.PLANNING_TOOL_SCHEMAS + [
            pipeline.PROPOSE_PIPELINE_TOOL_SCHEMA, designer.DESIGN_FLOW_TOOL_SCHEMA,
        ]
        tools = _stable + _planning_tools + _volatile
        # Inject live todos + scratchpad as a TRAILING context message, not into the
        # system prompt: the system block is prompt-cached, so mutating it each step
        # would bust the cache and re-bill the whole system prompt every iteration.
        # As a trailing message the cached system/tools/history prefix stays intact.
        state_block = planning.render_state(todos, scratchpad)
        call_messages = messages
        if state_block:
            call_messages = messages + [{
                "role": "user",
                "content": f"[Your live working state — plan + saved facts]\n{state_block}",
                # Mutates every iteration: the provider keeps the cache breakpoint on the
                # stable history before this message so the prefix stays a cache hit.
                "_ephemeral": True,
            }]
        try:
            t0 = time.perf_counter()
            with console.status(f"[dim]{random.choice(_THINKING_WORDS)}…[/dim]", spinner="dots") as _status:
                _cycle = asyncio.create_task(_cycle_status(_status))
                try:
                    response = await llm.complete(call_messages, tools, system=SYSTEM_PROMPT)
                finally:
                    _cycle.cancel()
            elapsed = time.perf_counter() - t0
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[dim]Interrupted.[/dim]")
            raise
        except Exception as e:
            # A transient LLM/API error (timeout, dropped connection, 5xx) must NOT kill the
            # session. The failed call appended nothing, so message history is still valid —
            # abandon this turn and let the user re-send. Provider already retried.
            console.print(f"\n[red]✗ LLM request failed:[/red] {type(e).__name__}: {e}")
            console.print("[dim]Turn aborted — your session and context are intact. "
                          "Re-send your last message to retry.[/dim]")
            break

        iter_prompt = 0
        iter_completion = 0
        cache_read = 0
        cache_write = 0
        if response["usage"]:
            iter_prompt = response["usage"]["prompt_tokens"]
            iter_completion = response["usage"]["completion_tokens"]
            cache_read = response["usage"].get("cache_read_tokens", 0) or 0
            cache_write = response["usage"].get("cache_creation_tokens", 0) or 0
            prompt_tokens += iter_prompt
            completion_tokens += iter_completion
        cache_str = f" · 📦 r={cache_read:,} w={cache_write:,}" if cache_read or cache_write else ""
        console.print(
            f"[dim]⏱ {elapsed:.1f}s · ↑{iter_prompt:,} ↓{iter_completion:,} tokens{cache_str}[/dim]"
        )

        if response["tool_calls"]:
            for tc in response["tool_calls"]:
                args_str = json.dumps(tc['arguments'], indent=None)
                if len(args_str) > 300:
                    args_str = args_str[:300] + "…"
                console.print(f"[dim]→ {tc['name']}({args_str})[/dim]")
                sink.tool_call(tc['name'], tc['arguments'])

            messages.append(_assistant_tool_call_message(response["tool_calls"]))

            # Build credential overrides once per turn — reused by create_flow, update_flow, clone_starter_template
            _credential_overrides: dict = {}
            if settings.azure_anthropic_api_key:
                _credential_overrides["AnthropicModel"] = {
                    "api_key": settings.azure_anthropic_api_key,
                    "base_url": settings.azure_anthropic_endpoint,
                    "model_name": settings.azure_anthropic_deployment,
                }
                _credential_overrides["Agent"] = {
                    "api_key": settings.azure_anthropic_api_key,
                }
            if settings.azure_openai_endpoint:
                _credential_overrides["AzureOpenAIModel"] = {
                    "azure_endpoint": settings.azure_openai_endpoint,
                    "api_key": settings.azure_openai_api_key,
                    "azure_deployment": settings.azure_openai_deployment,
                    "api_version": settings.azure_openai_api_version,
                }

            _last_build_flow_id: str | None = None
            _intended_edges: list[dict] = []  # transient: edges from this turn's create/replace
            _starter_template_msg_ids: set[str] = set()
            _plan_rejected = False  # per-response: skip rest of THIS batch, not future re-plans
            for tc in response["tool_calls"]:
                args = tc["arguments"]
                _tool_t0 = time.perf_counter()  # pure-Python exec timer for this tool
                _canvas_graph: dict | None = None  # slim graph for the live canvas, set when a tool returns flow data

                # Plan rejected this turn: every queued tool_call still needs a tool
                # result (Anthropic requires one per tool_use), but we run none of them.
                if _plan_rejected:
                    messages.append(_tool_result_message(tc["id"], "skipped — plan not approved; awaiting revision"))
                    continue

                # Auto-enrich nodes with full component schemas + credentials before sending to Langflow.
                # create_flow and update_flow have distinct semantics:
                #   create: build-from-scratch, dedup structural singletons, inject hardcoded stub IDs
                #   update: fetch existing, merge delta, preserve existing IDs (no destructive dedup)
                enrich_error: str | None = None
                build_block: str | None = None

                def _node_outputs_langmodel(n: dict) -> bool:
                    return any(
                        "LanguageModel" in (o.get("types") or o.get("output_types") or [])
                        for o in n.get("data", {}).get("node", {}).get("outputs", [])
                    )

                def _enrich_create_data(data: dict, from_design: bool = False) -> dict:
                    # ``from_design``: the build came from an approved design_flow design
                    # (user already reviewed the exact node/edge graph). The design is
                    # AUTHORITATIVE — skip the structural rewrites below (singleton dedup,
                    # LLM dedup, Agent injection) that assume a single-agent build. Those
                    # heuristics collapse legitimately-distinct same-type stages (e.g. two
                    # LLMs: intent-classify + SQL-gen, or two vector stores), which silently
                    # severs downstream edges and orphans the terminal nodes. Additive
                    # enrichment (schemas, creds, prompt {var} handles, edge types) still runs.
                    # Approved designs carry Prompt templates two ways: a PER-NODE `template`
                    # on each Prompt stub (preferred — a multi-prompt pipeline has a distinct
                    # template per Prompt node), and a single design-level `prompt_template`
                    # (legacy / single-prompt fallback). Capture both and strip the control
                    # keys so they never leak into the flow payload. Per-node templates are
                    # applied as each node's own prompt-field value after enrichment, so
                    # apply_prompt_fields materializes the right {vars} on the right node
                    # instead of stamping ONE template onto every prompt (the bug that made
                    # 3 prompts identical).
                    _prompt_template = data.pop("prompt_template", "") if isinstance(data, dict) else ""
                    data.pop("vars", None)
                    _node_templates: dict[str, str] = {}
                    for _n in (data.get("nodes") or []):
                        if isinstance(_n, dict) and _n.get("template"):
                            _node_templates[_n.get("id", "")] = _n.pop("template")
                    # Deduplicate structural singletons (non-LLM) by type name. Skip for an
                    # approved design — it is authoritative about how many of each node exist.
                    if not from_design:
                        _STRUCTURAL_SINGLETONS = {"ChatInput", "ChatOutput", "Agent"}
                        seen_singletons: set[str] = set()
                        deduped: list[dict] = []
                        removed_ids: set[str] = set()
                        for _n in data["nodes"]:
                            _nt = _n.get("data", {}).get("type") or _n.get("type", "")
                            if _nt in _STRUCTURAL_SINGLETONS and _nt in seen_singletons:
                                removed_ids.add(_n.get("id", ""))
                                console.print(f"[yellow]↳ dedup: removed extra '{_nt}' node[/yellow]")
                                continue
                            seen_singletons.add(_nt)
                            deduped.append(_n)
                        if removed_ids:
                            data["nodes"] = deduped
                    # Drop orphan edges before enrichment
                    valid_node_ids = {n.get("id", "") for n in data["nodes"]}
                    orphans_before = len(data.get("edges", []))
                    data["edges"] = [
                        e for e in data.get("edges", [])
                        if e.get("source") in valid_node_ids and e.get("target") in valid_node_ids
                    ]
                    orphans_dropped = orphans_before - len(data["edges"])
                    if orphans_dropped:
                        console.print(f"[yellow]↳ dropped {orphans_dropped} orphan edge(s) with missing node references[/yellow]")
                    data["nodes"] = mcp.enrich_nodes(data["nodes"], credential_overrides=_credential_overrides)
                    # Dynamic LLM dedup (post-enrichment): always keeps AzureOpenAIModel.
                    llm_nodes = [n for n in data["nodes"] if _node_outputs_langmodel(n)]
                    azure_llm = next((n for n in llm_nodes if n.get("data", {}).get("type") == "AzureOpenAIModel"), None)
                    # Force the configured provider for hand builds only; a design's chosen
                    # provider (and node count) is authoritative.
                    if not from_design and not azure_llm:
                        if llm_nodes:
                            remove_ids = {n.get("id", "") for n in llm_nodes}
                            data["nodes"] = [n for n in data["nodes"] if n.get("id", "") not in remove_ids]
                            data["edges"] = [
                                e for e in data.get("edges", [])
                                if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                            ]
                            for rid in remove_ids:
                                console.print(f"[yellow]↳ replaced non-Azure LLM '{rid}' with AzureOpenAIModel[/yellow]")
                        removed_pos = next(
                            (n.get("position", {"x": 250, "y": 200}) for n in llm_nodes),
                            {"x": 250, "y": 200},
                        )
                        azure_stub = [{"id": "AzureOpenAIModel-1", "type": "AzureOpenAIModel", "position": removed_pos, "data": {"type": "AzureOpenAIModel", "id": "AzureOpenAIModel-1"}}]
                        data["nodes"].extend(mcp.enrich_nodes(azure_stub, credential_overrides=_credential_overrides))
                    # Wire AzureOpenAI to ALL nodes with ModelInput fields. Skip for an
                    # approved design — the designer owns the wiring, and hardcoding the
                    # source to AzureOpenAIModel-1 cross-wires a multi-LLM pipeline.
                    if not from_design:
                        for _n in data["nodes"]:
                            if _n.get("data", {}).get("type") == "AzureOpenAIModel":
                                continue
                            _mf = _find_model_field(_n)
                            if _mf:
                                data["edges"].append({
                                    "source": "AzureOpenAIModel-1",
                                    "target": _n.get("id"),
                                    "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                                    "targetHandle": {"fieldName": _mf, "id": _n.get("id"), "inputTypes": ["LanguageModel"], "type": "model"},
                                })
                        console.print("[dim]↳ wired AzureOpenAIModel → all ModelInput fields[/dim]")
                    # Inject Agent when tool-only components exist without one (hand builds only)
                    _cf_has_agent = any(n.get("data", {}).get("type") == "Agent" for n in data["nodes"])
                    _cf_has_tool_only = any(_has_only_tool_outputs(n) for n in data["nodes"])
                    if not from_design and _cf_has_tool_only and not _cf_has_agent:
                        _agent_stub = [{"id": "Agent-1", "type": "Agent", "position": {"x": 670, "y": 540}, "data": {"type": "Agent", "id": "Agent-1"}}]
                        data["nodes"].extend(mcp.enrich_nodes(_agent_stub, credential_overrides=_credential_overrides))
                        _ci = next((n for n in data["nodes"] if n.get("data", {}).get("type") == "ChatInput"), None)
                        if _ci:
                            data["edges"].append({
                                "source": _ci.get("id"), "target": "Agent-1",
                                "sourceHandle": {"dataType": "ChatInput", "id": _ci.get("id"), "name": "message", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": "Agent-1", "inputTypes": ["Message"], "type": "str"},
                            })
                        data["edges"].append({
                            "source": "AzureOpenAIModel-1", "target": "Agent-1",
                            "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                            "targetHandle": {"fieldName": "model", "id": "Agent-1", "inputTypes": ["LanguageModel"], "type": "model"},
                        })
                        _co = next((n for n in data["nodes"] if n.get("data", {}).get("type") == "ChatOutput"), None)
                        if _co:
                            data["edges"].append({
                                "source": "Agent-1", "target": _co.get("id"),
                                "sourceHandle": {"dataType": "Agent", "id": "Agent-1", "name": "response", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": _co.get("id"), "inputTypes": ["Data", "JSON", "DataFrame", "Table", "Message"], "type": "other"},
                            })
                        console.print("[dim]↳ injected Agent + wired ChatInput→Agent→ChatOutput[/dim]")
                    # Keep ONLY one LLM for hand-built creates (the model often double-emits).
                    # NEVER for an approved design — a multi-stage pipeline legitimately reuses
                    # the same LLM type across distinct stages (intent-classify + SQL-gen); this
                    # was deleting the SQL-gen LLM and orphaning the executor + ChatOutput.
                    if not from_design and azure_llm and len(llm_nodes) > 1:
                        extra_llm_ids = {n.get("id", "") for n in llm_nodes if n is not azure_llm}
                        data["nodes"] = [n for n in data["nodes"] if n.get("id", "") not in extra_llm_ids]
                        data["edges"] = [
                            e for e in data.get("edges", [])
                            if e.get("source") not in extra_llm_ids and e.get("target") not in extra_llm_ids
                        ]
                        for eid in extra_llm_ids:
                            console.print(f"[yellow]↳ dedup: removed extra LLM node '{eid}'[/yellow]")
                    # Set each Prompt node's OWN template value (from its stub) onto its
                    # prompt-type field, so apply_prompt_fields materializes that node's vars
                    # rather than the single global template. Schema-driven: finds the prompt
                    # field by type, never a hardcoded field/component name.
                    if _node_templates:
                        for _n in data["nodes"]:
                            _t = _node_templates.get(_n.get("id", ""))
                            if not _t:
                                continue
                            _ntmpl = _n.get("data", {}).get("node", {}).get("template", {})
                            for _f in (_ntmpl.values() if isinstance(_ntmpl, dict) else []):
                                if isinstance(_f, dict) and _f.get("type") == "prompt":
                                    _f["value"] = _t
                                    break
                    # Build dynamic {var} handles on Prompt-like nodes BEFORE edge
                    # enrichment, so enrich_edges resolves the var field types and the saved
                    # node keeps the handles the inbound edges target. Per-node values set
                    # above win; the design-level template is the fallback for any prompt
                    # node without its own.
                    mcp.apply_prompt_fields(data["nodes"], _prompt_template)
                    data["edges"] = mcp.ensure_tool_edges(data["nodes"], data.get("edges", []))
                    if "edges" in data:
                        data["edges"] = mcp.enrich_edges(data["edges"], data["nodes"])
                    mcp.fix_selected_outputs(data["nodes"], data.get("edges", []))
                    return data

                def _enrich_update_merge(existing_data: dict, payload_data: dict) -> dict:
                    existing_nodes = list(existing_data.get("nodes", []) or [])
                    existing_edges = list(existing_data.get("edges", []) or [])

                    # Honor explicit removal lists in the payload. Lets LLM combine
                    # "add X / remove Y" in one update_flow call without falling into
                    # the union-only merge trap.
                    remove_ids: set[str] = set(payload_data.get("remove_node_ids") or [])
                    remove_types: set[str] = set(payload_data.get("remove_types") or [])
                    if remove_types:
                        for n in existing_nodes:
                            t = (n.get("data") or {}).get("type") or n.get("type", "")
                            if t in remove_types and n.get("id"):
                                remove_ids.add(n["id"])
                    if remove_ids:
                        existing_nodes = [n for n in existing_nodes if n.get("id") not in remove_ids]
                        existing_edges = [
                            e for e in existing_edges
                            if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                        ]
                        console.print(f"[dim]↳ removing {len(remove_ids)} node(s): {sorted(remove_ids)}[/dim]")
                        # Strip control fields so they don't leak into PATCH body
                        payload_data = {k: v for k, v in payload_data.items() if k not in ("remove_node_ids", "remove_types")}

                    existing_node_ids = {n.get("id", "") for n in existing_nodes if n.get("id")}
                    existing_edge_ids = {e.get("id", "") for e in existing_edges if e.get("id")}

                    def _edge_fingerprint(e: dict) -> tuple:
                        """Structural identity: same source node + target node + target field = same edge.
                        Used to dedup edges the LLM resends without IDs (ID-based dedup misses these)."""
                        th = e.get("targetHandle") or {}
                        if isinstance(th, str):
                            try:
                                th = json.loads(th.replace('œ', '"'))
                            except Exception:
                                th = {}
                        return (e.get("source", ""), e.get("target", ""), th.get("fieldName", ""))

                    existing_edge_fingerprints = {_edge_fingerprint(e) for e in existing_edges}

                    # Type → existing-node-id map. LLMs routinely invent fresh IDs
                    # ("ChatInput-1") for components that already exist in the flow under
                    # UUIDs. Without remapping we'd duplicate every structural node.
                    existing_by_type: dict[str, str] = {}
                    for _n in existing_nodes:
                        _t = _n.get("data", {}).get("type") or _n.get("type", "")
                        if _t and _t not in existing_by_type:
                            existing_by_type[_t] = _n.get("id", "")

                    id_map: dict[str, str] = {}
                    for _n in payload_data.get("nodes", []):
                        _nid = _n.get("id", "")
                        if not _nid or _nid in existing_node_ids:
                            continue
                        _t = _n.get("data", {}).get("type") or _n.get("type", "")
                        if _t in existing_by_type:
                            id_map[_nid] = existing_by_type[_t]

                    if id_map:
                        for _src, _dst in id_map.items():
                            console.print(f"[dim]↳ remap payload id '{_src}' → existing '{_dst}' (same type)[/dim]")

                    def _remap_id(node_id: str) -> str:
                        return id_map.get(node_id, node_id)

                    def _remap_edge(e: dict) -> dict:
                        e2 = dict(e)
                        if e2.get("source") in id_map:
                            e2["source"] = id_map[e2["source"]]
                        if e2.get("target") in id_map:
                            e2["target"] = id_map[e2["target"]]
                        for hk in ("sourceHandle", "targetHandle"):
                            h = e2.get(hk)
                            if isinstance(h, dict) and h.get("id") in id_map:
                                h = dict(h)
                                h["id"] = id_map[h["id"]]
                                e2[hk] = h
                        return e2

                    addition_nodes = [
                        n for n in payload_data.get("nodes", [])
                        if n.get("id")
                        and n.get("id") not in existing_node_ids
                        and n.get("id") not in id_map
                    ]
                    addition_edges = [
                        _remap_edge(e) for e in payload_data.get("edges", [])
                        if ((not e.get("id")) or e.get("id") not in existing_edge_ids)
                        and _edge_fingerprint(_remap_edge(e)) not in existing_edge_fingerprints
                    ]
                    console.print(f"[dim]↳ merging {len(addition_nodes)} new node(s), {len(addition_edges)} new edge(s) into flow[/dim]")

                    existing_llm = mcp.find_llm_node(existing_nodes)
                    existing_agent = mcp.find_agent_node(existing_nodes)

                    mcp.offset_new_positions(existing_nodes, addition_nodes)
                    enriched_additions = (
                        mcp.enrich_nodes(addition_nodes, credential_overrides=_credential_overrides)
                        if addition_nodes else []
                    )

                    if existing_llm is None:
                        added_llm = next((n for n in enriched_additions if _node_outputs_langmodel(n)), None)
                        if added_llm and added_llm.get("data", {}).get("type") == "AzureOpenAIModel":
                            llm_id = added_llm.get("id")
                        elif added_llm:
                            # Non-Azure LLM in additions → replace with AzureOpenAIModel stub
                            remove_id = added_llm.get("id", "")
                            pos = added_llm.get("position", {"x": 250, "y": 200})
                            enriched_additions = [n for n in enriched_additions if n.get("id", "") != remove_id]
                            addition_edges = [
                                e for e in addition_edges
                                if e.get("source") != remove_id and e.get("target") != remove_id
                            ]
                            console.print(f"[yellow]↳ replaced non-Azure LLM '{remove_id}' with AzureOpenAIModel[/yellow]")
                            stub_id = f"AzureOpenAIModel-{int(time.time() * 1000) % 1000000}"
                            stub = [{"id": stub_id, "type": "AzureOpenAIModel", "position": pos, "data": {"type": "AzureOpenAIModel", "id": stub_id}}]
                            enriched_additions.extend(mcp.enrich_nodes(stub, credential_overrides=_credential_overrides))
                            llm_id = stub_id
                        else:
                            needs_model = any(_find_model_field(n) for n in enriched_additions)
                            if needs_model:
                                stub_id = f"AzureOpenAIModel-{int(time.time() * 1000) % 1000000}"
                                stub = [{"id": stub_id, "type": "AzureOpenAIModel", "position": {"x": 250, "y": 200}, "data": {"type": "AzureOpenAIModel", "id": stub_id}}]
                                enriched_additions.extend(mcp.enrich_nodes(stub, credential_overrides=_credential_overrides))
                                llm_id = stub_id
                                console.print(f"[dim]↳ injected AzureOpenAIModel '{stub_id}' for new ModelInput fields[/dim]")
                            else:
                                llm_id = None
                    else:
                        llm_id = existing_llm.get("id")
                        # Drop duplicate LLMs from additions (existing wins)
                        dup_llm_ids = {n.get("id", "") for n in enriched_additions if _node_outputs_langmodel(n)}
                        if dup_llm_ids:
                            enriched_additions = [n for n in enriched_additions if n.get("id", "") not in dup_llm_ids]
                            addition_edges = [
                                e for e in addition_edges
                                if e.get("source") not in dup_llm_ids and e.get("target") not in dup_llm_ids
                            ]
                            for rid in dup_llm_ids:
                                console.print(f"[yellow]↳ dropped duplicate LLM '{rid}' from additions (existing LLM kept)[/yellow]")

                    # Wire any new ModelInput fields → discovered llm_id (not hardcoded)
                    if llm_id:
                        llm_node_lookup = mcp.find_node_by_type(existing_nodes + enriched_additions, "AzureOpenAIModel")
                        llm_type = (llm_node_lookup or existing_llm or {}).get("data", {}).get("type", "AzureOpenAIModel")
                        for n in enriched_additions:
                            if n.get("id") == llm_id:
                                continue
                            mf = _find_model_field(n)
                            if mf:
                                addition_edges.append({
                                    "source": llm_id, "target": n.get("id"),
                                    "sourceHandle": {"dataType": llm_type, "id": llm_id, "name": "model_output", "output_types": ["LanguageModel"]},
                                    "targetHandle": {"fieldName": mf, "id": n.get("id"), "inputTypes": ["LanguageModel"], "type": "model"},
                                })

                    # Inject Agent only when additions need one AND none exists anywhere
                    has_tool_only = any(_has_only_tool_outputs(n) for n in enriched_additions)
                    has_agent_anywhere = existing_agent is not None or any(
                        n.get("data", {}).get("type") == "Agent" for n in enriched_additions
                    )
                    if has_tool_only and not has_agent_anywhere:
                        stub_id = f"Agent-{int(time.time() * 1000) % 1000000}"
                        stub = [{"id": stub_id, "type": "Agent", "position": {"x": 670, "y": 540}, "data": {"type": "Agent", "id": stub_id}}]
                        enriched_additions.extend(mcp.enrich_nodes(stub, credential_overrides=_credential_overrides))
                        ci = mcp.find_node_by_type(existing_nodes + enriched_additions, "ChatInput")
                        if ci:
                            addition_edges.append({
                                "source": ci.get("id"), "target": stub_id,
                                "sourceHandle": {"dataType": "ChatInput", "id": ci.get("id"), "name": "message", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": stub_id, "inputTypes": ["Message"], "type": "str"},
                            })
                        if llm_id:
                            addition_edges.append({
                                "source": llm_id, "target": stub_id,
                                "sourceHandle": {"dataType": "AzureOpenAIModel", "id": llm_id, "name": "model_output", "output_types": ["LanguageModel"]},
                                "targetHandle": {"fieldName": "model", "id": stub_id, "inputTypes": ["LanguageModel"], "type": "model"},
                            })
                        co = mcp.find_node_by_type(existing_nodes + enriched_additions, "ChatOutput")
                        if co:
                            addition_edges.append({
                                "source": stub_id, "target": co.get("id"),
                                "sourceHandle": {"dataType": "Agent", "id": stub_id, "name": "response", "output_types": ["Message"]},
                                "targetHandle": {"fieldName": "input_value", "id": co.get("id"), "inputTypes": ["Data", "JSON", "DataFrame", "Table", "Message"], "type": "other"},
                            })
                        console.print(f"[dim]↳ injected Agent '{stub_id}' for tool-only additions[/dim]")

                    merged_nodes = existing_nodes + enriched_additions
                    # Materialize {var} handles for any newly added Prompt-like nodes (uses
                    # the node's own template value — update payloads carry no design ref).
                    mcp.apply_prompt_fields(enriched_additions)
                    merged_edges = existing_edges + addition_edges
                    merged_edges = mcp.ensure_tool_edges(merged_nodes, merged_edges)
                    merged_edges = mcp.enrich_edges(merged_edges, merged_nodes)
                    mcp.fix_selected_outputs(merged_nodes, merged_edges)

                    # Regression guard: every existing id must survive merge
                    final_ids = {n.get("id", "") for n in merged_nodes}
                    dropped = existing_node_ids - final_ids
                    if dropped:
                        raise RuntimeError(f"merge would drop existing nodes: {sorted(dropped)}")

                    return {**payload_data, "nodes": merged_nodes, "edges": merged_edges}

                if tc["name"] == "create_flow":
                    data = args.get("data", {})
                    # Resolve an approved design reference → its stored nodes/edges, so the
                    # agent builds the confirmed design without re-emitting the payload.
                    _from_design = isinstance(data, dict) and bool(data.get("_design_ref"))
                    if _from_design:
                        ref = data["_design_ref"]
                        approved = _approved_designs.get(ref)
                        if approved:
                            data = dict(approved)
                            args = {**args, "data": data}
                            console.print(f"[dim]↳ building approved design {ref}[/dim]")
                        else:
                            build_block = json.dumps({
                                "error": "unknown_design_ref",
                                "note": f"No approved design {ref!r}. Call design_flow first.",
                            })
                    # Complex builds must go through design_flow (graph review) — enforce
                    # the design+confirm path. Small builds + template clones stay direct.
                    if (not build_block and not _from_design and isinstance(data, dict)
                            and len(data.get("nodes") or []) >= 5):
                        build_block = json.dumps({
                            "error": "design_required",
                            "note": "Complex builds (>=5 nodes) must go through design_flow "
                                    "first so the user can review the graph. Call "
                                    "design_flow(request=...), then create_flow with "
                                    "{\"data\":{\"_design_ref\":<ref>}}.",
                        })
                    if not build_block and isinstance(data, dict) and "nodes" in data:
                        try:
                            data = _enrich_create_data(data, from_design=_from_design)
                            args = {**args, "data": data}
                            _intended_edges = list(data.get("edges", []))  # for post-build strip diff
                            console.print("[dim]↳ enriched nodes with component schemas + credentials[/dim]")
                        except Exception as enrich_err:
                            enrich_error = str(enrich_err)
                            console.print(f"[red]✗ schema enrichment failed: {enrich_err}[/red]")
                            console.print("[dim]↳ skipping call to prevent broken flow[/dim]")

                elif tc["name"] == "update_flow":
                    payload_data = args.get("data")
                    flow_id = args.get("flow_id")
                    if isinstance(payload_data, dict) and "nodes" in payload_data and flow_id:
                        try:
                            existing_raw = await mcp.call_tool("get_flow", {"flow_id": flow_id})
                            existing = json.loads(existing_raw) if isinstance(existing_raw, str) else existing_raw
                            existing_data = existing.get("data", {}) if isinstance(existing, dict) else {}
                            existing_node_ids = {n.get("id", "") for n in existing_data.get("nodes", []) if n.get("id")}

                            mode = mcp.classify_update_payload(payload_data, existing_node_ids)
                            console.print(f"[dim]↳ update_flow mode: {mode}[/dim]")

                            if mode == "full_replace":
                                payload_data = _enrich_create_data(payload_data)
                                _intended_edges = list(payload_data.get("edges", []))
                            else:
                                payload_data = _enrich_update_merge(existing_data, payload_data)
                                _intended_edges = list(payload_data.get("edges", []))
                            args = {**args, "data": payload_data}
                        except Exception as enrich_err:
                            enrich_error = str(enrich_err)
                            console.print(f"[red]✗ update_flow enrichment failed: {enrich_err}[/red]")
                            console.print("[dim]↳ skipping call to prevent broken flow[/dim]")

                # Legacy hard-block (schema-driven): refuse to ship agent-authored legacy
                # components. Scoped to create_flow/update_flow — clone_starter_template is
                # Langflow-authored and exempt. Forces decomposition into modern primitives.
                if tc["name"] in ("create_flow", "update_flow") and not enrich_error and not build_block:
                    _data = args.get("data") if isinstance(args.get("data"), dict) else {}
                    _legacy = sorted({
                        t for n in (_data.get("nodes") or [])
                        if (t := ((n.get("data") or {}).get("type") or n.get("type")))
                        and mcp.is_legacy(t)
                    })
                    if _legacy:
                        console.print(f"[red]✗ legacy components blocked: {_legacy}[/red]")
                        build_block = json.dumps({
                            "error": "legacy_components",
                            "offending": _legacy,
                            "note": "These are legacy and hard-blocked. Rebuild from non-legacy "
                                    "primitives — decompose any legacy mega-component into "
                                    "explicit stages (e.g. one Prompt with {vars} → LLM → "
                                    "executor → ChatOutput). Verify a replacement is "
                                    "legacy:false via get_component_schema, then retry.",
                        })

                if enrich_error:
                    # Refuse to send broken nodes to Langflow. Return error as tool result so LLM retries.
                    result = f"ERROR: {enrich_error} Do NOT retry blindly — call list_components first and find the exact 'type' string."
                elif build_block:
                    result = build_block
                elif tc["name"] == "get_component_schema":
                    result = json.dumps(mcp.get_component_schema(args.get("type_name", "")))
                    _schema_msg_ids.add(tc["id"])
                elif tc["name"] == "clone_starter_template":
                    # Server-side clone: fetch template → enrich → POST directly, zero LLM token cost
                    name_or_id = (args.get("name_or_id") or "").strip()
                    flow_name = (args.get("name") or name_or_id).strip()
                    flow_desc = args.get("description", "")
                    key = name_or_id.lower()
                    # Lookup chain: in-memory cache → Redis/HTTP via client
                    template = _starter_cache.get(key)
                    if not template:
                        for k, v in _starter_cache.items():
                            if key in k or k in key:
                                template = v
                                break
                    if not template:
                        template = await mcp.fetch_starter(name_or_id)
                    if not template:
                        result = json.dumps({"error": f"Template '{name_or_id}' not found. Call get_basic_examples to populate cache first."})
                    else:
                        try:
                            tdata = json.loads(json.dumps(template.get("data", {})))  # deep copy
                            if not tdata.get("nodes"):
                                result = json.dumps({"error": "Template has no nodes."})
                            else:
                                tdata["nodes"] = mcp.enrich_nodes(tdata["nodes"], credential_overrides=_credential_overrides)
                                # Always ensure AzureOpenAIModel is the LLM.
                                # Simple Agent uses built-in ModelInput (no separate LLM node) — must inject.
                                # Other templates may have a non-Azure LLM node — must replace.
                                llm_nodes = [n for n in tdata["nodes"] if _is_langmodel_node(n)]
                                azure_llm = next((n for n in llm_nodes if n.get("data", {}).get("type") == "AzureOpenAIModel"), None)
                                if not azure_llm:
                                    # Remove any non-Azure LLM nodes that exist
                                    if llm_nodes:
                                        remove_ids = {n.get("id", "") for n in llm_nodes}
                                        tdata["nodes"] = [n for n in tdata["nodes"] if n.get("id", "") not in remove_ids]
                                        tdata["edges"] = [
                                            e for e in tdata.get("edges", [])
                                            if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                                        ]
                                    # Position: near removed LLM or sensible default
                                    removed_pos = next(
                                        (n.get("position", {"x": 250, "y": 200}) for n in llm_nodes),
                                        {"x": 250, "y": 200},
                                    )
                                    azure_stub = [{"id": "AzureOpenAIModel-1", "type": "AzureOpenAIModel", "position": removed_pos, "data": {"type": "AzureOpenAIModel", "id": "AzureOpenAIModel-1"}}]
                                    tdata["nodes"].extend(mcp.enrich_nodes(azure_stub, credential_overrides=_credential_overrides))
                                # Wire AzureOpenAI to ALL nodes with ModelInput fields (Agent, StructuredOutput, etc.)
                                for _n in tdata["nodes"]:
                                    if _n.get("data", {}).get("type") == "AzureOpenAIModel":
                                        continue
                                    _mf = _find_model_field(_n)
                                    if _mf:
                                        tdata["edges"].append({
                                            "source": "AzureOpenAIModel-1",
                                            "target": _n.get("id"),
                                            "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                                            "targetHandle": {"fieldName": _mf, "id": _n.get("id"), "inputTypes": ["LanguageModel"], "type": "model"},
                                        })
                                console.print("[dim]↳ wired AzureOpenAIModel → all ModelInput fields[/dim]")
                                # Inject Agent when tool-only components exist without one.
                                # Nodes whose only outputs are component_as_tool/api_build_tool cannot
                                # connect via data edges — an Agent is mandatory to use them.
                                _has_agent = any(n.get("data", {}).get("type") == "Agent" for n in tdata["nodes"])
                                _has_tool_only = any(_has_only_tool_outputs(n) for n in tdata["nodes"])
                                if _has_tool_only and not _has_agent:
                                    _agent_stub = [{"id": "Agent-1", "type": "Agent", "position": {"x": 670, "y": 540}, "data": {"type": "Agent", "id": "Agent-1"}}]
                                    tdata["nodes"].extend(mcp.enrich_nodes(_agent_stub, credential_overrides=_credential_overrides))
                                    _ci = next((n for n in tdata["nodes"] if n.get("data", {}).get("type") == "ChatInput"), None)
                                    if _ci:
                                        tdata["edges"].append({
                                            "source": _ci.get("id"), "target": "Agent-1",
                                            "sourceHandle": {"dataType": "ChatInput", "id": _ci.get("id"), "name": "message", "output_types": ["Message"]},
                                            "targetHandle": {"fieldName": "input_value", "id": "Agent-1", "inputTypes": ["Message"], "type": "str"},
                                        })
                                    tdata["edges"].append({
                                        "source": "AzureOpenAIModel-1", "target": "Agent-1",
                                        "sourceHandle": {"dataType": "AzureOpenAIModel", "id": "AzureOpenAIModel-1", "name": "model_output", "output_types": ["LanguageModel"]},
                                        "targetHandle": {"fieldName": "model", "id": "Agent-1", "inputTypes": ["LanguageModel"], "type": "model"},
                                    })
                                    _co = next((n for n in tdata["nodes"] if n.get("data", {}).get("type") == "ChatOutput"), None)
                                    if _co:
                                        tdata["edges"].append({
                                            "source": "Agent-1", "target": _co.get("id"),
                                            "sourceHandle": {"dataType": "Agent", "id": "Agent-1", "name": "response", "output_types": ["Message"]},
                                            "targetHandle": {"fieldName": "input_value", "id": _co.get("id"), "inputTypes": ["Data", "JSON", "DataFrame", "Table", "Message"], "type": "other"},
                                        })
                                    console.print("[dim]↳ injected Agent + wired ChatInput→Agent→ChatOutput[/dim]")
                                tdata["edges"] = mcp.ensure_tool_edges(tdata["nodes"], tdata.get("edges", []))
                                tdata["edges"] = mcp.enrich_edges(tdata["edges"], tdata["nodes"])
                                mcp.fix_selected_outputs(tdata["nodes"], tdata["edges"])
                                created = mcp._create_flow_direct(
                                    name=flow_name or template.get("name", "Cloned Flow"),
                                    description=flow_desc or template.get("description", ""),
                                    data=tdata,
                                )
                                _last_build_flow_id = None  # reset; build_flow will set it
                                result = json.dumps({
                                    "flow_id": created.get("id"),
                                    "name": created.get("name"),
                                    "node_count": len(tdata["nodes"]),
                                    "edge_count": len(tdata["edges"]),
                                })
                                console.print(f"[dim]↳ cloned '{template.get('name')}' → {created.get('id')} ({len(tdata['nodes'])} nodes, {len(tdata['edges'])} edges)[/dim]")
                                sink.flow_built(created.get('id'), graph=slim_graph({"data": tdata}))
                                if mcp._redis_cache and created.get("id") and created.get("name"):
                                    try:
                                        await mcp._redis_cache.upsert_flow(
                                            flow_id=created["id"],
                                            name=created["name"],
                                            description=created.get("description") or "",
                                        )
                                    except Exception:
                                        pass
                        except Exception as clone_err:
                            result = f"ERROR cloning template: {clone_err}"
                elif tc["name"] == "get_starter_template":
                    # Virtual tool — look up cached starter data, return full template for ONE winner
                    key = (args.get("name_or_id") or "").strip().lower()
                    match = _starter_cache.get(key)
                    if not match:
                        # Try partial name match
                        for k, v in _starter_cache.items():
                            if key in k or k in key:
                                match = v
                                break
                    if match:
                        result = json.dumps({"id": match.get("id"), "name": match.get("name"), "data": match.get("data", {})})
                        _starter_template_msg_ids.add(tc["id"])
                        console.print(f"[dim]↳ starter template '{match.get('name')}' served from cache[/dim]")
                    else:
                        result = json.dumps({"error": f"Template '{args.get('name_or_id')}' not found in cache. Call get_basic_examples first."})
                elif tc["name"] == "propose_pipeline":
                    # Real-life multi-stage build: render the stage→component map, capture it
                    # to scratchpad, and tell the model to ask the user about ambiguous stages
                    # (or proceed to design_flow once none remain). This is the ask→loop step.
                    stages = pipeline.normalize_stages(args.get("stages"))
                    console.print(Panel(
                        Markdown(pipeline.render_pipeline(stages)),
                        title="[bold]Proposed Pipeline[/bold]", border_style="cyan",
                    ))
                    planning.remember(scratchpad, "pipeline", pipeline.render_pipeline(stages))
                    resolved, open_questions = pipeline.split(stages)
                    if open_questions:
                        _pipeline_asked = True
                    elif _pipeline_asked:
                        # The user has now answered the earlier questions → interpretation is
                        # aligned. Auto-confirm the downstream design graph (one human gate).
                        _plan_confirmed = True
                    result = json.dumps(pipeline.build_result(resolved, open_questions))
                elif tc["name"] == "design_flow":
                    # Delegate graph design to the in-process sub-agent (isolated context,
                    # reuses the cached provider). Then render the graph + confirm gate.
                    try:
                        with console.status("[dim]designing flow…[/dim]", spinner="dots"):
                            spec = await designer.design_flow(
                                args.get("request", ""), mcp, llm, feedback=args.get("feedback"),
                                resolved_stages=args.get("resolved_stages"),
                            )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        raise
                    except Exception as e:
                        # The designer makes its own LLM calls; a transient timeout/error must
                        # not crash the session. Return it as a tool result so the model can
                        # retry design_flow, and tool_use/tool_result pairing stays intact.
                        console.print(f"[red]✗ design_flow failed:[/red] {type(e).__name__}: {e}")
                        spec = {"error": f"designer failed: {type(e).__name__}: {e}. Retry design_flow."}
                    if spec.get("error"):
                        result = json.dumps(spec)
                    else:
                        console.print(Panel(
                            Markdown(designer.render_design(spec)),
                            title="[bold]Proposed Design[/bold]", border_style="cyan",
                        ))
                        approve = _plan_confirmed
                        if not approve:
                            try:
                                answer = console.input(
                                    "[bold cyan]Proceed with this design? [y/n][/bold cyan] "
                                ).strip().lower()
                            except (KeyboardInterrupt, EOFError):
                                answer = "n"
                            if answer in ("y", "yes", ""):
                                approve = True
                                _plan_confirmed = True
                        if approve:
                            ref = f"d{len(_approved_designs) + 1}"
                            _approved_designs[ref] = {
                                "nodes": spec.get("nodes", []),
                                "edges": spec.get("edges", []),
                                "prompt_template": spec.get("prompt_template", ""),
                                "vars": spec.get("vars", []),
                            }
                            result = json.dumps({
                                "approved": True,
                                "design_ref": ref,
                                "summary": spec.get("summary", ""),
                                "next": f'Build it now: call create_flow with {{"data":{{"_design_ref":"{ref}"}}}}.',
                            })
                        else:
                            feedback = console.input(
                                "[dim]What should change? (enter to skip)[/dim] "
                            ).strip()
                            _plan_rejected = True
                            result = json.dumps({
                                "approved": False,
                                "feedback": feedback or "User rejected the design. Revise it.",
                                "note": "Call design_flow again with this feedback.",
                            })
                elif tc["name"] in planning.PLANNING_TOOL_NAMES:
                    _todos_before = [dict(t) for t in todos]
                    result = planning.dispatch(tc["name"], args, todos, scratchpad)
                    _todos_changed = tc["name"] == "write_todos" and todos != _todos_before
                    if _todos_changed and todos:
                        # Show the plan to the user only when it actually changes (a
                        # no-op re-call renders nothing — avoids panel spam / loops).
                        console.print(Panel(
                            Markdown(planning.render_todos(todos)),
                            title="[bold]Plan[/bold]", border_style="cyan",
                        ))
                        # Confirm-gate ONCE per request, only for a real multi-step
                        # plan. After approval the agent auto-locks and won't re-prompt.
                        needs_gate = (
                            not _plan_confirmed
                            and len(todos) >= 2
                            and any(t["status"] != "completed" for t in todos)
                        )
                        if needs_gate:
                            try:
                                answer = console.input(
                                    "[bold cyan]Proceed with this plan? [y/n][/bold cyan] "
                                ).strip().lower()
                            except (KeyboardInterrupt, EOFError):
                                answer = "n"
                            if answer in ("y", "yes", ""):
                                _plan_confirmed = True
                            else:
                                feedback = console.input(
                                    "[dim]What should change? (enter to skip)[/dim] "
                                ).strip()
                                _plan_rejected = True
                                result = json.dumps({
                                    "ok": False,
                                    "rejected_by_user": True,
                                    "feedback": feedback or "User rejected the plan. Revise it.",
                                })
                else:
                    try:
                        result = await mcp.call_tool(tc["name"], args)
                    except Exception as e:
                        result = f"ERROR: {e}"

                # Immediately strip template list results — cache full data, send index only to LLM
                if tc["name"] in ("list_starter_projects", "get_basic_examples"):
                    try:
                        parsed = json.loads(result) if isinstance(result, str) else result
                        if isinstance(parsed, list):
                            for t in parsed:
                                tid = str(t.get("id", "")).lower()
                                tname = str(t.get("name", "")).lower()
                                if tid:
                                    _starter_cache[tid] = t
                                if tname:
                                    _starter_cache[tname] = t
                            result = json.dumps([
                                {"id": t.get("id"), "name": t.get("name"), "description": t.get("description", "")}
                                for t in parsed
                            ])
                            console.print(f"[dim]↳ template index: {len(parsed)} templates cached, full data stripped from context[/dim]")
                    except Exception:
                        pass

                # download_* tools (flows/project/folder) return the full export JSON.
                # Write it to disk and hand the LLM only a path receipt — otherwise the
                # entire flow JSON floods context (defeats the dynamic-tools bloat goal).
                if tc["name"].startswith("download_") and not str(result).startswith("ERROR"):
                    try:
                        parsed = json.loads(result) if isinstance(result, str) else result
                        # MCP returns a JSON error envelope ({"error": true, "message": ...})
                        # on failure — surface it to the LLM, never persist it as an export.
                        if isinstance(parsed, dict) and parsed.get("error"):
                            raise RuntimeError(parsed.get("message") or "download failed")
                        export_dir = Path.cwd() / "exports"
                        export_dir.mkdir(exist_ok=True)
                        items = parsed if isinstance(parsed, list) else [parsed]
                        written = []
                        for item in items:
                            if not isinstance(item, dict) or item.get("error"):
                                continue  # skip non-flow payloads / per-item error envelopes
                            raw_name = str(item.get("name") or item.get("id") or tc["name"])
                            fid = str(item.get("id") or "")[:8]
                            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-") or "export"
                            fname = f"{slug}-{fid}.json" if fid else f"{slug}.json"
                            path = export_dir / fname
                            path.write_text(json.dumps(item, indent=2))
                            written.append({"name": raw_name, "path": str(path), "bytes": path.stat().st_size})
                        if not written:
                            raise RuntimeError("download returned no flow data")
                        console.print(f"[dim]↳ exported {len(written)} file(s) → {export_dir}[/dim]")
                        result = json.dumps({
                            "exported": written,
                            "_note": "Export JSON written to disk; full content not included here.",
                        })
                    except Exception as e:
                        result = f"ERROR writing export: {e}"

                if tc["name"] == "build_flow":
                    _last_build_flow_id = tc["arguments"].get("flow_id")
                    sink.flow_built(_last_build_flow_id)
                    # build_flow result is large job metadata — LLM only needs the job_id
                    try:
                        bdata = json.loads(result) if isinstance(result, str) else result
                        if isinstance(bdata, dict) and "id" in bdata:
                            result = json.dumps({"id": bdata["id"], "status": bdata.get("status", "")})
                    except Exception:
                        pass

                # Strip list_components to type+display_name only — full schema is ~1M tokens
                if tc["name"] == "list_components":
                    try:
                        components = json.loads(result) if isinstance(result, str) else result
                        if isinstance(components, list):
                            def _entry(c):
                                e = {"type": c.get("type"), "display_name": c.get("display_name", c.get("type"))}
                                # Tag legacy (schema-driven) so component selection is
                                # quality-aware from the first look. Only when true — keeps
                                # the stripped list compact (legacy is hard-blocked at build).
                                if mcp.is_legacy(c.get("type")):
                                    e["legacy"] = True
                                return e
                            result = json.dumps([_entry(c) for c in components])
                    except Exception:
                        pass

                # After create/update: LLM has consumed template data — strip get_starter_template
                # results from message history to reclaim the ~63KB they occupy in every subsequent call
                if tc["name"] in ("create_flow", "update_flow") and _starter_template_msg_ids:
                    for msg in messages:
                        if msg.get("role") == "tool" and msg.get("tool_call_id") in _starter_template_msg_ids:
                            try:
                                parsed = json.loads(msg["content"])
                                if isinstance(parsed, dict):
                                    msg["content"] = json.dumps({
                                        "id": parsed.get("id"),
                                        "name": parsed.get("name"),
                                        "_note": "full data stripped after use",
                                    })
                            except Exception:
                                pass
                    _starter_template_msg_ids.clear()

                # After create/update: component schemas have been consumed to build the
                # edges, so the ~5K-each schema results are dead weight — stub them.
                if tc["name"] in ("create_flow", "update_flow") and _schema_msg_ids:
                    for msg in messages:
                        if msg.get("role") == "tool" and msg.get("tool_call_id") in _schema_msg_ids:
                            msg["content"] = '{"_note":"component schema consumed during build — stripped"}'
                    _schema_msg_ids.clear()

                # Remember intended edges keyed by the created/updated flow id, so the
                # post-build verifier (which runs in a later turn) can diff intended vs
                # surviving edges and report the ones Langflow rejected as invalid.
                if tc["name"] in ("create_flow", "update_flow") and _intended_edges:
                    try:
                        _res = json.loads(result) if isinstance(result, str) else result
                        _fid = _res.get("id") if isinstance(_res, dict) else None
                        if _fid:
                            _intended_edges_by_flow[_fid] = _intended_edges
                    except (json.JSONDecodeError, AttributeError):
                        pass

                # Auto-capture flow determinants into the scratchpad so the flow_id
                # survives history summarization — the user can reference "that flow"
                # many turns later without the LLM having to remember it.
                if tc["name"] in ("create_flow", "update_flow", "clone_starter_template"):
                    try:
                        _res = json.loads(result) if isinstance(result, str) else result
                        if isinstance(_res, dict):
                            _fid = _res.get("id") or _res.get("flow_id")
                            _fname = _res.get("name")
                            if _fid:
                                label = re.sub(r"[^A-Za-z0-9]+", "_", str(_fname or "flow")).strip("_").lower()
                                planning.remember(scratchpad, f"flow:{label}", str(_fid))
                    except (json.JSONDecodeError, AttributeError):
                        pass

                # Strip data.node schemas from any tool that returns flow JSON.
                # create_flow/update_flow return the full updated flow — same schema bloat as get_flow.
                # Must run before _inject_node_check (which appends text and breaks JSON parsing).
                if tc["name"] in ("get_flow", "create_flow", "update_flow"):
                    try:
                        flow = json.loads(result) if isinstance(result, str) else result
                        if isinstance(flow, dict) and "data" in flow:
                            # Capture the canvas graph BEFORE stripping data.node (labels live there).
                            _canvas_graph = slim_graph(flow)
                            for node in flow.get("data", {}).get("nodes", []):
                                # Preserve data.node for noteNodes: their visible text is
                                # stored in data.node.description, not a component schema.
                                # Stripping it blanks the note permanently on full_replace.
                                if node.get("type") == "noteNode" or node.get("data", {}).get("type") in ("note", "noteNode"):
                                    continue
                                node.get("data", {}).pop("node", None)
                            result = json.dumps({
                                "id": flow.get("id"),
                                "name": flow.get("name"),
                                "data": flow.get("data"),
                            })
                        # Keep Redis cache in sync so search_flows/list_flows reflect
                        # renames and new flows without waiting for background sync.
                        if isinstance(flow, dict) and flow.get("id") and flow.get("name") and mcp._redis_cache:
                            try:
                                await mcp._redis_cache.upsert_flow(
                                    flow_id=flow["id"],
                                    name=flow["name"],
                                    description=flow.get("description") or "",
                                    folder_id=str(flow.get("folder_id") or ""),
                                    updated_at=str(flow.get("updated_at") or ""),
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass

                # After get_flow following a build: check nodes + test-run (runs after truncation)
                if tc["name"] == "get_flow" and _last_build_flow_id:
                    console.print("[dim]↳ verifying flow execution…[/dim]")
                    result = _inject_node_check(
                        result, mcp, _last_build_flow_id,
                        intended_edges=_intended_edges_by_flow.get(_last_build_flow_id, []),
                    )
                    _last_build_flow_id = None

                # Supersede older flow snapshots: only the newest get_flow/update state
                # is useful, so stub the prior ones in history (each ~5K of node JSON).
                if tc["name"] in ("get_flow", "create_flow", "update_flow"):
                    for _mid in _flow_snapshot_ids:
                        for _m in messages:
                            if _m.get("role") == "tool" and _m.get("tool_call_id") == _mid:
                                _m["content"] = '{"_note":"flow snapshot superseded by a newer one — omitted to save context"}'
                    _flow_snapshot_ids.append(tc["id"])

                # Pure-Python per-tool execution time, shown on the same dim metrics line
                # style as the LLM ⏱/token/cache line. (Tools with a confirm gate include the
                # human y/n wait — that's real wall-clock for that step.)
                _tool_dt = time.perf_counter() - _tool_t0
                tool_time += _tool_dt
                console.print(f"[dim]  ⏱ exec {tc['name']} {_tool_dt:.2f}s[/dim]")

                messages.append(_tool_result_message(tc["id"], str(result)))

                # Canvas sync: after create_flow (else path) capture new flow id;
                # after any write op on the current flow, bust the iframe cache.
                _tc_name = tc["name"]
                _result_str = str(result)
                if _tc_name == "create_flow":
                    try:
                        _r = json.loads(_result_str) if isinstance(_result_str, str) else _result_str
                        if isinstance(_r, dict) and _r.get("id"):
                            sink.flow_built(_r["id"], graph=_canvas_graph)
                    except Exception:
                        pass
                elif sink.flow_id and not _result_str.startswith("ERROR"):
                    _is_write = any(_tc_name.startswith(p) for p in ("delete_", "add_", "update_", "remove_", "patch_"))
                    # Only push to the canvas when the op targets the flow on the canvas —
                    # a get_flow/edit on some other flow must not overwrite what's shown.
                    _target = args.get("flow_id") if isinstance(args, dict) else None
                    _on_canvas = (not _target) or (_target == sink.flow_id)
                    if _on_canvas and (_is_write or _tc_name == "get_flow"):
                        graph = _canvas_graph
                        # delete_node etc. return only a status — refetch the graph so the
                        # canvas reflects the change.
                        if graph is None and _is_write:
                            try:
                                _raw = await mcp.call_tool("get_flow", {"flow_id": sink.flow_id})
                                _f = json.loads(_raw) if isinstance(_raw, str) else _raw
                                graph = slim_graph(_f) if isinstance(_f, dict) else None
                            except Exception:
                                graph = None
                        sink.flow_modified(graph=graph)

            # Stall guard: if an iteration only called planning tools (no real work),
            # the model is spinning on the plan instead of executing. Nudge, then abort.
            if all(tc["name"] in planning.PLANNING_TOOL_NAMES for tc in response["tool_calls"]):
                _stall += 1
            else:
                _stall = 0
            if _stall == 3:
                messages.append({
                    "role": "user",
                    "content": "STOP re-calling planning tools. The plan is recorded and "
                               "shown to you each step. Execute the next pending action NOW "
                               "with a real tool (clone_starter_template / create_flow / "
                               "update_flow / get_flow). Only call write_todos again when an "
                               "item actually completes.",
                })
            elif _stall >= 6:
                console.print("[yellow]⚠ Planning loop detected — pausing. Tell me how to proceed.[/yellow]")
                break

            iterations += 1
        else:
            if response["content"]:
                console.print(Markdown(response["content"]))
            messages.append({"role": "assistant", "content": response["content"]})
            sink.final(response["content"])
            if prompt_tokens or completion_tokens:
                total_elapsed = time.perf_counter() - turn_start
                console.print(
                    f"[dim]total: {total_elapsed:.1f}s (tools {tool_time:.1f}s · llm "
                    f"{max(total_elapsed - tool_time, 0):.1f}s) · ↑{prompt_tokens:,} "
                    f"↓{completion_tokens:,} tokens[/dim]"
                )
            # Compact fat flow snapshots (verification get_flow ~5K) before they carry
            # into the next turn, then apply summarization.
            compact_flow_snapshots(messages)
            # Context management: keep full history under the threshold, otherwise
            # summarize the older prefix and keep the recent tail verbatim. Replaces
            # the old last-2-user-turns trim, which discarded determinants and context.
            messages = await summarize_history(llm, messages, settings)
            break
    else:
        console.print("[yellow]⚠ Max tool iterations reached.[/yellow]")

    return messages


async def run_chat(llm: LLMProvider, mcp: LangflowMCPClient, settings: Settings) -> None:
    # Seed value for run_turn; it rebuilds tools each iteration so discovered tools appear next turn.
    tools = mcp.get_tool_schemas()
    messages: list[dict] = []
    _starter_cache: dict[str, dict] = {}  # name_lower/id → full template dict
    sink = ConsoleSink()
    # Persistent agent state — lives across turns so plans, saved facts, intended edges
    # and approved designs survive the per-turn history summarization run_turn applies.
    todos: list[dict] = []
    scratchpad: dict[str, str] = {}
    _intended_edges_by_flow: dict[str, list] = {}
    _approved_designs: dict[str, dict] = {}

    console.print(Panel(
        "[bold green]Langflow Coding Agent[/bold green]\n"
        "Type your request. Ctrl+C or 'exit' to quit.",
        border_style="green",
    ))

    while True:
        try:
            user_input = console.input("[bold cyan]nokia>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            messages = await run_turn(
                llm, mcp, settings, tools, messages, _starter_cache, sink,
                todos=todos, scratchpad=scratchpad,
                _intended_edges_by_flow=_intended_edges_by_flow,
                _approved_designs=_approved_designs,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            break
