"""Flow-designer sub-agent — a lightweight, in-process nested tool-loop (no langchain).

For a complex from-scratch build, the main agent delegates graph design here. The
sub-agent runs its OWN tool-loop with its OWN message list and a SMALL read-only toolset
(get_component_schema, list_components), then finishes by calling ``submit_design`` with a
create_flow-ready node/edge spec. Because it reuses the same provider, it gets its own
system+tools prompt-cache breakpoints; because its (verbose) schema-fetching lives in an
isolated context, only the compact final spec crosses back to the main thread — a net
token win on complex builds.

Design rules the sub-agent enforces (general, not per-case):
  * Never use a legacy component (schemas flag ``legacy: true``) — decompose into primitives.
  * Consolidate metadata/instructions/examples into ONE Prompt with ``{vars}`` — never
    CombineText or multiple text nodes.
  * Match the user's described pipeline as DISTINCT stages — never collapse into a single
    mega-component that hides them.
"""

import json

_DESIGNER_MAX_ITERS = 10
# Each design_invalid rejection grants one extra iteration (capped at _DESIGNER_MAX_ITERS + 3)
# so machine-validation never starves the sub-agent's exploration budget.
_MAX_VALIDATION_RETRIES = 3

DESIGNER_SYSTEM_PROMPT = """You are a Langflow flow-design specialist. Given a build request,
produce a correct node/edge graph and return it via `submit_design`. You do NOT build or
call create_flow — you only design.

## Hard rules
1. NEVER use a legacy component. `get_component_schema` and `list_components` flag legacy
   ones with `legacy: true` — avoid them. A legacy "mega-component" that hides multiple
   stages (e.g. a single Natural-Language-to-SQL node) must be DECOMPOSED into explicit
   modern primitives.
2. Consolidate static text (metadata, instructions, examples) for ONE logical prompt into a
   single Prompt component using `{variable}` placeholders in its `template`. NEVER chain
   CombineText or multiple text-input nodes. Langflow creates one input handle per `{var}` in
   the template — so set the template first, then wire edges into those var fields.
   IMPORTANT for multi-prompt pipelines: when the pipeline has SEVERAL distinct prompt stages
   (e.g. an intent-classification prompt AND a SQL-generation prompt), emit a SEPARATE Prompt
   node for each, and give EACH node its OWN `template` carrying ONLY that prompt's `{vars}`.
   Do NOT reuse one template across prompt nodes and do NOT merge several prompts' text or vars
   into a single template — each Prompt node's template and vars are independent.
3. Match the described data flow as DISTINCT stages. If the user says one component emits
   text and a separate one consumes/executes it (e.g. "LLM returns SQL, then run that SQL"),
   wire them as separate nodes. Never collapse into one self-contained *Agent.
4. Wire every REQUIRED pure-DATA input (schema field type `other` — Data/DataFrame/
   Embeddings/Tool/...). Config and credential fields are NOT data inputs: endpoints,
   deployment names, API keys, api versions, index/collection names — any `str` field, even
   when required and even when it exposes a Message handle — must be LEFT EMPTY and UNWIRED.
   They are injected from settings after design or filled by the user in the Langflow UI.
   NEVER create passthrough feeder nodes (TextInput or similar) to fill a config field.
5. TYPE-CHECK every edge BEFORE you submit. For each edge confirm from `get_component_schema`
   that (a) the target FIELD exists on the target component, and (b) the source output's type
   is listed in that field's `input_types`. If your chosen component has no field that fits the
   incoming data, or its output type does not fit the next stage's input, it is the WRONG
   component — pick one whose schema actually fits the stage's role (use `list_components`).
   A node that ends a pipeline (e.g. ChatOutput) must be reachable from ChatInput through the
   stages — never leave a stage with its output going nowhere or its real input port empty.
   `submit_design` is machine-validated against the live schemas; a `design_invalid` result
   lists violations with catalog-discovered fixes — resolve all of them and resubmit.
6. A STATEFUL STORE (any component whose schema has both a data-ingest input and a query
   input — vector stores, caches) holds its index INSIDE the node instance. Use exactly ONE
   instance per logical store and wire BOTH its ingest input and its query input on that same
   node. NEVER split "store" and "retrieve" pipeline stages into two instances of the same
   store component — each would keep a separate index and retrieval would find nothing. Two
   instances are only valid when they point at distinct collections (differing config values).

## Common type strings (use these directly — do NOT call list_components for them)
ChatInput, ChatOutput, AzureOpenAIModel (Azure LLM), Prompt (Prompt Template, the {var}
component), Parser.

To EXECUTE SQL use **`SQLComponent`** — inputs `query` (the generated SQL string) and
`database_url` (connection string), output `run_sql_query` (Table, which ChatOutput accepts).
WARNING: a different component, `SQLDatabase`, shares the display name "SQL Database" but only
wraps a connection (single input `uri`, output a `SQLDatabase` object) and CANNOT run a query —
never use it as the executor, and prefer the exact type string `SQLComponent` over the display
name to avoid the collision. For a NL→SQL pipeline: Prompt → LLM (produces SQL) →
`SQLComponent.query` (with `database_url` set) → ChatOutput.

## How to work — be efficient, you have a limited iteration budget
1. In ONE turn, batch `get_component_schema` for every component type you intend to use.
   Confirm each is `legacy:false` and read its inputs (name, input_types, required) + outputs.
2. Do NOT keep exploring once you have those schemas — guessing more type strings wastes the
   budget. Use `list_components` only if a needed component is not in the list above and you
   cannot guess it.
3. Then immediately call `submit_design` ONCE with the final spec. Submitting early with a
   correct graph is far better than running out of turns mid-exploration.

## Node format
{"id": "<Type>-1", "type": "<Type>", "position": {"x": <int>, "y": <int>}}
Provide only id/type/position — schemas + credentials are injected later. EACH Prompt node MUST
also carry its OWN `template` string in the node object, e.g.
{"id": "Prompt-intent", "type": "Prompt", "position": {...}, "template": "Classify intent: {user_query}"}
That per-node template defines exactly that node's `{var}` input handles before wiring — so two
different prompt stages get two different node templates (never one shared template).

## Edge format
{"source":"<src_id>",
 "sourceHandle":{"dataType":"<SrcType>","id":"<src_id>","name":"<output_name>","output_types":["<Type>"]},
 "target":"<tgt_id>",
 "targetHandle":{"fieldName":"<field>","id":"<tgt_id>","inputTypes":["<Type>"],"type":"<handle_type>"}}

## Finishing
Call `submit_design(summary, nodes, edges, prompt_template, vars)`:
- summary: 1-2 sentences describing the pipeline stages.
- nodes / edges: the full create_flow-ready arrays above. Each Prompt node carries its OWN
  `template` (this is how multi-prompt pipelines stay distinct).
- prompt_template / vars: LEGACY single-prompt fallback only — set them ONLY when the whole
  flow has exactly one Prompt node and you did not put a `template` on it. With per-node
  templates (the normal case), leave prompt_template="" and vars=[].
"""

_GET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_component_schema",
        "description": "Inspect one component's inputs/outputs and legacy flag. Returns "
                       "{type, inputs:[{field,type,required,input_types}], outputs, legacy?}.",
        "parameters": {
            "type": "object",
            "properties": {"type_name": {"type": "string", "description": "Exact component type string"}},
            "required": ["type_name"],
        },
    },
}

_LIST_COMPONENTS = {
    "type": "function",
    "function": {
        "name": "list_components",
        "description": "List all available component types (with display_name and legacy flag) "
                       "to find an exact type string. Use sparingly — large result.",
        "parameters": {"type": "object", "properties": {}},
    },
}

_SUBMIT_DESIGN = {
    "type": "function",
    "function": {
        "name": "submit_design",
        "description": "Return the final flow design. Call ONCE when the graph is complete and "
                       "every required input is wired.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "1-2 sentence pipeline description"},
                "nodes": {"type": "array", "items": {"type": "object"}, "description": "create_flow-ready nodes[]"},
                "edges": {"type": "array", "items": {"type": "object"}, "description": "create_flow-ready edges[]"},
                "prompt_template": {"type": "string", "description": "Prompt component template, or ''"},
                "vars": {"type": "array", "items": {"type": "string"}, "description": "Declared {var} names, or []"},
            },
            "required": ["summary", "nodes", "edges"],
        },
    },
}

_DESIGNER_TOOLS = [_GET_SCHEMA, _LIST_COMPONENTS, _SUBMIT_DESIGN]

# Virtual tool the MAIN agent calls to delegate complex graph design here. Stable (cached
# region). Dispatched specially in run_chat (runs the sub-agent + confirm gate), not via
# planning.dispatch.
DESIGN_FLOW_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "design_flow",
        "description": (
            "Delegate complex from-scratch flow design to the flow-designer sub-agent. Call "
            "this FIRST for any complex build (>=5 nodes or an explicit multi-stage pipeline) "
            "instead of hand-building create_flow — it picks modern (non-legacy) components, "
            "consolidates text into one Prompt with {vars}, and keeps pipeline stages "
            "distinct. Pass the user's full described pipeline as `request`. The user reviews "
            "and confirms the graph. On approval you receive a `design_ref`; then call "
            "create_flow with {\"data\":{\"_design_ref\":<ref>}} — do NOT re-emit nodes/edges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "request": {"type": "string", "description": "The full build request / described pipeline"},
                "feedback": {"type": "string", "description": "Optional: revision notes if redesigning after a rejection"},
                "resolved_stages": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional: the user-aligned stage→component map returned by "
                                   "propose_pipeline (status 'ok' stages). Pass it so the designer "
                                   "honors the agreed components instead of re-picking.",
                },
            },
            "required": ["request"],
        },
    },
}


def render_design(spec: dict) -> str:
    """Compact human-readable graph sketch for the confirm panel: stages, edges, prompt vars."""
    lines = [f"## Proposed flow design\n\n{spec.get('summary', '').strip()}", ""]
    nodes = spec.get("nodes") or []
    lines.append("**Nodes:** " + ", ".join(
        f"`{(n.get('data') or {}).get('type') or n.get('type')}`" for n in nodes
    ))
    edges = spec.get("edges") or []
    if edges:
        lines.append("\n**Wiring:**")
        for e in edges:
            sh = e.get("sourceHandle") or {}
            th = e.get("targetHandle") or {}
            src = e.get("source", "?")
            tgt = e.get("target", "?")
            out = sh.get("name", "out") if isinstance(sh, dict) else "out"
            fld = th.get("fieldName", "?") if isinstance(th, dict) else "?"
            lines.append(f"- `{src}`.{out} → `{tgt}`.{fld}")
    # Per-node templates (the normal multi-prompt case) — show each distinctly so the user can
    # confirm the prompts are NOT identical.
    node_templates = [(n.get("id", "?"), n.get("template")) for n in nodes if isinstance(n, dict) and n.get("template")]
    if node_templates:
        lines.append("\n**Prompt templates (per node):**")
        for nid, tmpl in node_templates:
            lines.append(f"- `{nid}`: `{tmpl}`")
    elif spec.get("prompt_template"):
        lines.append(f"\n**Prompt template:** `{spec['prompt_template']}`")
        if spec.get("vars"):
            lines.append("**Prompt vars:** " + ", ".join(f"`{{{v}}}`" for v in spec["vars"]))
    # Validation warnings the sub-agent could not resolve within its retry budget — surfaced so
    # the human confirm gate sees them (never auto-approved when present).
    violations = spec.get("violations") or []
    if violations:
        lines.append("\n**⚠ Unresolved validation warnings:**")
        for v in violations:
            lines.append(f"- {v}")
    return "\n".join(lines)


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
                    # provider returns arguments as a dict; history needs a JSON string
                    "arguments": json.dumps(tc.get("arguments") or {}),
                },
            }
            for tc in tool_calls
        ],
    }


def _tool_result_message(tool_call_id: str, result: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": result or "null"}


async def _list_components_compact(mcp) -> str:
    """list_components stripped to {type, display_name, legacy?} — same shape the main loop
    surfaces, so the sub-agent's selection is quality-aware without the ~1M-token raw blob."""
    raw = await mcp.call_tool("list_components", {})
    try:
        comps = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return raw if isinstance(raw, str) else "[]"
    if not isinstance(comps, list):
        return json.dumps(comps)
    out = []
    for c in comps:
        e = {"type": c.get("type"), "display_name": c.get("display_name", c.get("type"))}
        if mcp.is_legacy(c.get("type")):
            e["legacy"] = True
        out.append(e)
    return json.dumps(out)


async def design_flow(request: str, mcp, llm, feedback: str | None = None,
                      resolved_stages: list | None = None) -> dict:
    """Run the designer sub-agent. Returns the compact spec dict from submit_design, or
    {"error": ...} if it does not converge. ``feedback`` (from a rejected design) is folded
    into the opening message so the sub-agent revises rather than restarts blindly.
    ``resolved_stages`` (from an approved propose_pipeline alignment) pins each stage's chosen
    component/source so the designer materializes the agreed graph instead of re-guessing.
    """
    opening = f"Build request:\n{request}"
    if resolved_stages:
        opening += (
            "\n\nThe user already aligned on this stage→component map — honor it ONE-TO-ONE: each "
            "stage becomes exactly ONE node. Do NOT expand a stage into several nodes (e.g. do "
            "not turn a single 'Gateway' stage into a Prompt + an LLM) and do not collapse "
            "stages. EXCEPTION: when SEVERAL stages name the SAME stateful-store component "
            "(e.g. a 'store/ingest' stage and a 'retrieve/search' stage both on one vector "
            "store), they are two ROLES of ONE node — emit a single instance and wire both its "
            "ingest and query inputs (hard rule 6). "
            "Keep the stages DISTINCT and do not swap a valid choice. Verify every named "
            "`component` exists via get_component_schema; if a name is not a real catalog type "
            "(e.g. 'SQLExecutorComponent', or a user's custom 'LLM Gateway'), resolve it to the "
            "correct existing type for that role, or use a single `CustomComponent` placeholder "
            "for a genuinely custom node — never a multi-node substitute. TYPE-CHECK its I/O. "
            "Then fill wiring, schemas, and per-node Prompt templates:\n"
            + json.dumps(resolved_stages, indent=2)
        )
    if feedback:
        opening += f"\n\nThe previous design was rejected. Revise per this feedback:\n{feedback}"
    messages: list[dict] = [{"role": "user", "content": opening}]

    budget = _DESIGNER_MAX_ITERS
    validation_retries = 0
    iters = 0
    while iters < budget:
        iters += 1
        resp = await llm.complete(messages, _DESIGNER_TOOLS, system=DESIGNER_SYSTEM_PROMPT)
        tool_calls = resp.get("tool_calls") or []
        if not tool_calls:
            # No tool call — nudge toward submit_design rather than ending empty.
            messages.append({"role": "assistant", "content": resp.get("content") or ""})
            messages.append({"role": "user", "content": "Call submit_design with the final spec now."})
            continue

        messages.append(_assistant_tool_call_message(tool_calls))
        for tc in tool_calls:
            name, args = tc["name"], (tc.get("arguments") or {})
            if name == "submit_design":
                spec = {
                    "summary": args.get("summary", ""),
                    "nodes": args.get("nodes", []),
                    "edges": args.get("edges", []),
                    "prompt_template": args.get("prompt_template", ""),
                    "vars": args.get("vars", []),
                }
                violations = mcp.validate_design(
                    spec["nodes"], spec["edges"], node_templates=_spec_node_templates(spec)
                )
                if not isinstance(violations, list):  # MagicMock / stub guard
                    violations = []
                if not violations:
                    return spec  # valid — accept mid-batch (no further turns needed)
                validation_retries += 1
                if validation_retries > _MAX_VALIDATION_RETRIES:
                    # Out of retries — surface the spec WITH violations; the human confirm
                    # gate decides rather than hard-failing the loop.
                    spec["violations"] = violations
                    return spec
                # Rejection: feed violations back as this call's tool_result (pairing intact)
                # and grant +1 iteration so validation never starves exploration.
                messages.append(_tool_result_message(tc["id"], json.dumps({
                    "error": "design_invalid",
                    "violations": violations,
                    "note": "Fix every violation and resubmit. Bridge suggestions come from the "
                            "live catalog — verify the chosen one with get_component_schema before wiring.",
                })))
                budget = min(budget + 1, _DESIGNER_MAX_ITERS + _MAX_VALIDATION_RETRIES)
                continue
            if name == "get_component_schema":
                result = json.dumps(mcp.get_component_schema(args.get("type_name", "")))
            elif name == "list_components":
                result = await _list_components_compact(mcp)
            else:
                result = json.dumps({"error": f"unknown tool {name!r}"})
            messages.append(_tool_result_message(tc["id"], result))

    return {"error": "designer did not converge within iteration budget"}


def _spec_node_templates(spec: dict) -> dict:
    """node id → prompt template string for validation. Per-node ``template`` is the normal
    multi-prompt case; the design-level ``prompt_template`` is the single-prompt fallback,
    applied to any node lacking its own (harmless for non-dynamic nodes — they ignore it)."""
    pt = spec.get("prompt_template", "")
    out: dict[str, str] = {}
    for n in spec.get("nodes", []):
        if isinstance(n, dict) and n.get("id"):
            out[n["id"]] = n.get("template") or pt
    return out
