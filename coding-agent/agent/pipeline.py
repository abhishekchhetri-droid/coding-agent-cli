"""Pipeline-alignment step for real-life, multi-stage flow builds.

Real-world pipeline requests (e.g. NL→SQL: intent classify → pull table schema from Qdrant →
LLM gateway → SQL-gen prompt with few-shot examples → SQL executor → output) contain stages
whose component or data source is ambiguous: "intent classification" (a Prompt? a router?),
"schema from Qdrant" (which collection/retriever?), "LLM gateway" (which provider?). Before
delegating graph design to the designer sub-agent, the main agent maps EVERY stage to a concrete
non-legacy component and asks the user ONLY about the ambiguous ones (the ask→loop→design loop).

This module is the virtual ``propose_pipeline`` tool. It normalizes the stage map, renders it for
the user, and partitions resolved stages from open-question stages so the loop knows whether to
ask the user or proceed to ``design_flow``. Dispatch lives inline in ``run_chat`` (same pattern as
write_todos / design_flow). Judgement about which stages are ambiguous is the LLM's, decided from
the live component catalog — there is no hardcoded stage list here.
"""

import json

# A stage is "ok" once its component (and data source, if any) are pinned; "ask" while the
# mapping is still ambiguous and needs a user decision.
STAGE_STATUSES = ("ok", "ask")

PROPOSE_PIPELINE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_pipeline",
        "description": (
            "Call this FIRST for a real-life, multi-stage pipeline build (an explicit "
            "real-world data flow such as NL→SQL, RAG-with-retrieval, classify→route→act) "
            "BEFORE design_flow. Map EVERY described stage to a concrete non-legacy component "
            "and (if the stage reads/writes data) its source. Mark a stage `ask` — and give a "
            "one-line `question` — when its mapping is genuinely ambiguous: no single clear "
            "non-legacy component fits, the data source/collection is unspecified, or the model "
            "provider is unspecified. Mark the rest `ok`. Decide ambiguity from the actual "
            "component catalog, not a fixed list. The user sees the whole map and answers only "
            "the `ask` stages; re-call propose_pipeline with their answers folded in until no "
            "stage is `ask`, then call design_flow(request=..., resolved_stages=<the ok stages>). "
            "Skip this tool for near-exact template clones and simple/unambiguous builds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stages": {
                    "type": "array",
                    "description": "Ordered pipeline stages (full map, not just the ambiguous ones).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage": {"type": "string", "description": "What this stage does, in the user's terms"},
                            "component": {"type": "string", "description": "Chosen non-legacy component type, or '' if unresolved"},
                            "source": {"type": "string", "description": "Data source/collection if the stage reads/writes data, else ''"},
                            "status": {"type": "string", "enum": list(STAGE_STATUSES), "description": "ok | ask"},
                            "question": {"type": "string", "description": "For an `ask` stage: the one-line question to put to the user"},
                        },
                        "required": ["stage", "status"],
                    },
                }
            },
            "required": ["stages"],
        },
    },
}


def normalize_stages(raw) -> list[dict]:
    """Coerce arbitrary tool input into clean stage dicts. Unknown status → 'ask' (safer: an
    unrecognized status means the model was unsure, so surface it rather than silently proceed)."""
    stages: list[dict] = []
    if not isinstance(raw, list):
        return stages
    for item in raw:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "")).strip()
        if not stage:
            continue
        status = item.get("status", "ask")
        if status not in STAGE_STATUSES:
            status = "ask"
        s = {
            "stage": stage,
            "component": str(item.get("component", "")).strip(),
            "source": str(item.get("source", "")).strip(),
            "status": status,
        }
        q = str(item.get("question", "")).strip()
        if q:
            s["question"] = q
        # An `ok` stage with no component chosen is contradictory — treat as `ask`.
        if s["status"] == "ok" and not s["component"]:
            s["status"] = "ask"
            s.setdefault("question", f"Which component should handle: {stage}?")
        stages.append(s)
    return stages


def _stage_output_types(schema: dict) -> set[str]:
    """Union of all output types a component emits."""
    out: set[str] = set()
    for o in schema.get("outputs", []) or []:
        out |= set(o.get("types") or o.get("output_types") or [])
    return out


def verify_stages(stages: list[dict], mcp) -> list[dict]:
    """Schema-driven verification of a proposed stage map (mutates + returns ``stages``).

    Adaptive, never use-case-specific — every check reads the live catalog:
      * An `ok` stage whose component does not resolve to a real catalog type is flipped to
        `ask` (the existing ask→loop re-resolves it with the user — works for any custom name).
      * An `ok` stage whose component is `legacy: true` is flipped to `ask`, with non-legacy
        alternatives discovered from the catalog by matching the legacy component's I/O signature.
      * Soft, non-blocking chainability hint: for consecutive resolved stages, if no output of
        stage N intersects any input of stage N+1 AND no bridge component exists, annotate the
        later stage so the incompatibility surfaces before design even starts. Soft because
        stage order ≠ exact wiring; the hard check stays in the designer gate.
    """
    try:
        schemas = mcp._fetch_component_schemas()
    except Exception:
        return stages  # catalog unreachable — skip verification rather than block the loop

    for s in stages:
        if s["status"] != "ok" or not s.get("component"):
            continue
        resolved = mcp._resolve_type(s["component"], schemas)
        if resolved not in schemas:
            s["status"] = "ask"
            s["question"] = (
                f"{s['component']!r} for stage '{s['stage']}' is not a known catalog component — "
                f"which component should handle this stage?"
            )
            continue
        if schemas[resolved].get("legacy"):
            sig_in = mcp._component_input_types(schemas[resolved])
            sig_out = _stage_output_types(schemas[resolved])
            alts = mcp.find_bridges(sig_in, sig_out) if (sig_in and sig_out) else []
            alt_txt = ", ".join(a["display_name"] for a in alts) if alts else "a non-legacy equivalent"
            s["status"] = "ask"
            s["question"] = (
                f"{resolved} is legacy (hard-blocked at build) for stage '{s['stage']}' — "
                f"use {alt_txt} instead. Which component?"
            )
            continue
        s["component"] = resolved  # normalize to the canonical catalog key

    # Chainability hint over consecutive still-resolved stages.
    ok = [s for s in stages if s["status"] == "ok" and s.get("component") in schemas]
    for a, b in zip(ok, ok[1:]):
        a_out = _stage_output_types(schemas[a["component"]])
        b_in = mcp._component_input_types(schemas[b["component"]])
        if not a_out or not b_in:
            continue  # can't reason about untyped value-only stages
        if a_out & b_in:
            continue
        if mcp.find_bridges(a_out, b_in):
            continue  # a converter exists; design time can insert it
        b["hint"] = (
            f"⚠ no direct type path {a['component']}→{b['component']}; "
            f"a converter stage may be inserted at design time"
        )
    return stages


def split(stages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition into (resolved_ok_stages, open_questions). open_questions carry stage+question."""
    resolved = [s for s in stages if s["status"] == "ok"]
    open_questions = [
        {"stage": s["stage"], "question": s.get("question") or f"Which component for: {s['stage']}?"}
        for s in stages if s["status"] == "ask"
    ]
    return resolved, open_questions


def render_pipeline(stages: list[dict]) -> str:
    """Compact markdown stage map for the confirm panel. Ambiguous stages flagged with the
    question the user needs to answer."""
    if not stages:
        return "_(empty pipeline)_"
    lines = ["## Proposed pipeline", ""]
    for i, s in enumerate(stages, 1):
        if s["status"] == "ok":
            tail = f"`{s['component']}`"
            if s.get("source"):
                tail += f" ← {s['source']}"
            lines.append(f"{i}. **{s['stage']}** → {tail}")
            if s.get("hint"):
                lines.append(f"   - {s['hint']}")
        else:
            q = s.get("question") or "needs a decision"
            lines.append(f"{i}. **{s['stage']}** → ❓ _{q}_")
    return "\n".join(lines)


def build_result(resolved: list[dict], open_questions: list[dict]) -> dict:
    """The structured tool result the model sees: either the open questions to ask, or the
    resolved stage map plus the directive to proceed to design_flow."""
    if open_questions:
        return {
            "ready": False,
            "questions": open_questions,
            "note": "Ask the user these questions, then call propose_pipeline again with their "
                    "answers folded in (set those stages to status 'ok' with the chosen "
                    "component/source). Do NOT call design_flow yet.",
        }
    return {
        "ready": True,
        "resolved_stages": resolved,
        "note": "Pipeline aligned. Now call design_flow(request=<original request>, "
                "resolved_stages=<these stages>) — the designer will honor each chosen component.",
    }
