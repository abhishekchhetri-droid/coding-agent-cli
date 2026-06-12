"""Event sink boundary so the agent turn loop drives both the terminal REPL and
the AG-UI web server without duplicating the loop logic.

run_turn (agent/agent.py) keeps all its existing Rich console.print calls — those
remain the CLI's visible output and go to server stdout under the web server. On top
of that it emits three *structured* signals through an EventSink:

  - tool_call(name, arguments) — a tool is about to run
  - flow_built(flow_id)        — a flow was created/updated/built (drives the canvas)
  - usage(metrics)             — end-of-turn token/timing totals (drives the token meter)
  - notice(markdown)           — an intermediate artifact (proposed design, plan) to show in chat
  - final(text)                — the assistant's end-of-turn answer

ConsoleSink is a no-op: the CLI already renders everything via console.print, so the
structured signals add nothing there and CLI behaviour stays identical. The web sink
(AGUISink, in server/) maps these signals to AG-UI protocol events.
"""
import json
from typing import Protocol


def _handle_field(h, key: str):
    """Pull a field (e.g. 'name' / 'fieldName') from an edge handle that may be a dict
    or a œ/Å-encoded JSON string (Langflow's on-disk handle format)."""
    if isinstance(h, str):
        for ch in ("œ", "Å"):
            try:
                h = json.loads(h.replace(ch, '"'))
                break
            except Exception:
                continue
    if isinstance(h, dict):
        return h.get(key)
    return None


def _stringify_field_value(value, secret: bool) -> str | None:
    """Compact display string for a Langflow field value (None -> empty placeholder)."""
    if secret:
        return "••••••" if value else None
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 48 else value[:45] + "…"
    if isinstance(value, list):
        return f"[{len(value)} item{'s' if len(value) != 1 else ''}]"
    if isinstance(value, dict):
        return "{…}"
    return str(value)[:48]


def _slim_fields(node_meta: dict) -> list:
    """Extract the shown (non-advanced) fields of a node's template for display.

    Schema-driven: walks the template dict Langflow already shipped — no assumptions about
    specific component types. Mirrors what Langflow shows on the node face.
    """
    tmpl = node_meta.get("template")
    if not isinstance(tmpl, dict):
        return []
    out = []
    for fname, f in tmpl.items():
        if fname in ("code", "_type") or not isinstance(f, dict):
            continue
        if f.get("show") is False or f.get("advanced") is True:
            continue
        ftype = f.get("type") or ""
        secret = bool(f.get("password")) or ftype == "password"
        out.append({
            "key": fname,  # real template key — the canvas needs it to write values back
            "name": f.get("display_name") or fname,
            "type": ftype,
            "value": _stringify_field_value(f.get("value"), secret),
            "secret": secret,
        })
    return out


def _slim_inputs(node_meta: dict) -> list:
    """Connectable input fields (those carrying input_types) → target handles on the canvas.

    Schema-driven: a template field is a connection target iff Langflow gave it input_types
    AND its widget renders an edge port. `_input_type == 'ModelInput'` is a dropdown-only
    widget with no port — Langflow stores but strips/ignores edges to it (e.g. Agent.model),
    so it is excluded to match what the real Langflow canvas shows.
    `key` is the template key the edge's targetHandle.fieldName must reference.
    """
    tmpl = node_meta.get("template")
    if not isinstance(tmpl, dict):
        return []
    out = []
    for fname, f in tmpl.items():
        if fname in ("code", "_type") or not isinstance(f, dict):
            continue
        if f.get("show") is False:
            continue
        if f.get("_input_type") == "ModelInput":  # dropdown-only, no edge port
            continue
        its = f.get("input_types")
        if not its:
            continue
        out.append({
            "key": fname,
            "name": f.get("display_name") or fname,
            "input_types": its,
        })
    return out


def _slim_outputs(node_meta: dict) -> list:
    """Component outputs → source handles on the canvas. `name` is the sourceHandle.name."""
    outs = node_meta.get("outputs")
    if not isinstance(outs, list):
        return []
    res = []
    for o in outs:
        if not isinstance(o, dict):
            continue
        name = o.get("name")
        if not name:
            continue
        res.append({
            "name": name,
            "display": o.get("display_name") or name,
            "types": o.get("types") or [],
        })
    return res


def slim_graph(flow: dict) -> dict | None:
    """Reduce a full Langflow flow dict to the minimal graph the canvas needs.

    Returns {"nodes": [{id, position, label, type, fields}], "edges": [{id, source, target}]}
    or None if the flow has no usable graph data. Schema-driven: walks whatever nodes/edges
    exist rather than assuming specific component types.
    """
    if not isinstance(flow, dict):
        return None
    data = flow.get("data")
    if not isinstance(data, dict):
        return None
    nodes_out = []
    for n in data.get("nodes", []) or []:
        if not isinstance(n, dict) or not n.get("id"):
            continue
        nd = n.get("data") or {}
        node_meta = nd.get("node") or {}
        ntype = nd.get("type") or n.get("type") or ""
        is_note = "note" in ntype.lower()
        if is_note:
            # Note nodes carry their text in description, not a template.
            label = node_meta.get("description") or "Note"
            fields = []
            inputs: list = []
            outputs: list = []
        else:
            label = node_meta.get("display_name") or ntype or n["id"]
            fields = _slim_fields(node_meta)
            inputs = _slim_inputs(node_meta)
            outputs = _slim_outputs(node_meta)
        pos = n.get("position") or {}
        nodes_out.append({
            "id": n["id"],
            "position": {"x": pos.get("x", 0), "y": pos.get("y", 0)},
            "label": label,
            "type": ntype,
            "fields": fields,
            "inputs": inputs,
            "outputs": outputs,
        })
    edges_out = []
    for e in data.get("edges", []) or []:
        if not isinstance(e, dict) or not e.get("source") or not e.get("target"):
            continue
        edges_out.append({
            "id": e.get("id") or f"{e['source']}->{e['target']}",
            "source": e["source"],
            "target": e["target"],
            # Handle ids so the canvas attaches each edge to the right per-field handle
            # (source output name / target template key) when a node has several handles.
            "sourceHandle": _handle_field(e.get("sourceHandle"), "name"),
            "targetHandle": _handle_field(e.get("targetHandle"), "fieldName"),
        })
    if not nodes_out and not edges_out:
        return None
    return {"nodes": nodes_out, "edges": edges_out}


class EventSink(Protocol):
    flow_id: str | None
    interactive: bool  # True = a human can answer console confirm gates (CLI); False = web/headless
    def tool_call(self, name: str, arguments: dict) -> None: ...
    def flow_built(self, flow_id: str | None, graph: dict | None = None) -> None: ...
    def flow_modified(self, graph: dict | None = None) -> None: ...
    def usage(self, metrics: dict) -> None: ...
    def notice(self, markdown: str | None) -> None: ...
    def final(self, text: str | None) -> None: ...


class ConsoleSink:
    """No-op sink for the terminal REPL — existing console.print output is unchanged."""

    flow_id: str | None = None  # no canvas in the terminal; keeps run_turn's sink.flow_id checks falsy
    interactive: bool = True  # terminal REPL: confirm gates can read y/n from stdin

    def tool_call(self, name: str, arguments: dict) -> None:
        pass

    def flow_built(self, flow_id: str | None, graph: dict | None = None) -> None:
        pass

    def flow_modified(self, graph: dict | None = None) -> None:
        pass

    def usage(self, metrics: dict) -> None:
        pass  # CLI already prints the token/timing line via console.print

    def notice(self, markdown: str | None) -> None:
        pass  # CLI already renders the design/plan Panel via console.print

    def final(self, text: str | None) -> None:
        pass
