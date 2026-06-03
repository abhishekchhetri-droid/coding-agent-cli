"""Event sink boundary so the agent turn loop drives both the terminal REPL and
the AG-UI web server without duplicating the loop logic.

run_turn (agent/agent.py) keeps all its existing Rich console.print calls — those
remain the CLI's visible output and go to server stdout under the web server. On top
of that it emits three *structured* signals through an EventSink:

  - tool_call(name, arguments) — a tool is about to run
  - flow_built(flow_id)        — a flow was created/updated/built (drives the canvas)
  - final(text)                — the assistant's end-of-turn answer

ConsoleSink is a no-op: the CLI already renders everything via console.print, so the
structured signals add nothing there and CLI behaviour stays identical. The web sink
(AGUISink, in server/) maps these signals to AG-UI protocol events.
"""
from typing import Protocol


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
            "name": f.get("display_name") or fname,
            "type": ftype,
            "value": _stringify_field_value(f.get("value"), secret),
            "secret": secret,
        })
    return out


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
        else:
            label = node_meta.get("display_name") or ntype or n["id"]
            fields = _slim_fields(node_meta)
        pos = n.get("position") or {}
        nodes_out.append({
            "id": n["id"],
            "position": {"x": pos.get("x", 0), "y": pos.get("y", 0)},
            "label": label,
            "type": ntype,
            "fields": fields,
        })
    edges_out = []
    for e in data.get("edges", []) or []:
        if not isinstance(e, dict) or not e.get("source") or not e.get("target"):
            continue
        edges_out.append({
            "id": e.get("id") or f"{e['source']}->{e['target']}",
            "source": e["source"],
            "target": e["target"],
        })
    if not nodes_out and not edges_out:
        return None
    return {"nodes": nodes_out, "edges": edges_out}


class EventSink(Protocol):
    def tool_call(self, name: str, arguments: dict) -> None: ...
    def flow_built(self, flow_id: str | None, graph: dict | None = None) -> None: ...
    def flow_modified(self, graph: dict | None = None) -> None: ...
    def final(self, text: str | None) -> None: ...


class ConsoleSink:
    """No-op sink for the terminal REPL — existing console.print output is unchanged."""

    def tool_call(self, name: str, arguments: dict) -> None:
        pass

    def flow_built(self, flow_id: str | None, graph: dict | None = None) -> None:
        pass

    def flow_modified(self, graph: dict | None = None) -> None:
        pass

    def final(self, text: str | None) -> None:
        pass
