"""Apply direct canvas edits (drag / field edit / add+delete edge / delete node) to a
Langflow flow and persist them.

The web canvas is no longer a read-only mirror: the user can move nodes, edit field
values, wire edges and delete elements. Those are deterministic CRUD gestures, NOT
language, so they bypass the LLM run_turn loop entirely and land here via
POST /canvas/mutate.

One get_flow → mutate the full flow JSON in memory → one update_flow → build_flow, then
return a fresh slim_graph for the browser to reconcile against. update_flow/build_flow go
through the client's _raw_call to get a FULL REPLACE (the public call_tool path delta-merges
and is union-only — it cannot remove nodes/edges), mirroring _handle_delete_node.

The mutation math is schema-driven: add_edge supplies only {source, target, output, field}
and reuses LangflowMCPClient.enrich_edges, which derives the exact compact (œ-encoded)
handles from each node's own outputs/template — no per-component-type assumptions, no extra
schema fetch hand-rolled here.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from agent.events import slim_graph


def _coerce_value(raw: Any, ftype: str) -> Any:
    """Best-effort coerce an edited string back to the template field's type.

    The browser always sends strings; numeric/bool fields need their native type or
    Langflow may reject the value. Unknown types pass through unchanged.
    """
    if not isinstance(raw, str):
        return raw
    if ftype in ("int", "Integer"):
        try:
            return int(raw)
        except ValueError:
            return raw
    if ftype in ("float", "Float"):
        try:
            return float(raw)
        except ValueError:
            return raw
    if ftype in ("bool", "boolean"):
        return raw.strip().lower() in ("true", "1", "yes", "on")
    return raw


def _node_by_id(nodes: list[dict], node_id: str) -> dict | None:
    return next((n for n in nodes if n.get("id") == node_id), None)


def _drop_nodes(nodes: list[dict], edges: list[dict], ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Remove nodes by id plus any edge touching them. Same filter as
    LangflowMCPClient._handle_delete_node (kept local to avoid a server→mcpbridge import)."""
    kept_nodes = [n for n in nodes if n.get("id") not in ids]
    kept_edges = [
        e for e in edges
        if e.get("source") not in ids and e.get("target") not in ids
    ]
    return kept_nodes, kept_edges


def _build_edge_skeleton(op: dict, nodes: list[dict]) -> dict | None:
    """Minimal edge dict for enrich_edges to flesh out. Returns None if the connection is
    invalid (missing endpoints, or target field is not a real input handle)."""
    source = op.get("source")
    target = op.get("target")
    output = op.get("output")  # source handle id = output name
    field = op.get("field")    # target handle id = template field key
    if not (source and target and field):
        return None

    tgt = _node_by_id(nodes, target)
    if tgt is None:
        return None
    tmpl = (((tgt.get("data") or {}).get("node") or {}).get("template")) or {}
    tfield = tmpl.get(field)
    # Schema-driven guard: only allow drops onto fields with a real edge port — i.e. carrying
    # input_types and NOT a dropdown-only ModelInput (Langflow strips/ignores edges to those,
    # e.g. Agent.model). Mirrors the _slim_inputs handle filter so UI and backend agree.
    if not isinstance(tfield, dict) or not tfield.get("input_types"):
        return None
    if tfield.get("_input_type") == "ModelInput":
        return None

    return {
        "id": f"reactflow__edge-{source}-{target}-{uuid.uuid4().hex[:6]}",
        "source": source,
        "target": target,
        "sourceHandle": {"id": source, "name": output} if output else {"id": source},
        "targetHandle": {"id": target, "fieldName": field, "inputTypes": tfield.get("input_types")},
    }


def apply_ops(data: dict, ops: list[dict], enrich_edges: Callable[[list, list], list]) -> dict:
    """Pure in-memory mutation of a flow's `data` (nodes+edges). Returns the new data dict.

    `enrich_edges` is injected (LangflowMCPClient.enrich_edges) so this stays unit-testable
    without an MCP session. Unknown ops are ignored.
    """
    nodes = list(data.get("nodes") or [])
    edges = list(data.get("edges") or [])
    new_skeletons: list[dict] = []
    del_node_ids: set[str] = set()

    for op in ops:
        kind = op.get("op")
        if kind == "move":
            n = _node_by_id(nodes, op.get("id"))
            if n is not None and isinstance(op.get("position"), dict):
                n["position"] = {"x": op["position"].get("x", 0), "y": op["position"].get("y", 0)}
        elif kind == "edit_field":
            n = _node_by_id(nodes, op.get("id"))
            key = op.get("key")
            if n is not None and key:
                tmpl = (((n.get("data") or {}).get("node") or {}).get("template")) or {}
                field = tmpl.get(key)
                if isinstance(field, dict):
                    field["value"] = _coerce_value(op.get("value"), field.get("type") or "")
        elif kind == "delete_edge":
            eid = op.get("id")
            if eid:
                edges = [e for e in edges if e.get("id") != eid]
        elif kind == "delete_node":
            nid = op.get("id")
            if nid:
                del_node_ids.add(nid)
        elif kind == "add_edge":
            skel = _build_edge_skeleton(op, nodes)
            if skel is not None:
                new_skeletons.append(skel)

    if del_node_ids:
        nodes, edges = _drop_nodes(nodes, edges, del_node_ids)

    if new_skeletons:
        # enrich_edges fills compact œ-encoded handles + schema-correct types from `nodes`.
        edges = edges + enrich_edges(new_skeletons, nodes)

    return {**data, "nodes": nodes, "edges": edges}


async def apply_canvas_ops(mcp, flow_id: str, ops: list[dict]) -> dict | None:
    """Fetch the flow, apply canvas ops, full-replace persist, rebuild, return slim_graph."""
    flow = await mcp._session_call_json("get_flow", {"flow_id": flow_id})
    if not isinstance(flow, dict):
        return None
    data = flow.get("data") or {}
    new_data = apply_ops(data, ops, mcp.enrich_edges)

    # _raw_call → full replace (call_tool delta-merges and can't remove); build_flow
    # invalidates Langflow's canvas cache so a later "Open in Langflow" reflects the edit.
    await mcp._raw_call("update_flow", {"flow_id": flow_id, "data": new_data})
    await mcp._raw_call("build_flow", {"flow_id": flow_id})
    return slim_graph({**flow, "data": new_data})
