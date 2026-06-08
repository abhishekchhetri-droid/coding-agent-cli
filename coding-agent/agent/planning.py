"""Lightweight planning primitives ported from deepagents — a todo list and a
scratchpad — implemented as plain in-process loop state, no langchain dependency.

The agent loop (``run_chat``) holds a ``todos`` list and a ``scratchpad`` dict and
renders both into the system prompt every iteration. Because they live in loop state
(not in the message history), they survive the turn-end summarization/trim — which is
exactly how the agent keeps planning context and determinants (flow_ids, decisions)
across long conversations.

Tool schemas here are concatenated onto ``mcp.get_tool_schemas()`` in the loop, so the
MCP bridge stays untouched. Dispatch is handled inline in ``run_chat`` (same pattern as
the other virtual tools) via :func:`dispatch`, which mutates the caller's state.
"""

import json

# Status values a todo may take. One in_progress at a time is a convention the prompt
# enforces; we don't hard-block it here so the model can correct itself freely.
TODO_STATUSES = ("pending", "in_progress", "completed")

_STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}

PLANNING_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "write_todos",
            "description": (
                "Create or replace your task plan for a complex / multi-step request. "
                "Pass the FULL list every time — it replaces the previous list, it does "
                "not append. Call this FIRST for any build that spans multiple stages "
                "(e.g. ≥5 nodes or an explicit pipeline) before touching create/build "
                "tools, then update statuses as you progress. Keep exactly one item "
                "'in_progress' at a time. The current list is shown back to you every "
                "step, so it is how you remember what is left to do."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete todo list (replaces the prior one).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Short imperative task description"},
                                "status": {
                                    "type": "string",
                                    "enum": list(TODO_STATUSES),
                                    "description": "pending | in_progress | completed",
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_write",
            "description": (
                "Persist a small fact you will need later — a flow_id, a chosen "
                "component type, a user decision. The scratchpad survives history "
                "trimming, so write anything here that must outlive the recent message "
                "window. Writing an existing key overwrites it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short identifier, e.g. 'rag_flow_id'"},
                    "content": {"type": "string", "description": "The value/note to store"},
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_read",
            "description": (
                "Read back persisted notes. Omit 'key' to list everything; pass a 'key' "
                "to fetch one note. The scratchpad is also rendered into your system "
                "prompt each step, so usually you can just read it there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Specific note to read; omit for all"},
                },
                "required": [],
            },
        },
    },
]

PLANNING_TOOL_NAMES = frozenset(s["function"]["name"] for s in PLANNING_TOOL_SCHEMAS)


def _normalize_todos(raw) -> list[dict]:
    """Coerce arbitrary tool input into a clean list of {content, status} dicts."""
    todos: list[dict] = []
    if not isinstance(raw, list):
        return todos
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        status = item.get("status", "pending")
        if status not in TODO_STATUSES:
            status = "pending"
        todos.append({"content": content, "status": status})
    return todos


def render_todos(todos: list[dict]) -> str:
    """Compact markdown checklist for injection into the system prompt. Empty → ""."""
    if not todos:
        return ""
    lines = ["## Current Plan (your todo list — keep it updated via write_todos)"]
    for t in todos:
        lines.append(f"- {_STATUS_MARK.get(t['status'], '[ ]')} {t['content']}")
    return "\n".join(lines)


def render_scratchpad(pad: dict[str, str]) -> str:
    """Compact markdown of persisted determinants. Empty → ""."""
    if not pad:
        return ""
    lines = ["## Scratchpad (persisted facts — survive history trimming)"]
    for k, v in pad.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def render_state(todos: list[dict], pad: dict[str, str]) -> str:
    """Combined todos + scratchpad block. Injected as a trailing context message each
    iteration (NOT into the cached system prompt). Empty → ""."""
    parts = [p for p in (render_todos(todos), render_scratchpad(pad)) if p]
    return "\n\n".join(parts)


def dispatch(name: str, args: dict, todos: list[dict], pad: dict[str, str]) -> str:
    """Handle a planning tool call in-process, mutating ``todos`` / ``pad`` in place.

    Returns a string tool result for the message history. ``todos`` is mutated via slice
    assignment so the caller's list object stays the same reference across the loop.
    """
    if name == "write_todos":
        new = _normalize_todos(args.get("todos"))
        # No-op guard: re-sending the same list signals the model is stuck re-planning
        # instead of executing. Tell it plainly to move on (and don't re-render the panel).
        if new == todos:
            return json.dumps({
                "ok": True,
                "noop": True,
                "note": "Plan unchanged — do NOT call write_todos again. Execute the "
                        "in_progress item now with a real tool (e.g. clone_starter_template / "
                        "create_flow / get_flow).",
            })
        todos[:] = new
        done = sum(1 for t in todos if t["status"] == "completed")
        # Compact result: the full plan is injected as live state every step, so echoing
        # it back would duplicate tokens. The directive keeps the model executing.
        return json.dumps({
            "ok": True,
            "count": len(todos),
            "completed": done,
            "note": "Plan saved. Proceed with the in_progress item; only call write_todos "
                    "again when an item's status actually changes.",
        })

    if name == "scratchpad_write":
        key = str(args.get("key", "")).strip()
        if not key:
            return json.dumps({"error": "key is required"})
        pad[key] = str(args.get("content", ""))
        return json.dumps({"ok": True, "key": key})

    if name == "scratchpad_read":
        key = args.get("key")
        if key:
            key = str(key).strip()
            if key in pad:
                return json.dumps({"key": key, "content": pad[key]})
            return json.dumps({"error": f"no note for key {key!r}", "keys": list(pad)})
        return json.dumps({"notes": pad})

    return json.dumps({"error": f"unknown planning tool {name!r}"})


def remember(pad: dict[str, str], key: str, content: str) -> None:
    """Code-side scratchpad write used by the loop to auto-capture determinants
    (e.g. flow_ids) without relying on the model to call scratchpad_write."""
    if key and content:
        pad[key] = str(content)
