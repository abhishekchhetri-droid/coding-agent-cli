"""Turn-end context management — summarize older history instead of dropping it.

The original loop trimmed history to the last two user turns plus the final assistant
message (``messages = prior_user[-2:] + [assistant]``). That reclaims tokens but throws
away determinants (flow_ids, prior decisions) and any context older than two turns — the
"long conversation loses context" problem.

:func:`summarize_history` replaces that trim: below a threshold the full history is kept
verbatim (better fidelity than the old trim); above it, the older prefix is compressed
into one summary note via the existing :class:`LLMProvider` and the recent tail is kept
verbatim. The tail boundary is snapped to a ``user`` message so a ``tool`` result is never
orphaned from its assistant ``tool_calls`` (Anthropic rejects a tool_result without its
tool_use).
"""

import json

# How many trailing messages to keep verbatim (snapped outward to a user boundary).
_KEEP_RECENT = 8

_SUMMARY_SYSTEM = (
    "You compress an AI agent's working conversation into a dense memo. The agent builds "
    "Langflow flows. Preserve every durable fact the agent will need to continue: flow_ids, "
    "flow names, component types chosen, user decisions and preferences, what was built and "
    "verified, and any open/unfinished tasks. Drop chit-chat and tool mechanics. Output a "
    "compact bullet memo, no preamble."
)


def _tail_start_index(messages: list[dict], keep_recent: int) -> int:
    """Index where the verbatim tail should begin: at most ``keep_recent`` from the end,
    snapped backward to the nearest ``user`` message so the tail never opens on a ``tool``
    result whose ``tool_calls`` parent got summarized away."""
    naive = max(0, len(messages) - keep_recent)
    for i in range(naive, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return 0


def _render_for_summary(messages: list[dict]) -> str:
    """Flatten a message slice into compact text for the summarizer prompt."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            lines.append(f"USER: {m.get('content', '')}")
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                calls = ", ".join(tc["function"]["name"] for tc in tcs)
                if m.get("content"):
                    lines.append(f"ASSISTANT: {m['content']}")
                lines.append(f"ASSISTANT→tools: {calls}")
            elif m.get("content"):
                lines.append(f"ASSISTANT: {m['content']}")
        elif role == "tool":
            content = str(m.get("content", ""))
            if len(content) > 400:
                content = content[:400] + "…"
            lines.append(f"TOOL_RESULT: {content}")
    return "\n".join(lines)


def _compact_flow_content(content: str) -> str | None:
    """If ``content`` is a flow-JSON tool result (optionally followed by a verifier verdict),
    return a compact summary that keeps the node id/type map + verdict but drops the heavy
    per-node template JSON. Returns None when content isn't a flow snapshot."""
    if not content or '"nodes"' not in content:
        return None
    # _inject_node_check appends "\n\n✅/⚠ …" after the JSON; the JSON is the first chunk.
    head = content.split("\n\n", 1)[0]
    try:
        obj = json.loads(head)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    data = obj.get("data")
    nodes = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(nodes, list):
        return None
    summary = {
        "id": obj.get("id"),
        "name": obj.get("name"),
        "node_count": len(nodes),
        "edge_count": len(data.get("edges", []) if isinstance(data, dict) else []),
        "nodes": [
            {"id": n.get("id"), "type": (n.get("data", {}) or {}).get("type") or n.get("type")}
            for n in nodes
        ],
        "_note": "full node JSON compacted to save context",
    }
    out = json.dumps(summary)
    verdict = content[len(head):].strip()
    if verdict:
        out += "\n\n" + verdict
    return out


def compact_flow_snapshots(messages: list[dict]) -> list[dict]:
    """Shrink every flow-JSON tool result in history to a node/type summary (+ verdict).

    Run at turn end so the verification ``get_flow`` (~5K) doesn't get re-sent fat on the
    next turn. Non-flow tool results are left untouched. Mutates message dicts in place and
    returns the same list."""
    for m in messages:
        if m.get("role") != "tool":
            continue
        compacted = _compact_flow_content(str(m.get("content", "")))
        if compacted is not None:
            m["content"] = compacted
    return messages


async def summarize_history(llm, messages: list[dict], settings) -> list[dict]:
    """Return a context-managed copy of ``messages`` for the next turn.

    Below ``settings.summarize_threshold_messages`` the history is returned unchanged.
    Above it, the older prefix is summarized into a single ``user`` memo and the recent
    tail is preserved verbatim. On any summarizer failure, falls back to keeping the tail
    (never raises into the turn).
    """
    threshold = getattr(settings, "summarize_threshold_messages", 30)
    if len(messages) <= threshold:
        return messages

    cut = _tail_start_index(messages, _KEEP_RECENT)
    if cut <= 0:
        return messages

    older, tail = messages[:cut], messages[cut:]
    rendered = _render_for_summary(older)
    if not rendered.strip():
        return messages

    try:
        resp = await llm.complete(
            [{"role": "user", "content": f"Compress this agent conversation:\n\n{rendered}"}],
            tools=[],
            system=_SUMMARY_SYSTEM,
        )
        summary = (resp.get("content") or "").strip()
    except Exception:
        summary = ""

    if not summary:
        # Summarizer unavailable — fall back to keeping the tail (still better-bounded
        # than the old [-2:] trim, and tool pairing stays intact via the user-snapped cut).
        return tail

    memo = {"role": "user", "content": f"[Earlier conversation summary]\n{summary}"}
    return [memo] + tail
