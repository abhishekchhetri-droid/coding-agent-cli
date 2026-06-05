"use client";

import { memo, useCallback, useEffect, useRef, useState } from "react";
import { CopilotKit, useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  useReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  type Node,
  type Edge,
  type NodeProps,
  type Connection,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Slim graph streamed from the Python server's STATE_SNAPSHOT events.
type GraphField = { key?: string; name: string; type?: string; value?: string | null; secret?: boolean };
type GraphInput = { key: string; name?: string; input_types?: string[] };
type GraphOutput = { name: string; display?: string; types?: string[] };
type GraphNode = {
  id: string;
  position: { x: number; y: number };
  label: string;
  type?: string;
  fields?: GraphField[];
  inputs?: GraphInput[];
  outputs?: GraphOutput[];
};
type GraphEdge = {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};
type FlowGraph = { nodes: GraphNode[]; edges: GraphEdge[] };

type FlowState = {
  flow_id?: string;
  flow_url?: string;
  graph?: FlowGraph;
};

// One canvas-edit op sent to /api/canvas (persisted to Langflow, bypassing the LLM).
type CanvasOp =
  | { op: "move"; id: string; position: { x: number; y: number } }
  | { op: "edit_field"; id: string; key: string; value: string }
  | { op: "add_edge"; source: string; target: string; output?: string | null; field?: string | null }
  | { op: "delete_edge"; id: string }
  | { op: "delete_node"; id: string };

const THREAD_KEY = "nokia-flow-thread-id";

function getOrCreateThreadId(): string {
  if (typeof window === "undefined") return "";
  const existing = localStorage.getItem(THREAD_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(THREAD_KEY, id);
  return id;
}

// Map a Langflow component type to an icon, accent color and category label.
function nodeMeta(type?: string, label?: string) {
  const t = `${type ?? ""} ${label ?? ""}`.toLowerCase();
  if (t.includes("chatinput")) return { icon: "💬", accent: "#22c55e", kind: "Input" };
  if (t.includes("chatoutput")) return { icon: "💬", accent: "#3b82f6", kind: "Output" };
  if (t.includes("agent")) return { icon: "🧠", accent: "#a855f7", kind: "Agent" };
  if (t.includes("calculator")) return { icon: "🧮", accent: "#f59e0b", kind: "Tool" };
  if (t.includes("url")) return { icon: "🌐", accent: "#06b6d4", kind: "Tool" };
  if (t.includes("azure") || t.includes("openai") || t.includes("model") || t.includes("llm"))
    return { icon: "✨", accent: "#14b8a6", kind: "Model" };
  if (t.includes("prompt")) return { icon: "📋", accent: "#ec4899", kind: "Prompt" };
  if (t.includes("note")) return { icon: "📝", accent: "#eab308", kind: "Note" };
  return { icon: "⬡", accent: "#64748b", kind: "Component" };
}

// Callback a node uses to push a field edit up to the canvas (debounced → POST).
type EditFieldFn = (nodeId: string, key: string, value: string) => void;

type NodeData = {
  label: string;
  ctype?: string;
  fields?: GraphField[];
  inputs?: GraphInput[];
  outputs?: GraphOutput[];
  onEditField?: EditFieldFn;
  nodeId: string;
};

const handleStyle: React.CSSProperties = {
  width: 9,
  height: 9,
  background: "#0f0f14",
  border: "2px solid #64748b",
};

const MAX_FIELDS = 6;

// Field types we let the user edit inline; everything else (tables, secrets, blobs) stays
// read-only so a truncated display value is never written back.
const EDITABLE_TYPES = new Set([
  "str", "Text", "text", "int", "integer", "float", "number", "Message", "message", "prompt",
]);

function isEditable(f: GraphField): boolean {
  if (f.secret || !f.key) return false;
  return !f.type || EDITABLE_TYPES.has(f.type);
}

function FieldRow({ f, nodeId, onEdit }: { f: GraphField; nodeId: string; onEdit?: EditFieldFn }) {
  const empty = f.value == null || f.value === "";
  const editable = Boolean(onEdit) && isEditable(f);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 10,
        padding: "5px 12px",
      }}
    >
      <span style={{ fontSize: 11, color: "#9aa0ad", whiteSpace: "nowrap" }}>{f.name}</span>
      {editable ? (
        <input
          className="nodrag"
          defaultValue={empty ? "" : (f.value as string)}
          placeholder="—"
          onMouseDown={(e) => e.stopPropagation()}
          onChange={(e) => onEdit!(nodeId, f.key!, e.target.value)}
          style={{
            fontSize: 11,
            width: 130,
            padding: "2px 8px",
            borderRadius: 6,
            background: "#0f1117",
            border: "1px solid #2c2f3a",
            color: "#d7dae2",
            outline: "none",
          }}
        />
      ) : (
        <span
          style={{
            fontSize: 11,
            maxWidth: 130,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            padding: "2px 8px",
            borderRadius: 6,
            background: "#0f1117",
            border: "1px solid #2c2f3a",
            color: empty ? "#5b606e" : f.secret ? "#c4b5fd" : "#d7dae2",
            fontFamily: f.secret ? "monospace" : undefined,
          }}
        >
          {empty ? "—" : f.value}
        </span>
      )}
    </div>
  );
}

// Spread N handles evenly down a node edge so each input/output is a distinct drop target.
function handleTop(i: number, n: number): string {
  return `${((i + 1) / (n + 1)) * 100}%`;
}

const LangflowNode = memo(({ data }: NodeProps<Node<NodeData>>) => {
  const m = nodeMeta(data.ctype, data.label);
  const fields = data.fields ?? [];
  const inputs = data.inputs ?? [];
  const outputs = data.outputs ?? [];
  const shown = fields.slice(0, MAX_FIELDS);
  const extra = fields.length - shown.length;
  return (
    <div
      style={{
        width: 230,
        borderRadius: 12,
        background: "#191b22",
        border: "1px solid #2c2f3a",
        boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
        color: "#e5e7eb",
        overflow: "visible",
        position: "relative",
      }}
    >
      {/* Target handles (left) — one per connectable input field, or a fallback. */}
      {inputs.length > 0 ? (
        inputs.map((inp, i) => (
          <Handle
            key={inp.key}
            type="target"
            position={Position.Left}
            id={inp.key}
            title={inp.name || inp.key}
            style={{ ...handleStyle, top: handleTop(i, inputs.length) }}
          />
        ))
      ) : (
        <Handle type="target" position={Position.Left} style={handleStyle} />
      )}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "9px 12px",
          borderBottom: fields.length ? "1px solid #2c2f3a" : "none",
          borderLeft: `3px solid ${m.accent}`,
          borderTopLeftRadius: 12,
          borderTopRightRadius: 12,
        }}
      >
        <span
          style={{
            display: "grid",
            placeItems: "center",
            width: 24,
            height: 24,
            borderRadius: 7,
            fontSize: 14,
            background: `${m.accent}22`,
          }}
        >
          {m.icon}
        </span>
        <div style={{ lineHeight: 1.15, minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 600, whiteSpace: "nowrap" }}>{data.label}</div>
          <div style={{ fontSize: 10, color: "#8b8f9c", textTransform: "uppercase", letterSpacing: 0.4 }}>
            {m.kind}
          </div>
        </div>
      </div>
      {fields.length > 0 && (
        <div style={{ padding: "6px 0" }}>
          {shown.map((f, i) => (
            <FieldRow key={f.key ?? `${f.name}-${i}`} f={f} nodeId={data.nodeId} onEdit={data.onEditField} />
          ))}
          {extra > 0 && (
            <div style={{ padding: "3px 12px 1px", fontSize: 10, color: "#6b7080" }}>
              +{extra} more field{extra > 1 ? "s" : ""}
            </div>
          )}
        </div>
      )}
      {/* Source handles (right) — one per output, or a fallback. */}
      {outputs.length > 0 ? (
        outputs.map((out, i) => (
          <Handle
            key={out.name}
            type="source"
            position={Position.Right}
            id={out.name}
            title={out.display || out.name}
            style={{ ...handleStyle, top: handleTop(i, outputs.length) }}
          />
        ))
      ) : (
        <Handle type="source" position={Position.Right} style={handleStyle} />
      )}
    </div>
  );
});
LangflowNode.displayName = "LangflowNode";

const NoteNode = memo(({ data }: NodeProps<Node<NodeData>>) => (
  <div
    style={{
      maxWidth: 220,
      padding: "10px 12px",
      borderRadius: 8,
      background: "#3a341b",
      border: "1px solid #6b5e2a",
      color: "#f5e9b8",
      fontSize: 12,
      lineHeight: 1.3,
    }}
  >
    📝 {data.label}
  </div>
));
NoteNode.displayName = "NoteNode";

const nodeTypes = { langflow: LangflowNode, note: NoteNode };

function isNote(n: GraphNode): boolean {
  const t = `${n.type ?? ""} ${n.label ?? ""}`.toLowerCase();
  return t.includes("note");
}

function toRFNodes(graph: FlowGraph | undefined, onEditField: EditFieldFn): Node<NodeData>[] {
  return (graph?.nodes ?? []).map((n) => ({
    id: n.id,
    position: n.position,
    type: isNote(n) ? "note" : "langflow",
    data: {
      label: n.label,
      ctype: n.type,
      fields: n.fields,
      inputs: n.inputs,
      outputs: n.outputs,
      onEditField,
      nodeId: n.id,
    },
  }));
}

function toRFEdges(graph?: FlowGraph): Edge[] {
  return (graph?.edges ?? []).map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle ?? undefined,
    targetHandle: e.targetHandle ?? undefined,
    type: "smoothstep",
    animated: true,
    style: { stroke: "#52556a", strokeWidth: 1.6 },
  }));
}

function LiveCanvas({ graph, threadId }: { graph?: FlowGraph; threadId: string }) {
  const { fitView } = useReactFlow();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<NodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // While a mutation is in flight, ignore inbound snapshots so optimistic edits aren't
  // clobbered; sigRef skips re-applying an unchanged streamed graph (breaks edit→POST→
  // snapshot→re-merge loops).
  const pendingRef = useRef(0);
  const sigRef = useRef("");
  const fieldTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fieldBuf = useRef<Map<string, CanvasOp>>(new Map());

  // POST canvas ops; apply the authoritative slim_graph the server returns.
  const mutate = useCallback(
    async (ops: CanvasOp[]) => {
      if (!threadId || ops.length === 0) return;
      pendingRef.current++;
      try {
        const res = await fetch("/api/canvas", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ thread_id: threadId, ops }),
        });
        const json = await res.json();
        if (json?.graph) {
          setNodes(toRFNodes(json.graph, onEditFieldRef.current));
          setEdges(toRFEdges(json.graph));
        }
      } catch {
        // Network/persist error: leave optimistic local state; next snapshot reconciles.
      } finally {
        pendingRef.current--;
      }
    },
    [threadId, setNodes, setEdges],
  );

  // Field edits: optimistic input keeps the typed value (uncontrolled); coalesce per
  // node+key and flush ~500ms after typing stops as a single mutate.
  const onEditField = useCallback<EditFieldFn>(
    (nodeId, key, value) => {
      fieldBuf.current.set(`${nodeId}::${key}`, { op: "edit_field", id: nodeId, key, value });
      if (fieldTimer.current) clearTimeout(fieldTimer.current);
      fieldTimer.current = setTimeout(() => {
        const ops = Array.from(fieldBuf.current.values());
        fieldBuf.current.clear();
        void mutate(ops);
      }, 500);
    },
    [mutate],
  );
  // Stable ref so rebuilding nodes from a snapshot doesn't need mutate in its deps.
  const onEditFieldRef = useRef(onEditField);
  useEffect(() => {
    onEditFieldRef.current = onEditField;
  }, [onEditField]);

  // Reconcile inbound streamed graph → local interactive state.
  useEffect(() => {
    if (pendingRef.current > 0) return;
    const sig = JSON.stringify(graph ?? null);
    if (sig === sigRef.current) return;
    sigRef.current = sig;
    setNodes(toRFNodes(graph, onEditFieldRef.current));
    setEdges(toRFEdges(graph));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph]);

  // Refit only when the SET of nodes changes (added/removed), not on drags/edits.
  const lastNodeKey = useRef<string>("");
  useEffect(() => {
    const key = nodes.map((n) => n.id).sort().join("|");
    if (key && key !== lastNodeKey.current) {
      lastNodeKey.current = key;
      requestAnimationFrame(() => fitView({ padding: 0.25, duration: 300 }));
    }
  }, [nodes, fitView]);

  const onNodeDragStop = useCallback(
    (_: unknown, node: Node) => {
      void mutate([{ op: "move", id: node.id, position: node.position }]);
    },
    [mutate],
  );

  const onConnect = useCallback(
    (conn: Connection) => {
      setEdges((eds) =>
        addEdge(
          { ...conn, type: "smoothstep", animated: true, style: { stroke: "#52556a", strokeWidth: 1.6 } },
          eds,
        ),
      );
      void mutate([
        {
          op: "add_edge",
          source: conn.source!,
          target: conn.target!,
          output: conn.sourceHandle,
          field: conn.targetHandle,
        },
      ]);
    },
    [mutate, setEdges],
  );

  const onNodesDelete = useCallback(
    (deleted: Node[]) => {
      void mutate(deleted.map((n) => ({ op: "delete_node", id: n.id })));
    },
    [mutate],
  );

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      void mutate(deleted.map((e) => ({ op: "delete_edge", id: e.id })));
    },
    [mutate],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onNodeDragStop={onNodeDragStop}
      onNodesDelete={onNodesDelete}
      onEdgesDelete={onEdgesDelete}
      nodesDraggable
      nodesConnectable
      elementsSelectable
      deleteKeyCode={["Backspace", "Delete"]}
      fitView
      minZoom={0.1}
      proOptions={{ hideAttribution: true }}
      style={{ background: "#0f0f14" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#2a2d3a" />
      <Controls
        style={{
          // dark-ify the default controls
          // @ts-expect-error CSS var for control button colors
          "--xy-controls-button-background-color-default": "#191b22",
          "--xy-controls-button-color-default": "#e5e7eb",
          "--xy-controls-button-border-color-default": "#2c2f3a",
          "--xy-controls-button-background-color-hover-default": "#2c2f3a",
        }}
      />
    </ReactFlow>
  );
}

function FlowCanvas({ threadId }: { threadId: string }) {
  const { state, running } = useCoAgent<FlowState>({ name: "langflow", initialState: {} });
  const hasFlow = Boolean(state.flow_id);
  const hasGraph = Boolean(state.graph?.nodes?.length);

  return (
    <div style={styles.canvasWrap}>
      {!hasFlow && (
        <div style={styles.placeholder}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🔧</div>
          <h2 style={{ margin: "0 0 8px" }}>No flow yet</h2>
          <p style={{ maxWidth: 360, textAlign: "center", color: "#666" }}>
            Describe the flow you want in the chat — e.g.{" "}
            <em>"build me a research agent with web search"</em>. It will appear here as the
            agent builds it.
          </p>
          {running && <p style={{ color: "#0a7" }}>Working…</p>}
        </div>
      )}
      {hasFlow && (
        <div style={styles.canvasBar}>
          <span style={{ fontWeight: 600 }}>Flow</span>
          <code style={{ fontSize: 12, color: "#888" }}>{state.flow_id}</code>
          {running && <span style={{ fontSize: 12, color: "#0a7" }}>Working…</span>}
          {state.flow_url && (
            <a href={state.flow_url} target="_blank" rel="noreferrer" style={styles.openLink}>
              Open in Langflow ↗
            </a>
          )}
        </div>
      )}
      {hasFlow && (
        <div style={{ flex: 1, minHeight: 0 }}>
          {hasGraph ? (
            <ReactFlowProvider>
              <LiveCanvas graph={state.graph} threadId={threadId} />
            </ReactFlowProvider>
          ) : (
            <div style={styles.placeholder}>
              <p style={{ color: "#666" }}>Loading flow…</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [threadId, setThreadId] = useState("");
  useEffect(() => { setThreadId(getOrCreateThreadId()); }, []);

  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="langflow" threadId={threadId || undefined}>
      <main style={styles.main}>
        <section style={styles.chatPane}>
          <CopilotChat
            labels={{
              title: "Flow Builder",
              initial: "Hi! Tell me what flow to build and watch it render on the right.",
            }}
            className="nokia-chat"
          />
        </section>
        <section style={styles.canvasPane}>
          <FlowCanvas threadId={threadId} />
        </section>
      </main>
    </CopilotKit>
  );
}

const styles: Record<string, React.CSSProperties> = {
  main: { display: "flex", height: "100vh", width: "100vw" },
  chatPane: {
    width: 420,
    minWidth: 320,
    borderRight: "1px solid #e5e5e5",
    height: "100%",
    overflow: "hidden",
  },
  canvasPane: { flex: 1, height: "100%", background: "#0f0f14" },
  canvasWrap: { display: "flex", flexDirection: "column", height: "100%" },
  canvasBar: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "8px 16px",
    borderBottom: "1px solid #2c2f3a",
    background: "#15171d",
    color: "#e5e7eb",
  },
  openLink: { marginLeft: "auto", fontSize: 13, color: "#34d399", textDecoration: "none" },
  placeholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    color: "#444",
  },
};
