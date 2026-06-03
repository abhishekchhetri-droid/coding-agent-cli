"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";
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
  type Node,
  type Edge,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Slim graph streamed from the Python server's STATE_SNAPSHOT events.
type GraphField = { name: string; type?: string; value?: string | null; secret?: boolean };
type GraphNode = {
  id: string;
  position: { x: number; y: number };
  label: string;
  type?: string;
  fields?: GraphField[];
};
type GraphEdge = { id: string; source: string; target: string };
type FlowGraph = { nodes: GraphNode[]; edges: GraphEdge[] };

type FlowState = {
  flow_id?: string;
  flow_url?: string;
  graph?: FlowGraph;
};

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

type NodeData = { label: string; ctype?: string; fields?: GraphField[] };

const handleStyle: React.CSSProperties = {
  width: 9,
  height: 9,
  background: "#0f0f14",
  border: "2px solid #64748b",
};

const MAX_FIELDS = 6;

function FieldRow({ f }: { f: GraphField }) {
  const empty = f.value == null || f.value === "";
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
    </div>
  );
}

const LangflowNode = memo(({ data }: NodeProps<Node<NodeData>>) => {
  const m = nodeMeta(data.ctype, data.label);
  const fields = data.fields ?? [];
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
        overflow: "hidden",
      }}
    >
      <Handle type="target" position={Position.Left} style={handleStyle} />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "9px 12px",
          borderBottom: fields.length ? "1px solid #2c2f3a" : "none",
          borderLeft: `3px solid ${m.accent}`,
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
            <FieldRow key={`${f.name}-${i}`} f={f} />
          ))}
          {extra > 0 && (
            <div style={{ padding: "3px 12px 1px", fontSize: 10, color: "#6b7080" }}>
              +{extra} more field{extra > 1 ? "s" : ""}
            </div>
          )}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={handleStyle} />
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

function toRFNodes(graph?: FlowGraph): Node<NodeData>[] {
  return (graph?.nodes ?? []).map((n) => ({
    id: n.id,
    position: n.position,
    type: isNote(n) ? "note" : "langflow",
    data: { label: n.label, ctype: n.type, fields: n.fields },
  }));
}

function toRFEdges(graph?: FlowGraph): Edge[] {
  return (graph?.edges ?? []).map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    type: "smoothstep",
    animated: true,
    style: { stroke: "#52556a", strokeWidth: 1.6 },
  }));
}

function LiveCanvas({ graph }: { graph?: FlowGraph }) {
  const { fitView } = useReactFlow();
  const nodes = useMemo(() => toRFNodes(graph), [graph]);
  const edges = useMemo(() => toRFEdges(graph), [graph]);

  // Refit only when the SET of nodes changes (added/removed), not on every snapshot.
  const lastNodeKey = useRef<string>("");
  useEffect(() => {
    const key = nodes.map((n) => n.id).sort().join("|");
    if (key && key !== lastNodeKey.current) {
      lastNodeKey.current = key;
      requestAnimationFrame(() => fitView({ padding: 0.25, duration: 300 }));
    }
  }, [nodes, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
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

function FlowCanvas() {
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
              <LiveCanvas graph={state.graph} />
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
          <FlowCanvas />
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
