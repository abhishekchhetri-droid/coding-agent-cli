"use client";

import { CopilotKit, useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";

// Shared agent state mirrored from the Python server's STATE_SNAPSHOT events.
type FlowState = {
  flow_id?: string;
  flow_url?: string;
};

function FlowCanvas() {
  const { state, running } = useCoAgent<FlowState>({
    name: "langflow",
    initialState: {},
  });

  if (!state?.flow_url) {
    return (
      <div style={styles.placeholder}>
        <div style={{ fontSize: 48, marginBottom: 16 }}>🔧</div>
        <h2 style={{ margin: "0 0 8px" }}>No flow yet</h2>
        <p style={{ maxWidth: 360, textAlign: "center", color: "#666" }}>
          Describe the flow you want in the chat — e.g.{" "}
          <em>“build me a research agent with web search”</em>. It will appear
          here as the agent builds it.
        </p>
        {running && <p style={{ color: "#0a7" }}>Working…</p>}
      </div>
    );
  }

  return (
    <div style={styles.canvasWrap}>
      <div style={styles.canvasBar}>
        <span style={{ fontWeight: 600 }}>Flow</span>
        <code style={{ fontSize: 12, color: "#888" }}>{state.flow_id}</code>
        <a
          href={state.flow_url}
          target="_blank"
          rel="noreferrer"
          style={styles.openLink}
        >
          Open in Langflow ↗
        </a>
      </div>
      <iframe
        // key forces a reload when the agent rebuilds / switches the flow
        key={state.flow_url}
        src={state.flow_url}
        style={styles.iframe}
        title="Langflow flow editor"
      />
    </div>
  );
}

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="langflow">
      <main style={styles.main}>
        <section style={styles.chatPane}>
          <CopilotChat
            labels={{
              title: "Flow Builder",
              initial:
                "Hi! Tell me what flow to build and watch it render on the right.",
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
  canvasPane: { flex: 1, height: "100%", background: "#fafafa" },
  canvasWrap: { display: "flex", flexDirection: "column", height: "100%" },
  canvasBar: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "8px 16px",
    borderBottom: "1px solid #e5e5e5",
    background: "#fff",
  },
  openLink: { marginLeft: "auto", fontSize: 13, color: "#0a7", textDecoration: "none" },
  iframe: { flex: 1, width: "100%", border: "none" },
  placeholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    color: "#444",
  },
};
