import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

// The agent name "langflow" must match the `agent` prop on <CopilotKit> and the
// `name` passed to useCoAgent on the frontend.
const runtime = new CopilotRuntime({
  agents: {
    langflow: new HttpAgent({
      url: process.env.AGENT_URL ?? "http://localhost:8000/agent",
    }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
