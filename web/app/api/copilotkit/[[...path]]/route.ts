import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

const runtime = new CopilotRuntime({
  agents: {
    langflow: new HttpAgent({
      url: process.env.AGENT_URL ?? "http://localhost:8000/agent",
    }),
  },
});

function makeHandler(req: NextRequest) {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
}

export const GET = (req: NextRequest) => {
  const { pathname } = new URL(req.url);
  // Single-route mode only handles POST; return empty threads list so the UI doesn't 405.
  if (pathname.endsWith("/threads")) {
    return Response.json({ threads: [], nextCursor: null });
  }
  return makeHandler(req);
};

export const POST = (req: NextRequest) => makeHandler(req);
