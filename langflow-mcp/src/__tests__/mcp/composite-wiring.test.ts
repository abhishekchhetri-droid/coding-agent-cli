import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the Langflow client (same pattern as server-new-tools.test.ts).
const clientMock: Record<string, ReturnType<typeof vi.fn>> = {};
vi.mock('../../services/langflow-client', () => ({
  LangflowClient: vi.fn(function (this: Record<string, unknown>) {
    Object.assign(this, clientMock);
  }),
}));

import { LangflowMCPServer } from '../../mcp/server';

function getCallToolHandler(server: LangflowMCPServer): (req: any) => Promise<any> {
  const sdkServer = (server as any).server;
  const handler = sdkServer._requestHandlers.get('tools/call');
  if (!handler) throw new Error('tools/call handler not registered');
  return (req: any) => handler(req, {});
}

function getListToolsHandler(server: LangflowMCPServer): (req: any) => Promise<any> {
  const sdkServer = (server as any).server;
  const handler = sdkServer._requestHandlers.get('tools/list');
  if (!handler) throw new Error('tools/list handler not registered');
  return (req: any) => handler(req, {});
}

async function callTool(server: LangflowMCPServer, name: string, args: Record<string, unknown>) {
  return getCallToolHandler(server)({ method: 'tools/call', params: { name, arguments: args } });
}

describe('composite tools wired into the MCP server', () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
    process.env.LANGFLOW_BASE_URL = 'http://localhost:7860';
    process.env.LANGFLOW_API_KEY = 'test-api-key-123';
    for (const key of Object.keys(clientMock)) delete clientMock[key];
    clientMock.listFlows = vi.fn().mockResolvedValue([{ id: '1', name: 'RAG flow', description: '' }]);
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.clearAllMocks();
  });

  it('lists the composite tools alongside the standard tools', async () => {
    const server = new LangflowMCPServer();
    const res = await getListToolsHandler(server)({ method: 'tools/list', params: {} });
    const names = res.tools.map((t: any) => t.name);
    for (const n of ['search_flows', 'get_component_schema', 'delete_node', 'get_starter_template']) {
      expect(names).toContain(n);
    }
    expect(names).not.toContain('clone_starter_template'); // Layer-A dependent, stays in agent
  });

  it('dispatches search_flows to the composite handler', async () => {
    const server = new LangflowMCPServer();
    const res = await callTool(server, 'search_flows', { query: 'rag' });
    expect(clientMock.listFlows).toHaveBeenCalledTimes(1);
    // formatSuccessResponse wraps the result as JSON text content
    const text = res.content[0].text;
    expect(text).toContain('RAG flow');
  });
});
