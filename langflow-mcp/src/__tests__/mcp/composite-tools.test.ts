import { describe, it, expect, vi } from 'vitest';
import {
  compositeTools,
  compositeToolNames,
  handleCompositeTool,
  resolveComponentType,
} from '../../mcp/composite-tools';

/**
 * Composite/convenience tools ported from the Python mcpbridge virtual tools.
 * These compose existing Langflow API calls server-side so any MCP client
 * (not just the coding-agent CLI) gets them. Tests exercise the dispatcher
 * with a plain mocked client — no live server.
 */

function mockClient(overrides: Record<string, any> = {}) {
  return {
    listFlows: vi.fn(),
    getFlow: vi.fn(),
    updateFlow: vi.fn(),
    buildFlow: vi.fn(),
    createFlow: vi.fn(),
    getBasicExamples: vi.fn(),
    getComponentSchemas: vi.fn(),
    ...overrides,
  };
}

describe('composite tool registration', () => {
  it('exposes the four portable tools', () => {
    const names = compositeTools.map(t => t.name).sort();
    expect(names).toEqual(
      ['delete_node', 'get_component_schema', 'get_starter_template', 'search_flows'].sort()
    );
  });

  it('compositeToolNames matches the tool definitions', () => {
    expect([...compositeToolNames].sort()).toEqual(compositeTools.map(t => t.name).sort());
  });

  it('does NOT include search_tools (client-side context manager, stays in agent)', () => {
    expect(compositeToolNames.has('search_tools')).toBe(false);
  });

  it('does NOT include clone_starter_template (depends on Layer-A enrichment, stays in agent)', () => {
    expect(compositeToolNames.has('clone_starter_template')).toBe(false);
  });
});

describe('search_flows', () => {
  it('filters flows by query in name or description and maps to {id,name,description}', async () => {
    const client = mockClient({
      listFlows: vi.fn().mockResolvedValue([
        { id: '1', name: 'RAG Agent', description: 'retrieval' },
        { id: '2', name: 'Chatbot', description: 'a rag pipeline' },
        { id: '3', name: 'Calculator', description: 'math' },
      ]),
    });
    const res = await handleCompositeTool('search_flows', { query: 'rag' }, client as any);
    expect(res).toEqual([
      { id: '1', name: 'RAG Agent', description: 'retrieval' },
      { id: '2', name: 'Chatbot', description: 'a rag pipeline' },
    ]);
  });

  it('respects the limit', async () => {
    const client = mockClient({
      listFlows: vi.fn().mockResolvedValue([
        { id: '1', name: 'a flow', description: '' },
        { id: '2', name: 'a flow', description: '' },
        { id: '3', name: 'a flow', description: '' },
      ]),
    });
    const res = await handleCompositeTool('search_flows', { query: 'flow', limit: 2 }, client as any);
    expect(res).toHaveLength(2);
  });
});

describe('get_starter_template', () => {
  it('returns the full matched example by name (case-insensitive substring)', async () => {
    const hybrid = { id: 'h1', name: 'Hybrid RAG Agent', data: { nodes: [{ id: 'n' }], edges: [] } };
    const client = mockClient({
      getBasicExamples: vi.fn().mockResolvedValue([{ id: 'x', name: 'Simple Agent' }, hybrid]),
    });
    const res = await handleCompositeTool('get_starter_template', { name_or_id: 'hybrid rag' }, client as any);
    expect(res).toEqual(hybrid);
  });

  it('returns an error object when no template matches', async () => {
    const client = mockClient({
      getBasicExamples: vi.fn().mockResolvedValue([{ id: 'x', name: 'Simple Agent' }]),
    });
    const res = await handleCompositeTool('get_starter_template', { name_or_id: 'nope' }, client as any);
    expect(res).toHaveProperty('error');
  });
});

describe('clone_starter_template is NOT served by the server', () => {
  it('throws — the agent runs the enriched clone (Layer A) itself', async () => {
    const client = mockClient();
    await expect(
      handleCompositeTool('clone_starter_template', { name_or_id: 'Simple Agent' }, client as any)
    ).rejects.toThrow(/Not a composite tool/);
  });
});

describe('delete_node', () => {
  it('drops nodes by id + dangling edges, then updates and builds', async () => {
    const client = mockClient({
      getFlow: vi.fn().mockResolvedValue({
        id: 'f1',
        data: {
          nodes: [{ id: 'keep' }, { id: 'drop' }],
          edges: [
            { id: 'e1', source: 'keep', target: 'drop' },
            { id: 'e2', source: 'keep', target: 'keep' },
          ],
        },
      }),
      updateFlow: vi.fn().mockResolvedValue({}),
      buildFlow: vi.fn().mockResolvedValue({}),
    });
    const res = await handleCompositeTool('delete_node', { flow_id: 'f1', node_ids: ['drop'] }, client as any);
    expect(res).toEqual({ flow_id: 'f1', removed_node_ids: ['drop'], removed_edge_count: 1 });
    const updateArg = client.updateFlow.mock.calls[0][1];
    expect(updateArg.data.nodes.map((n: any) => n.id)).toEqual(['keep']);
    expect(updateArg.data.edges.map((e: any) => e.id)).toEqual(['e2']);
    expect(client.buildFlow).toHaveBeenCalledWith('f1', expect.anything(), expect.anything());
  });

  it('resolves nodes by component type', async () => {
    const client = mockClient({
      getFlow: vi.fn().mockResolvedValue({
        id: 'f1',
        data: {
          nodes: [
            { id: 'calc', data: { type: 'CalculatorComponent' } },
            { id: 'chat', data: { type: 'ChatInput' } },
          ],
          edges: [],
        },
      }),
      updateFlow: vi.fn().mockResolvedValue({}),
      buildFlow: vi.fn().mockResolvedValue({}),
    });
    const res = await handleCompositeTool(
      'delete_node',
      { flow_id: 'f1', types: ['CalculatorComponent'] },
      client as any
    );
    expect(res.removed_node_ids).toEqual(['calc']);
  });

  it('no-ops cleanly when nothing matches', async () => {
    const client = mockClient({
      getFlow: vi.fn().mockResolvedValue({ id: 'f1', data: { nodes: [{ id: 'a' }], edges: [] } }),
      updateFlow: vi.fn(),
      buildFlow: vi.fn(),
    });
    const res = await handleCompositeTool('delete_node', { flow_id: 'f1', node_ids: ['ghost'] }, client as any);
    expect(res.removed_node_ids).toEqual([]);
    expect(client.updateFlow).not.toHaveBeenCalled();
  });
});

describe('get_component_schema', () => {
  const schemas = {
    SplitText: {
      template: {
        chunk_size: { type: 'int', display_name: 'Chunk Size', show: true, required: false, input_types: [] },
        secret: { type: 'str', advanced: true, show: true },
        code: { type: 'code', show: true },
      },
      outputs: [{ name: 'chunks', output_types: ['Data'], tool_mode: false }],
    },
    SQLGenerator: { legacy: true, template: {}, outputs: [] },
    SQLComponent: { legacy: false, display_name: 'SQL Database', template: {}, outputs: [] },
  };

  it('returns compact inputs/outputs, hiding advanced/code/prompt fields', async () => {
    const client = mockClient({ getComponentSchemas: vi.fn().mockResolvedValue(schemas) });
    const res = await handleCompositeTool('get_component_schema', { type_name: 'SplitText' }, client as any);
    expect(res.type).toBe('SplitText');
    expect(res.inputs.map((i: any) => i.field)).toEqual(['chunk_size']); // advanced + code excluded
    expect(res.outputs).toEqual([{ name: 'chunks', types: ['Data'], tool_mode: false }]);
  });

  it('flags legacy components and omits the flag when modern', async () => {
    const client = mockClient({ getComponentSchemas: vi.fn().mockResolvedValue(schemas) });
    expect((await handleCompositeTool('get_component_schema', { type_name: 'SQLGenerator' }, client as any)).legacy).toBe(true);
    expect(await handleCompositeTool('get_component_schema', { type_name: 'SplitText' }, client as any)).not.toHaveProperty('legacy');
  });

  it('resolves display-name variants via resolveComponentType', async () => {
    const client = mockClient({ getComponentSchemas: vi.fn().mockResolvedValue(schemas) });
    const res = await handleCompositeTool('get_component_schema', { type_name: 'SQL Database' }, client as any);
    expect(res.type).toBe('SQLComponent');
  });

  it('returns an error for unknown types', async () => {
    const client = mockClient({ getComponentSchemas: vi.fn().mockResolvedValue(schemas) });
    const res = await handleCompositeTool('get_component_schema', { type_name: 'Nope' }, client as any);
    expect(res).toHaveProperty('error');
  });
});

describe('resolveComponentType (port of Python _resolve_type)', () => {
  const schemas: Record<string, any> = {
    SQLComponent: { display_name: 'SQL Database', legacy: false },
    SQLDatabase: { display_name: 'SQLDatabase', legacy: false },
    OldThing: { display_name: 'Cool Thing', legacy: true },
    NewThing: { display_name: 'CoolThing', legacy: false },
  };

  it('exact key wins', () => {
    expect(resolveComponentType('SQLComponent', schemas)).toBe('SQLComponent');
  });

  it('disambiguates display-name collision by exact display (spaces kept)', () => {
    expect(resolveComponentType('SQL Database', schemas)).toBe('SQLComponent');
    expect(resolveComponentType('SQLDatabase', schemas)).toBe('SQLDatabase');
  });

  it('normalized display collision prefers the non-legacy twin', () => {
    expect(resolveComponentType('cool_thing', schemas)).toBe('NewThing');
  });

  it('returns the raw type unchanged when unknown', () => {
    expect(resolveComponentType('Mystery', schemas)).toBe('Mystery');
  });
});
