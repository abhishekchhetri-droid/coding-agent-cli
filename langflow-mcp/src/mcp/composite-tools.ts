import { z } from 'zod';
import { ToolDefinition } from '../types';
import { LangflowClient } from '../services/langflow-client';
import { logger } from '../utils/logger';

/**
 * Composite / convenience tools.
 *
 * Ported from the Python `mcpbridge` virtual tools so any MCP client — not only the
 * coding-agent CLI — gets them. Each one composes existing Langflow API calls server-side.
 * Kept in a dedicated module (not folded into the generic `tools.ts` passthrough list) so
 * langflow-mcp's thin-wrapper identity stays clear and these opinionated helpers are easy to
 * track separately.
 *
 * NOTE: `search_tools` is intentionally NOT here — it is a client-side context manager
 * (active-tool-window + prompt-cache partitioning) and has no meaning on the server.
 *
 * NOTE: `clone_starter_template` is also NOT here. The agent's clone is not a bare clone — it
 * runs the full Layer-A enrichment pipeline (schema injection, AzureOpenAIModel wiring, Agent
 * injection, edge enrichment) before saving. That logic stays in the Python agent, so a faithful
 * server port would require porting Layer A. The 4 tools below are the genuinely portable ones.
 */

// ── Validation schemas ───────────────────────────────────────────────────────

export const SearchFlowsSchema = z.object({
  query: z.string(),
  limit: z.number().int().positive().default(15),
});

export const GetStarterTemplateSchema = z.object({
  name_or_id: z.string(),
});

export const DeleteNodeSchema = z.object({
  flow_id: z.string(),
  node_ids: z.array(z.string()).optional(),
  types: z.array(z.string()).optional(),
});

export const GetComponentSchemaSchema = z.object({
  type_name: z.string(),
});

// ── Tool definitions ─────────────────────────────────────────────────────────

export const compositeTools: ToolDefinition[] = [
  {
    name: 'search_flows',
    description:
      'Search user flows by keyword. Returns [{id, name, description}] for matching flows. ' +
      'Use instead of list_flows when looking for a specific flow by name or topic.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Keyword to search flow names and descriptions' },
        limit: { type: 'integer', description: 'Max results (default 15)' },
      },
      required: ['query'],
    },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: true },
  },
  {
    name: 'get_component_schema',
    description:
      'Get exact input field names and output handle names for a specific component type. ' +
      'Call this for ANY component before building edges to/from it. Prevents invalid connections.',
    inputSchema: {
      type: 'object',
      properties: {
        type_name: {
          type: 'string',
          description: "Exact component type string (e.g. 'SplitText', 'Chroma', 'AzureOpenAIEmbeddings')",
        },
      },
      required: ['type_name'],
    },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: true },
  },
  {
    name: 'delete_node',
    description:
      'Remove one or more nodes from a flow in a single round-trip. Fetches the flow, drops ' +
      'matched nodes + any edges referencing them, PATCHes the result, then rebuilds. Use this ' +
      "for 'remove X' / 'delete X' requests instead of update_flow (whose merge can only ADD). " +
      'Pass node_ids (exact IDs from get_flow) OR types; type→ID resolution happens server-side. ' +
      'Returns {flow_id, removed_node_ids, removed_edge_count}.',
    inputSchema: {
      type: 'object',
      properties: {
        flow_id: { type: 'string', description: 'Flow UUID' },
        node_ids: { type: 'array', items: { type: 'string' }, description: 'Exact node IDs to delete' },
        types: { type: 'array', items: { type: 'string' }, description: 'Component types — every node of these types is removed' },
      },
      required: ['flow_id'],
    },
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true },
  },
  {
    name: 'get_starter_template',
    description:
      'Get full nodes[] and edges[] for ONE specific starter template by name or id. Call this ' +
      'AFTER scoring templates from list_starter_projects to fetch the winning template’s full ' +
      'data. Cheaper than re-calling list_starter_projects — returns only the one template.',
    inputSchema: {
      type: 'object',
      properties: {
        name_or_id: { type: 'string', description: "Template name (e.g. 'Hybrid RAG Agent') or id" },
      },
      required: ['name_or_id'],
    },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: true },
  },
];

export const compositeToolNames = new Set(compositeTools.map(t => t.name));

// ── Type resolution (port of Python LangflowMCPClient._resolve_type) ───────────

/**
 * Resolve a type string to a canonical schema key against the live /api/v1/all schema.
 * exact → case-insensitive key → exact display_name → normalized display_name → prefix → raw.
 * On a normalized-display collision the non-legacy twin wins (deterministic). No static aliases.
 */
export function resolveComponentType(rawType: string, schemas: Record<string, any>): string {
  if (rawType in schemas) return rawType;
  const lower = rawType.toLowerCase();

  const lowerIndex: Record<string, string> = {};
  const displayExact: Record<string, string> = {};
  const displayNorm: Record<string, string> = {};

  const prefer = (index: Record<string, string>, k: string, key: string, schema: any) => {
    const prev = index[k];
    if (prev === undefined || (Boolean(schemas[prev]?.legacy) && !Boolean(schema?.legacy))) {
      index[k] = key;
    }
  };

  for (const [key, schema] of Object.entries(schemas)) {
    lowerIndex[key.toLowerCase()] = key;
    const dn: string = (schema as any)?.display_name || '';
    if (dn) {
      prefer(displayExact, dn.toLowerCase(), key, schema);
      prefer(displayNorm, dn.toLowerCase().replace(/ /g, ''), key, schema);
    }
  }

  if (lower in lowerIndex) return lowerIndex[lower];
  if (lower in displayExact) return displayExact[lower];
  const normalized = lower.replace(/ /g, '').replace(/_/g, '').replace(/-/g, '');
  if (normalized in displayNorm) return displayNorm[normalized];

  const candidates = Object.entries(lowerIndex)
    .filter(([lk]) => lk.startsWith(lower) || lower.startsWith(lk))
    .map(([, key]) => key);
  if (candidates.length === 1) return candidates[0];
  return rawType;
}

// ── Per-tool handlers ──────────────────────────────────────────────────────────

function findStarter(examples: any[], nameOrId: string): any | undefined {
  const q = nameOrId.trim().toLowerCase();
  return examples.find(
    s => s?.id === nameOrId.trim() || (s?.name || '').toLowerCase().includes(q)
  );
}

async function searchFlows(args: z.infer<typeof SearchFlowsSchema>, client: LangflowClient): Promise<any[]> {
  const flows = await client.listFlows();
  const q = args.query.toLowerCase();
  const filtered = (flows || [])
    .filter((f: any) => (f.name || '').toLowerCase().includes(q) || (f.description || '').toLowerCase().includes(q))
    .map((f: any) => ({ id: f.id, name: f.name || '', description: f.description || '' }));
  return filtered.slice(0, args.limit);
}

async function getStarterTemplate(args: z.infer<typeof GetStarterTemplateSchema>, client: LangflowClient): Promise<any> {
  const examples = await client.getBasicExamples();
  const match = findStarter(examples || [], args.name_or_id);
  if (!match) {
    return { error: `No starter template matched ${JSON.stringify(args.name_or_id)}.` };
  }
  return match;
}

async function deleteNode(args: z.infer<typeof DeleteNodeSchema>, client: LangflowClient): Promise<any> {
  const nodeIds = new Set<string>(args.node_ids || []);
  const types = new Set<string>(args.types || []);
  if (nodeIds.size === 0 && types.size === 0) {
    return { error: 'flow_id and (node_ids or types) required' };
  }

  const flow: any = await client.getFlow(args.flow_id);
  const data = flow?.data || {};
  const nodes: any[] = data.nodes || [];
  const edges: any[] = data.edges || [];

  if (types.size > 0) {
    for (const n of nodes) {
      const t = n?.data?.type || n?.type || '';
      if (types.has(t) && n?.id) nodeIds.add(n.id);
    }
  }

  // Only act on ids that actually exist in the flow — a delete of absent ids must not
  // trigger a pointless PATCH + rebuild, nor report phantom removals.
  const presentIds = new Set([...nodeIds].filter(id => nodes.some(n => n?.id === id)));
  if (presentIds.size === 0) {
    return { flow_id: args.flow_id, removed_node_ids: [], removed_edge_count: 0, note: 'no matching nodes' };
  }
  nodeIds.clear();
  for (const id of presentIds) nodeIds.add(id);

  const keptNodes = nodes.filter(n => !nodeIds.has(n?.id));
  const keptEdges = edges.filter(e => !nodeIds.has(e?.source) && !nodeIds.has(e?.target));
  const removedEdgeCount = edges.length - keptEdges.length;

  const newData = { ...data, nodes: keptNodes, edges: keptEdges };
  await client.updateFlow(args.flow_id, { data: newData } as any);
  // Rebuild so Langflow invalidates its canvas cache and the UI reflects the removal.
  await client.buildFlow(args.flow_id, {} as any, {} as any);

  return {
    flow_id: args.flow_id,
    removed_node_ids: [...nodeIds].sort(),
    removed_edge_count: removedEdgeCount,
  };
}

async function getComponentSchema(args: z.infer<typeof GetComponentSchemaSchema>, client: LangflowClient): Promise<any> {
  const schemas = await client.getComponentSchemas();
  const resolved = resolveComponentType(args.type_name, schemas);
  if (!(resolved in schemas)) {
    return { error: `Unknown type: ${JSON.stringify(args.type_name)}. Call list_components to find the exact type string.` };
  }
  const schema = schemas[resolved];
  const tmpl = schema.template || {};
  const inputs = Object.entries(tmpl)
    .filter(
      ([, v]: [string, any]) =>
        v && typeof v === 'object' && !v.advanced && (v.show ?? true) && v.type !== 'code' && v.type !== 'prompt'
    )
    .map(([k, v]: [string, any]) => ({
      field: k,
      type: v.type || '',
      display: v.display_name || k,
      required: v.required || false,
      input_types: v.input_types || [],
    }));
  const outputs = (schema.outputs || []).map((o: any) => ({
    name: o.name,
    types: o.output_types || [],
    tool_mode: o.tool_mode || false,
  }));
  const result: any = { type: resolved, inputs, outputs };
  if (schema.legacy) result.legacy = true;
  if (schema.beta) result.beta = true;
  return result;
}

// ── Dispatcher ───────────────────────────────────────────────────────────────

/**
 * Dispatch a composite tool by name. Caller should gate on `compositeToolNames.has(name)`.
 * Returns the raw result object; the server wraps it via formatSuccessResponse.
 */
export async function handleCompositeTool(name: string, args: unknown, client: LangflowClient): Promise<any> {
  // Origin marker: lets you confirm a call was served by langflow-mcp (this server) and not by
  // the coding-agent's in-process Python virtual tool. Visible when LOG_LEVEL permits (e.g. via
  // MCP Inspector); suppressed under the agent, which runs the server at LOG_LEVEL=error.
  logger.info(`[composite-tool] served by langflow-mcp: ${name}`);
  switch (name) {
    case 'search_flows':
      return searchFlows(SearchFlowsSchema.parse(args), client);
    case 'get_starter_template':
      return getStarterTemplate(GetStarterTemplateSchema.parse(args), client);
    case 'delete_node':
      return deleteNode(DeleteNodeSchema.parse(args), client);
    case 'get_component_schema':
      return getComponentSchema(GetComponentSchemaSchema.parse(args), client);
    default:
      throw new Error(`Not a composite tool: ${name}`);
  }
}
