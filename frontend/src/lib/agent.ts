// The LLM suite (D-117): client for the suite's control-api routes plus the pure logic the
// agent tab needs —
// grouping the tool catalogue, deciding which pending calls can be approved, and the confirmation
// text shown before a mutating call runs.
//
// Two rules live here rather than in the page, because they are what keeps the agent safe to use
// and they are worth testing:
//   * an approval is for one call with exactly those arguments, so the page can never send a
//     blanket "yes" (`approvalKey`);
//   * nothing mutating is offered while trading mode is `live` (`canApprove`).
import { delJson, postJson, putJson, q } from './api';
import { getApi } from './console';

export interface ToolInfo {
	name: string;
	access: 'read' | 'mutating';
	description: string;
	/** Set when the tool comes from an external MCP server (D-120). */
	server?: string;
}

export interface McpTool {
	name: string;
	remote: string;
	description: string;
	read_only_hint: boolean;
}

export interface McpServer {
	id: string;
	url: string;
	label: string;
	access: 'read' | 'mutating';
	enabled: boolean;
	/** The token is write-only: the API reports only whether one is stored. */
	token_set: boolean;
	status: 'pending' | 'disabled' | 'ok' | 'error';
	error: string | null;
	probed_at: string | null;
	info: string | null;
	protocol: string | null;
	tools: McpTool[];
}

export interface McpList {
	servers: McpServer[];
	protocol: string;
	max_tools_per_server: number;
	note: string;
}

export interface McpForm {
	url: string;
	label?: string;
	access?: 'read' | 'mutating';
	enabled?: boolean;
	token?: string;
	clear_token?: boolean;
}

export interface LlmStatus {
	configured: boolean;
	model: string | null;
	embedding_model: string | null;
	embedding_dim: number | null;
	endpoint: string | null;
	key_set: boolean;
	trading_mode: string;
	mutating_tools_available: boolean;
	max_rounds: number;
	tools: ToolInfo[];
	mcp_servers: number;
}

export interface PendingCall {
	tool: string;
	arguments: Record<string, unknown>;
}

export interface ToolRecord {
	tool: string;
	arguments: Record<string, unknown>;
	outcome: string;
	detail: unknown;
}

export interface TurnResponse {
	conversation_id: string;
	reply: string;
	tool_calls: ToolRecord[];
	pending: PendingCall[];
	rounds: number;
	usage: { prompt_tokens: number; completion_tokens: number };
}

export interface ConversationSummary {
	conversation_id: string;
	title: string;
	created_at: string;
	updated_at: string;
	messages: number;
}

export interface ConversationDetail {
	conversation_id: string;
	messages: { seq: number; role: string; content: string; created_at: string }[];
	tool_calls: (ToolRecord & { created_at: string })[];
}

export interface EmbeddingHealth {
	total: number;
	by_provider: { provider: string; count: number; newest: string | null }[];
	hashed_share: number;
}

export interface EndpointForm {
	base_url?: string;
	model?: string;
	embedding_model?: string;
	embedding_dim?: number;
	api_key?: string;
}

export const agentApi = {
	status: () => getApi<LlmStatus>('/api/llm/status'),
	embeddings: () => getApi<EmbeddingHealth>('/api/llm/embeddings'),
	conversations: () => getApi<{ conversations: ConversationSummary[] }>('/api/llm/conversations'),
	conversation: (id: string) => getApi<ConversationDetail>(`/api/llm/conversations/${encodeURIComponent(id)}`),
	chat: (message: string, conversation_id: string | null, approvals: PendingCall[]) =>
		postJson<TurnResponse>('/api/llm/chat', { message, conversation_id, approvals }),
	/** The key is write-only: no route returns it, so the form sends it and forgets it. */
	setEndpoint: (form: EndpointForm) =>
		putJson<{ ok: boolean; fields: string[]; restart_required: boolean; note: string }>('/api/llm/endpoint', form),
	mcp: () => getApi<McpList>('/api/llm/mcp'),
	saveMcp: (id: string, form: McpForm) =>
		putJson<{ ok: boolean; server: McpServer | null; status: string }>(
			`/api/llm/mcp/${encodeURIComponent(id)}`,
			form
		),
	removeMcp: (id: string) => delJson<{ ok: boolean; removed: number }>(`/api/llm/mcp/${encodeURIComponent(id)}`),
	probeMcp: (id: string) =>
		postJson<{ status: string; server: McpServer | null }>(`/api/llm/mcp/${encodeURIComponent(id)}/probe`, {})
};

/**
 * The same slug rule the suite and the proxy apply, checked here so the form says why before a
 * round trip. The id becomes part of a tool name (`mcp__<id>__<tool>`), which the chat API
 * restricts to `[A-Za-z0-9_-]`.
 */
export function validServerId(id: string): boolean {
	return /^[a-z0-9][a-z0-9_-]{0,31}$/.test(id);
}

/** One line per MCP server: what it is, whether it answered, and how many tools it brought. */
export function mcpStatusNote(s: McpServer): string {
	if (!s.enabled) return 'disabled — its tools are not offered to the agent';
	switch (s.status) {
		case 'ok':
			return `${s.tools.length} tool${s.tools.length === 1 ? '' : 's'}${s.info ? ` · ${s.info}` : ''}`;
		case 'error':
			return `unreachable: ${s.error ?? 'unknown error'}`;
		default:
			return 'not probed yet';
	}
}

/**
 * What approving a call on this server means, shown next to the access control.
 *
 * `read` is the operator asserting the server only reads; the server's own `readOnlyHint` is
 * reported per tool and deliberately not used to decide this.
 */
export function accessNote(access: 'read' | 'mutating', hinted: number, total: number): string {
	if (access === 'read') {
		return 'read: the agent may call these without asking. You are asserting this, not the server.';
	}
	const hint = total > 0 ? ` ${hinted} of ${total} call themselves read-only.` : '';
	return `mutating: every call needs your approval and none is possible in live mode.${hint}`;
}

/** Identity of one approvable call: tool name plus its arguments, key-order independent. */
export function approvalKey(call: PendingCall): string {
	const args = call.arguments ?? {};
	const sorted = Object.keys(args)
		.sort()
		.map((k) => `${k}=${JSON.stringify(args[k])}`)
		.join('&');
	return `${call.tool}(${sorted})`;
}

/** Whether this pending call may be approved at all. Live mode removes mutating tools entirely. */
export function canApprove(status: LlmStatus | null, call: PendingCall): boolean {
	if (!status) return false;
	if (!status.mutating_tools_available) return false;
	return status.tools.some((t) => t.name === call.tool && t.access === 'mutating');
}

/** What the operator is asked to confirm, naming the tool and every argument. */
export function confirmText(call: PendingCall): string {
	const args = Object.entries(call.arguments ?? {})
		.map(([k, v]) => `${k} = ${typeof v === 'string' ? v : JSON.stringify(v)}`)
		.join(', ');
	return `Run \`${call.tool}\`${args ? ` with ${args}` : ''}? This changes the running platform and is written to the audit log.`;
}

/** Tools split into the two groups the panel shows, each sorted by name. */
export function groupTools(tools: ToolInfo[]): { read: ToolInfo[]; mutating: ToolInfo[] } {
	const by = (a: ToolInfo, b: ToolInfo) => a.name.localeCompare(b.name);
	return {
		read: tools.filter((t) => t.access === 'read').sort(by),
		mutating: tools.filter((t) => t.access === 'mutating').sort(by)
	};
}

/** One line describing the endpoint state, for the header. */
export function endpointLabel(s: LlmStatus | null): string {
	if (!s) return 'unknown';
	if (!s.configured) return 'no endpoint configured';
	const key = s.key_set ? 'key set' : 'no key';
	return `${s.endpoint} · ${s.model} · ${key}`;
}

/** How to read the embedding mix: a high fallback share means the text expert is not reading the model's vectors. */
export function embeddingNote(h: EmbeddingHealth | null): string {
	if (!h || h.total === 0) return 'no news embedded yet';
	const pct = Math.round(h.hashed_share * 100);
	if (pct === 0) return `${h.total.toLocaleString()} items, all from the endpoint`;
	if (pct === 100) return `${h.total.toLocaleString()} items, all hashed fallback — no endpoint reached`;
	return `${h.total.toLocaleString()} items, ${pct}% hashed fallback`;
}

export { q };
