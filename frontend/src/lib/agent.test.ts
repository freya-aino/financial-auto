import { describe, expect, it } from 'vitest';
import {
	accessNote,
	approvalKey,
	canApprove,
	confirmText,
	embeddingNote,
	endpointLabel,
	groupTools,
	mcpStatusNote,
	validServerId,
	type LlmStatus,
	type McpServer
} from './agent';

const status = (over: Partial<LlmStatus> = {}): LlmStatus => ({
	configured: true,
	model: 'gpt-4o-mini',
	embedding_model: 'text-embedding-3-small',
	embedding_dim: 1536,
	endpoint: 'https://api.example.com/v1',
	key_set: true,
	trading_mode: 'paper',
	mutating_tools_available: true,
	max_rounds: 6,
	mcp_servers: 0,
	tools: [
		{ name: 'trading_risk', access: 'read', description: 'risk state' },
		{ name: 'ingest_control', access: 'mutating', description: 'start or stop a connector' },
		{ name: 'audit', access: 'read', description: 'audit log' }
	],
	...over
});

describe('approvalKey', () => {
	it('is independent of key order but not of values', () => {
		const a = approvalKey({ tool: 'ingest_control', arguments: { source: 'kraken', action: 'stop' } });
		const b = approvalKey({ tool: 'ingest_control', arguments: { action: 'stop', source: 'kraken' } });
		expect(a).toBe(b);
		expect(a).not.toBe(approvalKey({ tool: 'ingest_control', arguments: { source: 'yahoo', action: 'stop' } }));
		expect(a).not.toBe(approvalKey({ tool: 'ingest_control', arguments: { source: 'kraken', action: 'start' } }));
		expect(a).not.toBe(approvalKey({ tool: 'cancel_job', arguments: { source: 'kraken', action: 'stop' } }));
	});

	it('handles a call with no arguments', () => {
		expect(approvalKey({ tool: 'x', arguments: {} })).toBe('x()');
	});
});

describe('canApprove', () => {
	const call = { tool: 'ingest_control', arguments: { source: 'kraken', action: 'stop' } };

	it('allows an approvable mutating tool in paper mode', () => {
		expect(canApprove(status(), call)).toBe(true);
	});

	it('refuses everything in live mode', () => {
		expect(canApprove(status({ trading_mode: 'live', mutating_tools_available: false }), call)).toBe(false);
	});

	it('refuses a tool that is not in the catalogue, or is a read tool', () => {
		expect(canApprove(status(), { tool: 'not_a_tool', arguments: {} })).toBe(false);
		expect(canApprove(status(), { tool: 'trading_risk', arguments: {} })).toBe(false);
	});

	it('refuses when the status is unknown', () => {
		expect(canApprove(null, call)).toBe(false);
	});
});

describe('confirmText', () => {
	it('names the tool, every argument and the audit consequence', () => {
		const t = confirmText({ tool: 'ingest_control', arguments: { source: 'kraken', action: 'stop' } });
		expect(t).toContain('ingest_control');
		expect(t).toContain('source = kraken');
		expect(t).toContain('action = stop');
		expect(t).toContain('audit log');
	});
});

describe('groupTools', () => {
	it('splits and sorts', () => {
		const g = groupTools(status().tools);
		expect(g.read.map((t) => t.name)).toEqual(['audit', 'trading_risk']);
		expect(g.mutating.map((t) => t.name)).toEqual(['ingest_control']);
	});
});

describe('endpointLabel', () => {
	it('says when nothing is configured', () => {
		expect(endpointLabel(status({ configured: false }))).toBe('no endpoint configured');
		expect(endpointLabel(null)).toBe('unknown');
	});

	it('shows the endpoint and whether a key is set, never the key', () => {
		expect(endpointLabel(status())).toBe('https://api.example.com/v1 · gpt-4o-mini · key set');
		expect(endpointLabel(status({ key_set: false }))).toContain('no key');
	});
});

describe('embeddingNote', () => {
	it('distinguishes none, all-endpoint, mixed and all-fallback', () => {
		expect(embeddingNote(null)).toBe('no news embedded yet');
		expect(embeddingNote({ total: 0, by_provider: [], hashed_share: 0 })).toBe('no news embedded yet');
		expect(embeddingNote({ total: 10, by_provider: [], hashed_share: 0 })).toContain('all from the endpoint');
		expect(embeddingNote({ total: 10, by_provider: [], hashed_share: 1 })).toContain('all hashed fallback');
		expect(embeddingNote({ total: 10, by_provider: [], hashed_share: 0.3 })).toContain('30% hashed fallback');
	});
});

const server = (over: Partial<McpServer> = {}): McpServer => ({
	id: 'docs',
	url: 'https://host/mcp',
	label: '',
	access: 'mutating',
	enabled: true,
	token_set: false,
	status: 'ok',
	error: null,
	probed_at: '2026-11-17T09:00:00Z',
	info: 'docs 1.0',
	protocol: '2025-06-18',
	tools: [{ name: 'mcp__docs__search', remote: 'search', description: 'search', read_only_hint: true }],
	...over
});

describe('MCP servers', () => {
	it('accepts only ids that can become a tool name', () => {
		for (const ok of ['docs', 'my-server', 'a_b', 's3']) expect(validServerId(ok), ok).toBe(true);
		for (const bad of ['', 'Docs', '-x', '_x', 'has space', 'a/b', 'x'.repeat(33)])
			expect(validServerId(bad), bad).toBe(false);
	});

	it('says what a probe found, and what a disabled server means', () => {
		expect(mcpStatusNote(server())).toContain('1 tool');
		expect(mcpStatusNote(server())).toContain('docs 1.0');
		expect(mcpStatusNote(server({ enabled: false }))).toContain('not offered');
		expect(mcpStatusNote(server({ status: 'error', error: 'connection refused' }))).toContain('connection refused');
		expect(mcpStatusNote(server({ status: 'pending' }))).toContain('not probed');
	});

	it('puts the authority on the operator, not on the server', () => {
		// The point of the wording: a read classification is an assertion the operator makes.
		expect(accessNote('read', 0, 0)).toContain('You are asserting this');
		const note = accessNote('mutating', 1, 2);
		expect(note).toContain('needs your approval');
		expect(note).toContain('live mode');
		// A self-declared read-only hint is reported as a claim, never as a grant.
		expect(note).toContain('1 of 2 call themselves read-only');
	});

	it('treats an MCP tool as any other tool in the approval rule', () => {
		const s = status({
			mcp_servers: 1,
			tools: [
				{ name: 'mcp__docs__search', access: 'read', description: 'search', server: 'docs' },
				{ name: 'mcp__ops__deploy', access: 'mutating', description: 'deploy', server: 'ops' }
			]
		});
		expect(canApprove(s, { tool: 'mcp__ops__deploy', arguments: {} })).toBe(true);
		expect(canApprove(s, { tool: 'mcp__docs__search', arguments: {} })).toBe(false);
		expect(canApprove({ ...s, mutating_tools_available: false }, { tool: 'mcp__ops__deploy', arguments: {} })).toBe(
			false
		);
		expect(groupTools(s.tools).mutating.map((t) => t.name)).toEqual(['mcp__ops__deploy']);
	});
});
