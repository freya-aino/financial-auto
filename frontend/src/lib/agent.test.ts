import { describe, expect, it } from 'vitest';
import {
	approvalKey,
	canApprove,
	confirmText,
	embeddingNote,
	endpointLabel,
	groupTools,
	type LlmStatus
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
