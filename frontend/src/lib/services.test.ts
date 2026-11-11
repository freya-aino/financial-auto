import { describe, expect, it } from 'vitest';
import { actionBody, actionGate, confirmText, describeResult, groupCounters, primitiveShares } from './services';

describe('heartbeat counter grouping', () => {
	it('groups gauge prefixes, folds per-item families and keeps the rest', () => {
		const g = groupCounters({
			'primitives.graph': 10,
			'primitives.fallback': 5,
			'hub.writes': 3,
			'strategy.s1.mean': 0.1,
			'strategy.s2.mean': 0.2,
			'trades.md.trade.x': 7,
			signals: 2,
			'validation.pbo': 0.3
		});
		expect(g.gauges.map((x) => x.prefix)).toEqual(['primitives', 'validation', 'hub']);
		expect(g.gauges[0].items).toEqual([
			{ key: 'fallback', value: 5 },
			{ key: 'graph', value: 10 }
		]);
		expect(g.other).toEqual([{ key: 'signals', value: 2 }]);
		expect(g.folded).toEqual([
			{ prefix: 'strategy', count: 2, sum: 0.30000000000000004 },
			{ prefix: 'trades.md', count: 1, sum: 7 }
		]);
	});

	it('computes primitive shares', () => {
		expect(primitiveShares({ 'primitives.graph': 3, 'primitives.decoder': 1, 'primitives.fallback': 0 })).toEqual({
			graph: 0.75,
			decoder: 0.25,
			fallback: 0
		});
		expect(primitiveShares({})).toBeNull();
	});
});

describe('service control gating', () => {
	const sup = { available: true };
	const running = { name: 'fa-risk-engine-1', id: 'x', image: 'i', state: 'running', status: 'Up' };
	const exited = { ...running, state: 'exited' };
	it('refuses infrastructure and needs a socket and a container', () => {
		expect(actionGate('postgres', 'restart', running, sup).enabled).toBe(false);
		expect(actionGate('control-api', 'restart', running, sup).why).toContain('itself');
		expect(actionGate('ingestor', 'restart', running, { available: false, detail: 'ENOENT' }).why).toContain('ENOENT');
		expect(actionGate('ingestor', 'restart', null, sup).enabled).toBe(false);
	});
	it('enables only state-changing actions', () => {
		expect(actionGate('ingestor', 'start', running, sup).enabled).toBe(false);
		expect(actionGate('ingestor', 'stop', running, sup).enabled).toBe(true);
		expect(actionGate('ingestor', 'stop', exited, sup).enabled).toBe(false);
		expect(actionGate('ingestor', 'start', exited, sup).enabled).toBe(true);
		expect(actionGate('ingestor', 'restart', exited, sup).enabled).toBe(true);
	});
	it('risk-engine stop/restart names the halt and sends confirm', () => {
		expect(confirmText('risk-engine', 'stop')).toContain('halts trading');
		expect(confirmText('risk-engine', 'start')).not.toContain('halts trading');
		expect(actionBody('risk-engine', 'restart')).toEqual({ confirm: true });
		expect(actionBody('ingestor', 'restart')).toEqual({});
	});
	it('describes results and API errors', () => {
		expect(describeResult('ingestor', 'stop', { ok: true, state: 'exited', noop: false, audited: true })).toBe('ingestor stop: state exited · audited');
		expect(describeResult('nats', 'stop', null, { status: 409, message: 'refused', body: { error: 'refused', message: 'infrastructure services are not controlled from the API' } })).toContain('409');
		expect(describeResult('x', 'start', null, { status: 501, message: 'docker_unavailable', body: { error: 'docker_unavailable', detail: 'no socket', hint: 'mount it' } })).toContain('mount it');
	});
});
