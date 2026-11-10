import { describe, expect, it } from 'vitest';
import { groupCounters, primitiveShares } from './services';

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
