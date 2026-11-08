import { describe, expect, it } from 'vitest';
import type { OrderView, SignalView } from './api';
import { deriveStrategies } from './strategies';

const now = Date.parse('2026-11-08T12:00:00Z');
const signal = (id: string, exposures: [string, number][], asOf: string | null = '2026-11-08T11:59:00Z'): SignalView => ({
	strategy_id: id,
	genome_id: `g-${id}`,
	horizon: '1d',
	as_of: asOf,
	exposures: exposures.map(([instrument_id, mean]) => ({ instrument_id, mean, stddev: 0.1 }))
});
const order = (strategy_ids: string[], status: string, filled = 0): OrderView =>
	({ intent_id: Math.random().toString(36), strategy_ids, status, filled_quantity: filled, instrument_id: 'x', side: 'buy' }) as OrderView;

describe('deriveStrategies', () => {
	it('aggregates exposure stats and attributes orders via strategy_ids', () => {
		const rows = deriveStrategies(
			[signal('s1', [['a', 0.5], ['b', -0.3]]), signal('s2', [['a', 0.1]], null)],
			[order(['s1'], 'filled', 2), order(['s1', 's2'], 'rejected'), order(['s3'], 'working')],
			now
		);
		expect(rows.map((r) => r.strategy_id)).toEqual(['s1', 's2', 's3']);
		const s1 = rows[0];
		expect(s1.instruments).toBe(2);
		expect(s1.mean_abs_exposure).toBeCloseTo(0.4);
		expect(s1.max_exposure).toEqual({ instrument_id: 'a', mean: 0.5 });
		expect(s1.last_signal_ms).toBe(60_000);
		expect(s1).toMatchObject({ orders: 2, filled: 1, rejected: 1 });
		expect(rows[1]).toMatchObject({ last_signal_ms: null, orders: 1, rejected: 1 });
		// order-only strategy gets a stub row
		expect(rows[2]).toMatchObject({ genome_id: '', instruments: 0, orders: 1, filled: 0 });
	});

	it('returns nothing without signals or orders', () => {
		expect(deriveStrategies([], [], now)).toEqual([]);
	});
});
