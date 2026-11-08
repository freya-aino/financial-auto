// Strategies table rows derived from /api/trading/signals + /api/trading/orders (spec 51 §6.5).
// There is no /api/trading/strategies yet (501), so status, sleeve and gates are unknown.
import type { OrderView, SignalView } from './api';

export interface StrategyRow {
	strategy_id: string;
	genome_id: string;
	horizon: string;
	/** Age of the last signal in ms, or null when the signal carries no `as_of`. */
	last_signal_ms: number | null;
	instruments: number;
	/** Mean of |exposure mean| over the strategy's instruments. */
	mean_abs_exposure: number;
	/** Largest |exposure| and the instrument carrying it. */
	max_exposure: { instrument_id: string; mean: number } | null;
	orders: number;
	filled: number;
	rejected: number;
}

export const STRATEGY_ACTIONS = ['pause', 'resume', 'flatten', 'promote'] as const;

export function deriveStrategies(signals: SignalView[], orders: OrderView[], now = Date.now()): StrategyRow[] {
	const byStrategy = new Map<string, { orders: number; filled: number; rejected: number }>();
	for (const o of orders) {
		for (const id of o.strategy_ids) {
			const e = byStrategy.get(id) ?? { orders: 0, filled: 0, rejected: 0 };
			e.orders++;
			if (o.status === 'filled' || o.filled_quantity > 0) e.filled++;
			if (o.status === 'rejected') e.rejected++;
			byStrategy.set(id, e);
		}
	}
	const rows: StrategyRow[] = signals.map((s) => {
		const abs = s.exposures.map((e) => Math.abs(e.mean));
		const max = s.exposures.reduce<StrategyRow['max_exposure']>(
			(m, e) => (m === null || Math.abs(e.mean) > Math.abs(m.mean) ? { instrument_id: e.instrument_id, mean: e.mean } : m),
			null
		);
		const o = byStrategy.get(s.strategy_id) ?? { orders: 0, filled: 0, rejected: 0 };
		byStrategy.delete(s.strategy_id);
		return {
			strategy_id: s.strategy_id,
			genome_id: s.genome_id,
			horizon: s.horizon,
			last_signal_ms: s.as_of ? Math.max(0, now - Date.parse(s.as_of)) : null,
			instruments: s.exposures.length,
			mean_abs_exposure: abs.length ? abs.reduce((a, b) => a + b, 0) / abs.length : 0,
			max_exposure: max,
			...o
		};
	});
	// Strategies that only appear on orders (no live signal) still get a row.
	for (const [id, o] of byStrategy) {
		rows.push({
			strategy_id: id,
			genome_id: '',
			horizon: '',
			last_signal_ms: null,
			instruments: 0,
			mean_abs_exposure: 0,
			max_exposure: null,
			...o
		});
	}
	return rows.sort((a, b) => b.mean_abs_exposure - a.mean_abs_exposure || a.strategy_id.localeCompare(b.strategy_id));
}
