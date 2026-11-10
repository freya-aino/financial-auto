// Heartbeat counter grouping for the Ops services table (D-053/D-055 gauges): prefixes like
// `primitives.*`, `validation.*`, `pareto.*`, `hub.*`, `gpu.*` become small gauge groups,
// per-strategy and per-subject counters are folded so the table stays readable.
export interface GaugeGroup {
	prefix: string;
	items: { key: string; value: number }[];
}

/** Prefixes rendered as named gauge groups (order = display order). */
export const GAUGE_PREFIXES = ['primitives', 'validation', 'pareto', 'hub', 'gpu', 'connectors', 'archive', 'tsdb'];

/** Per-item counters that would flood the table; folded into `<prefix>.* (n)`. */
const FOLD_PREFIXES = ['strategy.', 'trades.md.', 'states_published.', 'region_events.', 'regions_live.', 'transition_rows.', 'transitions_published.', 'bars_closed.', 'unresolved.', 'events.'];

export interface GroupedCounters {
	gauges: GaugeGroup[];
	/** Everything else, sorted by key. */
	other: { key: string; value: number }[];
	/** Folded families: prefix → number of counters hidden. */
	folded: { prefix: string; count: number; sum: number }[];
}

export function groupCounters(counters: Record<string, number>): GroupedCounters {
	const groups = new Map<string, { key: string; value: number }[]>();
	const other: { key: string; value: number }[] = [];
	const folded = new Map<string, { count: number; sum: number }>();
	for (const [key, value] of Object.entries(counters).sort(([a], [b]) => a.localeCompare(b))) {
		const fold = FOLD_PREFIXES.find((p) => key.startsWith(p) && key.length > p.length);
		if (fold) {
			const f = folded.get(fold) ?? { count: 0, sum: 0 };
			f.count++;
			f.sum += value;
			folded.set(fold, f);
			continue;
		}
		const dot = key.indexOf('.');
		const prefix = dot > 0 ? key.slice(0, dot) : '';
		if (prefix && GAUGE_PREFIXES.includes(prefix)) {
			const g = groups.get(prefix) ?? [];
			g.push({ key: key.slice(dot + 1), value });
			groups.set(prefix, g);
		} else other.push({ key, value });
	}
	return {
		gauges: GAUGE_PREFIXES.filter((p) => groups.has(p)).map((p) => ({ prefix: p, items: groups.get(p)! })),
		other,
		folded: [...folded].map(([prefix, f]) => ({ prefix: prefix.replace(/\.$/, ''), ...f }))
	};
}

/** Share of primitive calls answered from the graph / decoder cache vs the price fallback (D-053). */
export function primitiveShares(counters: Record<string, number>): { graph: number; decoder: number; fallback: number } | null {
	const g = counters['primitives.graph'] ?? 0;
	const d = counters['primitives.decoder'] ?? 0;
	const f = counters['primitives.fallback'] ?? 0;
	const t = g + d + f;
	if (!t) return null;
	return { graph: g / t, decoder: d / t, fallback: f / t };
}

export const fmtCounter = (v: number) => (Number.isInteger(v) ? String(v) : Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1));
