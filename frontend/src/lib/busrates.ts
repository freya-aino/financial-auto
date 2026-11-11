// Per-subject message rates for the bus monitor (spec 51 §6.2 "rate sparkline per subject"):
// counts per second over a sliding window, keyed by the subject cut to `depth` tokens so
// per-instrument subjects (`md.trade.<id>`) fold into one family row when wanted.

export interface SubjectRate {
	key: string;
	/** msg/s per second, oldest first (length = window). */
	series: number[];
	/** Last full second. */
	last: number;
	/** Mean over the window. */
	mean: number;
	total: number;
}

/** `md.trade.binance.btc` at depth 2 → `md.trade.>`; depth ≤ 0 keeps the full subject. */
export function subjectKey(subject: string, depth: number): string {
	if (depth <= 0) return subject;
	const t = subject.split('.');
	return t.length > depth ? `${t.slice(0, depth).join('.')}.>` : subject;
}

export class SubjectRates {
	private current = new Map<string, number>();
	private rings = new Map<string, { series: number[]; total: number; idle: number }>();

	constructor(
		public window = 60,
		public depth = 3,
		private maxKeys = 200
	) {}

	hit(subject: string): void {
		const k = subjectKey(subject, this.depth);
		this.current.set(k, (this.current.get(k) ?? 0) + 1);
	}

	/** Closes the current second; keys idle for a whole window are dropped. */
	tick(): void {
		for (const k of this.current.keys()) {
			if (!this.rings.has(k) && this.rings.size < this.maxKeys) this.rings.set(k, { series: new Array<number>(this.window).fill(0), total: 0, idle: 0 });
		}
		for (const [k, r] of this.rings) {
			const c = this.current.get(k) ?? 0;
			r.series.push(c);
			if (r.series.length > this.window) r.series.shift();
			r.total += c;
			r.idle = c ? 0 : r.idle + 1;
			if (r.idle >= this.window) this.rings.delete(k);
		}
		this.current.clear();
	}

	reset(depth = this.depth): void {
		this.depth = depth;
		this.current.clear();
		this.rings.clear();
	}

	/** Busiest keys over the window first. */
	top(n = 12): SubjectRate[] {
		return [...this.rings]
			.map(([key, r]) => {
				const sum = r.series.reduce((a, b) => a + b, 0);
				return { key, series: [...r.series], last: r.series[r.series.length - 1] ?? 0, mean: sum / r.series.length, total: r.total };
			})
			.sort((a, b) => b.mean - a.mean || a.key.localeCompare(b.key))
			.slice(0, n);
	}

	get size(): number {
		return this.rings.size;
	}
}

/** SVG polyline points for a tiny sparkline of `values` in a w × h box (max-scaled, 0 at bottom). */
export function sparkPoints(values: number[], w: number, h: number): string {
	if (!values.length) return '';
	const max = Math.max(1, ...values);
	const dx = values.length > 1 ? w / (values.length - 1) : 0;
	return values.map((v, i) => `${(i * dx).toFixed(1)},${(h - 1 - (v / max) * (h - 2)).toFixed(1)}`).join(' ');
}
