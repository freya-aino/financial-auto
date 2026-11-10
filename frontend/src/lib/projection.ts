// Colour and scale helpers for the latent projection scatter and the expert correlation
// heatmap (spec 51 §6.3). Pure so the mappings are unit-tested.

/** Categorical palette (12 hues, dark/light safe), assigned to region ids in first-seen order. */
export const REGION_PALETTE = [
	'#4fd1a5',
	'#e6b450',
	'#7aa2f7',
	'#f0716b',
	'#bb9af7',
	'#2ac3de',
	'#ff9e64',
	'#9ece6a',
	'#f7768e',
	'#73daca',
	'#c0caf5',
	'#e0af68'
];

/** Deterministic colour per category: stable for the lifetime of the map (new ids append). */
export class CategoryColors {
	private index = new Map<string, number>();
	constructor(private palette: string[] = REGION_PALETTE) {}

	color(key: string): string {
		let i = this.index.get(key);
		if (i === undefined) {
			i = this.index.size;
			this.index.set(key, i);
		}
		return this.palette[i % this.palette.length];
	}

	/** Categories seen so far, in assignment order (legend). */
	get keys(): string[] {
		return [...this.index.keys()];
	}
}

/** Sorts region ids so the colour assignment does not depend on fetch order. */
export function assignRegionColors(regionIds: Iterable<string>, palette = REGION_PALETTE): Map<string, string> {
	const c = new CategoryColors(palette);
	const sorted = [...new Set(regionIds)].sort();
	for (const id of sorted) c.color(id);
	return new Map(sorted.map((id) => [id, c.color(id)]));
}

/**
 * Diverging colour for a correlation in [-1, 1]: negative → danger red, zero → neutral,
 * positive → accent green, with opacity ∝ |v| so the matrix reads at a glance.
 */
export function heatColor(v: number): { color: string; alpha: number } {
	const x = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
	return { color: x < 0 ? 'var(--danger)' : 'var(--accent)', alpha: Math.round(Math.abs(x) * 100) / 100 };
}

export interface Extent {
	lo: number;
	hi: number;
}

/** Min/max with a relative pad; degenerate ranges get ±1. */
export function extent(values: number[], pad = 0.06): Extent {
	if (!values.length) return { lo: -1, hi: 1 };
	let lo = Infinity;
	let hi = -Infinity;
	for (const v of values) {
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	if (hi === lo) return { lo: lo - 1, hi: hi + 1 };
	const p = (hi - lo) * pad;
	return { lo: lo - p, hi: hi + p };
}

/** Maps `v` in `e` onto `[a, b]`. */
export function scale(v: number, e: Extent, a: number, b: number): number {
	return a + ((v - e.lo) / (e.hi - e.lo)) * (b - a);
}

/** `explained_variance_ratio` as a label: "PC1 59.5% · PC2 36.6% (96.1%)". */
export function explainedLabel(ratios: number[]): string {
	if (!ratios.length) return '';
	const parts = ratios.map((r, i) => `PC${i + 1} ${(100 * r).toFixed(1)}%`);
	const total = ratios.reduce((a, b) => a + b, 0);
	return `${parts.join(' · ')} (${(100 * total).toFixed(1)}%)`;
}
