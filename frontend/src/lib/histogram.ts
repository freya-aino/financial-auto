// Equal-width binning for the score histogram (spec 51 §6.4).

export interface Bins {
	counts: number[];
	lo: number;
	hi: number;
	/** Largest count, never 0 so it can be a divisor. */
	max: number;
}

export function histogram(values: number[], bins = 12): Bins {
	if (!values.length || bins < 1) return { counts: [], lo: 0, hi: 0, max: 1 };
	let lo = Infinity;
	let hi = -Infinity;
	for (const v of values) {
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	const w = (hi - lo || 1) / bins;
	const counts = Array<number>(bins).fill(0);
	for (const v of values) counts[Math.min(bins - 1, Math.floor((v - lo) / w))]++;
	return { counts, lo, hi, max: Math.max(1, ...counts) };
}
