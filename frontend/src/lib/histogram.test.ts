import { describe, expect, it } from 'vitest';
import { histogram } from './histogram';

describe('histogram', () => {
	it('bins values into equal widths with the max in the last bin', () => {
		const h = histogram([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5);
		expect(h.lo).toBe(0);
		expect(h.hi).toBe(10);
		expect(h.counts).toEqual([2, 2, 2, 2, 3]);
		expect(h.counts.reduce((a, b) => a + b)).toBe(11);
		expect(h.max).toBe(3);
	});

	it('puts identical values into the first bin without dividing by zero', () => {
		const h = histogram([2, 2, 2], 4);
		expect(h.counts).toEqual([3, 0, 0, 0]);
		expect(h.max).toBe(3);
	});

	it('returns an empty, safe result for no values', () => {
		expect(histogram([])).toEqual({ counts: [], lo: 0, hi: 0, max: 1 });
		expect(histogram([1, 2], 0).counts).toEqual([]);
	});

	it('handles negative ranges', () => {
		const h = histogram([-3, -1, 1, 3], 2);
		expect(h.counts).toEqual([2, 2]);
		expect(h.lo).toBe(-3);
	});
});
