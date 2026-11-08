import { describe, expect, it } from 'vitest';
import { MINIMISE, nonDominated } from './pareto';

describe('nonDominated', () => {
	const pts = [
		{ x: 1, y: 1 },
		{ x: 2, y: 2 }, // dominates (1,1)
		{ x: 3, y: 0 },
		{ x: 0, y: 3 },
		{ x: 2, y: 2 } // duplicate of the front point: both stay
	];

	it('maximising both: the front is the upper-right envelope', () => {
		expect(nonDominated(pts)).toEqual([false, true, true, true, true]);
	});

	it('minimising both flips the front', () => {
		expect(nonDominated(pts, false, false)).toEqual([true, false, true, true, false]);
	});

	it('mixed directions', () => {
		// maximise x, minimise y: (3,0) dominates everything else.
		expect(nonDominated(pts, true, false)).toEqual([false, false, true, false, false]);
	});

	it('handles empty and singleton inputs', () => {
		expect(nonDominated([])).toEqual([]);
		expect(nonDominated([{ x: 0, y: 0 }])).toEqual([true]);
	});

	it('knows which objectives are minimised', () => {
		expect(MINIMISE.has('crps')).toBe(true);
		expect(MINIMISE.has('sharpe')).toBe(false);
	});
});
