import { describe, expect, it } from 'vitest';
import { assignRegionColors, CategoryColors, explainedLabel, extent, heatColor, REGION_PALETTE, scale } from './projection';

describe('projection colours', () => {
	it('assigns stable colours per category and wraps the palette', () => {
		const c = new CategoryColors(['a', 'b']);
		expect(c.color('r1')).toBe('a');
		expect(c.color('r2')).toBe('b');
		expect(c.color('r3')).toBe('a');
		expect(c.color('r1')).toBe('a');
		expect(c.keys).toEqual(['r1', 'r2', 'r3']);
	});

	it('assigns by sorted id so fetch order does not matter', () => {
		const a = assignRegionColors(['1m-e0-r003', '1m-e0-r001']);
		const b = assignRegionColors(['1m-e0-r001', '1m-e0-r003']);
		expect(a.get('1m-e0-r001')).toBe(REGION_PALETTE[0]);
		expect(a).toEqual(b);
	});

	it('heat colour is red for negative, green for positive, alpha = |v|', () => {
		expect(heatColor(-0.5)).toEqual({ color: 'var(--danger)', alpha: 0.5 });
		expect(heatColor(1)).toEqual({ color: 'var(--accent)', alpha: 1 });
		expect(heatColor(2).alpha).toBe(1);
		expect(heatColor(NaN).alpha).toBe(0);
	});

	it('extent pads and scale maps linearly', () => {
		expect(extent([])).toEqual({ lo: -1, hi: 1 });
		expect(extent([2, 2])).toEqual({ lo: 1, hi: 3 });
		const e = extent([0, 10], 0);
		expect(scale(5, e, 0, 100)).toBe(50);
		expect(scale(0, e, 100, 0)).toBe(100);
	});

	it('labels explained variance', () => {
		expect(explainedLabel([0.5947, 0.3657])).toBe('PC1 59.5% · PC2 36.6% (96.0%)');
		expect(explainedLabel([])).toBe('');
	});
});
