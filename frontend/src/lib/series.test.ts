import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { drawdownSeries, linePath, loadHistory, pushSample, RingBuffer, saveHistory, yFor, type RiskSample } from './series';

const sample = (t: number, equity = 100): RiskSample => ({ t, equity, pnl: equity - 100, drawdown: 0, exposure: 0 });

describe('RingBuffer', () => {
	it('keeps the newest `capacity` items in order', () => {
		const b = new RingBuffer<number>(3);
		[1, 2, 3, 4, 5].forEach((n) => b.push(n));
		expect(b.toArray()).toEqual([3, 4, 5]);
		expect(b.length).toBe(3);
		expect(b.last).toBe(5);
		b.replace([9, 8, 7, 6]);
		expect(b.toArray()).toEqual([8, 7, 6]);
	});
});

describe('pushSample', () => {
	it('drops samples arriving faster than the minimum gap', () => {
		const b = new RingBuffer<RiskSample>(10);
		expect(pushSample(b, sample(0))).toBe(true);
		expect(pushSample(b, sample(1000))).toBe(false);
		expect(pushSample(b, sample(2000))).toBe(true);
		expect(b.length).toBe(2);
	});
});

describe('history persistence', () => {
	const store = new Map<string, string>();
	beforeEach(() =>
		vi.stubGlobal('localStorage', {
			getItem: (k: string) => store.get(k) ?? null,
			setItem: (k: string, v: string) => void store.set(k, v)
		})
	);
	afterEach(() => {
		store.clear();
		vi.unstubAllGlobals();
	});

	it('persists the newest 300 samples and restores only well-formed ones', () => {
		const b = new RingBuffer<RiskSample>(1000);
		for (let i = 0; i < 400; i++) b.push(sample(i * 2000, 100 + i));
		saveHistory(b);
		const saved = JSON.parse(store.get('fa.ui.trading.history')!);
		expect(saved).toHaveLength(300);
		expect(saved[0].equity).toBe(200);
		store.set('fa.ui.trading.history', JSON.stringify([sample(1), { t: 'nope' }, null, sample(2)]));
		expect(loadHistory(5).toArray()).toEqual([sample(1), sample(2)]);
		store.set('fa.ui.trading.history', '"garbage"');
		expect(loadHistory().length).toBe(0);
	});
});

describe('drawdownSeries', () => {
	it('is peak-to-trough as a fraction, never negative', () => {
		const dd = drawdownSeries([sample(0, 100), sample(1, 120), sample(2, 90), sample(3, 130), sample(4, 130)]);
		expect(dd.map((v) => +v.toFixed(3))).toEqual([0, 0, 0.25, 0, 0]);
	});
});

describe('linePath', () => {
	it('scales values into the box, oldest on the left', () => {
		const s = linePath([0, 5, 10], 100, 50);
		expect(s.pts).toEqual([
			{ x: 0, y: 50 },
			{ x: 50, y: 25 },
			{ x: 100, y: 0 }
		]);
		expect(s.path).toBe('M0.00 50.00 L50.00 25.00 L100.00 0.00');
	});

	it('centres a flat series and includes the baseline in the range', () => {
		const flat = linePath([3, 3], 100, 50);
		expect(flat.pts.every((p) => p.y === 25)).toBe(true);
		const withZero = linePath([5, 10], 100, 50, [0]);
		expect(withZero.lo).toBe(0);
		expect(yFor(0, withZero, 50)).toBe(50);
		expect(linePath([], 10, 10)).toEqual({ path: '', pts: [], lo: 0, hi: 0 });
	});
});
