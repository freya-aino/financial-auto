// In-memory time series for the Trading Desk charts (spec 51 §6.5): a ring buffer of
// risk samples built from the polled /api/trading/risk payload, plus SVG path helpers.
import { loadPref, savePref } from './prefs';

export interface RiskSample {
	/** Sample time (ms since epoch; the poll time, since the paper figures carry no as_of). */
	t: number;
	equity: number;
	pnl: number;
	drawdown: number;
	exposure: number;
}

export class RingBuffer<T> {
	private items: T[] = [];
	constructor(public readonly capacity: number) {}

	push(item: T): void {
		this.items.push(item);
		if (this.items.length > this.capacity) this.items.splice(0, this.items.length - this.capacity);
	}

	/** Oldest first. */
	toArray(): T[] {
		return this.items.slice();
	}

	get length(): number {
		return this.items.length;
	}

	get last(): T | undefined {
		return this.items[this.items.length - 1];
	}

	replace(items: T[]): void {
		this.items = items.slice(-this.capacity);
	}
}

const PREF_KEY = 'trading.history';
const PERSISTED = 300;

/** Restores the persisted samples (newest `PERSISTED`) into a buffer of `capacity`. */
export function loadHistory(capacity = 900): RingBuffer<RiskSample> {
	const buf = new RingBuffer<RiskSample>(capacity);
	const saved = loadPref<unknown>(PREF_KEY, []);
	if (Array.isArray(saved)) {
		buf.replace(saved.filter(isSample));
	}
	return buf;
}

export function saveHistory(buf: RingBuffer<RiskSample>): void {
	savePref(PREF_KEY, buf.toArray().slice(-PERSISTED));
}

function isSample(v: unknown): v is RiskSample {
	if (!v || typeof v !== 'object') return false;
	const s = v as Record<string, unknown>;
	return ['t', 'equity', 'pnl', 'drawdown', 'exposure'].every((k) => typeof s[k] === 'number' && Number.isFinite(s[k]));
}

/**
 * Appends a sample unless it repeats the last one within `minGapMs` (polls arrive every 2 s,
 * so an unchanged book should not produce a flat staircase of identical points faster than that).
 */
export function pushSample(buf: RingBuffer<RiskSample>, s: RiskSample, minGapMs = 1500): boolean {
	const last = buf.last;
	if (last && s.t - last.t < minGapMs) return false;
	buf.push(s);
	return true;
}

/** Peak-to-trough drawdown series computed client-side from equity (fraction, >= 0). */
export function drawdownSeries(samples: RiskSample[]): number[] {
	let peak = -Infinity;
	return samples.map((s) => {
		peak = Math.max(peak, s.equity);
		return peak > 0 ? Math.max(0, 1 - s.equity / peak) : 0;
	});
}

export interface Scaled {
	path: string;
	/** Scaled points, same order as the input. */
	pts: { x: number; y: number }[];
	lo: number;
	hi: number;
}

/**
 * Maps `values` (in order) onto a `w`×`h` box as an SVG polyline path. A flat series sits in
 * the vertical middle. `include` forces extra values (e.g. 0 for a P&L baseline) into the range.
 */
export function linePath(values: number[], w: number, h: number, include: number[] = []): Scaled {
	if (!values.length) return { path: '', pts: [], lo: 0, hi: 0 };
	let lo = Math.min(...values, ...include);
	let hi = Math.max(...values, ...include);
	if (hi === lo) {
		lo -= 1;
		hi += 1;
	}
	const n = values.length;
	const pts = values.map((v, i) => ({
		x: n === 1 ? w / 2 : (i / (n - 1)) * w,
		y: h - ((v - lo) / (hi - lo)) * h
	}));
	const path = pts.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ');
	return { path, pts, lo, hi };
}

/** Y coordinate of `value` in the same scale `linePath` produced. */
export function yFor(value: number, scaled: Scaled, h: number): number {
	return scaled.hi === scaled.lo ? h / 2 : h - ((value - scaled.lo) / (scaled.hi - scaled.lo)) * h;
}
