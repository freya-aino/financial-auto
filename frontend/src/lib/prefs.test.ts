import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { loadPref, savePref } from './prefs';

function fakeStorage(): Storage {
	const m = new Map<string, string>();
	return {
		getItem: (k) => m.get(k) ?? null,
		setItem: (k, v) => void m.set(k, String(v)),
		removeItem: (k) => void m.delete(k),
		clear: () => m.clear(),
		key: (i) => [...m.keys()][i] ?? null,
		get length() {
			return m.size;
		}
	};
}

describe('prefs', () => {
	beforeEach(() => vi.stubGlobal('localStorage', fakeStorage()));
	afterEach(() => vi.unstubAllGlobals());

	it('round-trips JSON under the fa.ui. prefix', () => {
		savePref('evo.x', 'sharpe');
		savePref('trading.history', [{ t: 1, equity: 2 }]);
		expect(localStorage.getItem('fa.ui.evo.x')).toBe('"sharpe"');
		expect(loadPref('evo.x', '')).toBe('sharpe');
		expect(loadPref('trading.history', [])).toEqual([{ t: 1, equity: 2 }]);
	});

	it('returns the fallback for missing or corrupt values', () => {
		expect(loadPref('missing', 42)).toBe(42);
		localStorage.setItem('fa.ui.bad', '{not json');
		expect(loadPref('bad', 'fallback')).toBe('fallback');
	});

	it('never throws when storage is unavailable', () => {
		vi.stubGlobal('localStorage', {
			getItem: () => {
				throw new Error('denied');
			},
			setItem: () => {
				throw new Error('quota');
			}
		});
		expect(() => savePref('x', 1)).not.toThrow();
		expect(loadPref('x', 'fb')).toBe('fb');
		vi.stubGlobal('localStorage', undefined);
		expect(loadPref('x', 'fb')).toBe('fb');
		expect(() => savePref('x', 1)).not.toThrow();
	});
});
