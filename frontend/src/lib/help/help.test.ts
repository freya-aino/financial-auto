// The ⓘ contract: every key a page uses resolves to text, and entries stay short enough to
// read in a popover (the ⓘ explains, the spec documents).
import { describe, expect, it } from 'vitest';

import { HELP } from './index';

// Every component and route source, as raw text (resolved by Vite at test time).
const sources = import.meta.glob('../../**/*.svelte', { query: '?raw', import: 'default', eager: true }) as Record<
	string,
	string
>;

describe('help registry', () => {
	it('entries are short, plain sentences', () => {
		for (const [k, v] of Object.entries(HELP)) {
			expect(v.trim().length, k).toBeGreaterThan(20);
			expect(v.length, `${k} should stay a short explanation`).toBeLessThanOrEqual(260);
		}
	});

	it('every <Info k> used in a page resolves', () => {
		const used = new Set<string>();
		for (const src of Object.values(sources)) {
			for (const m of src.matchAll(/<Info\s+k="([^"]+)"/g)) used.add(m[1]);
		}
		expect(used.size).toBeGreaterThan(0);
		const missing = [...used].filter((k) => !(k in HELP));
		expect(missing, 'keys used without a registry entry').toEqual([]);
	});
});
