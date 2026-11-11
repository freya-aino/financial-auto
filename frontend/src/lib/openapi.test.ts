// Spec 50 Q3 / spec 51 Q2: the fetch layer stays hand-written and is checked against the
// OpenAPI document control-api generates from its route declarations (committed copy
// `openapi.json`, kept current by `cargo test -p control-api openapi_snapshot_is_current`).
// Every `/api/…` literal in the frontend sources must name a documented path, with the method
// its helper uses (getJson/getApi → GET, postJson → POST, putJson → PUT).
import { describe, expect, it } from 'vitest';
import doc from './openapi.json';

type Method = 'get' | 'post' | 'put' | 'delete' | 'any';

const paths = doc.paths as Record<string, Record<string, unknown>>;
const templates = Object.keys(paths).map((p) => ({ p, segs: p.split('/') }));

/** `/api/x/${a}/y${q(...)}` → `/api/x/{}/y` (query strings and template tails dropped). */
export function normalise(lit: string): string {
	const noQuery = lit.replace(/\$\{q\([^}]*\)\}|\$\{q\(\{[^}]*\}\)\}/g, '').split('?')[0];
	return noQuery.replace(/\$\{[^}]*\}/g, '{}').replace(/\/+$/, '');
}

/** Documented path matching a normalised client path (`{}` and `{name}` match any segment),
 *  preferring one that serves `method`. */
export function match(path: string, method: Method = 'any'): string | null {
	const segs = path.split('/');
	const hits = templates
		.filter((t) => t.segs.length === segs.length && t.segs.every((s, i) => s === segs[i] || s.startsWith('{') || segs[i] === '{}' || (segs[i].includes('{}') && s.startsWith('{'))))
		.map((t) => t.p);
	return hits.find((p) => method === 'any' || paths[p][method]) ?? hits[0] ?? null;
}

/**
 * Routes the UI already calls but control-api does not serve yet (backend work listed in spec 51
 * / frontend/README). The UI treats 404/501 from them as "unavailable". Remove an entry once the
 * route is in the regenerated `openapi.json` (the test below fails on a stale entry).
 */
export const PENDING_ROUTES: Record<string, string> = {};

// every source of the app, raw (Vite glob; tests excluded below)
const SOURCES = import.meta.glob(['/src/**/*.ts', '/src/**/*.svelte', '!/src/**/*.test.ts'], { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

interface Call {
	file: string;
	lit: string;
	path: string;
	method: Method;
}

function calls(): Call[] {
	const out: Call[] = [];
	for (const [file, text] of Object.entries(SOURCES)) {
		const re = /(getJson|getApi|postJson|putJson|sendJson|fetch)?\s*(?:<[^>()]*(?:<[^>]*>[^>()]*)*>)?\(?\s*(['`])(\/api\/[^'`]*)\2/g;
		for (const m of text.matchAll(re)) {
			const helper = m[1] ?? '';
			const method: Method = helper === 'postJson' ? 'post' : helper === 'putJson' ? 'put' : helper === 'getJson' || helper === 'getApi' ? 'get' : 'any';
			out.push({ file, lit: m[3], path: normalise(m[3]), method });
		}
	}
	return out;
}

describe('frontend ↔ OpenAPI contract', () => {
	it('normalises template literals', () => {
		expect(normalise('/api/latent/snapshots/${enc(id)}/shadow')).toBe('/api/latent/snapshots/{}/shadow');
		expect(normalise('/api/trading/orphans${q({ status })}')).toBe('/api/trading/orphans');
		expect(match('/api/mcmc/{}/{}')).not.toBeNull();
		expect(match('/api/no/such/route')).toBeNull();
	});

	it('the document is the generated one', () => {
		expect(doc.openapi).toBe('3.1.0');
		expect(paths['/api/openapi.json']?.get).toBeTruthy();
		expect(Object.keys(paths).length).toBeGreaterThan(90);
	});

	it('every /api call of the frontend is a documented route with that method', () => {
		const found = calls();
		expect(found.length).toBeGreaterThan(80);
		const problems: string[] = [];
		for (const c of found) {
			// prose mentions (`Resolve via POST /api/…`) inside markup are not calls
			if (c.lit.includes(' ') || c.lit.includes('&#123;')) continue;
			const p = match(c.path, c.method);
			if (!p && PENDING_ROUTES[`${c.method.toUpperCase()} ${c.path}`]) continue;
			if (!p) {
				problems.push(`${c.file}: ${c.lit} → no documented path`);
				continue;
			}
			if (c.method !== 'any' && !paths[p][c.method]) problems.push(`${c.file}: ${c.method.toUpperCase()} ${c.lit} → ${p} has ${Object.keys(paths[p]).join(', ')}`);
		}
		expect(problems).toEqual([]);
	});

	it('pending routes are still missing from the document (else drop them from PENDING_ROUTES)', () => {
		for (const key of Object.keys(PENDING_ROUTES)) {
			const [method, path] = key.split(' ');
			const p = match(path, method.toLowerCase() as Method);
			expect(p && paths[p][method.toLowerCase()] ? key : null).toBeNull();
		}
	});
});
