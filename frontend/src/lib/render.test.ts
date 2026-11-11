// SSR smoke render of every view and panel at its initial (empty) state: catches template errors
// that `svelte-check` cannot (undefined access in markup, bad snippets). The app itself runs
// client-only (`ssr = false`); `onMount`/`$effect` do not run here, so no network is touched.
import type { Component } from 'svelte';
import { render } from 'svelte/server';
import { describe, expect, it, vi } from 'vitest';

/** Renders any component with loosely typed props (the shapes are the API's, asserted in api.ts). */
const ssr = (c: unknown, props: Record<string, unknown> = {}) => render(c as Component<Record<string, unknown>>, { props });

vi.stubGlobal('location', { protocol: 'http:', host: 'test' });
vi.stubGlobal('localStorage', { getItem: () => null, setItem: () => {} });
vi.stubGlobal('fetch', () => new Promise(() => {}));

const pages = {
	ops: () => import('../routes/+page.svelte'),
	latent: () => import('../routes/latent/+page.svelte'),
	evolution: () => import('../routes/evolution/+page.svelte'),
	trading: () => import('../routes/trading/+page.svelte'),
	alerts: () => import('../routes/alerts/+page.svelte'),
	refdata: () => import('../routes/refdata/+page.svelte'),
	mcmc: () => import('../routes/latent/mcmc/+page.svelte'),
	snapshots: () => import('../routes/latent/snapshots/+page.svelte'),
	decoders: () => import('../routes/evolution/decoders/+page.svelte'),
	pipeline: () => import('../routes/trading/pipeline/+page.svelte'),
	risk: () => import('../routes/trading/risk/+page.svelte'),
	ledger: () => import('../routes/trading/ledger/+page.svelte')
};

describe('views render at their empty state', () => {
	for (const [name, load] of Object.entries(pages)) {
		it(name, async () => {
			const { default: Page } = await load();
			const { body } = ssr(Page);
			expect(body).toContain('panel');
			expect(body.length).toBeGreaterThan(500);
		});
	}
});

describe('panels render with live-shaped props', () => {
	it('StrategiesTable shows gate and actions', async () => {
		const { default: C } = await import('./StrategiesTable.svelte');
		const { body } = ssr(C, {
				now: Date.now(),
				view: {
					source: 'risk-engine',
					count: 1,
					status_counts: { shadow: 1 },
					signals_without_state: [],
					strategies: [
						{
							strategy_id: 's1', status: 'shadow', prev_status: null, since: new Date().toISOString(), first_seen: null,
							sleeve_eur: 1000, weight: 0.1, pnl_eur: -5, peak_pnl_eur: 2, drawdown: 0.01, max_drawdown: 0.02, last_reason: '',
							shadow: { days: 1.5, horizons_completed: 3, since: new Date().toISOString() },
							gate: { green: false, missing: ['min_horizons'], items: [{ name: 'min_horizons', pass: false, advisory: false, detail: '3 of 20' }] },
							signal: { strategy_id: 's1', genome_id: 'g', horizon: '5x1s', as_of: null, exposures: [{ instrument_id: 'i', mean: 0.2, stddev: 0.1 }] }
						}
				]
			}
		});
		expect(body).toContain('1 missing');
		expect(body).toContain('promotion blocked');
		expect(body).toContain('disabled');
	});

	it('Generations renders ws-backed rows with survivors per rung', async () => {
		const { default: C } = await import('./Generations.svelte');
		const frame = {
			generation_id: 'g1', kind: 'evo.eval.strategy', generation: 0, max_generations: 1, rung: 1, rungs: 2, final: true, candidates: 4,
			unranked: 0, objective_keys: ['sharpe'], front_sizes: [2, 2], survivors: ['a'], pareto_front: [{ candidate: 'a', objectives: { sharpe: 1 }, rank: 0, crowding: 'inf' }],
			next_population: 0, vary_job_id: '', elapsed_ms: 10, at: null, status: 'done', survivors_by_rung: { '0': ['a', 'b'], '1': ['a'] }, updated_at: new Date().toISOString()
		};
		const { body } = ssr(C, { rows: [], frames: new Map([['g1', frame]]), populations: [], now: Date.now() });
		expect(body).toContain('done');
		expect(body).toContain('2 / 2');
	});

	it('Heatmap and Experts render a correlation matrix', async () => {
		const { default: C } = await import('./Experts.svelte');
		const { body } = ssr(C, {
				view: {
					snapshot: null, source: 'helix+live', live: { messages: 3, last_as_of: null },
					experts: [{ expert_index: 0, expert_space_id: 's-e0', dim: 8, regions: 2, regions_live: 2, routing: { expert_index: 0, dims: 8, messages: 3, gate_mean: 0.5, top_share: 0.7, regions_seen: 2, norm_mean: 1, uncertainty_mean: 0.3 } }],
				covariance: { experts: [0], matrix: [[0.4]], correlation: [[1]], mean: [1], n: 3, feature: 'l2' }
			}
		});
		expect(body).toContain('E0');
		expect(body).toContain('1.00');
	});
});
