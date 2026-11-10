import { describe, expect, it } from 'vitest';
import type { GenerationFrame, GenerationRow, Population } from './api';
import { mergeGenerations, runActions, survivorsByRung } from './generations';

const frame = (over: Partial<GenerationFrame> = {}): GenerationFrame => ({
	generation_id: 'g1',
	kind: 'evo.eval.strategy',
	generation: 1,
	max_generations: 2,
	rung: 1,
	rungs: 2,
	final: true,
	candidates: 6,
	unranked: 0,
	objective_keys: ['sharpe'],
	front_sizes: [3, 3],
	survivors: ['a', 'b'],
	pareto_front: [],
	next_population: 0,
	vary_job_id: '',
	elapsed_ms: 1200,
	at: null,
	status: 'done',
	survivors_by_rung: { '1001': ['a', 'b'], '0': ['a', 'b', 'c'], '1000': ['a', 'b', 'c'] },
	updated_at: '2026-11-10T06:00:00Z',
	...over
});

describe('generations table', () => {
	it('decodes survivors_by_rung keys (generation * 1000 + rung) in order', () => {
		expect(survivorsByRung(frame()).map((s) => [s.generation, s.rung, s.ids.length])).toEqual([
			[0, 0, 3],
			[1, 0, 3],
			[1, 1, 2]
		]);
	});

	it('prefers ws frames over rest rows and joins registry stats by generation id', () => {
		const rows: GenerationRow[] = [
			{ id: 'g1', kind: 'evo.eval.strategy', generation: 0, max_generations: 2, rung: 0, rungs: 2, final: false, candidates: 6, front_sizes: [3], survivors: 3, status: 'running', at: null, updated_at: '2026-11-10T05:00:00Z' },
			{ id: 'g0', kind: 'evo.eval.decoder', generation: 0, max_generations: 1, rung: 0, rungs: 1, final: true, candidates: 2, front_sizes: [2], survivors: 2, status: 'done', at: null, updated_at: '2026-11-10T04:00:00Z' }
		];
		const pops: Population[] = [
			{ id: 'g0', kind: 'evo.eval.decoder', loops: 1, trials: 3, candidates: 2, first_at: '', last_at: '', stats: { n_trials: 3, deflated_sharpe: 0.0155, pbo: 0.5 } }
		];
		const out = mergeGenerations(rows, new Map([['g1', frame()]]), pops);
		expect(out.map((r) => [r.id, r.source, r.status, r.survivors])).toEqual([
			['g1', 'ws', 'done', 2],
			['g0', 'rest', 'done', 2]
		]);
		expect(out[1].validation?.pbo).toBe(0.5);
		expect(out[0].validation).toBeNull();
		expect(out[0].survivors_by_rung).toHaveLength(3);
	});

	it('offers pause/stop while running, resume/stop while paused, nothing when finished', () => {
		expect(runActions('running')).toEqual(['pause', 'stop']);
		expect(runActions('paused')).toEqual(['resume', 'stop']);
		expect(runActions('done')).toEqual([]);
		expect(runActions('stopped')).toEqual([]);
	});
});
