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

import { candidateLabel, directionFor, paretoPoints, plottableKeys } from './pareto';

describe('objective directions (D-049/D-052 keys)', () => {
	it('prefers the API direction map and falls back to the static list', () => {
		expect(directionFor('sharpe')).toBe('max');
		expect(directionFor('max_drawdown')).toBe('min');
		expect(directionFor('turnover')).toBe('min');
		expect(directionFor('deflated_sharpe', { deflated_sharpe: 'max' })).toBe('max');
		expect(directionFor('sharpe', { sharpe: 'min' })).toBe('min');
	});

	it('hides diagnostics from the pickers unless nothing else exists', () => {
		const members = [{ objectives: { sharpe: 1, n_obs: 10, max_drawdown: 0.1, psr: 0.5 } }];
		expect(plottableKeys(members)).toEqual(['max_drawdown', 'sharpe']);
		expect(plottableKeys([{ objectives: { n_obs: 1, psr: 2 } }])).toEqual(['n_obs', 'psr']);
		expect(plottableKeys([], ['crps', 'cost'])).toEqual(['cost', 'crps']);
	});

	it('labels inline-TOML strategies by id and truncates', () => {
		expect(candidateLabel('{ id = "gp-770a4384455337d3", genome_id = "gp-770a4384455337d3", steps = 5 }')).toBe('gp-770a4384455337d3');
		expect(candidateLabel('retrieval:ts=1s;h=3;lb=8;levels=0.05/0.25/0.5/0.75/0.95;kernel=16', 20)).toBe('retrieval:ts=1s;h=3…');
	});

	it('uses API ranks when present, else the non-dominated set under the directions', () => {
		const ranked = paretoPoints(
			[
				{ candidate: 'a', objectives: { sharpe: 1, turnover: 0.5 }, rank: 0, crowding: 'inf' },
				{ candidate: 'b', objectives: { sharpe: 0.5, turnover: 0.6 }, rank: 1, crowding: 0.2 },
				{ candidate: 'c', objectives: { sharpe: 0.5 }, rank: 1, crowding: 0.2 }
			],
			'sharpe',
			'turnover'
		);
		expect(ranked.map((p) => p.highlight)).toEqual([true, false]);
		const unranked = paretoPoints(
			[
				{ candidate: 'a', objectives: { sharpe: 1, turnover: 0.5 }, rank: -1 as never, crowding: 0 },
				{ candidate: 'b', objectives: { sharpe: 0.5, turnover: 0.1 }, rank: -1 as never, crowding: 0 },
				{ candidate: 'c', objectives: { sharpe: 0.4, turnover: 0.6 }, rank: -1 as never, crowding: 0 }
			],
			'sharpe',
			'turnover',
			{ sharpe: 'max', turnover: 'min' }
		);
		expect(unranked.map((p) => p.highlight)).toEqual([true, true, false]);
	});
});
