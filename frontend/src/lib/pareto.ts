// Pareto helpers for the Evolution Lab (spec 51 §6.4). Pure so it can be unit-tested.
import type { Direction, FrontMember } from './api';

/** Objectives where lower is better when the API sends no `directions` (spec 34 §6.3, spec 35 §6.2). */
export const MINIMISE = new Set([
	'reconstruction',
	'compute_cost',
	'crps',
	'nll',
	'loss',
	'max_drawdown',
	'turnover',
	'cost_sensitivity',
	'cost_share'
]);

/** Objective keys that are diagnostics rather than objectives; hidden from the pickers by default. */
export const DIAGNOSTIC_KEYS = new Set(['n_obs', 'n_trades', 'calibration', 'psr', 'cost_share']);

export interface XY {
	x: number;
	y: number;
}

/** Direction of `key`: the API's map first, then the static minimise list, else maximise. */
export function directionFor(key: string, directions?: Record<string, Direction> | null): Direction {
	const d = directions?.[key];
	if (d === 'min' || d === 'max') return d;
	return MINIMISE.has(key) ? 'min' : 'max';
}

/**
 * Marks the non-dominated points for two objectives. A point is dominated when another
 * point is at least as good on both objectives and strictly better on one. Duplicate
 * points are not dominated by each other.
 */
export function nonDominated(points: XY[], maximiseX = true, maximiseY = true): boolean[] {
	const sx = maximiseX ? 1 : -1;
	const sy = maximiseY ? 1 : -1;
	return points.map(
		(p) =>
			!points.some(
				(o) => sx * o.x >= sx * p.x && sy * o.y >= sy * p.y && (sx * o.x > sx * p.x || sy * o.y > sy * p.y)
			)
	);
}

/** Objective keys worth plotting: present on some member, not a pure diagnostic (unless nothing else). */
export function plottableKeys(members: { objectives: Record<string, number> }[], declared: string[] = []): string[] {
	const seen = new Set<string>(declared);
	for (const m of members) for (const k of Object.keys(m.objectives ?? {})) seen.add(k);
	const all = [...seen].sort();
	const main = all.filter((k) => !DIAGNOSTIC_KEYS.has(k));
	return main.length >= 2 ? main : all;
}

/** Short label for a candidate: the `id = "…"` of an inline-TOML strategy, else the string itself. */
export function candidateLabel(candidate: string, max = 40): string {
	const m = /\bid\s*=\s*"([^"]+)"/.exec(candidate);
	const s = m ? m[1] : candidate;
	return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

export interface ParetoPoint extends XY {
	label: string;
	highlight: boolean;
	rank: number;
	candidate: string;
	color?: string;
}

/**
 * Scatter points for two objectives over Pareto-front members (`fronts[][]` from the API or any
 * list with `rank`). `highlight` = rank 0 (the API's front) when ranks are present, else the
 * client-side non-dominated set under the given directions.
 */
export function paretoPoints(
	members: FrontMember[],
	xKey: string,
	yKey: string,
	directions?: Record<string, Direction> | null
): ParetoPoint[] {
	const pts = members
		.filter((m) => typeof m.objectives?.[xKey] === 'number' && typeof m.objectives?.[yKey] === 'number')
		.map((m) => ({
			x: m.objectives[xKey],
			y: m.objectives[yKey],
			label: candidateLabel(m.candidate),
			candidate: m.candidate,
			rank: typeof m.rank === 'number' ? m.rank : -1,
			highlight: false
		}));
	const haveRanks = pts.some((p) => p.rank >= 0);
	if (haveRanks) return pts.map((p) => ({ ...p, highlight: p.rank === 0 }));
	const front = nonDominated(pts, directionFor(xKey, directions) === 'max', directionFor(yKey, directions) === 'max');
	return pts.map((p, i) => ({ ...p, highlight: front[i] }));
}
