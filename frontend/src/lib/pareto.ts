// Pareto helpers for the Evolution Lab (spec 51 §6.4). Pure so it can be unit-tested.

/** Objectives where lower is better; everything else is maximised (spec 34). */
export const MINIMISE = new Set(['reconstruction', 'compute_cost', 'crps', 'nll', 'loss']);

export interface XY {
	x: number;
	y: number;
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
