// Evolution generations table rows (spec 51 §6.4, D-049/D-052): merges the REST list
// (`GET /api/evolution/generations`), the richer `/ws/evolution` `generation.<id>` frames and the
// registry statistics of `GET /api/evolution/populations` (deflated Sharpe / PBO per generation).
import type { GenerationFrame, GenerationRow, Population, ValidationStats } from './api';

export interface GenerationTableRow {
	id: string;
	kind: string;
	generation: number;
	max_generations: number;
	rung: number;
	rungs: number;
	final: boolean;
	candidates: number;
	front_sizes: number[];
	/** Survivor ids per rung (ascending rung key) when a frame is known, else only the count. */
	survivors_by_rung: { rung: number; generation: number; ids: string[] }[];
	survivors: number;
	status: string;
	validation: ValidationStats | null;
	artifact_key: string | null;
	elapsed_ms: number | null;
	at: string | null;
	updated_at: string;
	/** `ws` when a live frame backs the row. */
	source: 'ws' | 'rest';
}

export function survivorsByRung(frame: GenerationFrame): GenerationTableRow['survivors_by_rung'] {
	return Object.entries(frame.survivors_by_rung ?? {})
		.map(([k, ids]) => {
			const key = Number(k);
			return { generation: Math.floor(key / 1000), rung: key % 1000, ids };
		})
		.sort((a, b) => a.generation - b.generation || a.rung - b.rung);
}

export function mergeGenerations(
	rows: GenerationRow[],
	frames: Map<string, GenerationFrame>,
	populations: Population[] = []
): GenerationTableRow[] {
	const stats = new Map(populations.map((p) => [p.id, p.stats]));
	const ids = new Set<string>([...rows.map((r) => r.id), ...frames.keys()]);
	const out: GenerationTableRow[] = [];
	for (const id of ids) {
		const f = frames.get(id);
		const r = rows.find((x) => x.id === id);
		const v = f?.validation ?? stats.get(id) ?? null;
		if (f) {
			out.push({
				id,
				kind: f.kind || r?.kind || '',
				generation: f.generation,
				max_generations: f.max_generations,
				rung: f.rung,
				rungs: f.rungs,
				final: f.final,
				candidates: f.candidates,
				front_sizes: f.front_sizes,
				survivors_by_rung: survivorsByRung(f),
				survivors: f.survivors.length,
				status: f.status || r?.status || 'running',
				validation: v,
				artifact_key: f.artifact_key || null,
				elapsed_ms: f.elapsed_ms ?? null,
				at: f.at,
				updated_at: f.updated_at,
				source: 'ws'
			});
		} else if (r) {
			out.push({
				id,
				kind: r.kind,
				generation: r.generation,
				max_generations: r.max_generations,
				rung: r.rung,
				rungs: r.rungs,
				final: r.final,
				candidates: r.candidates,
				front_sizes: r.front_sizes,
				survivors_by_rung: [],
				survivors: Array.isArray(r.survivors) ? r.survivors.length : r.survivors,
				status: r.status,
				validation: v,
				artifact_key: null,
				elapsed_ms: null,
				at: r.at,
				updated_at: r.updated_at,
				source: 'rest'
			});
		}
	}
	return out.sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
}

/** Which run actions make sense for a status (the server enforces; this avoids pointless 409s). */
export function runActions(status: string): ('pause' | 'resume' | 'stop')[] {
	switch (status) {
		case 'running':
			return ['pause', 'stop'];
		case 'paused':
			return ['resume', 'stop'];
		default:
			return [];
	}
}
