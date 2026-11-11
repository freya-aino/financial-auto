// Genome lineage tree layout (spec 51 §6.4) from `GET /api/graph/lineage/{genome_id}`:
// `ancestors_by_depth[i]` are the ancestors i + 1 `ParentOf` hops away. Which ancestor at depth
// d + 1 is a parent of which genome at depth d is not in that reply, so the edges below depth 1
// come from the parents lists of `GET /api/evolution/genomes/{id}` (fetched per node, capped).

export interface LineageGenome {
	genome_id: string;
	kind: string;
	family: string;
	created_at?: string | null;
	status?: string | null;
}

export interface LineageView {
	genome: LineageGenome;
	ancestors_by_depth: LineageGenome[][];
	fitness: { generation: number; pareto_rank: number; objectives: string | Record<string, number> }[];
}

export interface PlacedNode extends LineageGenome {
	depth: number;
	x: number;
	y: number;
	root: boolean;
}

export interface LineageLayout {
	nodes: PlacedNode[];
	/** child → parent (child is the younger genome). */
	edges: { from: string; to: string }[];
	width: number;
	height: number;
	/** Nodes whose parent links are unknown (their parents were not fetched). */
	unlinked: number;
}

export const COL_W = 150;
export const ROW_H = 34;

/** Genomes whose parents are needed to draw every edge: depths 1 .. D − 1, nearest first. */
export function needParents(v: LineageView, limit = 40): string[] {
	const out: string[] = [];
	const seen = new Set<string>([v.genome.genome_id]);
	for (let d = 0; d < v.ancestors_by_depth.length - 1; d++) {
		for (const g of v.ancestors_by_depth[d]) {
			if (seen.has(g.genome_id)) continue;
			seen.add(g.genome_id);
			out.push(g.genome_id);
		}
	}
	return out.slice(0, limit);
}

/**
 * Columns by depth (oldest ancestors on the left, the genome on the right), rows ordered by the
 * barycentre of their already placed children so edges cross less. A genome reachable at several
 * depths (diamond ancestry) is placed once, at its nearest depth.
 */
export function layoutLineage(v: LineageView, parents: Map<string, string[]>): LineageLayout {
	const depthOf = new Map<string, number>([[v.genome.genome_id, 0]]);
	const byId = new Map<string, LineageGenome>([[v.genome.genome_id, v.genome]]);
	const levels: string[][] = [[v.genome.genome_id]];
	v.ancestors_by_depth.forEach((lvl, i) => {
		const ids: string[] = [];
		for (const g of lvl) {
			if (depthOf.has(g.genome_id)) continue;
			depthOf.set(g.genome_id, i + 1);
			byId.set(g.genome_id, g);
			ids.push(g.genome_id);
		}
		levels.push(ids);
	});
	while (levels.length > 1 && !levels[levels.length - 1].length) levels.pop();

	const edges: { from: string; to: string }[] = [];
	for (const id of levels[1] ?? []) edges.push({ from: v.genome.genome_id, to: id });
	let unlinked = 0;
	for (let d = 1; d < levels.length - 1; d++) {
		for (const child of levels[d]) {
			const ps = parents.get(child);
			if (!ps) {
				unlinked++;
				continue;
			}
			for (const p of ps) if ((depthOf.get(p) ?? -1) > d) edges.push({ from: child, to: p });
		}
	}

	// row order: barycentre of children (already ordered, nearer depth)
	const row = new Map<string, number>([[v.genome.genome_id, 0]]);
	for (let d = 1; d < levels.length; d++) {
		const bary = (id: string) => {
			const ys = edges.filter((e) => e.to === id && row.has(e.from)).map((e) => row.get(e.from)!);
			return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : Number.POSITIVE_INFINITY;
		};
		levels[d].sort((a, b) => bary(a) - bary(b) || a.localeCompare(b));
		levels[d].forEach((id, i) => row.set(id, i));
	}

	const tallest = Math.max(...levels.map((l) => l.length));
	const height = Math.max(1, tallest) * ROW_H + 10;
	const maxDepth = levels.length - 1;
	const nodes: PlacedNode[] = [];
	levels.forEach((ids, d) => {
		const offset = ((tallest - ids.length) * ROW_H) / 2;
		ids.forEach((id, i) => {
			nodes.push({ ...byId.get(id)!, depth: d, x: (maxDepth - d) * COL_W + 8, y: offset + i * ROW_H + ROW_H / 2 + 5, root: d === 0 });
		});
	});
	return { nodes, edges, width: (maxDepth + 1) * COL_W, height, unlinked };
}

/** Parent genome ids out of a `GET /api/evolution/genomes/{id}` `parents` list. */
export function parentIds(parents: unknown[]): string[] {
	return parents
		.map((p) => (typeof p === 'object' && p && 'genome_id' in p ? String((p as { genome_id: unknown }).genome_id) : typeof p === 'string' ? p : ''))
		.filter(Boolean);
}

/** Short, distinguishable node label: the `id = "…"` of an inline-table genome, a short id,
 *  or for kv genomes (`nn_forecast:ts=1s;h=4;…`) the family prefix plus a 5-hex FNV-1a hash. */
export function nodeLabel(genomeId: string, max = 18): string {
	const m = /\bid\s*=\s*"([^"]+)"/.exec(genomeId);
	if (m) return m[1].length > max ? `${m[1].slice(0, max - 1)}…` : m[1];
	if (genomeId.length <= max) return genomeId;
	let h = 0x811c9dc5;
	for (let i = 0; i < genomeId.length; i++) h = Math.imul(h ^ genomeId.charCodeAt(i), 0x01000193);
	const hash = (h >>> 0).toString(16).padStart(8, '0').slice(0, 5);
	const prefix = genomeId.split(/[:;={ ]/)[0].slice(0, max - 6) || genomeId.slice(0, max - 6);
	return `${prefix}#${hash}`;
}
