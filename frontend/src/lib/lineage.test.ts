import { describe, expect, it } from 'vitest';
import { layoutLineage, needParents, nodeLabel, parentIds, type LineageView } from './lineage';

const g = (id: string) => ({ genome_id: id, kind: 'strategy', family: 'gp' });
const view: LineageView = {
	genome: g('c'),
	ancestors_by_depth: [[g('p1'), g('p2')], [g('gp1'), g('gp2'), g('p1')]],
	fitness: []
};

describe('lineage layout', () => {
	it('asks for parents of every non-last level', () => {
		expect(needParents(view)).toEqual(['p1', 'p2']);
	});
	it('places each genome once and links known parents', () => {
		const l = layoutLineage(view, new Map([['p1', ['gp1']], ['p2', ['gp2', 'p1']]]));
		expect(l.nodes.map((n) => n.genome_id).sort()).toEqual(['c', 'gp1', 'gp2', 'p1', 'p2']);
		expect(l.nodes.find((n) => n.genome_id === 'p1')?.depth).toBe(1);
		expect(l.edges).toEqual(expect.arrayContaining([{ from: 'c', to: 'p1' }, { from: 'c', to: 'p2' }, { from: 'p1', to: 'gp1' }, { from: 'p2', to: 'gp2' }]));
		// p2 → p1 is not an ancestor edge across depths (same depth): dropped
		expect(l.edges).not.toContainEqual({ from: 'p2', to: 'p1' });
		expect(l.unlinked).toBe(0);
		// the root is the rightmost column
		const root = l.nodes.find((n) => n.root)!;
		expect(Math.max(...l.nodes.map((n) => n.x))).toBe(root.x);
	});
	it('counts unlinked nodes when parents were not fetched', () => {
		expect(layoutLineage(view, new Map()).unlinked).toBe(2);
	});
	it('reads parent ids from genome objects or strings', () => {
		expect(parentIds([{ genome_id: 'a' }, 'b', null])).toEqual(['a', 'b']);
	});
});

describe('lineage node labels', () => {
	it('distinguishes kv genomes by hash and keeps short ids', () => {
		const a = nodeLabel('nn_forecast:ts=1s;h=4;lb=11;seed=7');
		const b = nodeLabel('nn_forecast:ts=1s;h=4;lb=11;seed=8');
		expect(a.startsWith('nn_forecast#')).toBe(true);
		expect(a).not.toBe(b);
		expect(nodeLabel('{ id = "gp-abc", expr = "x" }')).toBe('gp-abc');
		expect(nodeLabel('g1')).toBe('g1');
	});
});
