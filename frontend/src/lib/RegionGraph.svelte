<script lang="ts">
	// Region transition graph (spec 51 §6.3): circular layout, edge width = posterior mean,
	// node radius = occupancy. Hovering a node highlights its outgoing transitions.
	import type { GraphEdge, GraphNode } from './api';

	interface Props {
		nodes: GraphNode[];
		edges: GraphEdge[];
		selected?: string | null;
		onselect?: (id: string | null) => void;
	}
	let { nodes, edges, selected = null, onselect }: Props = $props();

	const W = 520;
	const H = 420;
	const R = 160;
	const cx = W / 2;
	const cy = H / 2;

	let hover = $state<string | null>(null);
	const focus = $derived(hover ?? selected);

	const pos = $derived(
		new Map(
			nodes.map((n, i) => {
				const a = (2 * Math.PI * i) / Math.max(1, nodes.length) - Math.PI / 2;
				return [n.id, { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), a }];
			})
		)
	);
	const maxOcc = $derived(Math.max(1, ...nodes.map((n) => n.occupancy)));

	function edgePath(e: GraphEdge): string {
		const p = pos.get(e.from);
		const q = pos.get(e.to);
		if (!p || !q) return '';
		if (e.from === e.to) {
			// self-loop drawn outside the circle
			const ox = Math.cos(p.a) * 26;
			const oy = Math.sin(p.a) * 26;
			return `M ${p.x} ${p.y} C ${p.x + ox - oy} ${p.y + oy + ox}, ${p.x + ox + oy} ${p.y + oy - ox}, ${p.x} ${p.y}`;
		}
		// curve to the side so A→B and B→A don't overlap
		const mx = (p.x + q.x) / 2;
		const my = (p.y + q.y) / 2;
		const dx = q.x - p.x;
		const dy = q.y - p.y;
		const len = Math.hypot(dx, dy) || 1;
		const k = 0.18 * len;
		return `M ${p.x} ${p.y} Q ${mx - (dy / len) * k} ${my + (dx / len) * k} ${q.x} ${q.y}`;
	}

	const colorFor = (n: GraphNode) =>
		n.status === 'retired' ? 'var(--text-dim)' : n.status === 'unregistered' ? 'var(--warn)' : 'var(--accent)';
</script>

<svg viewBox="0 0 {W} {H}" role="img" aria-label="Region transition graph">
	<defs>
		<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
			<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-dim)" />
		</marker>
	</defs>
	{#each edges as e (e.from + '>' + e.to)}
		{@const active = !focus || e.from === focus}
		<path
			d={edgePath(e)}
			fill="none"
			stroke={active && focus ? 'var(--accent)' : 'var(--text-dim)'}
			stroke-width={0.5 + 7 * e.mean}
			stroke-opacity={active ? 0.25 + 0.6 * e.mean : 0.05}
			marker-end={e.from === e.to ? undefined : 'url(#arrow)'}
		>
			<title>{e.from} → {e.to}: {e.mean.toFixed(3)} [{e.lower.toFixed(3)}, {e.upper.toFixed(3)}] ({e.sampler_id})</title>
		</path>
	{/each}
	{#each nodes as n (n.id)}
		{@const p = pos.get(n.id)}
		{#if p}
			<g
				class="node"
				role="button"
				tabindex="0"
				onmouseenter={() => (hover = n.id)}
				onmouseleave={() => (hover = null)}
				onclick={() => onselect?.(selected === n.id ? null : n.id)}
				onkeydown={(ev) => ev.key === 'Enter' && onselect?.(selected === n.id ? null : n.id)}
			>
				<circle
					cx={p.x}
					cy={p.y}
					r={7 + 9 * (n.occupancy / maxOcc)}
					fill={colorFor(n)}
					fill-opacity={focus && focus !== n.id ? 0.35 : 0.9}
					stroke={selected === n.id ? 'var(--text)' : 'none'}
					stroke-width="2"
				>
					<title>{n.id} · lineage {n.lineage_id ?? '—'} · {n.status} · occupancy {n.occupancy}</title>
				</circle>
				<text x={cx + (R + 34) * Math.cos(p.a)} y={cy + (R + 34) * Math.sin(p.a)} text-anchor="middle" dominant-baseline="middle">
					{n.id}
				</text>
			</g>
		{/if}
	{/each}
</svg>

<style>
	svg {
		width: 100%;
		max-height: 440px;
		display: block;
	}
	text {
		font: 11px var(--mono);
		fill: var(--text-dim);
		pointer-events: none;
	}
	.node {
		cursor: pointer;
		outline: none;
	}
</style>
