<script lang="ts">
	// Genome lineage tree (spec 51 §6.4): `GET /api/graph/lineage/{genome_id}` for the ancestry
	// levels, parents of each ancestor from `GET /api/evolution/genomes/{id}` for the edges.
	import { getApi, errText } from './console';
	import { live } from './api';
	import { COL_W, layoutLineage, needParents, nodeLabel, parentIds, type LineageView } from './lineage';

	interface Props {
		genomeId: string;
		/** Open another genome (click on a node). */
		onopen?: (id: string) => void;
	}
	let { genomeId, onopen }: Props = $props();

	const PARENT_FETCH_LIMIT = 40;

	let view = $state<LineageView | null>(null);
	let parents = $state(new Map<string, string[]>());
	let error = $state('');
	let busy = $state(false);
	let pending = $state(0);

	$effect(() => {
		const id = genomeId;
		view = null;
		parents = new Map();
		error = '';
		busy = true;
		getApi<LineageView>(`/api/graph/lineage/${encodeURIComponent(id)}`)
			.then(async (v) => {
				if (id !== genomeId) return;
				view = v;
				const need = needParents(v, PARENT_FETCH_LIMIT);
				pending = need.length;
				const out = new Map<string, string[]>();
				// a few at a time: each is one Helix read
				for (let i = 0; i < need.length; i += 6) {
					const chunk = need.slice(i, i + 6);
					const res = await Promise.allSettled(chunk.map((g) => live.genome(g)));
					res.forEach((r, k) => r.status === 'fulfilled' && out.set(chunk[k], parentIds(r.value.parents)));
					if (id !== genomeId) return;
					parents = new Map(out);
					pending = Math.max(0, need.length - i - chunk.length);
				}
			})
			.catch((e) => {
				if (id === genomeId) error = errText(e);
			})
			.finally(() => {
				if (id === genomeId) busy = false;
			});
	});

	const layout = $derived(view ? layoutLineage(view, parents) : null);
	const pos = $derived(new Map((layout?.nodes ?? []).map((n) => [n.genome_id, n])));
	const statusClass = (s: string | null | undefined) => (s === 'active' ? 'ok' : s === 'retired' ? 'dim' : s ? 'warn' : '');
</script>

<section class="lineage">
	<h3>Lineage <span class="dim">GET /api/graph/lineage/{nodeLabel(genomeId)}{layout ? ` · ${layout.nodes.length - 1} ancestors over ${Math.max(0, ...layout.nodes.map((n) => n.depth))} generations` : ''}{pending ? ` · fetching parents (${pending} left)` : ''}</span></h3>
	{#if error}<p class="badge warn">{error}</p>{/if}
	{#if busy && !view}<p class="dim">loading…</p>{/if}
	{#if layout && layout.nodes.length > 1}
		<div class="scroll">
			<svg width={layout.width} height={layout.height} role="img" aria-label="lineage tree of {genomeId}">
				{#each layout.edges as e (e.from + '>' + e.to)}
					{@const a = pos.get(e.from)}
					{@const b = pos.get(e.to)}
					{#if a && b}
						<path d="M{a.x} {a.y} C{a.x - 30} {a.y}, {b.x + COL_W - 28 + 30} {b.y}, {b.x + COL_W - 28} {b.y}" class="edge" />
					{/if}
				{/each}
				{#each layout.nodes as n (n.genome_id)}
					<g transform="translate({n.x} {n.y - 11})" class="node" class:root={n.root} role="button" tabindex="0" onclick={() => !n.root && onopen?.(n.genome_id)} onkeydown={(e) => e.key === 'Enter' && !n.root && onopen?.(n.genome_id)}>
						<rect width={COL_W - 28} height="22" rx="4" />
						<text x="6" y="15" class="mono">{nodeLabel(n.genome_id)}</text>
						{#if n.status}<circle cx={COL_W - 36} cy="11" r="4" class={statusClass(n.status)} />{/if}
						<title>{n.genome_id} · {n.kind}/{n.family}{n.status ? ` · ${n.status}` : ''}{n.created_at ? ` · ${n.created_at}` : ''} — depth {n.depth}{n.root ? '' : ' (click to open)'}</title>
					</g>
				{/each}
			</svg>
		</div>
		{#if layout.unlinked}<p class="dim small">{layout.unlinked} ancestors without fetched parent links (limit {PARENT_FETCH_LIMIT} lookups); their columns are placed but not connected.</p>{/if}
	{:else if view}
		<p class="dim small">no ParentOf ancestors in HelixDB (a gen-0 or hand-written genome)</p>
	{/if}
</section>

<style>
	.lineage {
		margin-top: 12px;
	}
	.scroll {
		overflow: auto;
		max-height: 360px;
	}
	.edge {
		fill: none;
		stroke: var(--border);
		stroke-width: 1.5;
	}
	.node rect {
		fill: var(--bg-inset);
		stroke: var(--border);
	}
	.node.root rect {
		stroke: var(--accent);
		stroke-width: 2;
	}
	.node:not(.root) {
		cursor: pointer;
	}
	.node:not(.root):hover rect {
		stroke: var(--text-dim);
	}
	.node text {
		font-size: 11px;
		fill: var(--text);
	}
	circle {
		fill: var(--text-dim);
	}
	circle.ok {
		fill: var(--ok);
	}
	circle.warn {
		fill: var(--warn);
	}
	.small {
		font-size: 11px;
	}
</style>
