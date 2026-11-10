<script lang="ts">
	// 2-D PCA projection of one expert space (`GET /api/latent/{snapshot}/projection`), coloured
	// by region id, with the explained variance shown (spec 51 §6.3).
	import Empty from './Empty.svelte';
	import Scatter from './Scatter.svelte';
	import { live, short, type ProjectionView } from './api';
	import { loadPref, savePref } from './prefs';
	import { assignRegionColors, explainedLabel } from './projection';

	interface Props {
		snapshot: string;
		experts: number[];
		/** Region id pinned elsewhere (RegionGraph click) → highlighted here. */
		selected?: string | null;
		onselect?: (regionId: string | null) => void;
	}
	let { snapshot, experts, selected = null, onselect }: Props = $props();

	let expert = $state(loadPref('latent.proj.expert', 0));
	let range = $state(loadPref('latent.proj.range', 'all'));
	let n = $state(loadPref('latent.proj.n', 2000));
	let view = $state<ProjectionView | null>(null);
	let error = $state('');
	let busy = $state(false);
	$effect(() => savePref('latent.proj.expert', expert));
	$effect(() => savePref('latent.proj.range', range));
	$effect(() => savePref('latent.proj.n', n));

	const colors = $derived(assignRegionColors(view?.points.map((p) => p.region_id) ?? []));
	const points = $derived(
		(view?.points ?? []).map((p) => ({
			x: p.coords[0] ?? 0,
			y: p.coords[1] ?? 0,
			label: `${p.region_id} · ${short(p.instrument_id, 8)} · ${p.ts.slice(0, 16)} · σ ${p.uncertainty.toFixed(2)}`,
			color: colors.get(p.region_id),
			highlight: selected !== null && p.region_id === selected
		}))
	);
	const regionCounts = $derived.by(() => {
		const m = new Map<string, number>();
		for (const p of view?.points ?? []) m.set(p.region_id, (m.get(p.region_id) ?? 0) + 1);
		return [...m].sort((a, b) => b[1] - a[1]);
	});

	export async function refresh() {
		if (!snapshot) return;
		busy = true;
		try {
			view = await live.projection(snapshot, { expert, range, n, dims: 2 });
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			view = null;
		} finally {
			busy = false;
		}
	}

	$effect(() => {
		if (experts.length && !experts.includes(expert)) expert = experts[0];
	});
	$effect(() => {
		// refetch whenever snapshot/expert/range/n change
		void snapshot;
		void expert;
		void range;
		void n;
		void refresh();
	});
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Projection</h2>
		<div class="controls">
			<label>expert
				<select bind:value={expert}>{#each experts as e}<option value={e}>E{e}</option>{/each}</select>
			</label>
			<label>range
				<select bind:value={range}>
					{#each ['15m', '1h', '6h', '1d', '7d', 'all'] as r}<option value={r}>{r}</option>{/each}
				</select>
			</label>
			<label>n
				<select bind:value={n}>{#each [500, 2000, 5000, 20000] as k}<option value={k}>{k}</option>{/each}</select>
			</label>
			<button onclick={refresh} disabled={busy}>{busy ? '…' : 'refresh'}</button>
		</div>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if view && points.length}
		<p class="dim small">
			{view.expert_space_id} · {view.fetched}/{view.total_in_space} points{view.cached ? ' (cached)' : ''} · PCA {view.dim}→{view.dims}: <span class="mono">{explainedLabel(view.pca.explained_variance_ratio)}</span>
		</p>
		<Scatter {points} xLabel="PC1 ({(100 * (view.pca.explained_variance_ratio[0] ?? 0)).toFixed(0)}%)" yLabel="PC2 ({(100 * (view.pca.explained_variance_ratio[1] ?? 0)).toFixed(0)}%)" radius={3} onselect={(i) => onselect?.(view?.points[i]?.region_id ?? null)} />
		<div class="legend">
			{#each regionCounts as [id, count] (id)}
				<button class="swatch" class:sel={selected === id} style="--c: {colors.get(id)}" onclick={() => onselect?.(selected === id ? null : id)} title="{count} points">
					<i></i><span class="mono">{id}</span><span class="dim">{count}</span>
				</button>
			{/each}
		</div>
	{:else if !error}
		<Empty producer="LatentVec rows in HelixDB for {snapshot}-e{expert} (model-runtime hub sink, D-048)" hint="wait for closed bars at 1m or slower" />
	{/if}
</section>

<style>
	.controls {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
	}
	.controls label {
		display: flex;
		gap: 4px;
		align-items: center;
		color: var(--text-dim);
	}
	.small {
		font-size: 11px;
		margin: 0 0 6px;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		margin-top: 6px;
	}
	.swatch {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		background: none;
		border: 1px solid transparent;
		padding: 0 4px;
		font-size: 11px;
	}
	.swatch.sel {
		border-color: var(--border);
		background: var(--bg-inset);
	}
	.swatch i {
		width: 9px;
		height: 9px;
		border-radius: 50%;
		background: var(--c);
	}
</style>
