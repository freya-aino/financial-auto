<script lang="ts">
	// Experts of a persisted snapshot (`GET /api/latent/{snapshot}/experts`): routing gate mean,
	// top share, regions, and the correlation matrix between experts as a heatmap (spec 51 §6.3).
	import Empty from './Empty.svelte';
	import Heatmap from './Heatmap.svelte';
	import { ago, type ExpertsView } from './api';

	interface Props {
		view: ExpertsView | null;
		error?: string;
		now?: number;
	}
	let { view, error = '', now = Date.now() }: Props = $props();

	const labels = $derived((view?.covariance?.experts ?? view?.experts.map((e) => e.expert_index) ?? []).map((i) => `E${i}`));
	let matrixKind = $state<'correlation' | 'matrix'>('correlation');
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Experts</h2>
		{#if view}
			<span class="dim">source {view.source} · {view.live.messages} live states{view.live.last_as_of ? ` · last ${ago(view.live.last_as_of, now)} ago` : ''}</span>
		{/if}
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if view?.experts.length}
		<table>
			<thead><tr><th>expert</th><th>space</th><th>dim</th><th>regions (live)</th><th>gate mean</th><th>top share</th><th>regions seen</th><th>‖μ‖ mean</th><th>σ̄</th></tr></thead>
			<tbody>
				{#each view.experts as e (e.expert_index)}
					{@const r = e.routing}
					<tr>
						<td class="mono">E{e.expert_index}</td>
						<td class="mono dim">{e.expert_space_id}</td>
						<td class="mono">{e.dim}</td>
						<td class="mono">{e.regions} <span class="dim">({e.regions_live})</span></td>
						<td>
							{#if r}
								<div class="bar"><div style="width: {100 * Math.min(1, r.gate_mean)}%"></div></div>
								<span class="mono">{r.gate_mean.toFixed(3)}</span>
							{:else}<span class="dim">—</span>{/if}
						</td>
						<td>
							{#if r}
								<div class="bar"><div class="top" style="width: {100 * Math.min(1, r.top_share)}%"></div></div>
								<span class="mono">{(100 * r.top_share).toFixed(1)}%</span>
							{:else}<span class="dim">—</span>{/if}
						</td>
						<td class="mono">{r?.regions_seen ?? '—'}</td>
						<td class="mono">{r ? r.norm_mean.toFixed(3) : '—'}</td>
						<td class="mono">{r ? r.uncertainty_mean.toFixed(3) : '—'}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		{#if view.covariance}
			<div class="heat-head">
				<h3 class="sub">Expert {matrixKind === 'correlation' ? 'correlation' : 'covariance'} <span class="dim">· n={view.covariance.n} · feature: {view.covariance.feature}</span></h3>
				<select bind:value={matrixKind}>
					<option value="correlation">correlation</option>
					<option value="matrix">covariance</option>
				</select>
			</div>
			<div class="heat">
				<Heatmap matrix={matrixKind === 'correlation' ? view.covariance.correlation : view.covariance.matrix} {labels} title="expert {matrixKind}" digits={matrixKind === 'correlation' ? 2 : 3} />
			</div>
		{:else}
			<p class="dim">no covariance yet (needs latent.state.* frames for this snapshot)</p>
		{/if}
	{:else if !error}
		<Empty producer="GET /api/latent/{'{snapshot}'}/experts (HelixDB ExpertSpace rows + latent.state.*)" />
	{/if}
</section>

<style>
	.bar {
		display: inline-block;
		width: 70px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
		margin-right: 6px;
	}
	.bar > div {
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
	}
	.bar > .top {
		background: var(--warn);
	}
	.sub {
		font-size: 12px;
	}
	.heat-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin: 12px 0 6px;
	}
	.heat {
		max-width: 520px;
	}
</style>
