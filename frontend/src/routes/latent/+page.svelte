<script lang="ts">
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import QuantileStrip from '../../lib/QuantileStrip.svelte';
	import RegionGraph from '../../lib/RegionGraph.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import {
		ago,
		live,
		poll,
		short,
		type DecoderOutputView,
		type GraphEdge,
		type GraphNode,
		type LatentStateView,
		type Region
	} from '../../lib/api';

	let configured = $state<string[]>([]);
	let seen = $state<string[]>([]);
	let ts = $state('');
	let nodes = $state<GraphNode[]>([]);
	let edges = $state<GraphEdge[]>([]);
	let regions = $state<Region[]>([]);
	let states = $state<LatentStateView[]>([]);
	let outputs = $state<DecoderOutputView[]>([]);
	let families = $state<string[]>([]);
	let family = $state('');
	let selected = $state<string | null>(null);
	let error = $state('');
	let now = $state(Date.now());

	const timescales = $derived([...new Set([...configured, ...seen])]);
	const selectedRows = $derived(edges.filter((e) => e.from === selected).sort((a, b) => b.mean - a.mean));

	async function refresh() {
		now = Date.now();
		try {
			const t = await live.timescales();
			configured = t.configured;
			seen = t.seen;
			if (!ts) ts = t.seen[0] ?? t.configured[0] ?? '1h';
			const [g, r, s, d] = await Promise.all([
				live.graph(ts),
				live.regions(ts),
				live.states(ts),
				live.decoderOutputs(family || undefined, 30)
			]);
			nodes = g.nodes;
			edges = g.edges;
			regions = r.regions;
			states = s.states;
			outputs = d.outputs;
			families = d.families;
			error = '';
		} catch (e) {
			error = String(e);
		}
	}

	onMount(() => poll(refresh, 2000));
</script>

<ViewHeader title="Latent & Graph" spec="51 §6.3" note="live from latent.* and decoder.output.*" />

<div class="toolbar">
	<label>
		timescale
		<select bind:value={ts} onchange={() => ((selected = null), refresh())}>
			{#each timescales as t}
				<option value={t}>{t}{seen.includes(t) ? '' : ' (no data)'}</option>
			{/each}
		</select>
	</label>
	<span class="dim">{nodes.length} regions · {edges.length} transitions · {states.length} instruments</span>
	{#if error}<span class="badge danger">{error}</span>{/if}
</div>

<div class="grid two">
	<section class="panel">
		<div class="panel-head">
			<h2>Region transition graph</h2>
			<span class="dim">width = posterior mean · hover for 90% interval · click to pin</span>
		</div>
		{#if nodes.length}
			<RegionGraph {nodes} {edges} {selected} onselect={(id) => (selected = id)} />
			{#if selected}
				<h3 class="sub">Outgoing from <code>{selected}</code></h3>
				<table>
					<thead><tr><th>to</th><th>mean</th><th>interval</th><th>sampler</th></tr></thead>
					<tbody>
						{#each selectedRows as e}
							<tr>
								<td class="mono">{e.to}</td>
								<td>
									<div class="bar"><div style="width: {100 * e.mean}%"></div></div>
									<span class="mono">{e.mean.toFixed(3)}</span>
								</td>
								<td class="mono dim">[{e.lower.toFixed(3)}, {e.upper.toFixed(3)}]</td>
								<td class="mono dim">{e.sampler_id}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		{:else}
			<Empty producer="latent.region.{ts} / latent.transition.{ts} (model-runtime MCMC stage)" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Regions</h2>
			<span class="dim">lineage survives retraining (D-007)</span>
		</div>
		{#if regions.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>region</th><th>lineage</th><th>expert</th><th>status</th><th>occupancy</th><th>snapshot</th></tr></thead>
					<tbody>
						{#each regions as r (r.region_id)}
							<tr class:sel={r.region_id === selected} onclick={() => (selected = r.region_id)}>
								<td class="mono">{r.region_id}</td>
								<td class="mono dim">{r.lineage_id}</td>
								<td class="mono">{r.expert_index}</td>
								<td><span class="badge" class:ok={r.status === 'created'} class:warn={r.status !== 'created'}>{r.status}</span></td>
								<td class="mono">{r.occupancy}</td>
								<td class="mono dim">{r.snapshot_id}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<Empty producer="latent.region.{ts}" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Expert states per instrument</h2>
			<span class="dim">router weights · region per expert · ‖μ‖ · σ̄</span>
		</div>
		{#if states.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>instrument</th><th>router</th><th>experts</th><th>as of</th></tr></thead>
					<tbody>
						{#each states as s (s.instrument_id)}
							<tr>
								<td class="mono" title={s.instrument_id}>{short(s.instrument_id, 12)}</td>
								<td>
									<div class="router" title={s.router_weights.map((w) => w.toFixed(2)).join(' / ')}>
										{#each s.router_weights as w, i}
											<div style="flex: {w}; opacity: {0.35 + 0.65 * (1 - i / Math.max(1, s.router_weights.length))}"></div>
										{/each}
									</div>
								</td>
								<td>
									{#each s.experts as e}
										<span
											class="expert mono"
											class:sel={e.region_id === selected}
											title="expert {e.expert_index}: region {e.region_id} (lineage {e.region_lineage_id}), dims {e.dims}, ‖μ‖ {e.norm.toFixed(2)}, σ̄ {e.uncertainty.toFixed(2)}"
											>E{e.expert_index}:{e.region_id} <span class="dim">‖μ‖{e.norm.toFixed(1)} σ{e.uncertainty.toFixed(2)}</span></span
										>
									{/each}
								</td>
								<td class="mono dim">{ago(s.as_of, now)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<Empty producer="latent.state.{ts}.* (model-runtime encoders)" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Decoder outputs</h2>
			<select bind:value={family} onchange={refresh}>
				<option value="">all families</option>
				{#each families as f}<option value={f}>{f}</option>{/each}
				<option value="text">text.*</option>
				<option value="visual">visual.*</option>
			</select>
		</div>
		{#if outputs.length}
			<div class="scroll tall feed">
				{#each outputs as o (o.received_at + o.decoder_id + o.family)}
					<div class="out">
						<div class="out-head">
							<span class="badge">{o.family}</span>
							<span class="mono dim">{o.decoder_id} · genome {o.genome_id} · {o.timescale}</span>
							<span class="mono dim right">{ago(o.received_at, now)} ago</span>
						</div>
						{#if o.quantiles.length}
							<div class="row"><span class="dim">forecast</span><QuantileStrip quantiles={o.quantiles} /><span class="mono dim">median {o.quantiles[Math.floor(o.quantiles.length / 2)].toFixed(2)}</span></div>
						{/if}
						{#if o.analogs.length}
							<table class="analogs">
								<thead><tr><th>analog window</th><th>similarity</th><th>time distance</th></tr></thead>
								<tbody>
									{#each o.analogs as a}
										<tr>
											<td class="mono">{a.window_id}</td>
											<td><div class="bar"><div style="width: {100 * Math.max(0, a.similarity)}%"></div></div> <span class="mono">{a.similarity.toFixed(3)}</span></td>
											<td class="mono">{a.time_distance_days.toFixed(0)} d</td>
										</tr>
									{/each}
								</tbody>
							</table>
						{/if}
						{#if o.output_refs.length}
							<div class="mono dim refs">{o.output_refs.join(' · ')}</div>
						{/if}
					</div>
				{/each}
			</div>
		{:else}
			<Empty producer="decoder.output.* (decoder population / evo-workers)" />
		{/if}
	</section>
</div>

<style>
	.toolbar {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-bottom: 12px;
	}
	.two {
		grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
	}
	.tall {
		max-height: 440px;
	}
	.sub {
		font-size: 12px;
		margin: 10px 0 6px;
	}
	.bar {
		display: inline-block;
		width: 80px;
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
	.router {
		display: flex;
		width: 120px;
		height: 10px;
		gap: 1px;
	}
	.router > div {
		background: var(--accent);
	}
	.expert {
		display: inline-block;
		margin: 0 8px 2px 0;
		white-space: nowrap;
	}
	.sel,
	tr.sel td {
		color: var(--accent);
	}
	tr {
		cursor: default;
	}
	.feed {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.out {
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 8px;
		background: var(--bg-inset);
	}
	.out-head,
	.row {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 4px;
	}
	.right {
		margin-left: auto;
	}
	.analogs th,
	.analogs td {
		background: transparent;
		padding: 2px 6px;
	}
	.refs {
		word-break: break-all;
	}
</style>
