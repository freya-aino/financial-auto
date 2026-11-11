<script lang="ts">
	// Latent & Graph (spec 51 §6.3). Live state per instrument arrives on /ws/latent
	// (`state.<snapshot>.<instrument>`, `region.<snapshot>.<id>`); REST polling is the fallback.
	// The persisted side (snapshots, experts, projection, MCMC graph, explorer) comes from HelixDB.
	import { onMount, untrack } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import Experts from '../../lib/Experts.svelte';
	import GraphExplorer from '../../lib/GraphExplorer.svelte';
	import Projection from '../../lib/Projection.svelte';
	import QuantileStrip from '../../lib/QuantileStrip.svelte';
	import RegionGraph from '../../lib/RegionGraph.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import {
		ago,
		live,
		poll,
		short,
		type DecoderOutputView,
		type ExpertsView,
		type GraphEdge,
		type GraphNode,
		type LatentStateView,
		type McmcGraph,
		type Region,
		type Snapshot
	} from '../../lib/api';
	import { loadPref, savePref } from '../../lib/prefs';
	import { runtime } from '../../lib/runtime.svelte';
	import { latentWs, onBus } from '../../lib/ws.svelte';

	let configured = $state<string[]>([]);
	let seen = $state<string[]>([]);
	let ts = $state(loadPref('latent.ts', ''));
	let nodes = $state<GraphNode[]>([]);
	let edges = $state<GraphEdge[]>([]);
	let regions = $state<Region[]>([]);
	let states = $state<LatentStateView[]>([]);
	let outputs = $state<DecoderOutputView[]>([]);
	let families = $state<string[]>([]);
	let family = $state(loadPref('latent.family', ''));
	let selected = $state<string | null>(null);
	let error = $state('');
	let now = $state(Date.now());

	// persisted (HelixDB)
	let snapshots = $state<Snapshot[]>([]);
	let snapshotsError = $state('');
	let snapshot = $state(loadPref('latent.snapshot', ''));
	let experts = $state<ExpertsView | null>(null);
	let expertsError = $state('');
	let mcmc = $state<McmcGraph | null>(null);
	let mcmcError = $state('');
	let graphSource = $state(loadPref<'live' | 'persisted'>('latent.graphSource', 'live'));
	let actionMsg = $state('');
	let actionOk = $state(true);
	let actionBusy = $state(false);

	$effect(() => savePref('latent.ts', ts));
	$effect(() => savePref('latent.family', family));
	$effect(() => savePref('latent.snapshot', snapshot));
	$effect(() => savePref('latent.graphSource', graphSource));

	const timescales = $derived([...new Set([...configured, ...seen])]);
	const forTimescale = $derived(snapshots.filter((s) => s.timescale === ts));
	const snapshotRow = $derived(snapshots.find((s) => s.snapshot_id === snapshot) ?? null);
	const expertIndices = $derived(experts?.experts.map((e) => e.expert_index) ?? [0, 1, 2]);

	/** Persisted MCMC graph in the RegionGraph's node/edge shape. */
	const persistedNodes = $derived<GraphNode[]>(
		(mcmc?.nodes ?? []).map((r) => ({
			id: r.region_id,
			lineage_id: r.lineage_id,
			expert_index: mcmc?.spaces.find((s) => s.expert_space_id === r.expert_space_id)?.index ?? null,
			status: r.status === 'live' ? 'created' : r.status,
			occupancy: 0
		}))
	);
	const persistedEdges = $derived<GraphEdge[]>(
		(mcmc?.edges ?? []).map((e) => ({ from: e.from_region_id, to: e.to_region_id, mean: e.mean, lower: e.lower, upper: e.upper, sampler_id: e.sampler_id }))
	);
	const graphNodes = $derived(graphSource === 'persisted' ? persistedNodes : nodes);
	const graphEdges = $derived(graphSource === 'persisted' ? persistedEdges : edges);
	const selectedRows = $derived(graphEdges.filter((e) => e.from === selected).sort((a, b) => b.mean - a.mean));

	// /ws/latent: latest state per instrument for the selected snapshot, region lifecycle events.
	const stateTopics = latentWs.topics<LatentStateView>('state.>');
	const regionTopic = latentWs.topic('region.>');
	$effect(() => {
		if (!stateTopics.count) return;
		const [snap, scale] = untrack(() => [snapshot, ts]);
		const fresh = new Map(untrack(() => states).map((s) => [s.instrument_id, s]));
		for (const f of stateTopics.entries.values()) {
			if (f.data.snapshot_id === snap || f.data.timescale === scale) fresh.set(f.data.instrument_id, f.data);
		}
		states = [...fresh.values()].sort((a, b) => a.instrument_id.localeCompare(b.instrument_id));
		now = Date.now();
	});
	$effect(() => {
		if (regionTopic.count) void refreshLive();
	});
	$effect(() => {
		if (snapshot) latentWs.setQuery({ snapshot });
	});

	async function refreshLive() {
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

	async function refreshSnapshots() {
		try {
			snapshots = (await live.snapshots()).snapshots.sort((a, b) => a.timescale.localeCompare(b.timescale) || b.created_at.localeCompare(a.created_at));
			snapshotsError = '';
		} catch (e) {
			snapshotsError = e instanceof Error ? e.message : String(e);
		}
	}

	async function refreshPersisted() {
		if (!snapshot) return;
		const [e, g] = await Promise.allSettled([live.experts(snapshot), live.mcmcGraph(snapshot, ts)]);
		if (e.status === 'fulfilled') {
			experts = e.value;
			expertsError = '';
		} else expertsError = e.reason instanceof Error ? e.reason.message : String(e.reason);
		if (g.status === 'fulfilled') {
			mcmc = g.value;
			mcmcError = '';
		} else {
			mcmc = null;
			mcmcError = g.reason instanceof Error ? g.reason.message : String(g.reason);
		}
	}

	// pick the active snapshot of the timescale whenever the timescale changes or the pick is stale
	$effect(() => {
		if (!forTimescale.length) return;
		if (!forTimescale.some((s) => s.snapshot_id === snapshot)) {
			snapshot = (forTimescale.find((s) => s.status === 'active') ?? forTimescale[0]).snapshot_id;
		}
	});
	$effect(() => {
		void snapshot;
		void ts;
		void refreshPersisted();
	});

	async function snapshotAction(action: 'promote' | 'rollback') {
		if (!snapshotRow) return;
		const what =
			action === 'promote'
				? `Promote ${snapshotRow.snapshot_id} to the active ${ts} snapshot? Other active snapshots of ${ts} are retired.`
				: `Roll back ${snapshotRow.snapshot_id}? It is retired and the newest other retired ${ts} snapshot becomes active again.`;
		if (!confirm(`${what} This is audited (latent.snapshot.${action}).`)) return;
		actionBusy = true;
		try {
			await live.snapshotAction(snapshotRow.snapshot_id, action);
			actionMsg = `${action} ok`;
			actionOk = true;
			await refreshSnapshots();
		} catch (e) {
			actionMsg = e instanceof Error ? e.message : String(e);
			actionOk = false;
		} finally {
			actionBusy = false;
		}
	}

	function refresh() {
		void refreshLive();
		void refreshPersisted();
	}

	onMount(() => {
		void refreshSnapshots();
		const stopPoll = poll(refreshLive, () => (latentWs.live ? runtime.pollMs * 10 : runtime.pollMs));
		const stopPersisted = poll(refreshPersisted, () => runtime.pollMs * 15);
		const stopSnapshots = poll(refreshSnapshots, () => runtime.pollMs * 30);
		const stopBus = onBus(['decoder.output.>'], refreshLive);
		return () => {
			stopPoll();
			stopPersisted();
			stopSnapshots();
			stopBus();
			stateTopics.close();
			regionTopic.close();
		};
	});
</script>

<ViewHeader title="Latent & Graph" spec="51 §6.3" note="live states on /ws/latent (state.<snapshot>.*), persisted graph from HelixDB; REST polling as fallback" />

<div class="toolbar">
	<label>
		timescale
		<select bind:value={ts} onchange={() => ((selected = null), refresh())}>
			{#each timescales as t}
				<option value={t}>{t}{seen.includes(t) ? '' : ' (no data)'}</option>
			{/each}
		</select>
	</label>
	<label>
		snapshot
		<select bind:value={snapshot} disabled={!forTimescale.length} title="GET /api/latent/snapshots (HelixDB)">
			{#each forTimescale as s (s.snapshot_id)}
				<option value={s.snapshot_id}>{s.snapshot_id} · {s.status} · gen {s.generation}</option>
			{:else}
				<option value="">no persisted snapshot for {ts}</option>
			{/each}
		</select>
	</label>
	{#if snapshotRow}
		<span class="badge" class:ok={snapshotRow.status === 'active'} class:warn={snapshotRow.status === 'candidate'}>{snapshotRow.status}</span>
		<button onclick={() => snapshotAction('promote')} disabled={actionBusy || snapshotRow.status === 'active'} title="POST /api/latent/snapshots/{snapshotRow.snapshot_id}/promote">promote</button>
		<button onclick={() => snapshotAction('rollback')} disabled={actionBusy || snapshotRow.status === 'retired'} title="POST /api/latent/snapshots/{snapshotRow.snapshot_id}/rollback">rollback</button>
		{#if actionMsg}<span class="badge" class:ok={actionOk} class:danger={!actionOk}>{actionMsg}</span>{/if}
	{/if}
	<span class="dim">{graphNodes.length} regions · {graphEdges.length} transitions · {states.length} instruments</span>
	<span class="badge" class:ok={latentWs.live} class:warn={!latentWs.live} title="/ws/latent?snapshot={snapshot} · {latentWs.frames} frames · {latentWs.lagged} superseded">ws {latentWs.status}</span>
	{#if error}<span class="badge danger">{error}</span>{/if}
	{#if snapshotsError}<span class="badge warn" title="GET /api/latent/snapshots">{snapshotsError}</span>{/if}
</div>

<div class="grid two">
	<section class="panel">
		<div class="panel-head">
			<h2>Region transition graph</h2>
			<div class="src">
				<select bind:value={graphSource} aria-label="graph source">
					<option value="live">live (latent.transition.{ts}, in-memory)</option>
					<option value="persisted">persisted (HelixDB /api/mcmc/{snapshot || '…'}/graph)</option>
				</select>
				<span class="dim">width = mean · hover for 90% interval · click to pin</span>
			</div>
		</div>
		{#if graphSource === 'persisted' && mcmcError}<p class="badge warn">{mcmcError}</p>{/if}
		{#if graphNodes.length}
			<RegionGraph nodes={graphNodes} edges={graphEdges} {selected} onselect={(id) => (selected = id)} />
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
		{:else if graphSource === 'persisted'}
			<Empty producer="HelixDB Transition edges for {snapshot} (model-runtime hub sink, D-048)" />
		{:else}
			<Empty producer="latent.region.{ts} / latent.transition.{ts} (model-runtime MCMC stage)" />
		{/if}
	</section>

	<Experts view={experts} error={expertsError} {now} />
	<Projection {snapshot} experts={expertIndices} {selected} {states} onselect={(id) => (selected = id)} />
	<GraphExplorer node={selected ?? snapshot} />

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
		flex-wrap: wrap;
	}
	.toolbar label {
		display: flex;
		gap: 6px;
		align-items: center;
		color: var(--text-dim);
	}
	.toolbar button {
		font-size: 11px;
		padding: 1px 8px;
	}
	.src {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
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
