<script lang="ts">
	// Growth (spec 51 §6, round 9): time series of the system's own growth over the guarded
	// VictoriaMetrics proxy (`GET /api/metrics/query_range`) — data in, data stored, bus usage
	// against caps, learned structure — next to the money that funds it: segment funds and
	// cumulative opex from `GET /api/accounting/balances` (D-104). One range picker drives every
	// chart; a chart whose query matches nothing names the series it would read.
	import { onMount } from 'svelte';
	import GrowthChart from '../../lib/GrowthChart.svelte';
	import Info from '../../lib/Info.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { api, bytes, eur, poll, q, type BusStream } from '../../lib/api';
	import { errText, getApi } from '../../lib/console';
	import {
		RANGES,
		fmtCount,
		generationsQuery,
		ingestEventsQuery,
		moneyView,
		rangeParams,
		regionsBornQuery,
		regionsLiveQuery,
		seriesFromMatrix,
		snapshotsQuery,
		strategiesQuery,
		streamBytesQuery,
		streamCaps,
		stripPrefix,
		tsdbWrittenQuery,
		vmBytesQuery,
		vmRowsQuery,
		vmSeriesQuery,
		type BalancesView,
		type RangeKey,
		type Series
	} from '../../lib/growth';
	import type { VmMatrix } from '../../lib/pointcloud';
	import { loadPref, savePref } from '../../lib/prefs';

	let range = $state<RangeKey>(loadPref<RangeKey>('growth.range', '24h'));
	$effect(() => savePref('growth.range', range));

	interface ChartSpec {
		id: string;
		section: 'data' | 'learning';
		label: string;
		query: (stepS: number) => string;
		format?: (v: number) => string;
		nameOf?: (metric: Record<string, string>) => string;
		withCaps?: boolean;
	}
	const SPECS: ChartSpec[] = [
		{ id: 'ingest', section: 'data', label: 'Ingest events / min by source', query: ingestEventsQuery, format: fmtCount, nameOf: (m) => stripPrefix(m.key ?? '', 'events.') },
		{ id: 'tsdb', section: 'data', label: 'Bars written to the TSDB / min', query: tsdbWrittenQuery, format: fmtCount },
		{ id: 'vmbytes', section: 'data', label: 'VictoriaMetrics data bytes', query: vmBytesQuery, format: bytes },
		{ id: 'vmrows', section: 'data', label: 'VictoriaMetrics rows and active series', query: vmRowsQuery, format: fmtCount },
		{ id: 'streams', section: 'data', label: 'NATS stream bytes vs caps', query: streamBytesQuery, format: bytes, nameOf: (m) => m.stream_name ?? '?', withCaps: true },
		{ id: 'regions', section: 'learning', label: 'Live latent regions by timescale', query: regionsLiveQuery, format: fmtCount, nameOf: (m) => stripPrefix(m.key ?? '', 'regions_live.') },
		{ id: 'born', section: 'learning', label: 'Regions born (cumulative)', query: regionsBornQuery, format: fmtCount },
		{ id: 'snapshots', section: 'learning', label: 'Encoder snapshots in lineage', query: snapshotsQuery, format: fmtCount },
		{ id: 'strategies', section: 'learning', label: 'Strategies loaded (total / neural)', query: strategiesQuery, format: fmtCount, nameOf: (m) => (m.key === 'strategies.neural' ? 'neural' : 'total') },
		{ id: 'generations', section: 'learning', label: 'Evolution generations / genomes / promotions', query: generationsQuery, format: fmtCount, nameOf: (m) => `${m.service ?? ''} ${m.key ?? ''}`.trim() }
	];

	let params = $state(rangeParams(loadPref<RangeKey>('growth.range', '24h')));
	let charts = $state<Record<string, { series: Series[]; query: string; error: string }>>({});
	// vm_rows gets the active-series line merged in so rows and series read together
	async function loadChart(spec: ChartSpec, p: { start: number; end: number; step: number }) {
		const query = spec.query(p.step);
		try {
			const r = await getApi<VmMatrix>(`/api/metrics/query_range${q({ query, start: p.start, end: p.end, step: p.step })}`);
			let series = seriesFromMatrix(r, spec.nameOf);
			if (spec.id === 'vmrows') {
				series = series.map((s) => ({ ...s, name: 'rows' }));
				const r2 = await getApi<VmMatrix>(`/api/metrics/query_range${q({ query: vmSeriesQuery(), start: p.start, end: p.end, step: p.step })}`);
				series = [...series, ...seriesFromMatrix(r2).map((s) => ({ ...s, name: 'active series (1h)' }))];
			}
			charts[spec.id] = { series, query, error: '' };
		} catch (e) {
			charts[spec.id] = { series: [], query, error: errText(e) };
		}
	}

	async function loadAll() {
		const p = rangeParams(range);
		params = p;
		await Promise.all(SPECS.map((s) => loadChart(s, p)));
	}
	$effect(() => {
		range;
		void loadAll();
	});

	// stream caps (reference lines) from the streams route the Streams panel already uses
	let streams = $state<BusStream[]>([]);
	async function loadStreams() {
		try {
			streams = (await api.streams()).streams;
		} catch {
			streams = [];
		}
	}
	const caps = $derived(streamCaps(streams));

	// money: segment funds and cumulative opex (D-104)
	let balances = $state<BalancesView | null>(null);
	let balancesError = $state('');
	async function loadBalances() {
		try {
			balances = await getApi<BalancesView>('/api/accounting/balances');
			balancesError = '';
		} catch (e) {
			balancesError = errText(e);
		}
	}
	const money = $derived(balances ? moneyView(balances) : null);

	onMount(() => {
		const stops = [poll(loadStreams, 60_000), poll(loadBalances, 60_000), poll(loadAll, 60_000)];
		return () => stops.forEach((s) => s());
	});

	const domain = $derived<[number, number]>([params.start * 1000, params.end * 1000]);
	const stepMs = $derived(params.step * 1000);
	const section = (name: 'data' | 'learning') => SPECS.filter((s) => s.section === name);
	const slot = (id: string) => charts[id] ?? { series: [], query: SPECS.find((s) => s.id === id)?.query(params.step) ?? '', error: '' };
</script>

<ViewHeader title="Growth" spec="51 §6 round 9" note="the system's own growth, and the money that funds it" />

<div class="controls">
	<span class="dim">range</span>
	{#each RANGES as r (r.key)}
		<button class:sel={range === r.key} onclick={() => (range = r.key)}>{r.key}</button>
	{/each}
	<span class="dim mono">step {params.step}s · via /api/metrics/query_range</span>
	<Info k="ops.growth.range" />
</div>

<div class="grid cols">
	<section class="panel">
		<div class="panel-head"><h2>Data <Info k="ops.growth.data" /></h2><span class="dim">ingest → bus → stores</span></div>
		<div class="charts">
			{#each section('data') as spec (spec.id)}
				<GrowthChart
					label={spec.label}
					series={slot(spec.id).series}
					{domain}
					{stepMs}
					rangeKey={range}
					format={spec.format}
					caps={spec.withCaps ? caps : []}
					query={slot(spec.id).query}
					error={slot(spec.id).error}
				/>
			{/each}
		</div>
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Learning <Info k="ops.growth.learning" /></h2><span class="dim">regions, snapshots, strategies, evolution</span></div>
		<div class="charts">
			{#each section('learning') as spec (spec.id)}
				<GrowthChart
					label={spec.label}
					series={slot(spec.id).series}
					{domain}
					{stepMs}
					rangeKey={range}
					format={spec.format}
					query={slot(spec.id).query}
					error={slot(spec.id).error}
				/>
			{/each}
		</div>
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Money <Info k="ops.growth.money" /></h2>
			<span class="dim">segment funds &amp; cumulative opex (D-104)</span>
		</div>
		{#if balancesError}<p class="badge danger">{balancesError}</p>{/if}
		{#if money}
			<div class="tiles">
				{#each money.segments as s (s.segment)}
					<div class="tile">
						<span class="dim">{s.segment}</span>
						<span class="mono" class:dim={s.funds === 0}>{eur(s.funds)}</span>
					</div>
				{/each}
			</div>
			{#if money.allRetained}
				<p class="dim">all retained — no split configured</p>
			{/if}
			<h3>Cumulative opex <Info k="ops.growth.opex" /></h3>
			{#if money.expenses.length}
				<table>
					<thead><tr><th>expense account</th><th>ref</th><th>EUR</th></tr></thead>
					<tbody>
						{#each money.expenses as e (e.account)}
							<tr>
								<td class="mono">{e.account}</td>
								<td class="mono dim">{e.external_ref ?? '—'}</td>
								<td class="mono">{eur(e.eur)}</td>
							</tr>
						{/each}
						<tr>
							<td class="dim">total</td>
							<td></td>
							<td class="mono">{eur(money.opexTotal)}</td>
						</tr>
					</tbody>
				</table>
			{:else}
				<p class="dim">no expense accounts yet — cumulative opex appears here once the accounting books an expense</p>
			{/if}
			<p class="dim small">
				trial balance <span class="badge" class:ok={Math.abs(money.trialBalance) < 0.005} class:danger={Math.abs(money.trialBalance) >= 0.005}>{eur(money.trialBalance)}</span>
				{#if balances?.entries !== undefined}· {balances.entries} journal entries{/if}
			</p>
		{:else if !balancesError}
			<p class="dim">loading balances…</p>
		{/if}
	</section>
</div>

<style>
	.controls {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}
	.controls button.sel {
		border-color: var(--accent);
		color: var(--accent);
	}
	.cols {
		grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
		align-items: start;
	}
	.charts {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
		gap: 8px;
		margin-bottom: 8px;
	}
	.tile {
		display: flex;
		flex-direction: column;
		gap: 2px;
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 8px 10px;
	}
	h3 {
		font-size: 13px;
		margin: 10px 0 6px;
	}
	.small {
		font-size: 11px;
	}
</style>
