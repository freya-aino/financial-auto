<script lang="ts">
	// PCA projection of one expert space (`GET /api/latent/{snapshot}/projection`, spec 51 §6.3):
	// 2-D or 3-D canvas point cloud with pan / zoom / orbit (up to 50 000 points), coloured by
	// region, instrument or time; live positions of selected instruments animated from the
	// `/ws/latent` states (placed at the centroid of their current region — the state frames carry
	// the region, not the latent vector); region occupancy over time from the fetched sample.
	import { onMount, untrack } from 'svelte';
	import Empty from './Empty.svelte';
	import PointCloud from './PointCloud.svelte';
	import Sparkline from './Sparkline.svelte';
	import { live, short, type LatentStateView, type ProjectionView } from './api';
	import { loadPref, savePref } from './prefs';
	import { assignRegionColors, explainedLabel } from './projection';
	import { getApi, errText } from './console';
	import { q } from './api';
	import { byTime, categorical, normaliser, normalise, occupancy, occupancyFromMatrix, regionCentroids, regionWeightQuery, tween, type Marker, type Occupancy, type VmMatrix } from './pointcloud';

	interface Props {
		snapshot: string;
		experts: number[];
		/** Region id pinned elsewhere (RegionGraph click) → highlighted here. */
		selected?: string | null;
		onselect?: (regionId: string | null) => void;
		/** Live expert states of the snapshot (instrument → region per expert). */
		states?: LatentStateView[];
	}
	let { snapshot, experts, selected = null, onselect, states: allStates = [] }: Props = $props();
	const states = $derived(allStates.filter((s) => s.snapshot_id === snapshot));

	/** spec 51 §7 `ui.projection_point_budget`; the API clamps to its own cap and says so. */
	const POINT_BUDGETS = [500, 2000, 5000, 20000, 50000];

	let expert = $state(loadPref('latent.proj.expert', 0));
	let range = $state(loadPref('latent.proj.range', 'all'));
	let n = $state(loadPref('latent.proj.n', 2000));
	let dims = $state<2 | 3>(loadPref<2 | 3>('latent.proj.dims', 2));
	let colorBy = $state<'region' | 'instrument' | 'time'>(loadPref('latent.proj.color', 'region'));
	let track = $state<string[]>(loadPref<string[]>('latent.proj.track', []));
	let view = $state<ProjectionView | null>(null);
	let error = $state('');
	let busy = $state(false);
	$effect(() => savePref('latent.proj.expert', expert));
	$effect(() => savePref('latent.proj.range', range));
	$effect(() => savePref('latent.proj.n', n));
	$effect(() => savePref('latent.proj.dims', dims));
	$effect(() => savePref('latent.proj.color', colorBy));
	$effect(() => savePref('latent.proj.track', track));

	// sequential ramp for time colouring, read from the theme tokens once mounted
	let seqStops = $state(['#2b4b73', '#34639b', '#4a80c0', '#6ea0dc', '#a9cdf5']);
	onMount(() => {
		const cs = getComputedStyle(document.documentElement);
		const read = [0, 1, 2, 3, 4].map((i) => cs.getPropertyValue(`--seq-${i}`).trim());
		if (read.every((c) => /^#[0-9a-f]{6}$/i.test(c))) seqStops = read;
	});

	const pts = $derived(view?.points ?? []);
	const coords = $derived(pts.map((p) => p.coords));
	const norm = $derived(normaliser(coords, dims));
	const cloud = $derived(normalise(coords, dims));
	const regionColors = $derived(assignRegionColors(pts.map((p) => p.region_id)));
	const instrumentCounts = $derived.by(() => {
		const m = new Map<string, number>();
		for (const p of pts) m.set(p.instrument_id, (m.get(p.instrument_id) ?? 0) + 1);
		return [...m].sort((a, b) => b[1] - a[1]);
	});
	const instrumentColors = $derived(assignRegionColors([...pts.map((p) => p.instrument_id), ...states.map((s) => s.instrument_id)]));
	const timeColoring = $derived(byTime(pts.map((p) => Date.parse(p.ts)), seqStops));
	const coloring = $derived(
		colorBy === 'time'
			? timeColoring
			: colorBy === 'instrument'
				? categorical(pts.map((p) => p.instrument_id), instrumentColors)
				: categorical(pts.map((p) => p.region_id), regionColors)
	);
	const emphasis = $derived(selected === null || !pts.some((p) => p.region_id === selected) ? null : Uint8Array.from(pts, (p) => (p.region_id === selected ? 1 : 0)));
	const regionCounts = $derived.by(() => {
		const m = new Map<string, number>();
		for (const p of pts) m.set(p.region_id, (m.get(p.region_id) ?? 0) + 1);
		return [...m].sort((a, b) => b[1] - a[1]);
	});
	// occupancy: from the fetched sample, or the `fa_region_weight` series in VictoriaMetrics
	// through control-api's metrics passthrough (spec 50 §5; region_weight written per closed bar)
	let occSource = $state<'sample' | 'vm'>(loadPref('latent.proj.occ', 'sample'));
	$effect(() => savePref('latent.proj.occ', occSource));
	let vmOcc = $state<Occupancy | null>(null);
	let vmError = $state('');
	const VM_RANGE: Record<string, string> = { '15m': '-15m', '1h': '-1h', '6h': '-6h', '1d': '-1d', '7d': '-7d', all: '-31d' };
	async function loadVm() {
		if (!view) return;
		try {
			const r = await getApi<VmMatrix>(`/api/metrics/query_range${q({ query: regionWeightQuery(view.expert_space_id), start: VM_RANGE[range] ?? '-1d' })}`);
			vmOcc = occupancyFromMatrix(r);
			vmError = vmOcc.times.length ? '' : `no fa_region_weight samples for ${view.expert_space_id} in the range`;
		} catch (e) {
			vmOcc = null;
			vmError = `VictoriaMetrics occupancy unavailable: ${errText(e)}`;
		}
	}
	$effect(() => {
		if (occSource === 'vm' && view) void loadVm();
	});
	const occ = $derived(occSource === 'vm' && vmOcc?.times.length ? vmOcc : occupancy(pts, 30));
	const occRegions = $derived(
		selected && occ.share.has(selected)
			? [selected]
			: [...occ.share.keys()].sort((a, b) => (occ.share.get(b) ?? []).reduce((x, y) => x + y, 0) - (occ.share.get(a) ?? []).reduce((x, y) => x + y, 0)).slice(0, 6)
	);
	const axes = $derived((view?.pca.explained_variance_ratio ?? []).map((r, i) => `PC${i + 1} ${(100 * r).toFixed(0)}%`));
	const capped = $derived(view !== null && view.requested < n);

	function describe(i: number): string {
		const p = pts[i];
		if (!p) return '';
		return `${p.region_id} · ${short(p.instrument_id, 8)} · ${p.ts.slice(0, 16)} · ${p.kind} · σ ${p.uncertainty.toFixed(2)}`;
	}

	// ------------------------------------------------------------ live positions
	const centroids = $derived(regionCentroids(pts));
	const liveInstruments = $derived(states.filter((s) => s.experts.some((e) => e.expert_index === expert)).map((s) => s.instrument_id));
	/** Region each tracked instrument is in now (for this expert). */
	const targets = $derived.by(() => {
		const out = new Map<string, { region: string; pos: [number, number, number] | null }>();
		for (const id of track) {
			const s = states.find((x) => x.instrument_id === id);
			const region = s?.experts.find((e) => e.expert_index === expert)?.region_id;
			if (!region) continue;
			const c = centroids.get(region);
			out.set(id, { region, pos: c ? norm(c) : null });
		}
		return out;
	});

	interface Anim {
		from: [number, number, number];
		to: [number, number, number];
		start: number;
		region: string;
		trail: [number, number, number][];
	}
	const anims = new Map<string, Anim>();
	let markers = $state<Marker[]>([]);
	let raf = 0;
	const DURATION = 900;

	function step() {
		raf = 0;
		const now = performance.now();
		let running = false;
		markers = [...anims].map(([id, a]) => {
			const t = (now - a.start) / DURATION;
			if (t < 1) running = true;
			return { pos: tween(a.from, a.to, t), label: `${short(id, 8)} → ${a.region}`, color: instrumentColors.get(id) ?? '#888888', trail: a.trail };
		});
		if (running) raf = requestAnimationFrame(step);
	}

	$effect(() => {
		const tg = targets;
		untrack(() => {
			const now = performance.now();
			for (const id of [...anims.keys()]) if (!tg.has(id) || !tg.get(id)!.pos) anims.delete(id);
			for (const [id, { region, pos }] of tg) {
				if (!pos) continue;
				const a = anims.get(id);
				if (!a) {
					anims.set(id, { from: pos, to: pos, start: now - DURATION, region, trail: [] });
				} else if (a.region !== region || a.to.some((v, k) => Math.abs(v - pos[k]) > 1e-9)) {
					const t = (now - a.start) / DURATION;
					const cur = tween(a.from, a.to, t);
					const trail = a.region !== region ? [...a.trail, a.to].slice(-6) : a.trail;
					anims.set(id, { from: cur, to: pos, start: now, region, trail });
				}
			}
			if (!raf && typeof requestAnimationFrame !== 'undefined') raf = requestAnimationFrame(step);
		});
	});
	onMount(() => () => raf && cancelAnimationFrame(raf));

	function toggleTrack(id: string) {
		track = track.includes(id) ? track.filter((x) => x !== id) : [...track, id].slice(-8);
	}
	// default: follow the first three live instruments once states arrive
	$effect(() => {
		if (!untrack(() => track.length) && liveInstruments.length) track = liveInstruments.slice(0, 3);
	});

	export async function refresh() {
		if (!snapshot) return;
		busy = true;
		try {
			view = await live.projection(snapshot, { expert, range, n, dims });
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
		// refetch whenever snapshot/expert/range/n/dims change
		void snapshot;
		void expert;
		void range;
		void n;
		void dims;
		void refresh();
	});

	const fmtPct = (v: number) => `${(100 * v).toFixed(0)}%`;
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
				<select bind:value={n}>{#each POINT_BUDGETS as k}<option value={k}>{k.toLocaleString()}</option>{/each}</select>
			</label>
			<label>view
				<select bind:value={dims}><option value={2}>2-D</option><option value={3}>3-D</option></select>
			</label>
			<label>colour
				<select bind:value={colorBy}><option value="region">region</option><option value="instrument">instrument</option><option value="time">time</option></select>
			</label>
			<button onclick={refresh} disabled={busy}>{busy ? '…' : 'refresh'}</button>
		</div>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if view && pts.length}
		<p class="dim small">
			{view.expert_space_id} · {view.fetched.toLocaleString()}/{view.total_in_space.toLocaleString()} points{view.cached ? ' (cached)' : ''} · PCA {view.dim}→{view.dims}: <span class="mono">{explainedLabel(view.pca.explained_variance_ratio)}</span>
			{#if capped}<span class="badge warn" title="the API clamps n (spec 50 projection cap)">server cap {view.requested.toLocaleString()} &lt; {n.toLocaleString()} requested</span>{/if}
		</p>
		<PointCloud {cloud} {coloring} {emphasis} {markers} {describe} {axes} onpick={(i) => onselect?.(pts[i]?.region_id === selected ? null : (pts[i]?.region_id ?? null))} />
		<p class="dim small">{dims === 3 ? 'drag to orbit · shift-drag to pan' : 'drag to pan'} · wheel or +/− to zoom · double-click or 0 resets · click a point to pin its region</p>
		{#if colorBy === 'time'}
			<div class="ramp">
				<span class="mono dim">{new Date(timeColoring.lo).toISOString().slice(0, 16)}</span>
				<i style="background: linear-gradient(to right, {seqStops.join(', ')})"></i>
				<span class="mono dim">{new Date(timeColoring.hi).toISOString().slice(0, 16)}</span>
			</div>
		{:else if colorBy === 'instrument'}
			<div class="legend">
				{#each instrumentCounts.slice(0, 24) as [id, count] (id)}
					<span class="swatch" style="--c: {instrumentColors.get(id)}" title="{id}: {count} points"><i></i><span class="mono">{short(id, 8)}</span><span class="dim">{count}</span></span>
				{/each}
				{#if instrumentCounts.length > 24}<span class="dim">+{instrumentCounts.length - 24} more (12 hues repeat)</span>{/if}
			</div>
		{:else}
			<div class="legend">
				{#each regionCounts as [id, count] (id)}
					<button class="swatch" class:sel={selected === id} style="--c: {regionColors.get(id)}" onclick={() => onselect?.(selected === id ? null : id)} title="{count} points">
						<i></i><span class="mono">{id}</span><span class="dim">{count}</span>
					</button>
				{/each}
			</div>
		{/if}

		<h3 class="sub">Live positions <span class="dim">(state.{snapshot}.&lt;instrument&gt; on /ws/latent; marker sits at the centroid of the instrument's current E{expert} region, trail = previous regions)</span></h3>
		{#if liveInstruments.length}
			<div class="legend">
				{#each liveInstruments as id (id)}
					<button class="swatch" class:sel={track.includes(id)} style="--c: {instrumentColors.get(id)}" onclick={() => toggleTrack(id)} title={id}>
						<i></i><span class="mono">{short(id, 8)}</span>{#if targets.get(id)}<span class="dim">{targets.get(id)?.region}{targets.get(id)?.pos ? '' : ' (not in sample)'}</span>{/if}
					</button>
				{/each}
			</div>
		{:else}
			<p class="dim small">no live states for expert E{expert} of {snapshot} yet</p>
		{/if}

		<h3 class="sub">
			Occupancy over time
			<span class="dim">({occSource === 'vm' && vmOcc?.times.length ? 'share of fa_region_weight per sample' : 'share of the fetched sample per time bucket'}{selected && occ.share.has(selected) ? `, region ${selected}` : ', six largest regions'})</span>
			<select bind:value={occSource} aria-label="occupancy source"><option value="sample">from the sample</option><option value="vm">from VictoriaMetrics</option></select>
		</h3>
		{#if occSource === 'vm' && vmError}<p class="badge warn">{vmError} — showing the sample</p>{/if}
		<div class="occ">
			{#each occRegions as r (r)}
				<Sparkline label="{r} · {occ.count.get(r) ?? 0} {occ === vmOcc ? 'samples' : 'pts'}" times={occ.times} values={occ.share.get(r) ?? []} format={fmtPct} height={44} fill />
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
	.sub {
		margin: 12px 0 4px;
		font-size: 12px;
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
	.ramp {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 11px;
		margin-top: 6px;
	}
	.ramp i {
		flex: 1;
		height: 8px;
		border-radius: 4px;
	}
	.occ {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
		gap: 8px 16px;
	}
</style>
