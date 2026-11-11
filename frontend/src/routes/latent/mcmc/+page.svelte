<script lang="ts">
	// MCMC & region explorer (D-069, spec 30, spec 33 §4.2): every query answers a posterior
	// distribution from the persisted Dirichlet kernels; samplers are scored prequentially and the
	// serving one is picked from the Pareto front. Region lineage (DescendsFrom), hierarchy
	// (REFINES), cross-asset priors, and the live soft memberships / lifecycle events on the bus.
	import { onMount, untrack } from 'svelte';
	import Empty from '../../../lib/Empty.svelte';
	import ViewHeader from '../../../lib/ViewHeader.svelte';
	import { live, num, pct, short, type Snapshot } from '../../../lib/api';
	import { since } from '../../../lib/alerts';
	import { consoleApi, errText, type CrossPrior, type Dist, type HierEdgeRow, type KernelRow, type LineageView, type McmcQuery, type RegionProb } from '../../../lib/console';
	import { LIFECYCLE_CLASS, cumulative, fmtDist, parseList, topMemberships } from '../../../lib/models';
	import { loadPref, savePref } from '../../../lib/prefs';
	import { bus } from '../../../lib/ws.svelte';

	type Membership = { snapshot_id: string; timescale: string; instrument_id: string; as_of: string | null; experts: { expert_index: number; novelty: number; out_of_region: boolean; weights: { region_id: string; weight: number }[] }[] };
	type Lifecycle = { kind: string; space: { snapshot_id: string; expert_index: number } | null; region_id: string; lineage_id: string; as_of: string | null; ancestors: { region_id: string; overlap: number; kind: string }[]; reason: string; mass: number; occupancy: number; related_region_id: string; regime_break: boolean };

	let snapshots = $state<Snapshot[]>([]);
	let snapshot = $state(loadPref('mcmc.snapshot', ''));
	let kernels = $state<KernelRow[]>([]);
	let kernelsError = $state('');
	let expert = $state(loadPref('mcmc.expert', 0));
	let query = $state<McmcQuery>(loadPref('mcmc.query', 'neighbors'));
	let from = $state('');
	let to = $state('');
	let k = $state(3);
	let n = $state(10);
	let targets = $state('');
	let horizon = $state(20);
	let pathText = $state('');
	let draws = $state(128);
	let level = $state(0.9);
	let seed = $state(1);
	let result = $state<Record<string, unknown> | null>(null);
	let resultError = $state('');
	let busy = $state(false);
	let hierarchy = $state<HierEdgeRow[]>([]);
	let hierError = $state('');
	let priors = $state<CrossPrior[]>([]);
	let priorsError = $state('');
	let lineageRegion = $state('');
	let lineage = $state<LineageView | null>(null);
	let lineageError = $state('');
	let memberships = $state<Record<string, Membership>>({});
	let lifecycle = $state<(Lifecycle & { seq: number })[]>([]);
	let now = $state(Date.now());
	$effect(() => savePref('mcmc.snapshot', snapshot));
	$effect(() => savePref('mcmc.expert', expert));
	$effect(() => savePref('mcmc.query', query));

	const snap = $derived(snapshots.find((s) => s.snapshot_id === snapshot) ?? null);
	const ts = $derived(snap?.timescale ?? '1m');
	const kernel = $derived(kernels.find((x) => x.expert_index === expert) ?? null);
	const regions = $derived(kernel?.stationary.map((r) => r.region_id) ?? []);
	const memberRows = $derived(Object.values(memberships).filter((m) => m.timescale === ts).sort((a, b) => a.instrument_id.localeCompare(b.instrument_id)));
	const QUERIES: { id: McmcQuery; label: string; needs: string[] }[] = [
		{ id: 'neighbors', label: 'neighbors (one step)', needs: ['region', 'n'] },
		{ id: 'transition', label: 'transition P(a → b in k)', needs: ['from', 'to', 'k'] },
		{ id: 'k-step', label: 'k-step distribution', needs: ['from', 'k', 'n'] },
		{ id: 'paths', label: 'sample paths', needs: ['from', 'k', 'n', 'seed'] },
		{ id: 'path-prob', label: 'path probability', needs: ['path'] },
		{ id: 'hitting', label: 'hitting time', needs: ['from', 'targets', 'horizon'] },
		{ id: 'stationary', label: 'stationary + relaxation', needs: [] }
	];
	const needs = $derived(QUERIES.find((x) => x.id === query)?.needs ?? []);

	async function loadSnapshots() {
		try {
			snapshots = (await live.snapshots()).snapshots.sort((a, b) => a.timescale.localeCompare(b.timescale) || b.created_at.localeCompare(a.created_at));
			if (!snapshots.some((s) => s.snapshot_id === snapshot)) snapshot = (snapshots.find((s) => s.status === 'active' && s.timescale === '1m') ?? snapshots.find((s) => s.status === 'active') ?? snapshots[0])?.snapshot_id ?? '';
		} catch (e) {
			kernelsError = errText(e);
		}
	}

	async function loadSnapshotData() {
		if (!snapshot) return;
		const [kk, h, p] = await Promise.allSettled([consoleApi.kernels(snapshot), consoleApi.hierarchy(snapshot), consoleApi.crossPriors(ts)]);
		if (kk.status === 'fulfilled') {
			kernels = kk.value.kernels.sort((a, b) => a.expert_index - b.expert_index);
			kernelsError = '';
			if (!kernels.some((x) => x.expert_index === expert) && kernels.length) expert = kernels[0].expert_index;
		} else {
			kernels = [];
			kernelsError = errText(kk.reason);
		}
		if (h.status === 'fulfilled') ((hierarchy = h.value.edges), (hierError = ''));
		else hierError = errText(h.reason);
		if (p.status === 'fulfilled') ((priors = p.value.proposals), (priorsError = ''));
		else priorsError = errText(p.reason);
	}

	$effect(() => {
		void snapshot;
		result = null;
		from = '';
		void loadSnapshotData();
	});

	$effect(() => {
		if (regions.length && !from) {
			from = regions[0];
			to = regions[1] ?? regions[0];
			targets = regions[1] ?? '';
			pathText = regions.slice(0, 3).join(', ');
			// first answer without a click: the default query on the most visited region, and its lineage
			const first = regions[0];
			untrack(() => {
				void runQuery();
				void loadLineage(first);
			});
		}
	});

	async function runQuery() {
		busy = true;
		const base = { expert, draws, level };
		const p: Record<string, string | number | undefined> = { ...base };
		if (needs.includes('from')) p.from = from;
		if (needs.includes('region')) p.region = from;
		if (needs.includes('to')) p.to = to;
		if (needs.includes('k')) p.k = k;
		if (needs.includes('n')) p.n = n;
		if (needs.includes('seed')) p.seed = seed;
		if (needs.includes('targets')) p.targets = parseList(targets).join(',');
		if (needs.includes('horizon')) p.horizon = horizon;
		if (needs.includes('path')) p.path = parseList(pathText).join(',');
		try {
			result = await consoleApi.mcmc(snapshot, query, p);
			resultError = '';
		} catch (e) {
			resultError = errText(e);
			result = null;
		} finally {
			busy = false;
		}
	}

	async function loadLineage(region = lineageRegion) {
		if (!region) return;
		lineageRegion = region;
		try {
			lineage = await consoleApi.lineage(region, 6);
			lineageError = '';
		} catch (e) {
			lineageError = errText(e);
			lineage = null;
		}
	}

	function pick(region: string) {
		from = region;
		void loadLineage(region);
	}

	onMount(() => {
		void loadSnapshots();
		const tick = setInterval(() => (now = Date.now()), 1000);
		let seq = 0;
		const offM = bus.subscribe('latent.membership.>', (f) => {
			const m = f.data.decoded as Membership | null;
			if (m && m.instrument_id) memberships = { ...memberships, [`${m.timescale}.${m.instrument_id}`]: m };
		});
		const offL = bus.subscribe('latent.lifecycle.>', (f) => {
			const e = f.data.decoded as Lifecycle | null;
			if (e && e.region_id) lifecycle = [{ ...e, seq: ++seq }, ...lifecycle].slice(0, 300);
		});
		return () => {
			clearInterval(tick);
			offM();
			offL();
		};
	});

	const d = (v: unknown) => v as Dist;
	const rp = (v: unknown) => (v ?? []) as RegionProb[];
</script>

<ViewHeader title="MCMC & regions" spec="33 §4.2 · 30 §4–5" note="posterior answers from persisted kernels (/api/mcmc/{snapshot}/…), lineage, hierarchy, cross priors, live memberships" />

<div class="toolbar panel">
	<label>
		snapshot
		<select bind:value={snapshot}>
			{#each snapshots as s (s.snapshot_id)}<option value={s.snapshot_id}>{s.timescale} · {s.snapshot_id} · {s.status}</option>{/each}
		</select>
	</label>
	<label>
		expert
		<select bind:value={expert}>
			{#each kernels as kr (kr.expert_index)}<option value={kr.expert_index}>E{kr.expert_index} · {kr.live_regions} regions</option>{/each}
			{#if !kernels.length}<option value={expert}>E{expert}</option>{/if}
		</select>
	</label>
	<label>draws <input type="number" bind:value={draws} min="16" max="1000" step="16" /></label>
	<label>level <input type="number" bind:value={level} min="0.5" max="0.99" step="0.05" /></label>
	{#if kernelsError}<span class="badge warn">{kernelsError}</span>{/if}
</div>

<div class="grid cols">
	<section class="panel wide">
		<div class="panel-head">
			<h2>Kernels <span class="dim">({kernels.length})</span></h2>
			<span class="dim">samplers scored prequentially (log score ↑, Brier ↓, ECE ↓, PIT KS ↓, cost ↓); serving from the Pareto front with hysteresis</span>
		</div>
		{#if kernels.length}
			<table>
				<thead><tr><th>expert</th><th>type</th><th>regions</th><th>obs. mass</th><th>relaxation</th><th>drift KL</th><th>sampler</th><th>log score</th><th>Brier</th><th>ECE</th><th>PIT KS</th><th>cost µs</th><th>evals</th></tr></thead>
				<tbody>
					{#each kernels as kr (kr.kernel_id)}
						{#each kr.scores as sc, i (sc.sampler_id)}
							<tr class:sel={kr.expert_index === expert}>
								{#if i === 0}
									<td rowspan={kr.scores.length} class="mono">E{kr.expert_index}</td>
									<td rowspan={kr.scores.length} class="mono dim">{kr.kernel_type}</td>
									<td rowspan={kr.scores.length} class="mono">{kr.live_regions}/{kr.n_rows}</td>
									<td rowspan={kr.scores.length} class="mono">{num(kr.observation_mass, 1)}</td>
									<td rowspan={kr.scores.length} class="mono">{num(kr.relaxation_steps, 2)} steps</td>
									<td rowspan={kr.scores.length} class="mono" class:warnc={kr.drift_kl > 0.1}>{num(kr.drift_kl, 3)}</td>
								{/if}
								<td class="mono">{sc.sampler_id}{#if sc.serving} <span class="badge ok">serving</span>{/if}{#if sc.pareto && !sc.serving} <span class="badge">pareto</span>{/if}</td>
								<td class="mono">{num(sc.log_score, 3)}</td>
								<td class="mono">{num(sc.brier, 3)}</td>
								<td class="mono">{num(sc.ece, 4)}</td>
								<td class="mono">{num(sc.pit_ks, 3)}</td>
								<td class="mono dim">{num(sc.cost_us, 1)}</td>
								<td class="mono dim">{sc.evals}</td>
							</tr>
						{:else}
							<tr><td class="mono">E{kr.expert_index}</td><td colspan="12" class="dim">no sampler scores yet</td></tr>
						{/each}
					{/each}
				</tbody>
			</table>
		{:else if !kernelsError}
			<Empty producer="RegionSet.kernel_state in HelixDB (model-runtime persists every 120 s)" hint="" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Query</h2><span class="dim">every answer is mean [lower, upper] at level {level}</span></div>
		<form class="qform" onsubmit={(e) => (e.preventDefault(), runQuery())}>
			<select bind:value={query} aria-label="query">{#each QUERIES as qd}<option value={qd.id}>{qd.label}</option>{/each}</select>
			{#if needs.includes('from') || needs.includes('region')}
				<label>{needs.includes('region') ? 'region' : 'from'} <input list="regions" bind:value={from} size="14" /></label>
			{/if}
			{#if needs.includes('to')}<label>to <input list="regions" bind:value={to} size="14" /></label>{/if}
			{#if needs.includes('k')}<label>k <input type="number" bind:value={k} min="1" max="512" /></label>{/if}
			{#if needs.includes('n')}<label>n <input type="number" bind:value={n} min="1" max="200" /></label>{/if}
			{#if needs.includes('seed')}<label>seed <input type="number" bind:value={seed} /></label>{/if}
			{#if needs.includes('targets')}<label>targets <input bind:value={targets} size="20" placeholder="r1, r2" /></label>{/if}
			{#if needs.includes('horizon')}<label>horizon <input type="number" bind:value={horizon} min="1" max="512" /></label>{/if}
			{#if needs.includes('path')}<label>path <input bind:value={pathText} size="30" placeholder="r1, r2, r3" /></label>{/if}
			<button type="submit" disabled={busy || !snapshot}>{busy ? '…' : 'ask'}</button>
			<datalist id="regions">{#each regions as r}<option value={r}></option>{/each}</datalist>
		</form>
		{#if resultError}<p class="badge danger">{resultError}</p>{/if}
		{#if result}
			{#if query === 'transition' || query === 'path-prob'}
				<p class="big mono">{fmtDist(d(result.prob), 4)}</p>
				<p class="dim small">sd {num(d(result.prob).sd, 4)} · median {num(d(result.prob).median, 4)} · {d(result.prob).n} draws</p>
			{:else if query === 'neighbors' || query === 'k-step' || query === 'stationary'}
				{@const rows = query === 'neighbors' ? rp(result.neighbors) : query === 'k-step' ? rp(result.distribution) : rp(result.stationary)}
				{@const scale = Math.max(...rows.map((r) => r.prob.upper), 1e-9)}
				{#if query === 'stationary'}<p class="mono">relaxation {fmtDist(d(result.relaxation_steps), 2)} steps</p>{/if}
				<div class="scroll mid">
					<table>
						<thead><tr><th>region</th><th>mean</th><th>interval</th></tr></thead>
						<tbody>
							{#each rows.slice(0, 60) as r (r.region_id)}
								<tr>
									<td class="mono"><button class="link" onclick={() => pick(r.region_id)}>{r.region_id}</button></td>
									<td><div class="ci" title="bar scaled to the largest upper bound ({num(scale, 4)})"><div class="span" style="left: {(100 * r.prob.lower) / scale}%; width: {Math.max(0.5, (100 * (r.prob.upper - r.prob.lower)) / scale)}%"></div><div class="pt" style="left: {(100 * r.prob.mean) / scale}%"></div></div> <span class="mono">{num(r.prob.mean, 4)}</span></td>
									<td class="mono dim">[{num(r.prob.lower, 4)}, {num(r.prob.upper, 4)}]</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else if query === 'paths'}
				{@const paths = (result.paths ?? []) as string[][]}
				{@const ends = (result.end_distribution ?? []) as { region_id: string; share: number }[]}
				<div class="scroll mid mono small">{#each paths as p, i (i)}<div>{p.join(' → ')}</div>{/each}</div>
				<h3 class="sub">End distribution</h3>
				<div class="ends">{#each ends as e (e.region_id)}<span class="badge mono">{e.region_id} {pct(e.share, 0)}</span>{/each}</div>
			{:else if query === 'hitting'}
				{@const h = result.hitting as Record<string, unknown>}
				{@const pmf = (h.pmf ?? []) as number[]}
				{@const cdf = cumulative(pmf)}
				<table>
					<tbody>
						<tr><th>P(hit within {h.horizon})</th><td class="mono">{fmtDist(d(h.p_hit))}</td></tr>
						<tr><th>E[steps | hit]</th><td class="mono">{fmtDist(d(h.conditional_steps), 2)}</td></tr>
						<tr><th>E[steps] (censored)</th><td class="mono">{fmtDist(d(h.expected_steps), 2)}</td></tr>
					</tbody>
				</table>
				<div class="pmf" title="P(first hit at step t); line = cumulative">
					{#each pmf as p, i}<div class="pbar" style="height: {100 * p / Math.max(...pmf, 1e-9)}%" title="t={i + 1}: {p.toFixed(4)} (cum {cdf[i].toFixed(3)})"></div>{/each}
				</div>
			{/if}
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Region lineage</h2><span class="dim">DescendsFrom carry · split · merge · partial</span></div>
		<form class="qform" onsubmit={(e) => (e.preventDefault(), loadLineage())}>
			<input list="regions" bind:value={lineageRegion} placeholder="region id" size="18" />
			<button type="submit">trace</button>
		</form>
		{#if lineageError}<p class="badge warn">{lineageError}</p>{/if}
		{#if lineage}
			<p>
				<span class="mono">{lineage.region.region_id}</span>
				<span class="badge" class:ok={lineage.region.status === 'live'}>{lineage.region.status}</span>
				<span class="dim">lineage {lineage.region.lineage_id} · {lineage.region.expert_space_id} · mass {num(lineage.region.mass ?? null, 1)} · occupancy {pct(lineage.region.occupancy ?? null, 1)} · {lineage.region.method ?? ''}</span>
			</p>
			<h3 class="sub">Ancestors ({lineage.ancestors.length})</h3>
			{#each lineage.ancestors as a, i (a.region_id + i)}
				<div class="mono small"><button class="link" onclick={() => loadLineage(a.region_id)}>{a.region_id}</button> <span class="badge">{a.kind ?? ''}</span> {a.overlap !== undefined ? `overlap ${num(a.overlap, 2)}` : ''} <span class="dim">{a.status}</span></div>
			{:else}<p class="dim small">none (born here)</p>{/each}
			<h3 class="sub">Descendants ({lineage.descendants.length})</h3>
			{#each lineage.descendants as a, i (a.region_id + i)}
				<div class="mono small"><button class="link" onclick={() => loadLineage(a.region_id)}>{a.region_id}</button> <span class="badge">{a.kind ?? ''}</span> <span class="dim">{a.status}</span></div>
			{:else}<p class="dim small">none</p>{/each}
			<h3 class="sub">Lineage members ({lineage.lineage.length})</h3>
			<div class="mono small dim">{lineage.lineage.map((r) => `${r.region_id} (${r.status})`).join(' · ')}</div>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Hierarchy <span class="dim">({hierarchy.length} REFINES edges)</span></h2></div>
		{#if hierError}<p class="badge warn">{hierError}</p>{/if}
		{#if hierarchy.length}
			<div class="scroll mid">
				<table>
					<thead><tr><th>fine</th><th>coarse</th><th>P(c|f)</th><th>P(f|c)</th><th>PMI bits</th><th>support</th></tr></thead>
					<tbody>
						{#each hierarchy as h, i (i)}
							<tr><td class="mono">{h.fine_region_id}</td><td class="mono">{h.coarse_region_id}</td><td class="mono">{num(h.p_coarse_given_fine ?? null, 3)}</td><td class="mono">{num(h.p_fine_given_coarse ?? null, 3)}</td><td class="mono">{num(h.pmi_bits ?? null, 2)}</td><td class="mono dim">{num(h.support ?? null, 0)}</td></tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !hierError}
			<p class="dim">No cross-timescale REFINES edges for this snapshot yet (bias-corrected MI needs co-occurring fine and coarse regions).</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Cross-asset priors <span class="dim">({priors.length}, {ts})</span></h2></div>
		{#if priorsError}<p class="badge warn">{priorsError}</p>{/if}
		{#if priors.length}
			<div class="scroll mid">
				<table>
					<thead><tr><th>source → target</th><th>corr (LW)</th><th>lead corr</th><th>ΔCRPS</th><th>score</th></tr></thead>
					<tbody>
						{#each priors as p, i (i)}
							<tr><td class="mono" title="{p.source_instrument_id} → {p.target_instrument_id}">{short(p.source_instrument_id)} → {short(p.target_instrument_id)}</td><td class="mono">{num(p.correlation ?? null, 3)}</td><td class="mono">{num(p.lead_corr ?? null, 3)}</td><td class="mono">{num(p.predictive_gain ?? null, 5)}</td><td class="mono">{num(p.score ?? null, 2)}</td></tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !priorsError}
			<p class="dim">No cross-asset prior proposals for {ts} yet (latent.crossprior.{ts}).</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Soft memberships <span class="dim">({memberRows.length} instruments, {ts})</span></h2>
			<span class="badge" class:ok={bus.status === 'open'} class:warn={bus.status !== 'open'}>latent.membership.{ts}.* {bus.status}</span>
		</div>
		{#if memberRows.length}
			<div class="scroll mid">
				<table>
					<thead><tr><th>instrument</th><th>top region per expert (weight · entropy · novelty)</th><th>as of</th></tr></thead>
					<tbody>
						{#each memberRows as m (m.instrument_id)}
							<tr>
								<td class="mono" title={m.instrument_id}>{short(m.instrument_id, 10)}</td>
								<td>
									{#each topMemberships(m) as t (t.expert_index)}
										<span class="memb mono" class:out={t.out} title="k={t.k}; novelty {t.novelty.toFixed(2)}{t.out ? ' (out of region)' : ''}">
											E{t.expert_index}:<button class="link" onclick={() => pick(t.region_id)}>{t.region_id}</button> {t.weight.toFixed(2)} <span class="dim">H{t.entropy.toFixed(2)} ν{t.novelty.toFixed(1)}</span>
										</span>
									{/each}
								</td>
								<td class="mono dim">{since(m.as_of, now)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="dim">Waiting for <code>latent.membership.{ts}.*</code> on the bus tap (published per closed bar).</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Lifecycle events <span class="dim">({lifecycle.length})</span></h2><span class="dim">latent.lifecycle.* (event-timed)</span></div>
		{#if lifecycle.length}
			<div class="scroll mid">
				<table>
					<thead><tr><th>as of</th><th>kind</th><th>region</th><th>ancestors</th><th>reason</th></tr></thead>
					<tbody>
						{#each lifecycle as e (e.seq)}
							<tr>
								<td class="mono dim">{since(e.as_of, now)}</td>
								<td><span class="badge {LIFECYCLE_CLASS[e.kind] ?? ''}">{e.kind}</span>{#if e.regime_break} <span class="badge danger">regime break</span>{/if}</td>
								<td class="mono"><button class="link" onclick={() => pick(e.region_id)}>{e.region_id}</button> <span class="dim">{e.space ? `E${e.space.expert_index}` : ''}</span></td>
								<td class="mono small">{e.ancestors.map((a) => `${a.region_id} (${a.kind} ${a.overlap.toFixed(2)})`).join(', ') || '—'}</td>
								<td class="small dim">{e.reason}{e.related_region_id ? ` → ${e.related_region_id}` : ''}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="dim">No lifecycle events since the page opened (births, splits, merges and deaths are rare between snapshots).</p>
		{/if}
	</section>
</div>

<style>
	.toolbar {
		display: flex;
		gap: 12px;
		align-items: center;
		flex-wrap: wrap;
		margin-bottom: 12px;
	}
	label {
		display: inline-flex;
		gap: 4px;
		align-items: center;
		color: var(--text-dim);
	}
	input[type='number'] {
		width: 72px;
	}
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
	}
	.wide {
		grid-column: 1 / -1;
	}
	.qform {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
		margin-bottom: 10px;
	}
	.mid {
		max-height: 380px;
	}
	.big {
		font-size: 18px;
		margin: 6px 0;
	}
	.small {
		font-size: 11px;
	}
	.sub {
		font-size: 12px;
		margin: 10px 0 4px;
	}
	.ci {
		position: relative;
		display: inline-block;
		width: 120px;
		height: 8px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
	}
	.ci .span {
		position: absolute;
		top: 1px;
		bottom: 1px;
		background: color-mix(in srgb, var(--accent) 45%, transparent);
	}
	.ci .pt {
		position: absolute;
		top: -1px;
		bottom: -1px;
		width: 2px;
		background: var(--accent);
	}
	.pmf {
		display: flex;
		align-items: flex-end;
		gap: 1px;
		height: 70px;
		margin-top: 8px;
		border-bottom: 1px solid var(--border);
	}
	.pbar {
		flex: 1;
		background: var(--accent);
		min-height: 1px;
	}
	.ends {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
	}
	.memb {
		display: inline-block;
		margin: 0 10px 2px 0;
		white-space: nowrap;
	}
	.memb.out {
		color: var(--warn);
	}
	button.link {
		background: none;
		border: none;
		color: var(--accent);
		padding: 0;
		font: inherit;
	}
	tr.sel td {
		background: color-mix(in srgb, var(--accent) 5%, transparent);
	}
	.warnc {
		color: var(--warn);
	}
</style>
