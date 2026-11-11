<script lang="ts">
	// Evolution Lab (spec 51 §6.4, D-049/D-052): generations (REST + /ws/evolution), start form for
	// decoder and strategy kinds, populations + Pareto fronts from the trial registry, genome
	// drawer, validation card, GPU arbiter and recent jobs.
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import GenomeDrawer from '../../lib/GenomeDrawer.svelte';
	import Generations from '../../lib/Generations.svelte';
	import Histogram from '../../lib/Histogram.svelte';
	import Scatter from '../../lib/Scatter.svelte';
	import Validation from '../../lib/Validation.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import {
		ago,
		live,
		poll,
		short,
		type GenerationFrame,
		type GenerationRow,
		type GpuStatus,
		type JobView,
		type ParetoView,
		type Population,
		type StartGenerationBody
	} from '../../lib/api';
	import { candidateLabel, directionFor, MINIMISE, nonDominated, paretoPoints, plottableKeys } from '../../lib/pareto';
	import { loadPref, savePref } from '../../lib/prefs';
	import { runtime } from '../../lib/runtime.svelte';
	import { evolutionWs, onBus } from '../../lib/ws.svelte';

	const STATUSES = ['queued', 'running', 'ok', 'failed', 'preempted', 'budget_exceeded'] as const;

	let counts = $state<Record<string, Record<string, number>>>({});
	let jobs = $state<JobView[]>([]);
	let objectives = $state<string[]>([]);
	let gpu = $state<GpuStatus | null>(null);
	let kindFilter = $state(loadPref('evo.kind', ''));
	let error = $state('');
	let now = $state(Date.now());

	let generations = $state<GenerationRow[]>([]);
	let populations = $state<Population[]>([]);
	let popError = $state('');
	let popId = $state(loadPref('evo.pop', ''));
	let popGen = $state<number | undefined>(undefined);
	let pareto = $state<ParetoView | null>(null);
	let paretoError = $state('');
	let genomeId = $state<string | null>(null);
	let validationCandidate = $state('');

	let xObj = $state(loadPref('evo.x', ''));
	let yObj = $state(loadPref('evo.y', ''));
	$effect(() => savePref('evo.kind', kindFilter));
	$effect(() => savePref('evo.x', xObj));
	$effect(() => savePref('evo.y', yObj));
	$effect(() => savePref('evo.pop', popId));

	// /ws/evolution: generation frames (full summaries) and job frames; REST stays the fallback.
	const genTopics = evolutionWs.topics<GenerationFrame>('generation.>');
	const jobTopic = evolutionWs.topic<JobView>('job.>');
	const frames = $derived(new Map([...genTopics.entries.values()].map((f) => [f.data.generation_id, f.data])));
	$effect(() => {
		if (jobTopic.count) void refreshJobs();
	});
	$effect(() => {
		// a new rung/summary may add registry stats: refetch populations on generation frames (debounced by raf)
		if (genTopics.count) void refreshPopulations();
	});

	// start-generation form (both kinds, D-049/D-052)
	let gKind = $state(loadPref('evo.start.kind', 'evo.eval.decoder'));
	let gSnapshot = $state(loadPref('evo.start.snapshot', ''));
	let gGenomes = $state('');
	let gEta = $state(3);
	let gMinBudget = $state(60_000);
	let gMaxBudget = $state<number | null>(null);
	let gMaxGenerations = $state(1);
	let gTotalBudget = $state<number | null>(null);
	let gDataWindow = $state<number | null>(6);
	let gResult = $state('');
	let gError = $state('');
	let gBusy = $state(false);
	$effect(() => savePref('evo.start.kind', gKind));
	$effect(() => savePref('evo.start.snapshot', gSnapshot));

	const kinds = $derived(Object.keys(counts).sort());
	const scored = $derived(jobs.filter((j) => Object.keys(j.objectives).length > 0));
	const scores = $derived(jobs.map((j) => j.score).filter((s): s is number => s !== null));
	const isStrategy = $derived(gKind === 'evo.eval.strategy');
	/** Genomes: one per line for strategies (inline TOML contains commas), whitespace/commas for decoders. */
	const genomeList = $derived(
		isStrategy
			? gGenomes.split(/\r?\n/).map((g) => g.trim()).filter(Boolean)
			: gGenomes.split(/[\s,]+/).map((g) => g.trim()).filter(Boolean)
	);

	const paretoKeys = $derived(pareto ? plottableKeys(pareto.fronts.flat(), pareto.objective_keys) : objectives);
	$effect(() => {
		const keys = paretoKeys;
		if (!keys.length) return;
		if (!xObj || !keys.includes(xObj)) xObj = keys[0];
		if (!yObj || !keys.includes(yObj) || yObj === xObj) yObj = keys.find((k) => k !== xObj) ?? keys[0];
	});

	/** Scatter points: the registry Pareto fronts when a population is selected, else the in-memory jobs. */
	const points = $derived.by(() => {
		if (!xObj || !yObj) return [];
		if (pareto) return paretoPoints(pareto.fronts.flat(), xObj, yObj, pareto.directions);
		const pts = scored
			.filter((j) => xObj in j.objectives && yObj in j.objectives)
			.map((j) => ({ x: j.objectives[xObj], y: j.objectives[yObj], label: `${j.genome_id} (${short(j.job_id)})`, candidate: j.genome_id, rank: -1 }));
		const front = nonDominated(pts, !MINIMISE.has(xObj), !MINIMISE.has(yObj));
		return pts.map((p, i) => ({ ...p, highlight: front[i] }));
	});
	const selectedPop = $derived(populations.find((p) => p.id === popId) ?? null);

	async function refreshJobs() {
		now = Date.now();
		try {
			const [j, g, gen] = await Promise.all([live.jobs(kindFilter || undefined, 150), live.gpu(), live.generations()]);
			counts = j.counts;
			jobs = j.jobs;
			objectives = j.objectives;
			gpu = g;
			generations = gen.generations;
			error = '';
		} catch (e) {
			error = String(e);
		}
	}

	async function refreshPopulations() {
		try {
			populations = (await live.populations()).populations;
			popError = '';
			if (popId && !populations.some((p) => p.id === popId)) popId = '';
		} catch (e) {
			popError = e instanceof Error ? e.message : String(e);
		}
	}

	async function loadPareto() {
		if (!popId) {
			pareto = null;
			return;
		}
		try {
			pareto = await live.pareto(popId, popGen);
			paretoError = '';
		} catch (e) {
			paretoError = e instanceof Error ? e.message : String(e);
			pareto = null;
		}
	}
	$effect(() => {
		void popId;
		void popGen;
		void loadPareto();
	});

	async function startGeneration(ev: SubmitEvent) {
		ev.preventDefault();
		gBusy = true;
		gResult = '';
		gError = '';
		try {
			const body: StartGenerationBody = {
				kind: gKind,
				snapshot_id: gSnapshot,
				genomes: genomeList,
				eta: gEta,
				min_budget_ms: gMinBudget,
				max_generations: gMaxGenerations
			};
			if (gMaxBudget) body.max_budget_ms = Number(gMaxBudget);
			if (gTotalBudget) body.total_budget_ms = Number(gTotalBudget);
			if (gDataWindow) body.data_window_hours = Number(gDataWindow);
			const r = await live.startGeneration(body);
			gResult = `generation ${r.generation_id} started`;
			await refreshJobs();
		} catch (e) {
			gError = e instanceof Error ? e.message : String(e);
		} finally {
			gBusy = false;
		}
	}

	function pickPoint(i: number) {
		const p = points[i];
		if (!p) return;
		const id = candidateLabel(p.candidate, 200);
		validationCandidate = id;
		genomeId = id;
	}

	const statusClass = (s: string) => (s === 'ok' ? 'ok' : s === 'failed' || s === 'budget_exceeded' ? 'danger' : s === 'running' ? '' : 'warn');

	onMount(() => {
		void refreshPopulations();
		const stopPoll = poll(refreshJobs, () => (evolutionWs.live ? runtime.pollMs * 10 : runtime.pollMs));
		const stopPops = poll(refreshPopulations, () => runtime.pollMs * 15);
		const stopBus = onBus('jobs.>', refreshJobs);
		return () => {
			stopPoll();
			stopPops();
			stopBus();
			genTopics.close();
			jobTopic.close();
		};
	});
</script>

<ViewHeader title="Evolution Lab" spec="51 §6.4" note="generations and jobs on /ws/evolution; registry (validation.trial) for populations and Pareto fronts; REST polling as fallback" />
{#if error}<p class="badge danger">{error}</p>{/if}

<section class="panel gens">
	<div class="panel-head">
		<h2>Generations <span class="dim">({generations.length})</span></h2>
		<span class="badge" class:ok={evolutionWs.live} class:warn={!evolutionWs.live} title="/ws/evolution · {evolutionWs.frames} frames · {evolutionWs.lagged} superseded">ws {evolutionWs.status}</span>
	</div>
	<Generations rows={generations} {frames} {populations} {now} onchanged={refreshJobs} onpick={(g) => (popId = populations.some((p) => p.id === g.id) ? g.id : popId)} />
	<p class="dim small">Rows come from <code>GET /api/evolution/generations</code> (<code>generation.summary</code> on jobs.result.evo.generation); the green dot marks rows backed by a live <code>generation.&lt;id&gt;</code> frame. Deflated Sharpe / PBO join from <code>GET /api/evolution/populations</code> when the registry has stats for the generation. Click a row to expand its Pareto front and select the population below.</p>
</section>

<div class="grid top">
	<section class="panel">
		<div class="panel-head"><h2>Start generation</h2><span class="dim">POST /api/evolution/generations → ctl.evo.start · successive halving (NSGA-II ranks)</span></div>
		<form onsubmit={startGeneration} class="form">
			<label>kind
				<select bind:value={gKind}>
					<option value="evo.eval.decoder">evo.eval.decoder (genome kv strings)</option>
					<option value="evo.eval.strategy">evo.eval.strategy (inline-TOML strategies)</option>
					<option value="evo.eval.allocator">evo.eval.allocator</option>
				</select>
			</label>
			<label>snapshot <input bind:value={gSnapshot} placeholder="encoder snapshot id (e.g. stat-v1-1m)" class="mono" /></label>
			<label class="wide">
				{isStrategy ? 'strategies (one inline-TOML table per line)' : 'genomes (comma / whitespace separated)'} <span class="dim">{genomeList.length} parsed</span>
				<textarea bind:value={gGenomes} rows="4" class="mono" placeholder={isStrategy ? '{ id = "gp-…", timescale = "1s", steps = 5, lookback_bars = 32, signal = "(neg (mul 400 analog_forecast))", uncertainty = "region_entropy" }' : 'retrieval:ts=1s;h=3;lb=8;levels=0.05/0.25/0.5/0.75/0.95;kernel=16;scale=1;shrink=0.5;k=10;tau=30;gap=5;metric=cos;lat=0.5'}></textarea>
			</label>
			<label>eta <input type="number" min="2" max="10" bind:value={gEta} /></label>
			<label>max generations <input type="number" min="1" max="100" bind:value={gMaxGenerations} /></label>
			<label>min budget (ms) <input type="number" min="1000" step="1000" bind:value={gMinBudget} /></label>
			<label>max budget (ms) <input type="number" min="1000" step="1000" bind:value={gMaxBudget} placeholder="orchestrator default" /></label>
			<label>total budget (ms) <input type="number" min="1000" step="60000" bind:value={gTotalBudget} placeholder="unbounded" /></label>
			<label>data window (hours) <input type="number" min="1" max="720" bind:value={gDataWindow} placeholder="6" /></label>
			<div class="wide actions">
				<button type="submit" disabled={gBusy || !genomeList.length}>{gBusy ? 'starting…' : `Start ${isStrategy ? 'strategy' : 'decoder'} generation (${genomeList.length})`}</button>
				{#if gResult}<span class="badge ok">{gResult}</span>{/if}
				{#if gError}<span class="badge danger">{gError}</span>{/if}
			</div>
		</form>
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>GPU arbiter</h2>
			{#if gpu?.available}<span class="badge ok">evo-orchestrator · {ago(gpu.last_seen, now)} ago</span>{:else}<span class="badge warn">unavailable</span>{/if}
		</div>
		{#if gpu?.available}
			<div class="tiles">
				<div><span class="dim">used</span><b class="mono">{gpu.used_mb ?? 0} MB{gpu.capacity_mb ? ` / ${gpu.capacity_mb}` : ''}</b></div>
				<div><span class="dim">active leases</span><b class="mono">{gpu.active ?? 0}</b></div>
				<div><span class="dim">queued</span><b class="mono">{gpu.queued ?? 0}</b></div>
				<div><span class="dim">generations</span><b class="mono">{gpu.generations_done ?? 0}/{gpu.generations_started ?? 0}</b></div>
			</div>
		{:else}
			<Empty producer="svc.heartbeat.evo-orchestrator" hint="go/bin/evo-orchestrator (or the compose stack)" />
		{/if}
		<div class="classes">
			<span class="dim">priority:</span>
			{#each gpu?.classes ?? [] as c, i}<span class="badge">{i + 1}. {c}</span>{/each}
		</div>
		<h3 class="sub">Jobs by kind <span class="dim">since control-api start</span></h3>
		{#if kinds.length}
			<table>
				<thead>
					<tr><th>kind</th>{#each STATUSES as s}<th>{s}</th>{/each}</tr>
				</thead>
				<tbody>
					{#each kinds as k}
						<tr>
							<td class="mono">{k}</td>
							{#each STATUSES as s}
								<td class="mono" class:dim={!counts[k][s]}>{counts[k][s] ?? 0}</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<Empty producer="jobs.* (evo-orchestrator / trainer)" />
		{/if}
	</section>
</div>

<div class="grid two">
	<section class="panel">
		<div class="panel-head">
			<h2>Populations &amp; Pareto front</h2>
			<div class="pickers">
				<select bind:value={popId} title="GET /api/evolution/populations (validation.trial per generation)">
					<option value="">in-memory jobs ({scored.length} scored)</option>
					{#each populations as p (p.id)}
						<option value={p.id}>{short(p.id, 12)} · {p.kind.replace('evo.eval.', '')} · {p.candidates} cand · {p.loops} loop{p.loops === 1 ? '' : 's'}</option>
					{/each}
				</select>
				{#if selectedPop}
					<select bind:value={popGen} title="loop (gen=)">
						<option value={undefined}>latest loop</option>
						{#each Array.from({ length: selectedPop.loops }, (_, i) => i) as i}<option value={i}>loop {i}</option>{/each}
					</select>
				{/if}
			</div>
		</div>
		{#if popError}<p class="badge warn">{popError}</p>{/if}
		{#if paretoError}<p class="badge danger">{paretoError}</p>{/if}
		{#if paretoKeys.length >= 2}
			<div class="pickers row">
				<select bind:value={xObj}>{#each paretoKeys as o}<option value={o}>{o} {directionFor(o, pareto?.directions) === 'min' ? '↓' : '↑'}</option>{/each}</select>
				<span class="dim">vs</span>
				<select bind:value={yObj}>{#each paretoKeys as o}<option value={o}>{o} {directionFor(o, pareto?.directions) === 'min' ? '↓' : '↑'}</option>{/each}</select>
				{#if pareto}<span class="dim">gen {pareto.gen} · {pareto.candidates} candidates · fronts {pareto.front_sizes.join(' / ')}</span>{/if}
			</div>
		{/if}
		{#if paretoKeys.length >= 2 && points.length}
			<Scatter {points} xLabel={xObj} yLabel={yObj} onselect={pickPoint} />
			<p class="dim small">Highlighted: {pareto ? 'rank-0 front from the registry (fa_evolution, directions per kind)' : `non-dominated for these two objectives (${[xObj, yObj].filter((k) => directionFor(k) === 'min').join(', ') || 'none'} minimised)`}. Click a point to open the genome drawer and load its validation reports.</p>
			{#if pareto?.stats}
				<p class="mono small">registry: n_trials {pareto.stats.n_trials} · deflated Sharpe {pareto.stats.deflated_sharpe?.toFixed(4) ?? '—'} · PBO {pareto.stats.pbo === null || pareto.stats.pbo === undefined ? '—' : (100 * pareto.stats.pbo).toFixed(0) + '%'} · best {candidateLabel(pareto.stats.best_candidate ?? '', 30)}</p>
			{/if}
		{:else if scores.length && !pareto}
			<Histogram values={scores} label="score" />
		{:else}
			<Empty producer="validation.trial rows (evo-worker) or jobs.result.* with objectives" />
		{/if}
		{#if populations.length}
			<div class="scroll pops">
				<table>
					<thead><tr><th>generation</th><th>kind</th><th>loops</th><th>trials</th><th>candidates</th><th>deflated Sharpe</th><th>PBO</th><th>best</th><th>last</th></tr></thead>
					<tbody>
						{#each populations as p (p.id)}
							<tr class:sel={p.id === popId} onclick={() => (popId = p.id)}>
								<td class="mono">{short(p.id, 12)}</td>
								<td class="mono dim">{p.kind.replace('evo.eval.', '')}</td>
								<td class="mono">{p.loops}</td>
								<td class="mono">{p.trials}</td>
								<td class="mono">{p.candidates}</td>
								<td class="mono">{p.stats?.deflated_sharpe?.toFixed(4) ?? '—'}</td>
								<td class="mono" class:warn={(p.stats?.pbo ?? 0) > 0.5}>{p.stats?.pbo === null || p.stats?.pbo === undefined ? '—' : `${(100 * p.stats.pbo).toFixed(0)}%`}</td>
								<td class="mono dim" title={p.stats?.best_candidate ?? ''}><button class="link" onclick={(e) => (e.stopPropagation(), (genomeId = candidateLabel(p.stats?.best_candidate ?? '', Infinity)), (validationCandidate = genomeId))}>{candidateLabel(p.stats?.best_candidate ?? '', 24) || '—'}</button></td>
								<td class="mono dim">{ago(p.last_at, now)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Recent jobs</h2>
			<select bind:value={kindFilter} onchange={refreshJobs}>
				<option value="">all kinds</option>
				{#each kinds as k}<option value={k}>{k}</option>{/each}
			</select>
		</div>
		{#if jobs.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>job</th><th>kind</th><th>genome</th><th>status</th><th>progress</th><th>gpu ms</th><th>score</th><th>updated</th></tr></thead>
					<tbody>
						{#each jobs as j (j.job_id)}
							<tr title={j.error || j.message}>
								<td class="mono">{short(j.job_id, 10)}</td>
								<td class="mono dim">{j.kind}</td>
								<td class="mono"><button class="link" onclick={() => (genomeId = j.genome_id)} disabled={!j.genome_id}>{candidateLabel(j.genome_id, 28) || '—'}</button></td>
								<td><span class="badge {statusClass(j.status)}">{j.status}</span></td>
								<td><div class="progress"><div style="width: {100 * j.progress}%" class:done={j.status === 'ok'}></div></div></td>
								<td class="mono dim">{j.used_gpu_ms || '—'}{j.budget_gpu_ms ? ` / ${j.budget_gpu_ms}` : ''}</td>
								<td class="mono">{j.score?.toFixed(3) ?? '—'}</td>
								<td class="mono dim">{ago(j.updated_at, now)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<Empty producer="jobs.*" />
		{/if}
	</section>

	<Validation candidate={validationCandidate} />
</div>

<GenomeDrawer {genomeId} onclose={() => (genomeId = null)} onopen={(id) => (genomeId = id)} />

<style>
	.gens {
		margin-bottom: 12px;
	}
	.top {
		grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
		margin-bottom: 12px;
	}
	.two {
		grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
	}
	.tall {
		max-height: 420px;
	}
	.pops {
		max-height: 220px;
		margin-top: 10px;
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 8px;
		margin-bottom: 10px;
	}
	.tiles > div {
		display: flex;
		flex-direction: column;
		background: var(--bg-inset);
		border-radius: 4px;
		padding: 6px 8px;
	}
	.tiles b {
		font-size: 16px;
	}
	.classes {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
		align-items: center;
	}
	.sub {
		font-size: 12px;
		margin: 12px 0 6px;
	}
	.form {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 8px;
	}
	.form label {
		display: flex;
		flex-direction: column;
		gap: 2px;
		color: var(--text-dim);
	}
	.wide {
		grid-column: 1 / -1;
	}
	textarea {
		font: 12px var(--mono);
		color: var(--text);
		background: var(--bg-inset);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 4px 8px;
		resize: vertical;
	}
	.actions {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.pickers {
		display: flex;
		gap: 6px;
		align-items: center;
		flex-wrap: wrap;
	}
	.pickers.row {
		margin-bottom: 6px;
	}
	.small {
		font-size: 11px;
		margin: 6px 0 0;
	}
	.warn {
		color: var(--warn);
	}
	tr.sel td {
		color: var(--accent);
	}
	.pops tbody tr {
		cursor: pointer;
	}
	.link {
		background: none;
		border: none;
		padding: 0;
		color: var(--accent);
		font: inherit;
	}
	.progress {
		width: 90px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
	}
	.progress > div {
		height: 100%;
		background: var(--warn);
		border-radius: 3px;
	}
	.progress > div.done {
		background: var(--ok);
	}
</style>
