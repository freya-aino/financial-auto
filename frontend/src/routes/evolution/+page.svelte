<script lang="ts">
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import Histogram from '../../lib/Histogram.svelte';
	import Scatter from '../../lib/Scatter.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { ago, live, poll, short, type GpuStatus, type JobView } from '../../lib/api';

	const STATUSES = ['queued', 'running', 'ok', 'failed', 'preempted', 'budget_exceeded'] as const;

	let counts = $state<Record<string, Record<string, number>>>({});
	let jobs = $state<JobView[]>([]);
	let objectives = $state<string[]>([]);
	let gpu = $state<GpuStatus | null>(null);
	let kindFilter = $state('');
	let error = $state('');
	let now = $state(Date.now());

	let xObj = $state('');
	let yObj = $state('');
	// Objectives where lower is better; everything else is maximised (spec 34).
	const MINIMISE = new Set(['reconstruction', 'compute_cost', 'crps', 'nll', 'loss']);

	// start-generation form
	let gKind = $state('evo.eval.decoder');
	let gSnapshot = $state('');
	let gGenomes = $state('');
	let gEta = $state(3);
	let gMinBudget = $state(60_000);
	let gResult = $state('');
	let gError = $state('');
	let gBusy = $state(false);

	const kinds = $derived(Object.keys(counts).sort());
	const scored = $derived(jobs.filter((j) => Object.keys(j.objectives).length > 0));
	const scores = $derived(jobs.map((j) => j.score).filter((s): s is number => s !== null));

	$effect(() => {
		if (!xObj && objectives.length) xObj = objectives[0];
		if (!yObj && objectives.length > 1) yObj = objectives[1];
	});

	/** Non-dominated points for the two chosen objectives. */
	const points = $derived.by(() => {
		if (!xObj || !yObj) return [];
		const sx = MINIMISE.has(xObj) ? -1 : 1;
		const sy = MINIMISE.has(yObj) ? -1 : 1;
		const pts = scored
			.filter((j) => xObj in j.objectives && yObj in j.objectives)
			.map((j) => ({ x: j.objectives[xObj], y: j.objectives[yObj], label: `${j.genome_id} (${short(j.job_id)})` }));
		return pts.map((p) => ({
			...p,
			highlight: !pts.some(
				(o) => sx * o.x >= sx * p.x && sy * o.y >= sy * p.y && (sx * o.x > sx * p.x || sy * o.y > sy * p.y)
			)
		}));
	});

	async function refresh() {
		now = Date.now();
		try {
			const [j, g] = await Promise.all([live.jobs(kindFilter || undefined, 150), live.gpu()]);
			counts = j.counts;
			jobs = j.jobs;
			objectives = j.objectives;
			gpu = g;
			error = '';
		} catch (e) {
			error = String(e);
		}
	}

	async function startGeneration(ev: SubmitEvent) {
		ev.preventDefault();
		gBusy = true;
		gResult = '';
		gError = '';
		try {
			const genomes = gGenomes
				.split(/[\s,]+/)
				.map((g) => g.trim())
				.filter(Boolean);
			const r = await live.startGeneration({
				kind: gKind,
				snapshot_id: gSnapshot,
				genomes,
				eta: gEta,
				min_budget_ms: gMinBudget
			});
			gResult = `generation ${r.generation_id} started`;
		} catch (e) {
			gError = e instanceof Error ? e.message : String(e);
		} finally {
			gBusy = false;
		}
	}

	const statusClass = (s: string) => (s === 'ok' ? 'ok' : s === 'failed' || s === 'budget_exceeded' ? 'danger' : s === 'running' ? '' : 'warn');

	onMount(() => poll(refresh, 2000));
</script>

<ViewHeader title="Evolution Lab" spec="51 §6.4" note="live from jobs.* and evo-orchestrator heartbeats" />
{#if error}<p class="badge danger">{error}</p>{/if}

<div class="grid top">
	<section class="panel">
		<div class="panel-head">
			<h2>Jobs by kind</h2>
			<span class="dim">since control-api start</span>
		</div>
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
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Start generation</h2><span class="dim">ctl.evo.start · successive halving</span></div>
		<form onsubmit={startGeneration} class="form">
			<label>kind
				<select bind:value={gKind}>
					<option value="evo.eval.decoder">evo.eval.decoder</option>
					<option value="evo.eval.strategy">evo.eval.strategy</option>
					<option value="evo.eval.allocator">evo.eval.allocator</option>
				</select>
			</label>
			<label>snapshot <input bind:value={gSnapshot} placeholder="encoder snapshot id" /></label>
			<label class="wide">genomes <textarea bind:value={gGenomes} rows="3" placeholder="g-1, g-2, g-3 …"></textarea></label>
			<label>eta <input type="number" min="2" max="10" bind:value={gEta} /></label>
			<label>min budget (ms) <input type="number" min="1000" step="1000" bind:value={gMinBudget} /></label>
			<div class="wide actions">
				<button type="submit" disabled={gBusy || !gGenomes.trim()}>{gBusy ? 'starting…' : 'Start generation'}</button>
				{#if gResult}<span class="badge ok">{gResult}</span>{/if}
				{#if gError}<span class="badge danger">{gError}</span>{/if}
			</div>
		</form>
	</section>
</div>

<div class="grid two">
	<section class="panel">
		<div class="panel-head">
			<h2>{objectives.length >= 2 ? 'Pareto view' : 'Score distribution'}</h2>
			{#if objectives.length >= 2}
				<div class="pickers">
					<select bind:value={xObj}>{#each objectives as o}<option>{o}</option>{/each}</select>
					<span class="dim">vs</span>
					<select bind:value={yObj}>{#each objectives as o}<option>{o}</option>{/each}</select>
				</div>
			{/if}
		</div>
		{#if objectives.length >= 2 && points.length}
			<Scatter {points} xLabel={xObj} yLabel={yObj} />
			<p class="dim">Highlighted: non-dominated for these two objectives ({[...MINIMISE].filter((m) => m === xObj || m === yObj).join(', ') || 'none'} minimised).</p>
		{:else if scores.length}
			<Histogram values={scores} label="score" />
		{:else}
			<Empty producer="jobs.result.* with {'{"score"}'} or {'{"objectives"}'} payloads" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Recent jobs</h2>
			<select bind:value={kindFilter} onchange={refresh}>
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
								<td class="mono">{j.genome_id}</td>
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
</div>

<style>
	.top {
		grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
		margin-bottom: 12px;
	}
	.two {
		grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
	}
	.tall {
		max-height: 420px;
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
