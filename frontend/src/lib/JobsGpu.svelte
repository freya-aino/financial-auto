<script lang="ts">
	// Ops "Jobs & GPU" (spec 51 §6.2): queue by priority class, running jobs, GPU arbiter state.
	// Job.priority indexes the arbiter's class list (spec 60, D-006).
	import { onMount } from 'svelte';
	import Empty from './Empty.svelte';
	import { ago, api, live, poll, short, type GpuStatus, type JobView } from './api';

	let jobs = $state<JobView[]>([]);
	let gpu = $state<GpuStatus | null>(null);
	let error = $state('');
	let cancelMsg = $state('');
	let now = $state(Date.now());

	const classes = $derived(gpu?.classes ?? []);
	const active = $derived(jobs.filter((j) => j.status === 'queued' || j.status === 'running'));
	const running = $derived(active.filter((j) => j.status === 'running'));
	const preempted = $derived(jobs.filter((j) => j.status === 'preempted').length);
	const byClass = $derived.by(() => {
		const rows = classes.map((c) => ({ name: c, queued: 0, running: 0 }));
		const other = { name: 'other', queued: 0, running: 0 };
		for (const j of active) {
			const row = rows[j.priority] ?? other;
			if (j.status === 'running') row.running++;
			else row.queued++;
		}
		return other.queued + other.running ? [...rows, other] : rows;
	});

	async function refresh() {
		now = Date.now();
		try {
			const [j, g] = await Promise.all([live.jobs(undefined, 200), live.gpu()]);
			jobs = j.jobs;
			gpu = g;
			error = '';
		} catch (e) {
			error = String(e);
		}
	}

	async function cancel(job: JobView) {
		if (!confirm(`Cancel job ${short(job.job_id, 10)} (${job.kind})?`)) return;
		try {
			await api.cancelJob(job.job_id);
			cancelMsg = `cancel requested for ${short(job.job_id, 10)}`;
		} catch (e) {
			cancelMsg = e instanceof Error ? e.message : String(e);
		}
	}

	onMount(() => poll(refresh, 2000));
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Jobs &amp; GPU</h2>
		{#if gpu?.available}
			<span class="badge ok">arbiter · {ago(gpu.last_seen, now)} ago</span>
		{:else}
			<span class="badge warn">arbiter offline</span>
		{/if}
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}

	<div class="tiles">
		<div><span class="dim">GPU used</span><b class="mono">{gpu?.used_mb ?? '—'} MB{gpu?.capacity_mb ? ` / ${gpu.capacity_mb}` : ''}</b></div>
		<div><span class="dim">leases / queued</span><b class="mono">{gpu?.active ?? '—'} / {gpu?.queued ?? '—'}</b></div>
		<div><span class="dim">running jobs</span><b class="mono">{running.length}</b></div>
		<div><span class="dim">preempted</span><b class="mono" class:warn={preempted > 0}>{preempted}</b></div>
	</div>

	{#if classes.length}
		<table>
			<thead><tr><th>#</th><th>priority class</th><th>running</th><th>queued</th></tr></thead>
			<tbody>
				{#each byClass as row, i (row.name)}
					<tr>
						<td class="mono dim">{i < classes.length ? i + 1 : '—'}</td>
						<td class="mono">{row.name}</td>
						<td class="mono" class:dim={!row.running}>{row.running}</td>
						<td class="mono" class:dim={!row.queued}>{row.queued}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}

	<h3 class="sub">Running</h3>
	{#if running.length}
		<table>
			<thead><tr><th>job</th><th>kind</th><th>worker</th><th>progress</th><th>gpu ms</th><th></th></tr></thead>
			<tbody>
				{#each running as j (j.job_id)}
					<tr title={j.message}>
						<td class="mono">{short(j.job_id, 10)}</td>
						<td class="mono dim">{j.kind}</td>
						<td class="mono dim">{j.worker_id ? short(j.worker_id, 8) : '—'}</td>
						<td><div class="progress"><div style="width: {100 * j.progress}%"></div></div></td>
						<td class="mono dim">{j.used_gpu_ms}{j.budget_gpu_ms ? ` / ${j.budget_gpu_ms}` : ''}</td>
						<td><button onclick={() => cancel(j)}>cancel</button></td>
					</tr>
				{/each}
			</tbody>
		</table>
	{:else if jobs.length}
		<p class="dim">nothing running · {active.length} queued · {jobs.length} seen</p>
	{:else}
		<Empty producer="jobs.* (evo-orchestrator / trainer)" />
	{/if}
	{#if cancelMsg}<p class="badge warn">{cancelMsg}</p>{/if}
	<p class="dim small">Full job history and Pareto views: <a href="/evolution/">Evolution Lab</a>.</p>
</section>

<style>
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
	.warn {
		color: var(--warn);
	}
	.sub {
		font-size: 12px;
		margin: 12px 0 6px;
	}
	.progress {
		width: 80px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
	}
	.progress > div {
		height: 100%;
		background: var(--warn);
		border-radius: 3px;
	}
	.small {
		font-size: 11px;
		margin: 10px 0 0;
	}
</style>
