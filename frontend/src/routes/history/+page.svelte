<script lang="ts">
	// Ops history (spec 20 §5.4, spec 60): every `jobs.result.>` stored in `ops.job_run` by
	// control-api's durable consumer, the job summary per kind × status, and the config snapshots
	// (`ops.config_snapshot`, one row per changed digest of a config section).
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { ago, short } from '../../lib/api';
	import { errText } from '../../lib/console';
	import { digestChanges, operatorApi, summaryMatrix, type ConfigSnapshotRow, type JobRun } from '../../lib/operator';
	import { loadPref, savePref } from '../../lib/prefs';

	let hours = $state(loadPref('history.hours', 24));
	let kind = $state(loadPref('history.kind', ''));
	let limit = $state(100);
	let service = $state(loadPref('history.service', ''));
	let withConfig = $state(false);
	$effect(() => savePref('history.hours', hours));
	$effect(() => savePref('history.kind', kind));
	$effect(() => savePref('history.service', service));

	let summary = $state<{ since: string; counts: { kind: string; status: string; count: number }[] } | null>(null);
	let summaryError = $state('');
	let runs = $state<JobRun[]>([]);
	let runsError = $state('');
	let snaps = $state<ConfigSnapshotRow[]>([]);
	let snapsError = $state('');
	let openRun = $state<string | null>(null);
	let openSnap = $state<number | null>(null);
	let now = $state(Date.now());

	const matrix = $derived(summaryMatrix(summary?.counts ?? []));
	const changes = $derived(digestChanges(snaps));
	const services = $derived([...new Set(snaps.map((s) => s.service))].sort());

	async function loadSummary() {
		try {
			summary = await operatorApi.jobSummary(hours);
			summaryError = '';
		} catch (e) {
			summaryError = errText(e);
		}
	}
	async function loadRuns() {
		try {
			runs = (await operatorApi.jobRuns(kind || undefined, limit)).runs;
			runsError = '';
		} catch (e) {
			runsError = errText(e);
		}
	}
	async function loadSnaps() {
		try {
			snaps = (await operatorApi.configSnapshots(service || undefined, 100, withConfig)).snapshots;
			snapsError = '';
		} catch (e) {
			snapsError = errText(e);
		}
	}

	$effect(() => {
		void hours;
		void loadSummary();
	});
	$effect(() => {
		void kind;
		void limit;
		void loadRuns();
	});
	$effect(() => {
		void service;
		void withConfig;
		void loadSnaps();
	});
	onMount(() => {
		const t = setInterval(() => (now = Date.now()), 5000);
		return () => clearInterval(t);
	});

	const statusClass = (s: string) => (s === 'ok' ? 'ok' : s === 'failed' || s === 'budget_exceeded' ? 'danger' : 'warn');
	const json = (v: unknown) => JSON.stringify(v, null, 2);
</script>

<ViewHeader title="Ops history" spec="20 §5.4 · 60" note="ops.job_run (durable jobs.result.> consumer), ops.config_snapshot (every 5 min on digest change)" />

<div class="grid cols">
	<section class="panel">
		<div class="panel-head">
			<h2>Job summary</h2>
			<label class="dim">last
				<select bind:value={hours}>{#each [1, 6, 24, 72, 168, 720] as h}<option value={h}>{h < 48 ? `${h} h` : `${h / 24} d`}</option>{/each}</select>
			</label>
		</div>
		{#if summaryError}<p class="badge danger">{summaryError}</p>{/if}
		{#if matrix.kinds.length}
			<div class="scroll">
				<table>
					<thead><tr><th>kind</th>{#each matrix.statuses as s}<th>{s}</th>{/each}<th>total</th></tr></thead>
					<tbody>
						{#each matrix.kinds as k (k.kind)}
							<tr class="click" onclick={() => (kind = kind === k.kind ? '' : k.kind)} class:sel={kind === k.kind}>
								<td class="mono">{k.kind}</td>
								{#each matrix.statuses as s}<td class="mono" class:dim={!k.counts[s]}><span class:danger={s !== 'ok' && k.counts[s]}>{k.counts[s] ?? 0}</span></td>{/each}
								<td class="mono">{k.total}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="dim small">since {summary?.since} · click a kind to filter the runs</p>
		{:else if !summaryError}
			<Empty producer="jobs.result.> (evo-worker, ingest backfill, …) via control-api's ops-job-history consumer" hint="" />
		{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head">
			<h2>Job runs <span class="dim">({runs.length})</span></h2>
			<div class="controls">
				<input class="mono" bind:value={kind} placeholder="kind (all)" size="22" aria-label="kind filter" />
				<select bind:value={limit} aria-label="limit">{#each [50, 100, 500, 2000] as l}<option value={l}>{l}</option>{/each}</select>
				<button onclick={loadRuns}>refresh</button>
			</div>
		</div>
		{#if runsError}<p class="badge danger">{runsError}</p>{/if}
		{#if runs.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>finished</th><th>job</th><th>kind</th><th>status</th><th>worker</th><th>GPU ms</th><th>outputs</th><th>error</th></tr></thead>
					<tbody>
						{#each runs as r (r.job_id + r.finished_at)}
							<tr class="click" onclick={() => (openRun = openRun === r.job_id ? null : r.job_id)}>
								<td class="mono dim" title={r.finished_at}>{ago(r.finished_at, now)} ago</td>
								<td class="mono">{short(r.job_id, 10)}</td>
								<td class="mono">{r.kind}</td>
								<td><span class="badge {statusClass(r.status)}">{r.status}</span></td>
								<td class="mono dim">{r.worker_id ?? '—'}</td>
								<td class="mono">{r.used_gpu_ms}</td>
								<td class="mono dim">{r.output_refs.length}</td>
								<td class="err">{r.error ?? ''}</td>
							</tr>
							{#if openRun === r.job_id}
								<tr class="detail"><td colspan="8"><pre class="mono">{json({ result: r.result, output_refs: r.output_refs, checkpoint_ref: r.checkpoint_ref, stream_seq: r.stream_seq })}</pre></td></tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !runsError}
			<p class="dim">no stored job results{kind ? ` of kind ${kind}` : ''}</p>
		{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head">
			<h2>Config snapshots <span class="dim">({snaps.length})</span></h2>
			<div class="controls">
				<select bind:value={service} aria-label="service">
					<option value="">all services</option>
					{#each services as s}<option value={s}>{s}</option>{/each}
					{#if service && !services.includes(service)}<option value={service}>{service}</option>{/if}
				</select>
				<label class="dim"><input type="checkbox" bind:checked={withConfig} /> include config</label>
			</div>
		</div>
		{#if snapsError}<p class="badge danger">{snapsError}</p>{/if}
		{#if changes.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>taken</th><th>service</th><th>section</th><th>digest</th><th>previous</th><th>actor</th></tr></thead>
					<tbody>
						{#each changes as c (c.id)}
							<tr class="click" onclick={() => (openSnap = openSnap === c.id ? null : c.id)}>
								<td class="mono dim" title={c.taken_at}>{ago(c.taken_at, now)} ago</td>
								<td class="mono">{c.service}</td>
								<td class="mono">{c.section}</td>
								<td class="mono">{c.digest.slice(0, 12)}</td>
								<td class="mono dim">{c.previous ? c.previous.slice(0, 12) : 'first in view'}</td>
								<td class="mono dim">{c.actor}</td>
							</tr>
							{#if openSnap === c.id}
								<tr class="detail"><td colspan="6">{#if c.config !== undefined}<pre class="mono">{json(c.config)}</pre>{:else}<span class="dim">tick "include config" to load the redacted section</span>{/if}</td></tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !snapsError}
			<p class="dim">no config snapshots stored yet</p>
		{/if}
	</section>
</div>

<style>
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
	}
	.wide {
		grid-column: 1 / -1;
	}
	.tall {
		max-height: 480px;
	}
	.controls {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.click {
		cursor: pointer;
	}
	.sel {
		background: var(--bg-inset);
	}
	.err {
		color: var(--danger);
		max-width: 320px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.danger {
		color: var(--danger);
	}
	pre {
		margin: 0;
		max-height: 300px;
		overflow: auto;
		font-size: 11px;
	}
	.small {
		font-size: 11px;
	}
</style>
