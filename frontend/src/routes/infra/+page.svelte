<script lang="ts">
	// Ops → Infra (spec 51 §6 round 9): real usage per store instead of TCP badges, the backup /
	// WAL / restore-drill card, the last maintenance run per job and the DLQ browser. Replay and
	// purge ask confirm() and show the API's answer (never optimistic).
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { ago, bytes, num, poll } from '../../lib/api';
	import { errText } from '../../lib/console';
	import {
		backupRows,
		connectionRows,
		connectionsSummary,
		fmtAge,
		infraApi,
		MAINT_LABELS,
		maintSummary,
		purgeConfirmText,
		replayConfirmText,
		vmBudget,
		type BackupView,
		type ConnectionsView,
		type DlqView,
		type MaintJob,
		type StoresView
	} from '../../lib/infra';

	let stores = $state<StoresView | null>(null);
	let storesError = $state('');
	let connections = $state<ConnectionsView | null>(null);
	let connectionsError = $state('');
	let openConn = $state<string | null>(null);
	let backup = $state<BackupView | null>(null);
	let backupError = $state('');
	let jobs = $state<MaintJob[]>([]);
	let jobsError = $state('');
	let dlq = $state<DlqView | null>(null);
	let dlqError = $state('');
	let dlqLimit = $state(50);
	let openSeq = $state<number | null>(null);
	let openJob = $state<string | null>(null);
	let busy = $state(false);
	let actionMsg = $state('');
	let actionOk = $state(false);
	let now = $state(Date.now());

	const vm = $derived(stores ? vmBudget(stores.stores.victoriametrics) : null);
	const backupCard = $derived(backup ? backupRows(backup) : []);
	const connRows = $derived(connections ? connectionRows(connections) : []);
	const connSummary = $derived(connRows.length ? connectionsSummary(connRows) : null);

	async function loadStores() {
		try {
			stores = await infraApi.stores();
			storesError = '';
		} catch (e) {
			storesError = errText(e);
		}
	}
	async function loadConnections() {
		try {
			connections = await infraApi.connections();
			connectionsError = '';
		} catch (e) {
			connectionsError = errText(e);
		}
	}
	async function loadBackup() {
		try {
			backup = await infraApi.backup();
			backupError = '';
		} catch (e) {
			backupError = errText(e);
		}
	}
	async function loadJobs() {
		try {
			jobs = (await infraApi.maintenance()).jobs;
			jobsError = '';
		} catch (e) {
			jobsError = errText(e);
		}
	}
	async function loadDlq() {
		try {
			dlq = await infraApi.dlq(dlqLimit);
			dlqError = '';
		} catch (e) {
			dlqError = errText(e);
		}
	}

	/** confirm() → API call → show the API's answer, then refresh the DLQ list. */
	async function act(question: string, done: string, call: () => Promise<unknown>) {
		if (!confirm(question)) return;
		busy = true;
		try {
			const r = await call();
			actionMsg = `${done}: ${JSON.stringify(r)}`;
			actionOk = true;
			await loadDlq();
		} catch (e) {
			actionMsg = `${done} failed: ${errText(e)}`;
			actionOk = false;
		} finally {
			busy = false;
		}
	}

	$effect(() => {
		void dlqLimit;
		void loadDlq();
	});
	onMount(() => {
		void loadStores();
		void loadConnections();
		void loadBackup();
		void loadJobs();
		const stopFast = poll(loadDlq, 15_000);
		const stopSlow = poll(async () => {
			await Promise.all([loadStores(), loadConnections(), loadBackup(), loadJobs()]);
		}, 30_000);
		const t = setInterval(() => (now = Date.now()), 5000);
		return () => {
			stopFast();
			stopSlow();
			clearInterval(t);
		};
	});

	const storeErr = (s: { error?: string } | undefined) => s?.error ?? null;
	const json = (v: unknown) => JSON.stringify(v, null, 2);
</script>

<ViewHeader title="Infra" spec="51 §6 round 9" note="GET /api/infra/stores|maintenance|backup · /api/dlq — real usage, backup health, maintenance runs, dead letters" />

<div class="grid cols">
	<section class="panel wide">
		<div class="panel-head">
			<h2>External connections</h2>
			{#if connSummary}
				<span class="badge {connSummary.missing ? 'danger' : 'ok'}">{connSummary.text}</span>
			{/if}
		</div>
		{#if connectionsError}<p class="badge danger">{connectionsError}</p>
		{:else if !connections}<p class="dim">loading…</p>
		{:else}
			<table>
				<thead><tr><th>connection</th><th>kind</th><th>status</th><th>credentials</th><th>account</th></tr></thead>
				<tbody>
					{#each connRows as c (c.id)}
						<tr class="click" onclick={() => (openConn = openConn === c.id ? null : c.id)}>
							<td>{c.name}</td>
							<td class="dim">{c.kind}</td>
							<td>{#if c.status}<span class="badge {c.statusCls}">{c.status}</span>{:else}<span class="dim">—</span>{/if}</td>
							<td>
								{#each c.credentials as f (f.name)}
									{#if f.set}
										<span class="cred mono" title="{f.secret_ref} · {f.env}">{f.name}{f.hint ? ` ${f.hint}` : ''} ✓</span>
									{:else if f.required}
										<span class="badge danger" title="set {f.secret_ref} in config/secrets.toml (or {f.env})">{f.name} missing</span>
									{/if}
								{/each}
							</td>
							<td class="dim">{c.account.map(([k, v]) => `${k}: ${v}`).join(' · ') || '—'}{c.note ? ` · ${c.note}` : ''}</td>
						</tr>
						{#if openConn === c.id}
							<tr class="detail"><td colspan="5">
								{#if c.missing.length}
									<p class="small">Missing credentials — owner-edited in <span class="mono">config/secrets.toml</span> (never writable through this API):</p>
									<ul class="small">
										{#each c.missing as f (f.name)}
											<li><span class="mono">{f.secret_ref}</span> <span class="dim">(env override <span class="mono">{f.env}</span>)</span></li>
										{/each}
									</ul>
								{:else}
									<p class="small dim">All credentials set. Values stay in config/secrets.toml; this view only shows masked tails.</p>
								{/if}
							</td></tr>
						{/if}
					{/each}
				</tbody>
			</table>
			<details>
				<summary class="dim small">Infra endpoints ({connections.endpoints.length}) — passwords redacted</summary>
				<table class="endpoints">
					<tbody>
						{#each connections.endpoints as e (e.name)}
							<tr><td>{e.name}</td><td class="mono dim">{e.url}</td></tr>
						{/each}
					</tbody>
				</table>
			</details>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Postgres</h2>{#if storeErr(stores?.stores.postgres)}<span class="badge danger">down</span>{/if}</div>
		{#if storesError}<p class="badge danger">{storesError}</p>
		{:else if !stores}<p class="dim">loading…</p>
		{:else if stores.stores.postgres.error}<p class="err">{stores.stores.postgres.error}</p>
		{:else}
			<p class="big mono">{bytes(stores.stores.postgres.size_bytes ?? 0)}</p>
			<p class="dim">{stores.stores.postgres.connections} / {stores.stores.postgres.max_connections} connections</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>VictoriaMetrics</h2>{#if storeErr(stores?.stores.victoriametrics)}<span class="badge danger">down</span>{/if}</div>
		{#if storesError}<p class="badge danger">{storesError}</p>
		{:else if !stores}<p class="dim">loading…</p>
		{:else if stores.stores.victoriametrics.error}<p class="err">{stores.stores.victoriametrics.error}</p>
		{:else if vm}
			<p class="big mono">{bytes(stores.stores.victoriametrics.data_size_bytes ?? 0)} <span class="dim">/ {stores.vm_disk_budget_gib} GiB budget</span></p>
			<div class="bar wide-bar" title="{vm.pct} of FA_VM_DISK_BUDGET_GIB"><div class:warn-fill={vm.cls === 'warn'} class:danger-fill={vm.cls === 'danger'} style="width: {Math.min(100, 100 * vm.ratio)}%"></div></div>
			<p class="dim">{num(stores.stores.victoriametrics.series, 0)} series · {num(stores.stores.victoriametrics.active_series, 0)} active this hour</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Garage</h2>{#if storeErr(stores?.stores.garage)}<span class="badge danger">down</span>{/if}</div>
		{#if storesError}<p class="badge danger">{storesError}</p>
		{:else if !stores}<p class="dim">loading…</p>
		{:else if stores.stores.garage.error}<p class="err">{stores.stores.garage.error}</p>
		{:else}
			<table>
				<thead><tr><th>bucket</th><th>objects</th><th>bytes</th></tr></thead>
				<tbody>
					{#each stores.stores.garage.buckets ?? [] as b (b.bucket)}
						<tr>
							<td class="mono">{b.bucket}</td>
							{#if b.error}<td colspan="2" class="err">{b.error}</td>
							{:else}<td class="mono">{b.objects}</td><td class="mono">{bytes(b.bytes ?? 0)}</td>{/if}
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Qdrant</h2>{#if storeErr(stores?.stores.qdrant)}<span class="badge danger">down</span>{/if}</div>
		{#if storesError}<p class="badge danger">{storesError}</p>
		{:else if !stores}<p class="dim">loading…</p>
		{:else if stores.stores.qdrant.error}<p class="err">{stores.stores.qdrant.error}</p>
		{:else if (stores.stores.qdrant.collections ?? []).length === 0}<p class="dim">no collections</p>
		{:else}
			<table>
				<thead><tr><th>collection</th><th>points</th><th>status</th></tr></thead>
				<tbody>
					{#each stores.stores.qdrant.collections ?? [] as c (c.name)}
						<tr>
							<td class="mono">{c.name}</td>
							{#if c.error}<td colspan="2" class="err">{c.error}</td>
							{:else}<td class="mono">{c.points ?? '—'}</td><td><span class="badge {c.status === 'green' ? 'ok' : 'warn'}">{c.status ?? '?'}</span></td>{/if}
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Helix</h2>{#if storeErr(stores?.stores.helix)}<span class="badge danger">down</span>{/if}</div>
		{#if storesError}<p class="badge danger">{storesError}</p>
		{:else if !stores}<p class="dim">loading…</p>
		{:else if stores.stores.helix.error}<p class="err">{stores.stores.helix.error}</p>
		{:else}
			<p class="big mono">{num(stores.stores.helix.nodes, 0)} <span class="dim">nodes</span> · {num(stores.stores.helix.edges, 0)} <span class="dim">edges</span></p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Backup & drills</h2></div>
		{#if backupError}<p class="badge danger">{backupError}</p>
		{:else if !backup}<p class="dim">loading…</p>
		{:else}
			<table>
				<tbody>
					{#each backupCard as r (r.id)}
						<tr>
							<td>{r.label}</td>
							<td><span class="badge {r.cls}">{r.age}</span></td>
							<td class="dim">{r.detail}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<p class="dim small">fa_backup_* · fa_wal_archive_* · fa_restore_drill_* from VictoriaMetrics</p>
		{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head"><h2>Maintenance</h2><span class="dim">VM retention · genome GC · media downsample · accounting close (audit.log)</span></div>
		{#if jobsError}<p class="badge danger">{jobsError}</p>
		{:else if !jobs.length}<p class="dim">loading…</p>
		{:else}
			<table>
				<thead><tr><th>job</th><th>last run</th><th>outcome</th><th>key numbers</th></tr></thead>
				<tbody>
					{#each jobs as j (j.action)}
						<tr class="click" onclick={() => (openJob = openJob === j.action ? null : j.action)}>
							<td>{MAINT_LABELS[j.action] ?? j.action}</td>
							<td class="mono dim" title={j.at ?? ''}>{j.at ? `${ago(j.at, now)} ago` : 'never'}</td>
							<td>
								{#if j.ok === null}<span class="badge">never ran</span>
								{:else if j.ok}<span class="badge ok">ok</span>
								{:else}<span class="badge danger">failed</span>{/if}
							</td>
							<td class="dim">{j.ok === false ? String(j.error ?? '') : maintSummary(j.report)}</td>
						</tr>
						{#if openJob === j.action}
							<tr class="detail"><td colspan="4"><pre class="mono">{json(j.ok === false ? j.error : j.report)}</pre></td></tr>
						{/if}
					{/each}
				</tbody>
			</table>
		{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head">
			<h2>Dead letters <span class="dim">{dlq ? `(${dlq.stream.messages} in DLQ · ${bytes(dlq.stream.bytes)})` : ''}</span></h2>
			<div class="controls">
				<select bind:value={dlqLimit} aria-label="limit">{#each [20, 50, 200, 500] as l}<option value={l}>{l}</option>{/each}</select>
				<button onclick={loadDlq}>refresh</button>
				<button
					disabled={busy || !dlq || dlq.stream.messages === 0}
					onclick={() => act(purgeConfirmText(dlq?.stream.messages ?? 0), 'purge', () => infraApi.dlqPurge())}
				>purge all…</button>
			</div>
		</div>
		{#if actionMsg}<p class="badge {actionOk ? 'ok' : 'danger'}">{actionMsg}</p>{/if}
		{#if dlqError}<p class="badge danger">{dlqError}</p>{/if}
		{#if dlq && dlq.messages.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>seq</th><th>age</th><th>stream</th><th>original subject</th><th>reason</th><th>size</th><th></th></tr></thead>
					<tbody>
						{#each dlq.messages as m (m.seq)}
							<tr class="click" onclick={() => (openSeq = openSeq === m.seq ? null : m.seq)}>
								<td class="mono">{m.seq}</td>
								<td class="mono dim">{fmtAge(m.age_s)}</td>
								<td class="mono">{m.original_stream ?? '?'}</td>
								<td class="mono">{m.original_subject}</td>
								<td class="dim">{m.reason ?? ''}{m.deliveries ? ` · ${m.deliveries} deliveries` : ''}</td>
								<td class="mono">{bytes(m.size)}</td>
								<td>
									<button
										disabled={busy}
										onclick={(e) => {
											e.stopPropagation();
											act(replayConfirmText(m), `replay seq ${m.seq}`, () => infraApi.dlqReplay(m.seq));
										}}
									>replay…</button>
								</td>
							</tr>
							{#if openSeq === m.seq}
								<tr class="detail"><td colspan="7">
									<p class="dim small mono">{m.dlq_subject}{m.consumer ? ` · consumer ${m.consumer}` : ''}{m.original_seq ? ` · source seq ${m.original_seq}` : ''}</p>
									<pre class="mono">{m.preview}{m.truncated ? '\n… (truncated)' : ''}</pre>
									{#if m.lossy}<p class="dim small">binary payload: preview is lossy UTF-8</p>{/if}
								</td></tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
		{:else if dlq && !dlqError}
			<Empty producer="fa_bus::dlq (poison messages land on dlq.<STREAM>.<subject> after max_deliver / terminate)" hint="" />
		{/if}
	</section>
</div>

<style>
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
	}
	.wide {
		grid-column: 1 / -1;
	}
	.tall {
		max-height: 420px;
	}
	.controls {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.click {
		cursor: pointer;
	}
	.detail td {
		background: var(--bg-inset);
	}
	.big {
		font-size: 1.3em;
		margin: 6px 0 2px;
	}
	.small {
		font-size: 0.85em;
	}
	.err {
		color: var(--danger);
	}
	.bar {
		display: inline-block;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
	}
	.wide-bar {
		width: 100%;
	}
	.bar > div {
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
	}
	.bar > div.warn-fill {
		background: var(--warn);
	}
	.bar > div.danger-fill {
		background: var(--danger);
	}
	pre {
		margin: 4px 0;
		white-space: pre-wrap;
		word-break: break-all;
	}
	.cred {
		margin-right: 10px;
		white-space: nowrap;
	}
	details {
		margin-top: 8px;
	}
	summary {
		cursor: pointer;
	}
	.endpoints {
		margin-top: 4px;
		width: auto;
	}
	.endpoints td:first-child {
		padding-right: 12px;
	}
</style>
