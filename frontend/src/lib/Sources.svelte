<script lang="ts">
	// Source catalog with live health (D-054): status ok|degraded|down, limiter saturation,
	// cursors, enable/disable, ingest start/stop and the Backfill dialog whose job lands in
	// Jobs & GPU (`kind=ingest.backfill`).
	import { onMount } from 'svelte';
	import Modal from './Modal.svelte';
	import { ago, api, poll, short, type Source, type SourceDetail } from './api';
	import { loadPref, savePref } from './prefs';
	import { runtime } from './runtime.svelte';
	import { onBus } from './ws.svelte';

	let sources = $state<Source[]>([]);
	let error = $state('');
	let now = $state(Date.now());
	let category = $state(loadPref('ops.category', ''));
	let status = $state(loadPref('ops.status', ''));
	let capability = $state(loadPref('ops.capability', ''));
	let query = $state('');
	let msg = $state<Record<string, { ok: boolean; text: string }>>({});
	let busy = $state<Record<string, boolean>>({});
	let detailId = $state<string | null>(null);
	let detail = $state<SourceDetail | null>(null);
	let detailError = $state('');

	// backfill dialog
	let bfSource = $state<Source | null>(null);
	let bfInstruments = $state('');
	let bfTimescale = $state('1m');
	let bfRange = $state('1d');
	let bfStart = $state('');
	let bfEnd = $state('');
	let bfMode = $state<'range' | 'absolute'>('range');
	let bfBusy = $state(false);
	let bfResult = $state('');
	let bfError = $state('');

	$effect(() => savePref('ops.category', category));
	$effect(() => savePref('ops.status', status));
	$effect(() => savePref('ops.capability', capability));

	const categories = $derived([...new Set(sources.map((s) => s.category))].sort());
	const statuses = $derived([...new Set(sources.map((s) => s.status))].sort());
	const filtered = $derived(
		sources.filter(
			(s) =>
				(!category || s.category === category) &&
				(!status || s.status === status) &&
				(!capability || s.capabilities.includes(capability)) &&
				(!query || `${s.id} ${s.name} ${s.notes}`.toLowerCase().includes(query.toLowerCase()))
		)
	);
	const healthCounts = $derived.by(() => {
		const c = { ok: 0, degraded: 0, down: 0 };
		for (const s of sources) if (s.health) c[s.health.status] = (c[s.health.status] ?? 0) + 1;
		return c;
	});

	async function refresh() {
		now = Date.now();
		try {
			sources = (await api.sources()).sources;
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
		if (detailId) void loadDetail(detailId);
	}

	async function loadDetail(id: string) {
		try {
			detail = await api.sourceDetail(id, 20);
			detailError = '';
		} catch (e) {
			detailError = e instanceof Error ? e.message : String(e);
		}
	}

	function toggleDetail(id: string) {
		if (detailId === id) {
			detailId = null;
			detail = null;
			return;
		}
		detailId = id;
		detail = null;
		void loadDetail(id);
	}

	async function act(id: string, label: string, fn: () => Promise<unknown>) {
		busy = { ...busy, [id]: true };
		try {
			await fn();
			msg = { ...msg, [id]: { ok: true, text: `${label} ok` } };
			await refresh();
		} catch (e) {
			msg = { ...msg, [id]: { ok: false, text: `${label}: ${e instanceof Error ? e.message : String(e)}` } };
		} finally {
			busy = { ...busy, [id]: false };
		}
	}

	function openBackfill(s: Source) {
		bfSource = s;
		bfInstruments = (s.config?.symbols ?? []).join(', ');
		bfTimescale = s.config?.timeframe ?? '1m';
		bfRange = '1d';
		bfStart = '';
		bfEnd = '';
		bfMode = 'range';
		bfResult = '';
		bfError = '';
	}

	async function submitBackfill(ev: SubmitEvent) {
		ev.preventDefault();
		if (!bfSource) return;
		bfBusy = true;
		bfError = '';
		bfResult = '';
		try {
			const instruments = bfInstruments
				.split(/[\s,]+/)
				.map((x) => x.trim())
				.filter(Boolean);
			const body: Parameters<typeof api.backfill>[1] = { timescale: bfTimescale };
			if (instruments.length) body.instruments = instruments;
			if (bfMode === 'range') body.range = bfRange.trim();
			else {
				if (bfStart) body.start = new Date(bfStart).toISOString();
				if (bfEnd) body.end = new Date(bfEnd).toISOString();
			}
			const r = await api.backfill(bfSource.id, body);
			bfResult = `job ${short(r.job_id, 12)} (${r.kind}) enqueued — follow it in Jobs & GPU`;
		} catch (e) {
			bfError = e instanceof Error ? e.message : String(e);
		} finally {
			bfBusy = false;
		}
	}

	const healthClass = (s: Source) => (s.health ? (s.health.status === 'ok' ? 'ok' : s.health.status === 'degraded' ? 'warn' : 'danger') : '');
	const satClass = (x: number) => (x > 0.8 ? 'danger' : x > 0.5 ? 'warn' : 'ok');

	onMount(() => {
		const stopPoll = poll(refresh, () => runtime.pollMs * 2);
		const stopBus = onBus(['ingest.health.>', 'risk.halt.>', 'risk.resume.>'], refresh, 500);
		const tick = setInterval(() => (now = Date.now()), 1000);
		return () => {
			stopPoll();
			stopBus();
			clearInterval(tick);
		};
	});
</script>

<section class="panel sources">
	<div class="panel-head">
		<h2>
			Sources <span class="dim">({filtered.length}/{sources.length})</span>
			<span class="badge ok">{healthCounts.ok} ok</span>
			{#if healthCounts.degraded}<span class="badge warn">{healthCounts.degraded} degraded</span>{/if}
			{#if healthCounts.down}<span class="badge danger">{healthCounts.down} down</span>{/if}
		</h2>
		<div class="filters">
			<input placeholder="search" bind:value={query} />
			<select bind:value={category} aria-label="category">
				<option value="">all categories</option>
				{#each categories as c}<option value={c}>{c}</option>{/each}
			</select>
			<select bind:value={status} aria-label="status">
				<option value="">all statuses</option>
				{#each statuses as s}<option value={s}>{s}</option>{/each}
			</select>
			<select bind:value={capability} aria-label="capability">
				<option value="">data + execution</option>
				<option value="data">data</option>
				<option value="execution">execution</option>
			</select>
		</div>
	</div>
	<p class="dim small">health from <code>ingest.health.&lt;source&gt;</code> (5 s per running connector); status from Postgres <code>catalog.source</code> (YAML seeds only). Click a row for cursors and runs.</p>
	{#if error}<p class="badge danger">{error}</p>{/if}
	<div class="scroll catalog">
		<table>
			<thead>
				<tr><th>id</th><th>name</th><th>category</th><th>status</th><th>health</th><th>last event</th><th>limiter</th><th>errors</th><th>capabilities</th><th>control</th><th>notes</th></tr>
			</thead>
			<tbody>
				{#each filtered as s (s.id)}
					{@const h = s.health}
					<tr class:sel={detailId === s.id} onclick={() => toggleDetail(s.id)}>
						<td class="mono">{s.id}</td>
						<td>{s.name}</td>
						<td class="dim">{s.category}</td>
						<td>
							<span class="badge" class:ok={s.status === 'live'} class:warn={s.status === 'stub'} class:danger={s.status === 'disabled'} title="yaml: {s.yaml_status ?? s.status}">{s.status}</span>
							{#if s.running}<span class="badge ok" title="connector running">running</span>{/if}
						</td>
						<td>
							{#if h}
								<span class="badge {healthClass(s)}" title={h.reason || h.last_error || `${h.mode} · period ${h.expected_period_s}s`}>{h.status}</span>
								{#if h.halted}<span class="badge danger" title="risk.halt.source:{s.id}">halted</span>{/if}
							{:else}
								<span class="dim" title="no ingest.health frame">—</span>
							{/if}
						</td>
						<td class="mono dim">{h?.last_event_at ? `${ago(h.last_event_at, now)} ago` : '—'}</td>
						<td>
							{#if h}
								<div class="bar" title="limiter saturation {(100 * h.limiter_saturation).toFixed(0)}% · {h.limiter_requests} req{s.rate_limit ? ` · ${s.rate_limit.per_second}/s burst ${s.rate_limit.burst}` : ''}">
									<div class={satClass(h.limiter_saturation)} style="width: {100 * Math.min(1, h.limiter_saturation)}%"></div>
								</div>
								<span class="mono dim">{(100 * h.limiter_saturation).toFixed(0)}%</span>
							{:else if s.rate_limit}
								<span class="mono dim">{s.rate_limit.per_second}/s</span>
							{:else}
								<span class="dim">—</span>
							{/if}
						</td>
						<td class="mono" class:neg={!!h?.errors_window}>{h ? `${h.errors_window}/${h.errors_total}` : '—'}</td>
						<td class="dim">{s.capabilities.join(', ')}</td>
						<td class="control" onclick={(e) => e.stopPropagation()}>
							{#if s.capabilities.includes('data') && s.status !== 'planned'}
								{#if s.status === 'disabled'}
									<button disabled={busy[s.id]} onclick={() => act(s.id, 'enable', () => api.sourceEnable(s.id, true))}>enable</button>
								{:else}
									<button disabled={busy[s.id]} onclick={() => confirm(`Disable ${s.id}? Persists catalog.source.status = disabled and stops the connector.`) && act(s.id, 'disable', () => api.sourceEnable(s.id, false))}>disable</button>
									{#if s.running}
										<button disabled={busy[s.id]} onclick={() => act(s.id, 'stop', () => api.ingest(s.id, 'stop'))}>stop</button>
									{:else}
										<button disabled={busy[s.id]} onclick={() => act(s.id, 'start', () => api.ingest(s.id, 'start'))}>start</button>
									{/if}
									<button disabled={busy[s.id]} onclick={() => openBackfill(s)}>backfill…</button>
								{/if}
								{#if msg[s.id]}
									<span class="badge" class:ok={msg[s.id].ok} class:danger={!msg[s.id].ok} title={msg[s.id].text}>{msg[s.id].text.slice(0, 70)}</span>
								{/if}
							{:else}
								<span class="dim">—</span>
							{/if}
						</td>
						<td class="dim notes">{s.notes}</td>
					</tr>
					{#if detailId === s.id}
						<tr class="detail">
							<td colspan="11">
								{#if detailError}<span class="badge danger">{detailError}</span>{/if}
								{#if detail}
									<div class="grid dgrid">
										<div>
											<h3 class="sub">Cursors <span class="dim">(ingest.backfill_cursor; live = watermark)</span></h3>
											{#if detail.cursors?.length}
												<table>
													<thead><tr><th>instrument</th><th>timescale</th><th>range</th><th>cursor</th><th>status</th><th>updated</th></tr></thead>
													<tbody>
														{#each detail.cursors as c}
															<tr>
																<td class="mono" title={c.instrument_id}>{short(c.instrument_id, 12)}</td>
																<td class="mono">{c.timescale}</td>
																<td class="mono dim">{c.live ? 'live' : `${c.range_start.slice(0, 16)} → ${c.range_end.slice(0, 16)}`}</td>
																<td class="mono">{c.cursor.slice(0, 16)}</td>
																<td><span class="badge" class:ok={c.status === 'done' || c.status === 'running'} class:warn={c.status !== 'done' && c.status !== 'running'}>{c.status}</span></td>
																<td class="mono dim">{ago(c.updated_at, now)} ago</td>
															</tr>
														{/each}
													</tbody>
												</table>
											{:else}
												<p class="dim">no cursors (Postgres rows written by the ingestor with -postgres)</p>
											{/if}
										</div>
										<div>
											<h3 class="sub">Connector runs <span class="dim">· quarantine open: {detail.quarantine_open ?? 0} · gaps {detail.gaps?.length ?? 0} · unresolved {detail.unresolved?.length ?? 0}</span></h3>
											{#if detail.runs?.length}
												<table>
													<thead><tr><th>run</th><th>started</th><th>ended</th><th>status</th><th>events</th><th>errors</th></tr></thead>
													<tbody>
														{#each detail.runs as r (r.id)}
															<tr title={r.last_error ?? ''}>
																<td class="mono">{short(r.id, 8)}</td>
																<td class="mono dim">{ago(r.started_at, now)} ago</td>
																<td class="mono dim">{r.ended_at ? `${ago(r.ended_at, now)} ago` : '—'}</td>
																<td><span class="badge" class:ok={r.status === 'running' || r.status === 'done'} class:warn={r.status === 'stopped'} class:danger={r.status === 'failed'}>{r.status}</span></td>
																<td class="mono">{r.events}</td>
																<td class="mono" class:neg={r.errors > 0}>{r.errors}</td>
															</tr>
														{/each}
													</tbody>
												</table>
											{:else}
												<p class="dim">no runs recorded</p>
											{/if}
											{#if detail.recent_errors?.length}
												<h3 class="sub">Recent errors</h3>
												<pre class="mono errs">{JSON.stringify(detail.recent_errors.slice(0, 5), null, 1)}</pre>
											{/if}
										</div>
									</div>
								{:else if !detailError}
									<span class="dim">loading GET /api/sources/{s.id}…</span>
								{/if}
							</td>
						</tr>
					{/if}
				{/each}
			</tbody>
		</table>
	</div>
</section>

<Modal open={bfSource !== null} title="Backfill {bfSource?.id ?? ''}" onclose={() => (bfSource = null)}>
	<form class="bf" onsubmit={submitBackfill}>
		<p class="dim small">POST /api/sources/{bfSource?.id}/backfill → <code>ctl.ingest.backfill</code> enqueues a <code>jobs.ingest.backfill</code> job (JOBS_INGEST work queue, D-054). Empty instruments = catalog <code>config.symbols</code>.</p>
		<label>instruments (source symbols, comma or space separated) <textarea rows="2" bind:value={bfInstruments} placeholder={(bfSource?.config?.symbols ?? []).join(', ') || 'AAPL, MSFT'}></textarea></label>
		<div class="row">
			<label>timescale
				<select bind:value={bfTimescale}>
					{#each ['1s', '1m', '5m', '15m', '1h', '1d', '1w'] as t}<option value={t}>{t}</option>{/each}
				</select>
			</label>
			<label>window
				<select bind:value={bfMode}>
					<option value="range">range back from now</option>
					<option value="absolute">start / end</option>
				</select>
			</label>
			{#if bfMode === 'range'}
				<label>range <input bind:value={bfRange} placeholder="1d" pattern="\d+[mhd]" title="<n>m|h|d" /></label>
			{:else}
				<label>start <input type="datetime-local" bind:value={bfStart} /></label>
				<label>end <input type="datetime-local" bind:value={bfEnd} /></label>
			{/if}
		</div>
		<div class="row">
			<button type="submit" disabled={bfBusy}>{bfBusy ? 'enqueuing…' : 'Enqueue backfill'}</button>
			{#if bfResult}<span class="badge ok">{bfResult}</span>{/if}
			{#if bfError}<span class="badge danger">{bfError}</span>{/if}
		</div>
	</form>
</Modal>

<style>
	.sources {
		margin-top: 12px;
	}
	.filters {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}
	.small {
		font-size: 11px;
		margin: 0 0 8px;
	}
	.catalog {
		max-height: 560px;
	}
	.notes {
		max-width: 300px;
	}
	.control {
		white-space: nowrap;
	}
	.control button {
		font-size: 11px;
		padding: 1px 6px;
	}
	tbody tr:not(.detail) {
		cursor: pointer;
	}
	tr.sel td {
		background: var(--bg-inset);
	}
	.detail td {
		background: var(--bg-inset);
	}
	.detail th {
		background: var(--bg-inset);
	}
	.dgrid {
		grid-template-columns: 1fr 1fr;
	}
	@media (max-width: 1100px) {
		.dgrid {
			grid-template-columns: 1fr;
		}
	}
	.sub {
		font-size: 12px;
		margin: 4px 0 6px;
	}
	.bar {
		display: inline-block;
		width: 48px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
		margin-right: 4px;
	}
	.bar > div {
		height: 100%;
		border-radius: 3px;
	}
	.bar .ok {
		background: var(--ok);
	}
	.bar .warn {
		background: var(--warn);
	}
	.bar .danger {
		background: var(--danger);
	}
	.neg {
		color: var(--danger);
	}
	.errs {
		max-height: 120px;
		overflow: auto;
		font-size: 11px;
	}
	.bf {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.bf label {
		display: flex;
		flex-direction: column;
		gap: 2px;
		color: var(--text-dim);
	}
	.bf textarea {
		font: 12px var(--mono);
		color: var(--text);
		background: var(--bg-inset);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 4px 8px;
	}
	.row {
		display: flex;
		gap: 10px;
		align-items: flex-end;
		flex-wrap: wrap;
	}
</style>
