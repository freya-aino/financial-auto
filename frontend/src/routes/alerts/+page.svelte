<script lang="ts">
	// Alerts (D-066, spec 60 §6.5): vmalert → control-api → `ops.alert`. Active alerts, recent
	// transitions, the alert pipeline's own health (Watchdog) and the live `ops.alert` tail.
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { poll } from '../../lib/api';
	import { applyAlertEvent, labelText, severityClass, since, sortAlerts } from '../../lib/alerts';
	import { consoleApi, errText, type AlertEvent, type AlertsView } from '../../lib/console';
	import { operatorApi } from '../../lib/operator';
	import { bus } from '../../lib/ws.svelte';

	let view = $state<AlertsView | null>(null);
	let error = $state('');
	let now = $state(Date.now());
	let tail = $state<(AlertEvent & { seq: number })[]>([]);
	let open = $state<string | null>(null);

	const active = $derived(sortAlerts(view?.active ?? []));

	// stored history (ops.alert_event): survives control-api restarts, unlike `recent`
	let hSince = $state('7d');
	let hName = $state('');
	let history = $state<AlertEvent[]>([]);
	let historyError = $state('');
	const SINCE: Record<string, number> = { '1h': 3_600_000, '24h': 86_400_000, '7d': 7 * 86_400_000, '30d': 30 * 86_400_000 };
	async function loadHistory() {
		try {
			const since = new Date(Date.now() - SINCE[hSince]).toISOString();
			history = (await operatorApi.alertHistory(since, hName.trim() || undefined, 1000)).transitions;
			historyError = '';
		} catch (e) {
			historyError = errText(e);
		}
	}
	$effect(() => {
		void hSince;
		void loadHistory();
	});
	const firingCount = $derived(history.filter((h) => h.status === 'firing').length);

	async function refresh() {
		now = Date.now();
		try {
			view = await consoleApi.alerts(100);
			error = '';
		} catch (e) {
			error = errText(e);
		}
	}

	onMount(() => {
		const stop = poll(refresh, 10_000);
		const tick = setInterval(() => (now = Date.now()), 1000);
		const off = bus.subscribe('ops.alert', (f) => {
			const ev = f.data.decoded as AlertEvent | null;
			if (!ev || typeof ev !== 'object' || !ev.alertname) return;
			tail = [{ ...ev, seq: f.seq }, ...tail].slice(0, 200);
			if (view) view = { ...view, active: applyAlertEvent(view.active, ev), recent: [ev, ...view.recent].slice(0, 100) };
		});
		return () => {
			stop();
			off();
			clearInterval(tick);
		};
	});
</script>

<ViewHeader title="Alerts" spec="60 §6.5 · 51 §6.1" note="vmalert rules → POST /api/alerts/vmalert → ops.alert + audit; GET /api/alerts" />

{#if error}<p class="badge danger">{error}</p>{/if}

<div class="grid cols">
	<section class="panel">
		<div class="panel-head">
			<h2>Alert pipeline</h2>
			{#if view}
				<span class="badge" class:ok={view.pipeline.ok} class:danger={!view.pipeline.ok}>{view.pipeline.ok ? 'Watchdog fresh' : 'PIPELINE DOWN'}</span>
			{/if}
		</div>
		{#if view}
			<table>
				<tbody>
					<tr><th>Watchdog last seen</th><td class="mono">{view.pipeline.watchdog_seen_at ?? 'never'} <span class="dim">({since(view.pipeline.watchdog_seen_at, now)} ago)</span></td></tr>
					<tr><th>notifications received</th><td class="mono">{view.pipeline.posts}</td></tr>
					<tr><th>last post</th><td class="mono">{view.pipeline.last_post_at ?? 'never'} <span class="dim">({since(view.pipeline.last_post_at, now)} ago)</span></td></tr>
					<tr><th>bus subject</th><td class="mono">{view.pipeline.subject}</td></tr>
				</tbody>
			</table>
			{#if !view.pipeline.ok}
				<p class="warnline">No Watchdog from vmalert: every other alert may be missing. Check <code>vmalert</code> (127.0.0.1:8880) and <code>scripts/alert-drill.sh</code>.</p>
			{/if}
		{:else}
			<Empty producer="GET /api/alerts (control-api)" hint="" />
		{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head">
			<h2>Active <span class="dim">({active.length})</span></h2>
			<span class="dim">critical first · click a row for labels</span>
		</div>
		{#if active.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>severity</th><th>alert</th><th>summary</th><th>firing for</th><th>refreshes</th><th></th></tr></thead>
					<tbody>
						{#each active as a (a.fingerprint)}
							<tr onclick={() => (open = open === a.fingerprint ? null : a.fingerprint)}>
								<td><span class="badge {severityClass(a.severity)}">{a.severity}</span></td>
								<td class="mono">{a.alertname}</td>
								<td>{a.summary}</td>
								<td class="mono">{since(a.starts_at, now)}</td>
								<td class="mono dim">{a.refreshes}</td>
								<td>{#if a.generator_url}<a href={a.generator_url} target="_blank" rel="noreferrer">rule</a>{/if}</td>
							</tr>
							{#if open === a.fingerprint}
								<tr class="detail"><td colspan="6" class="mono dim">{labelText(a.labels)}{#each Object.entries(a.annotations) as [k, v]}<br />{k}: {v}{/each}</td></tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
		{:else if view}
			<p class="dim">No active alerts.</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Recent transitions <span class="dim">({view?.recent.length ?? 0})</span></h2></div>
		{#if view?.recent.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>at</th><th>status</th><th>alert</th><th>severity</th></tr></thead>
					<tbody>
						{#each view.recent as r, i (r.fingerprint + r.at + i)}
							<tr>
								<td class="mono dim">{since(r.at, now)} ago</td>
								<td><span class="badge" class:danger={r.status === 'firing'} class:ok={r.status === 'resolved'}>{r.status}{r.expired ? ' (expired)' : ''}</span></td>
								<td class="mono">{r.alertname}</td>
								<td><span class="badge {severityClass(r.severity)}">{r.severity}</span></td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="dim">No transitions since control-api started.</p>
		{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head">
			<h2>History <span class="dim">({history.length} transitions, {firingCount} firing · ops.alert_event)</span></h2>
			<form
				class="controls"
				onsubmit={(e) => {
					e.preventDefault();
					void loadHistory();
				}}
			>
				<select bind:value={hSince} aria-label="since">{#each Object.keys(SINCE) as k}<option value={k}>last {k}</option>{/each}</select>
				<input class="mono" bind:value={hName} placeholder="alertname (all)" size="20" aria-label="alertname" />
				<button type="submit">filter</button>
			</form>
		</div>
		{#if historyError}<p class="badge danger">{historyError}</p>{/if}
		{#if history.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>at</th><th>status</th><th>alert</th><th>severity</th><th>summary</th><th>started</th></tr></thead>
					<tbody>
						{#each history as r, i (r.fingerprint + r.at + i)}
							<tr>
								<td class="mono dim" title={r.at}>{r.at.slice(0, 19).replace('T', ' ')}</td>
								<td><span class="badge" class:danger={r.status === 'firing'} class:ok={r.status === 'resolved'}>{r.status}{r.expired ? ' (expired)' : ''}</span></td>
								<td class="mono"><button class="link" onclick={() => ((hName = r.alertname), loadHistory())}>{r.alertname}</button></td>
								<td><span class="badge {severityClass(r.severity)}">{r.severity}</span></td>
								<td>{r.summary}</td>
								<td class="mono dim">{r.starts_at?.slice(0, 19).replace('T', ' ') ?? '—'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !historyError}
			<p class="dim">No stored transitions in the window.</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Live <code>ops.alert</code></h2>
			<span class="badge" class:ok={bus.status === 'open'} class:warn={bus.status !== 'open'}>/ws/bus {bus.status}</span>
		</div>
		{#if tail.length}
			<div class="scroll tall">
				{#each tail as t (t.seq)}
					<div class="mono line"><span class="badge" class:danger={t.status === 'firing'} class:ok={t.status === 'resolved'}>{t.status}</span> {t.alertname} <span class="dim">{t.summary}</span></div>
				{/each}
			</div>
		{:else}
			<p class="dim">Waiting for transitions on <code>ops.alert</code> (only changes are published).</p>
		{/if}
	</section>
</div>

<style>
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
	}
	.wide {
		grid-column: span 2;
	}
	@media (max-width: 1000px) {
		.wide {
			grid-column: auto;
		}
	}
	.tall {
		max-height: 460px;
	}
	.warnline {
		color: var(--danger);
	}
	tr {
		cursor: default;
	}
	.detail td {
		white-space: pre-wrap;
	}
	.controls {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.link {
		background: none;
		border: none;
		padding: 0;
		color: var(--accent);
		cursor: pointer;
		font: inherit;
	}
	.line {
		padding: 2px 0;
		border-bottom: 1px solid var(--border);
	}
</style>
