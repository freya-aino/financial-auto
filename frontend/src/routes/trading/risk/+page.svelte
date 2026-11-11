<script lang="ts">
	// Risk & halts (D-071, spec 40 §6, spec 51 §6.5): halts persisted in ledger.risk_state (they
	// survive restarts) merged with the ones on the bus, re-arm per scope behind the same
	// confirm flow as the kill switch (result from the API, never optimistic; audited), and the
	// RiskEvent feed: audit.log history plus the live risk.event / risk.halt / risk.resume tail.
	import { onMount } from 'svelte';
	import Empty from '../../../lib/Empty.svelte';
	import KillSwitch from '../../../lib/KillSwitch.svelte';
	import ViewHeader from '../../../lib/ViewHeader.svelte';
	import { live, poll, type RiskView } from '../../../lib/api';
	import { since } from '../../../lib/alerts';
	import { consoleApi, errText, type HaltsView, type RiskEventRow, type RiskEventsView } from '../../../lib/console';
	import { loadPref, savePref } from '../../../lib/prefs';
	import { busRiskEvent, eventKinds, haltClass, isRearmable, rearmConfirmText, rearmConsequence } from '../../../lib/riskdesk';
	import { runtime } from '../../../lib/runtime.svelte';
	import { bus } from '../../../lib/ws.svelte';

	let halts = $state<HaltsView | null>(null);
	let haltsError = $state('');
	let risk = $state<RiskView | null>(null);
	let events = $state<RiskEventsView | null>(null);
	let eventsError = $state('');
	let liveEvents = $state<RiskEventRow[]>([]);
	let windowSpec = $state(loadPref('risk.events.since', '24h'));
	let kindFilter = $state(loadPref('risk.events.kind', ''));
	let busy = $state<string | null>(null);
	let msg = $state('');
	let ok = $state(true);
	let open = $state<number | null>(null);
	let now = $state(Date.now());
	$effect(() => savePref('risk.events.since', windowSpec));
	$effect(() => savePref('risk.events.kind', kindFilter));

	const kinds = $derived(eventKinds(events?.events ?? []));
	const shownEvents = $derived([...liveEvents, ...(events?.events ?? [])].filter((e) => !kindFilter || e.kind.startsWith(kindFilter)).slice(0, 400));

	async function refreshHalts() {
		now = Date.now();
		const [h, r] = await Promise.allSettled([consoleApi.halts(), live.risk()]);
		if (h.status === 'fulfilled') {
			halts = h.value;
			haltsError = '';
		} else haltsError = errText(h.reason);
		if (r.status === 'fulfilled') risk = r.value;
	}

	async function refreshEvents() {
		try {
			events = await consoleApi.riskEvents(windowSpec, 500);
			eventsError = '';
			liveEvents = liveEvents.filter((e) => !events || e.at > (events.events[0]?.at ?? ''));
		} catch (e) {
			eventsError = errText(e);
		}
	}

	async function rearm(scope: string, reason: string) {
		if (!isRearmable(scope)) return;
		if (!confirm(rearmConfirmText(scope, reason))) return;
		const why = prompt(`Re-arm reason for ${scope} (recorded in the audit log):`, 'operator re-arm after review');
		if (why === null) return;
		busy = scope;
		try {
			const r = await consoleApi.rearm(scope, why);
			msg = `${r.resumed} re-armed${r.audited ? ' (audited)' : ' (audit not written: no Postgres)'}`;
			ok = true;
		} catch (e) {
			msg = `${scope}: ${errText(e)}`;
			ok = false;
		} finally {
			busy = null;
			await refreshHalts();
		}
	}

	onMount(() => {
		const stopH = poll(refreshHalts, () => runtime.pollMs * 2);
		const stopE = poll(refreshEvents, () => runtime.pollMs * 10);
		let seq = 0;
		const off = bus.subscribe(['risk.event.>', 'risk.halt.>', 'risk.resume.>'], (f) => {
			const row = busRiskEvent(f.data.subject, f.data.decoded as Record<string, unknown> | null, ++seq);
			if (row) liveEvents = [row, ...liveEvents].slice(0, 200);
			if (!f.data.subject.startsWith('risk.event.')) void refreshHalts();
		});
		return () => {
			stopH();
			stopE();
			off();
		};
	});
</script>

<ViewHeader title="Risk & halts" spec="40 §6 · 51 §6.5" note="halts persist in ledger.risk_state; re-arm publishes fa.v1.Resume on risk.resume.<scope> (confirm + audit)" />

<div class="killbar panel">
	<KillSwitch {risk} onchanged={refreshHalts} />
	<span class="dim">The kill switch halts the portfolio. Every other scope is raised by risk-engine itself (daily loss, feeds, model health, broker reconciliation, stale FX) and is re-armed here.</span>
	{#if msg}<span class="badge" class:ok class:danger={!ok}>{msg}</span>{/if}
</div>

<div class="grid cols">
	<section class="panel">
		<div class="panel-head">
			<h2>Halts <span class="dim">({halts?.halts.length ?? 0})</span></h2>
			{#if halts?.persisted}<span class="dim mono" title="ledger.risk_state ({halts.persisted.book})">persisted {since(halts.persisted.updated_at, now)} ago</span>{/if}
		</div>
		{#if haltsError}<p class="badge danger">{haltsError}</p>{/if}
		{#if halts && !halts.postgres}<p class="badge warn">Postgres unavailable: only halts seen on the bus since control-api start are listed.</p>{/if}
		{#if halts?.halts.length}
			<table>
				<thead><tr><th>scope</th><th>reason</th><th>since</th><th>source</th><th></th></tr></thead>
				<tbody>
					{#each halts.halts as h (h.scope)}
						<tr>
							<td><span class="badge {haltClass(h.scope)}">{h.scope}</span>{#if h.manual}<span class="dim"> manual</span>{/if}</td>
							<td>{h.reason}</td>
							<td class="mono dim" title={h.since ?? ''}>{since(h.since, now)}</td>
							<td class="dim">
								{#if h.persisted}<span class="badge ok" title="ledger.risk_state: survives a risk-engine restart">persisted</span>{/if}
								{#if h.live}<span class="badge" title="seen on risk.halt.* by control-api">bus</span>{/if}
							</td>
							<td>
								{#if h.rearmable}
									<button onclick={() => rearm(h.scope, h.reason)} disabled={busy !== null} title={rearmConsequence(h.scope)}>{busy === h.scope ? '…' : 're-arm'}</button>
								{:else}
									<span class="dim" title="not a scope risk-engine re-arms">—</span>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else if halts}
			<p class="dim">No active halts.</p>
		{:else}
			<Empty producer="GET /api/trading/halts" hint="" />
		{/if}
		<p class="dim small">Re-armable families: {(halts?.families ?? []).join(' · ')}. Strategy-level pauses are managed in the strategies table.</p>
	</section>

	<section class="panel wide">
		<div class="panel-head">
			<h2>Risk events <span class="dim">({shownEvents.length})</span></h2>
			<div class="ctl">
				<select bind:value={windowSpec} onchange={refreshEvents} aria-label="window">
					{#each ['1h', '6h', '24h', '7d'] as w}<option value={w}>{w}</option>{/each}
				</select>
				<select bind:value={kindFilter} aria-label="kind">
					<option value="">all kinds</option>
					{#each kinds as k}<option value={k.kind}>{k.kind} ({k.count})</option>{/each}
				</select>
				<span class="badge" class:ok={bus.status === 'open'} class:warn={bus.status !== 'open'} title="live tail of risk.event.>, risk.halt.>, risk.resume.>">live {bus.status}</span>
			</div>
		</div>
		{#if eventsError}<p class="badge danger">{eventsError}</p>{/if}
		{#if events}
			<div class="counts">
				{#each Object.entries(events.counts) as [a, n]}<span class="badge">{a.replace('risk.event.', '')} · {n}</span>{/each}
			</div>
		{/if}
		{#if shownEvents.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>at</th><th>kind</th><th>scope / instrument</th><th>actor</th><th>detail</th></tr></thead>
					<tbody>
						{#each shownEvents as e (e.id)}
							<tr onclick={() => (open = open === e.id ? null : e.id)} class:livep={e.id < 0}>
								<td class="mono dim" title={e.at}>{since(e.at, now)}</td>
								<td><span class="badge" class:danger={e.kind === 'halt'} class:ok={e.kind === 'resume' || e.kind === 'halt.rearm'} class:warn={e.kind.startsWith('feed') || e.kind === 'reconciliation.diff'}>{e.kind}</span></td>
								<td class="mono">{e.scope ?? '—'}</td>
								<td class="mono dim">{e.actor}</td>
								<td class="mono dim detail">{open === e.id ? JSON.stringify(e.detail, null, 1) : JSON.stringify(e.detail)?.slice(0, 120)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if events}
			<p class="dim">No risk events in {windowSpec}.</p>
		{/if}
	</section>
</div>

<style>
	.killbar {
		display: flex;
		align-items: center;
		gap: 14px;
		flex-wrap: wrap;
		margin-bottom: 12px;
	}
	.cols {
		grid-template-columns: minmax(420px, 1fr) minmax(520px, 1.5fr);
	}
	@media (max-width: 1000px) {
		.cols {
			grid-template-columns: 1fr;
		}
	}
	.tall {
		max-height: 560px;
	}
	.ctl {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.counts {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
		margin-bottom: 8px;
	}
	.detail {
		white-space: pre-wrap;
		word-break: break-all;
		max-width: 420px;
	}
	.livep td:first-child {
		box-shadow: inset 2px 0 0 var(--accent);
	}
	.small {
		font-size: 11px;
		margin-top: 8px;
	}
	button {
		font-size: 11px;
		padding: 1px 8px;
		white-space: nowrap;
	}
</style>
