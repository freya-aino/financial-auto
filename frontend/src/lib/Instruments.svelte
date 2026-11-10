<script lang="ts">
	// Reference-data search (`GET /api/instruments?q=`) and the quarantine review loop
	// (`GET /api/instruments/quarantine`, `POST …/{provisional}/resolve`; D-054).
	import { onMount } from 'svelte';
	import Modal from './Modal.svelte';
	import { ago, api, poll, short, type Instrument, type QuarantineRow } from './api';
	import { loadPref, savePref } from './prefs';
	import { ASSET_CLASSES, emptyResolveForm, INSTRUMENT_KINDS, validateResolve, type ResolveForm } from './quarantine';
	import { runtime } from './runtime.svelte';
	import { debounce } from './ws.svelte';

	let query = $state('');
	let results = $state<Instrument[]>([]);
	let searchError = $state('');
	let searching = $state(false);

	let qStatus = $state(loadPref<'open' | 'resolved' | 'ignored' | 'all'>('ops.quarantine', 'open'));
	let rows = $state<QuarantineRow[]>([]);
	let qError = $state('');
	let now = $state(Date.now());
	$effect(() => savePref('ops.quarantine', qStatus));

	// resolve dialog
	let target = $state<QuarantineRow | null>(null);
	let form = $state<ResolveForm>(emptyResolveForm());
	let busy = $state(false);
	let result = $state('');
	let resolveError = $state('');
	let mergeHits = $state<Instrument[]>([]);
	const validation = $derived(validateResolve(form));

	const search = debounce(async () => {
		const q = query.trim();
		if (!q) {
			results = [];
			return;
		}
		searching = true;
		try {
			results = (await api.instruments(q, undefined, 25)).instruments;
			searchError = '';
		} catch (e) {
			searchError = e instanceof Error ? e.message : String(e);
		} finally {
			searching = false;
		}
	}, 250);

	const mergeSearch = debounce(async () => {
		const q = form.merge_into.trim();
		if (!q || /^[0-9a-f-]{36}$/i.test(q)) {
			mergeHits = [];
			return;
		}
		try {
			mergeHits = (await api.instruments(q, 'active', 8)).instruments;
		} catch {
			mergeHits = [];
		}
	}, 250);

	async function refreshQuarantine() {
		now = Date.now();
		try {
			rows = (await api.quarantine(qStatus, 200)).quarantine;
			qError = '';
		} catch (e) {
			qError = e instanceof Error ? e.message : String(e);
		}
	}

	function openResolve(r: QuarantineRow) {
		target = r;
		form = { ...emptyResolveForm(), name: r.instrument_name.replace(/\s*\(.*\)$/, '') };
		result = '';
		resolveError = '';
		mergeHits = [];
	}

	async function submit(ev: SubmitEvent) {
		ev.preventDefault();
		if (!target || !validation.body) return;
		const verb = form.mode === 'merge' ? `merge into ${form.merge_into}` : form.mode;
		if (!confirm(`${verb} provisional instrument ${target.venue_symbol} (${target.source_id})? This is audited.`)) return;
		busy = true;
		resolveError = '';
		try {
			await api.resolveQuarantine(target.instrument_id, validation.body);
			result = `resolved (${form.mode})`;
			await refreshQuarantine();
			target = null;
		} catch (e) {
			resolveError = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	onMount(() => poll(refreshQuarantine, () => runtime.pollMs * 10));
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Instruments</h2>
		<form class="search" onsubmit={(e) => (e.preventDefault(), search())}>
			<input bind:value={query} oninput={search} placeholder="search names, identifiers, venue symbols" size="36" />
			{#if searching}<span class="dim">…</span>{/if}
		</form>
	</div>
	{#if searchError}<p class="badge danger">{searchError}</p>{/if}
	{#if results.length}
		<div class="scroll res">
			<table>
				<thead><tr><th>name</th><th>status</th><th>class / kind</th><th>identifiers</th><th>listings</th><th>id</th></tr></thead>
				<tbody>
					{#each results as i (i.id)}
						<tr>
							<td>{i.name}</td>
							<td><span class="badge" class:ok={i.status === 'active'} class:warn={i.status === 'provisional'} class:danger={i.status === 'merged' || i.status === 'ignored'}>{i.status}</span></td>
							<td class="dim">{i.asset_class ?? '—'} / {i.kind ?? '—'}</td>
							<td class="mono dim">{i.identifiers.map((x) => `${x.scheme}:${x.value}`).join(' ')}</td>
							<td class="mono dim">{i.listings.map((l) => `${l.source_id}:${l.venue_symbol}`).join(' ')}</td>
							<td class="mono dim" title={i.id}>{short(i.id, 8)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else if query.trim() && !searching && !searchError}
		<p class="dim">no instrument matches "{query}" (GET /api/instruments?q=)</p>
	{/if}

	<div class="panel-head qhead">
		<h3 class="sub">Quarantine <span class="dim">({rows.length})</span></h3>
		<select bind:value={qStatus} onchange={refreshQuarantine}>
			<option value="open">open</option>
			<option value="resolved">resolved</option>
			<option value="ignored">ignored</option>
			<option value="all">all</option>
		</select>
	</div>
	<p class="dim small">Provisional instruments created by the ingestor for unknown (source, symbol) pairs; activate with identifiers, merge into an existing instrument, or ignore (spec 12 §6, D-050/D-054).</p>
	{#if qError}<p class="badge danger">{qError}</p>{/if}
	{#if rows.length}
		<div class="scroll res">
			<table>
				<thead><tr><th>source</th><th>symbol</th><th>provisional</th><th>seen</th><th>last</th><th>status</th><th></th></tr></thead>
				<tbody>
					{#each rows as r (r.instrument_id + r.source_id + r.venue_symbol)}
						<tr title={r.sample_raw_ref ?? ''}>
							<td class="mono">{r.source_id}</td>
							<td class="mono">{r.venue_symbol}</td>
							<td class="mono dim" title={r.instrument_id}>{r.instrument_name} · {short(r.instrument_id, 8)}</td>
							<td class="mono">{r.count}</td>
							<td class="mono dim">{ago(r.last_seen, now)} ago</td>
							<td>
								<span class="badge" class:warn={r.status === 'open'} class:ok={r.status === 'resolved'}>{r.status}</span>
								{#if r.resolution}<span class="dim">{r.resolution}</span>{/if}
							</td>
							<td>{#if r.status === 'open'}<button onclick={() => openResolve(r)}>resolve…</button>{/if}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else if !qError}
		<p class="dim">no {qStatus === 'all' ? '' : qStatus} quarantine rows (ingest.quarantine)</p>
	{/if}
	{#if result}<p class="badge ok">{result}</p>{/if}
</section>

<Modal open={target !== null} title="Resolve {target?.venue_symbol ?? ''} ({target?.source_id ?? ''})" onclose={() => (target = null)}>
	<form class="rf" onsubmit={submit}>
		<p class="dim small">provisional <code>{target?.instrument_id}</code> · {target?.instrument_name}</p>
		<div class="modes">
			{#each ['activate', 'merge', 'ignore'] as m}
				<label class="radio"><input type="radio" bind:group={form.mode} value={m} /> {m}</label>
			{/each}
		</div>
		{#if form.mode === 'activate'}
			<div class="two">
				<label>ISIN <input bind:value={form.isin} placeholder="US0378331005" class="mono" /></label>
				<label>FIGI <input bind:value={form.figi} placeholder="BBG000B9XRY4" class="mono" /></label>
				<label>asset class
					<input list="asset-classes" bind:value={form.asset_class} />
					<datalist id="asset-classes">{#each ASSET_CLASSES as a}<option value={a}></option>{/each}</datalist>
				</label>
				<label>kind
					<input list="instrument-kinds" bind:value={form.kind} />
					<datalist id="instrument-kinds">{#each INSTRUMENT_KINDS as k}<option value={k}></option>{/each}</datalist>
				</label>
				<label class="wide">name <input bind:value={form.name} /></label>
				<label>base ccy <input bind:value={form.base_ccy} placeholder="USD" maxlength="3" /></label>
				<label>quote ccy <input bind:value={form.quote_ccy} placeholder="EUR" maxlength="3" /></label>
			</div>
		{:else if form.mode === 'merge'}
			<label>merge into (instrument UUID, or type a name to search active instruments)
				<input bind:value={form.merge_into} oninput={mergeSearch} class="mono" placeholder="92ab6643-…" />
			</label>
			{#if mergeHits.length}
				<ul class="hits">
					{#each mergeHits as h (h.id)}
						<li><button type="button" onclick={() => ((form.merge_into = h.id), (mergeHits = []))}>{h.name} <span class="mono dim">{short(h.id, 8)} · {h.identifiers.map((x) => x.value).join(' ')}</span></button></li>
					{/each}
				</ul>
			{/if}
			<p class="dim small">Listings, identifiers and cursors are re-pointed to the target; the provisional row becomes <code>merged</code>.</p>
		{:else}
			<p class="dim small">Closes the quarantine row with resolution <code>ignored</code>; the provisional instrument stays provisional.</p>
		{/if}
		{#if validation.errors.length}
			<ul class="errs">{#each validation.errors as e}<li>{e}</li>{/each}</ul>
		{/if}
		<div class="row">
			<button type="submit" disabled={busy || !validation.body}>{busy ? 'sending…' : `Resolve: ${form.mode}`}</button>
			{#if resolveError}<span class="badge danger">{resolveError}</span>{/if}
		</div>
	</form>
</Modal>

<style>
	.search {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.res {
		max-height: 260px;
	}
	.qhead {
		margin-top: 12px;
	}
	.sub {
		font-size: 12px;
	}
	.small {
		font-size: 11px;
		margin: 0 0 8px;
	}
	button {
		font-size: 11px;
		padding: 1px 6px;
	}
	.rf {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.rf label {
		display: flex;
		flex-direction: column;
		gap: 2px;
		color: var(--text-dim);
	}
	.modes {
		display: flex;
		gap: 14px;
	}
	.radio {
		flex-direction: row !important;
		align-items: center;
		gap: 4px !important;
	}
	.two {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 8px;
	}
	.wide {
		grid-column: 1 / -1;
	}
	.hits {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.hits button {
		width: 100%;
		text-align: left;
	}
	.errs {
		margin: 0;
		padding-left: 18px;
		color: var(--warn);
	}
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.rf .row button {
		font-size: 13px;
		padding: 4px 12px;
	}
</style>
