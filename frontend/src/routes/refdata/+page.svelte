<script lang="ts">
	// Reference data (D-067, spec 12 §4): point-in-time resolve (`at` = real-world instant,
	// `known_at` = what we knew then), instrument as-of, FX as-of, venue calendars (table rows or
	// built-in rule), corporate actions with entry / cancel and symbol changes. Every mutation is
	// confirmed, audited by control-api (actor ui) and announced on ref.changed.<entity>.
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { api, num, short, type Instrument } from '../../lib/api';
	import { consoleApi, errText, type CorporateAction, type SessionRow, type Venue } from '../../lib/console';

	// as-of resolve
	let rVenue = $state('');
	let rSource = $state('yahoo');
	let rSymbol = $state('AAPL');
	let rAt = $state('');
	let rKnown = $state('');
	let resolved = $state<Record<string, unknown> | null>(null);
	let resolveErr = $state('');

	// instrument as-of
	let search = $state('');
	let hits = $state<Instrument[]>([]);
	let instId = $state('');
	let asOf = $state<Record<string, unknown> | null>(null);
	let asOfErr = $state('');

	// fx
	let ccy = $state('USD');
	let fx = $state<{ rate_to_eur: number; rate_time: string; source: string } | null>(null);
	let fxErr = $state('');

	// calendars
	let venues = $state<Venue[]>([]);
	let venuesErr = $state('');
	let venue = $state('xetr');
	let calFrom = $state('');
	let calTo = $state('');
	let sessions = $state<SessionRow[]>([]);
	let calRule = $state<string | null>(null);
	let calTz = $state('');
	let calErr = $state('');
	let calMsg = $state('');

	// corporate actions
	let actions = $state<CorporateAction[]>([]);
	let actionsErr = $state('');
	let pit = $state(false);
	let ca = $state({ kind: 'split', ex_date: '', ratio: '', amount: '', ccy: '', source_id: 'manual', status: 'confirmed' });
	let caMsg = $state('');
	let caOk = $state(true);
	let sc = $state({ venue: '', old_symbol: '', new_symbol: '', effective: '' });
	let scMsg = $state('');

	const KINDS = ['split', 'reverse_split', 'stock_dividend', 'cash_dividend', 'spinoff', 'merger', 'symbol_change'];
	const iso = (v: string) => (v ? (v.length === 10 ? v : new Date(v).toISOString()) : undefined);
	const segText = (s: SessionRow) => (s.segments ?? []).map((g) => `${g.kind} ${g.start.slice(11, 16)}–${g.end.slice(11, 16)}`).join(' · ');

	async function resolve() {
		try {
			resolved = await consoleApi.resolve({ venue: rVenue || undefined, source: rVenue ? undefined : rSource || undefined, symbol: rSymbol, at: iso(rAt), known_at: iso(rKnown) });
			resolveErr = '';
			const r = resolved.resolved as { result: string; instrument_id?: string };
			if (r.result === 'one' && r.instrument_id && !instId) void loadInstrument(r.instrument_id);
		} catch (e) {
			resolveErr = errText(e);
			resolved = null;
		}
	}

	async function find() {
		try {
			hits = (await api.instruments(search, undefined, 15)).instruments;
		} catch (e) {
			asOfErr = errText(e);
		}
	}

	async function loadInstrument(id = instId) {
		if (!id) return;
		instId = id;
		try {
			asOf = await consoleApi.instrumentAsOf(id, iso(rAt), iso(rKnown));
			asOfErr = '';
			await loadActions();
		} catch (e) {
			asOfErr = errText(e);
			asOf = null;
		}
	}

	async function loadFx() {
		try {
			fx = await consoleApi.fx(ccy, iso(rAt), iso(rKnown));
			fxErr = '';
		} catch (e) {
			fxErr = errText(e);
			fx = null;
		}
	}

	async function loadVenues() {
		try {
			venues = (await consoleApi.venues()).venues;
			venuesErr = '';
		} catch (e) {
			venuesErr = errText(e);
		}
	}

	async function loadCalendar() {
		try {
			const c = await consoleApi.calendar(venue, calFrom || undefined, calTo || undefined);
			sessions = c.sessions;
			calRule = c.rule;
			calTz = c.timezone;
			calErr = '';
		} catch (e) {
			calErr = errText(e);
			sessions = [];
		}
	}

	async function materialize() {
		const today = new Date().toISOString().slice(0, 10);
		const f = calFrom || today;
		const t = calTo || new Date(Date.now() + 90 * 86_400_000).toISOString().slice(0, 10);
		if (!confirm(`Materialise ${venue} sessions ${f} → ${t} from the built-in ${calRule ?? ''} rule into ref.calendar_session? Audited (ref.calendar.materialized).`)) return;
		try {
			const r = await consoleApi.materialize(venue, f, t);
			calMsg = `${r.rows} rows written`;
			await Promise.all([loadCalendar(), loadVenues()]);
		} catch (e) {
			calMsg = errText(e);
		}
	}

	async function loadActions() {
		try {
			const r = await consoleApi.corporateActions(instId || undefined, iso(rKnown));
			actions = r.actions;
			pit = r.point_in_time;
			actionsErr = '';
		} catch (e) {
			actionsErr = errText(e);
		}
	}

	async function addAction() {
		if (!instId) {
			caMsg = 'pick an instrument first';
			caOk = false;
			return;
		}
		const body: Record<string, unknown> = { instrument_id: instId, kind: ca.kind, ex_date: ca.ex_date, source_id: ca.source_id || undefined, status: ca.status };
		if (ca.ratio) body.ratio = Number(ca.ratio);
		if (ca.amount) body.amount = Number(ca.amount);
		if (ca.ccy) body.ccy = ca.ccy.toUpperCase();
		if (!confirm(`Record ${ca.kind} for ${instId} ex ${ca.ex_date}${ca.ratio ? ` ratio ${ca.ratio}` : ''}${ca.amount ? ` amount ${ca.amount} ${ca.ccy}` : ''}?\nPrices are adjusted on read from then on; audited and announced on ref.changed.corporate_action.`)) return;
		try {
			const r = await consoleApi.addCorporateAction(body);
			caMsg = `recorded ${String(r.id ?? '')}${r.created === false ? ' (already known)' : ''}`;
			caOk = true;
			await loadActions();
		} catch (e) {
			caMsg = errText(e);
			caOk = false;
		}
	}

	async function cancelAction(a: CorporateAction) {
		if (!confirm(`Cancel ${a.kind} ex ${a.ex_date} (${a.id})? A cancellation is a new fact; as-of queries before now still see the action. Audited.`)) return;
		try {
			await consoleApi.cancelCorporateAction(a.id);
			caMsg = 'cancelled';
			caOk = true;
			await loadActions();
		} catch (e) {
			caMsg = errText(e);
			caOk = false;
		}
	}

	async function symbolChange() {
		if (!sc.venue || !sc.old_symbol || !sc.new_symbol) return;
		if (!confirm(`Symbol change on ${sc.venue}: ${sc.old_symbol} → ${sc.new_symbol}${sc.effective ? ` effective ${sc.effective}` : ''}? The old listing's validity ends; a recycled ticker resolves to the new instrument only after the change. Audited.`)) return;
		try {
			const r = await consoleApi.symbolChange({ venue: sc.venue, old_symbol: sc.old_symbol, new_symbol: sc.new_symbol, effective: iso(sc.effective) });
			scMsg = JSON.stringify(r).slice(0, 160);
		} catch (e) {
			scMsg = errText(e);
		}
	}

	onMount(() => {
		void loadVenues();
		void loadCalendar();
		void loadActions();
		void loadFx();
		void resolve();
	});
</script>

<ViewHeader title="Reference data" spec="12 §4" note="point-in-time: at = real-world instant, known_at = knowledge instant (bitemporal); mutations audited + ref.changed.*" />

<div class="toolbar panel">
	<label title="real-world instant (validity)">at <input type="datetime-local" bind:value={rAt} /></label>
	<label title="knowledge instant (record time); empty = now">known at <input type="datetime-local" bind:value={rKnown} /></label>
	<button onclick={() => (resolve(), loadFx(), loadInstrument(), loadActions())}>re-evaluate as of</button>
	<span class="dim">Empty = now. Every query below uses these instants.</span>
</div>

<div class="grid cols">
	<section class="panel">
		<div class="panel-head"><h2>Resolve symbol</h2><span class="dim">none · one · ambiguous</span></div>
		<form class="row" onsubmit={(e) => (e.preventDefault(), resolve())}>
			<label>venue <input bind:value={rVenue} size="6" placeholder="xetr" /></label>
			<label>or source <input bind:value={rSource} size="7" disabled={!!rVenue} /></label>
			<label>symbol <input bind:value={rSymbol} size="8" /></label>
			<button type="submit">resolve</button>
		</form>
		{#if resolveErr}<p class="badge danger">{resolveErr}</p>{/if}
		{#if resolved}
			{@const r = resolved.resolved as Record<string, unknown>}
			<p>
				<span class="badge" class:ok={r.result === 'one'} class:warn={r.result === 'ambiguous'} class:danger={r.result === 'none'}>{r.result}</span>
				{#if r.instrument_id}<button class="link mono" onclick={() => loadInstrument(String(r.instrument_id))}>{r.instrument_id}</button>{/if}
				{#if Array.isArray(r.candidates)}<span class="mono dim">{(r.candidates as unknown[]).map((c) => JSON.stringify(c)).join(' · ')}</span>{/if}
			</p>
		{/if}
		<h3 class="sub">FX to EUR</h3>
		<form class="row" onsubmit={(e) => (e.preventDefault(), loadFx())}>
			<input bind:value={ccy} size="4" aria-label="currency" />
			<button type="submit">rate</button>
			{#if fx}<span class="mono">1 {ccy.toUpperCase()} = {num(fx.rate_to_eur, 6)} EUR</span> <span class="dim mono">({fx.source}, {fx.rate_time.slice(0, 10)})</span>{/if}
			{#if fxErr}<span class="badge warn">{fxErr}</span>{/if}
		</form>
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Instrument as of</h2></div>
		<form class="row" onsubmit={(e) => (e.preventDefault(), find())}>
			<input bind:value={search} placeholder="search name / symbol" />
			<button type="submit">search</button>
			<input bind:value={instId} placeholder="or instrument UUID" size="36" class="mono" />
			<button type="button" onclick={() => loadInstrument()} disabled={!instId}>load</button>
		</form>
		{#if hits.length}
			<div class="hits">{#each hits as h (h.id)}<button class="link" onclick={() => ((hits = []), loadInstrument(h.id))}>{h.name} <span class="dim mono">{short(h.id)}</span></button>{/each}</div>
		{/if}
		{#if asOfErr}<p class="badge danger">{asOfErr}</p>{/if}
		{#if asOf}
			{@const i = asOf.instrument as Record<string, unknown>}
			<table>
				<tbody>
					<tr><th>name</th><td>{i.name} <span class="badge">{i.status}</span></td></tr>
					<tr><th>class / kind</th><td class="mono">{i.asset_class} / {i.kind} · {i.base_ccy ?? '—'}/{i.quote_ccy ?? '—'}</td></tr>
					<tr><th>listings</th><td class="mono small">{((i.listings as Record<string, unknown>[]) ?? []).map((l) => `${l.venue}:${l.venue_symbol}`).join(' · ') || '—'}</td></tr>
					<tr><th>identifiers</th><td class="mono small">{((i.identifiers as Record<string, unknown>[]) ?? []).map((l) => `${l.scheme}=${l.value}`).join(' · ') || '—'}</td></tr>
					<tr><th>contract specs</th><td class="mono small">{((asOf.contract_specs as unknown[]) ?? []).length}</td></tr>
					<tr><th>recorded</th><td class="mono dim">{i.recorded_at}</td></tr>
				</tbody>
			</table>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Calendar</h2>
			<span class="dim">{calRule ? `rule ${calRule}` : 'no built-in rule'}{calTz ? ` · ${calTz}` : ''}</span>
		</div>
		{#if venuesErr}<p class="badge danger">{venuesErr}</p>{/if}
		<form class="row" onsubmit={(e) => (e.preventDefault(), loadCalendar())}>
			<select bind:value={venue} onchange={loadCalendar} aria-label="venue">
				{#each venues as v (v.code)}<option value={v.code}>{v.code} · {v.name}{v.rule ? '' : ' (no rule)'} · {v.materialized.days} d stored</option>{/each}
				{#if !venues.length}<option value={venue}>{venue}</option>{/if}
			</select>
			<input type="date" bind:value={calFrom} aria-label="from" />
			<input type="date" bind:value={calTo} aria-label="to" />
			<button type="submit">show</button>
			<button type="button" onclick={materialize} disabled={!calRule} title="POST /api/refdata/calendar/{venue}/materialize">materialise…</button>
			{#if calMsg}<span class="badge">{calMsg}</span>{/if}
		</form>
		{#if calErr}<p class="badge danger">{calErr}</p>{/if}
		{#if sessions.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>date</th><th>basis</th><th>open → close (UTC)</th><th>segments</th></tr></thead>
					<tbody>
						{#each sessions as s (s.date)}
							<tr class:closed={!s.open}>
								<td class="mono">{s.date} <span class="dim">{new Date(s.date + 'T12:00:00Z').toUTCString().slice(0, 3)}</span></td>
								<td><span class="badge" class:ok={s.basis === 'table'} class:warn={s.basis === 'missing_calendar'}>{s.basis}</span></td>
								<td class="mono">
									{#if s.is_holiday}<span class="badge warn">holiday{s.holiday ? `: ${s.holiday}` : ''}</span>
									{:else if s.open}{s.open.slice(11, 16)} → {s.close?.slice(11, 16)}{#if s.early_close} <span class="badge warn">early close</span>{/if}
									{:else}<span class="dim">closed</span>{/if}
								</td>
								<td class="mono dim small">{segText(s)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head">
			<h2>Corporate actions <span class="dim">({actions.length}{pit ? `, as known ${rKnown || 'now'} for ${short(instId)}` : ', newest recorded'})</span></h2>
			<button onclick={() => ((instId = ''), loadActions())} disabled={!instId}>all instruments</button>
		</div>
		{#if actionsErr}<p class="badge danger">{actionsErr}</p>{/if}
		{#if actions.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>instrument</th><th>kind</th><th>ex date</th><th>ratio / amount</th><th>status</th><th>source</th><th>recorded</th><th></th></tr></thead>
					<tbody>
						{#each actions as a (a.id)}
							<tr>
								<td class="mono" title={a.instrument_id}>{a.instrument_name ?? short(a.instrument_id)}</td>
								<td><span class="badge">{a.kind}</span>{#if a.old_symbol}<span class="mono dim"> {a.old_symbol} → {a.new_symbol}</span>{/if}</td>
								<td class="mono">{a.ex_date}</td>
								<td class="mono">{a.ratio ?? ''}{a.amount !== null ? `${a.amount} ${a.ccy ?? ''}` : ''}</td>
								<td><span class="badge" class:ok={a.status === 'confirmed'} class:warn={a.status === 'announced'} class:danger={a.status === 'cancelled'}>{a.status}</span></td>
								<td class="mono dim">{a.source_id ?? '—'}</td>
								<td class="mono dim">{a.recorded_at.slice(0, 16)}</td>
								<td>{#if a.status !== 'cancelled'}<button onclick={() => cancelAction(a)}>cancel…</button>{/if}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !actionsErr}
			<p class="dim">No corporate actions{instId ? ' for this instrument' : ''} recorded.</p>
		{/if}
		<h3 class="sub">Enter a corporate action {instId ? `for ${short(instId, 12)}` : '(load an instrument first)'}</h3>
		<form class="row" onsubmit={(e) => (e.preventDefault(), addAction())}>
			<select bind:value={ca.kind} aria-label="kind">{#each KINDS as k}<option>{k}</option>{/each}</select>
			<label>ex date <input type="date" bind:value={ca.ex_date} required /></label>
			<label title="splits: new shares per old share (2-for-1 → 2; 1-for-10 → 0.1)">ratio <input bind:value={ca.ratio} size="5" /></label>
			<label>amount <input bind:value={ca.amount} size="6" /></label>
			<label>ccy <input bind:value={ca.ccy} size="4" /></label>
			<select bind:value={ca.status} aria-label="status"><option>confirmed</option><option>announced</option></select>
			<label>source <input bind:value={ca.source_id} size="7" /></label>
			<button type="submit" disabled={!instId || !ca.ex_date}>record…</button>
			{#if caMsg}<span class="badge" class:ok={caOk} class:danger={!caOk}>{caMsg}</span>{/if}
		</form>
		<h3 class="sub">Symbol change</h3>
		<form class="row" onsubmit={(e) => (e.preventDefault(), symbolChange())}>
			<label>venue <input bind:value={sc.venue} size="6" /></label>
			<label>old <input bind:value={sc.old_symbol} size="7" /></label>
			<label>new <input bind:value={sc.new_symbol} size="7" /></label>
			<label>effective <input type="date" bind:value={sc.effective} /></label>
			<button type="submit" disabled={!sc.venue || !sc.old_symbol || !sc.new_symbol}>apply…</button>
			{#if scMsg}<span class="mono dim small">{scMsg}</span>{/if}
		</form>
	</section>
</div>

{#if !venues.length && !venuesErr}<Empty producer="GET /api/refdata/venues (Postgres ref.venue)" hint="" />{/if}

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
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
	}
	.wide {
		grid-column: 1 / -1;
	}
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
		margin-bottom: 8px;
	}
	.sub {
		font-size: 12px;
		margin: 12px 0 6px;
	}
	.tall {
		max-height: 380px;
	}
	.small {
		font-size: 11px;
	}
	.hits {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		margin-bottom: 8px;
	}
	button.link {
		background: none;
		border: none;
		color: var(--accent);
		padding: 0 2px;
	}
	tr.closed td {
		opacity: 0.6;
	}
	th {
		position: static;
	}
</style>
