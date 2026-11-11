<script lang="ts">
	// Ledger reports (D-067, spec 42 §4): P&L with the FX split per book and strategy, tax lots
	// (FIFO), the statement (JSON summary + CSV download) and orphan fills.
	import { onMount } from 'svelte';
	import Empty from '../../../lib/Empty.svelte';
	import ViewHeader from '../../../lib/ViewHeader.svelte';
	import { eur, num, poll, short } from '../../../lib/api';
	import { consoleApi, errText, type PnlBreakdown, type PnlParts } from '../../../lib/console';
	import { canBook, orphanConfirmText, orphanResolveBody, orphanRow, payoutBody, payoutConfirmText } from '../../../lib/ledgerForms';
	import { loadPref, savePref } from '../../../lib/prefs';
	import { runtime } from '../../../lib/runtime.svelte';

	let book = $state(loadPref('ledger.book', 'paper'));
	let from = $state(loadPref('ledger.from', ''));
	let to = $state(loadPref('ledger.to', ''));
	let pnl = $state<PnlBreakdown | null>(null);
	let pnlError = $state('');
	let lots = $state<Record<string, unknown>[]>([]);
	let lotsError = $state('');
	let statement = $state<Record<string, unknown> | null>(null);
	let statementError = $state('');
	let orphans = $state<Record<string, unknown>[]>([]);
	let orphansError = $state('');
	$effect(() => savePref('ledger.book', book));
	$effect(() => savePref('ledger.from', from));
	$effect(() => savePref('ledger.to', to));

	const PARTS: (keyof PnlParts)[] = ['realised_eur', 'unrealised_eur', 'fx_realised_eur', 'fx_unrealised_eur', 'fx_cash_eur', 'fees_eur', 'funding_eur', 'borrow_eur', 'other_eur', 'net_eur'];
	const label = (k: string) => k.replace(/_eur$/, '').replace(/_/g, ' ');
	const strategies = $derived(Object.entries(pnl?.strategies ?? {}).sort((a, b) => a[1].net_eur - b[1].net_eur));
	const fills = $derived(((statement?.fills ?? []) as Record<string, unknown>[]));
	const closing = $derived((statement?.closing ?? null) as Record<string, unknown> | null);
	const opening = $derived((statement?.opening ?? null) as Record<string, unknown> | null);
	const fxFills = $derived(fills.filter((f) => f.ccy !== 'EUR').length);
	const feeSum = $derived(fills.reduce((s, f) => s + (typeof f.fee_eur === 'number' ? f.fee_eur : 0), 0));

	// operator forms: orphan resolution and payout booking (confirm first, show the API's answer)
	const orphanRows = $derived(orphans.map(orphanRow));
	let dismissReason = $state<Record<string, string>>({});
	let actionMsg = $state('');
	let actionOk = $state(true);
	let busy = $state(false);
	let payoutDay = $state(new Date().toISOString().slice(0, 10));
	let payoutAmount = $state('');

	async function act(question: string, call: () => Promise<unknown>, done: string) {
		if (!confirm(question)) return;
		busy = true;
		try {
			const r = await call();
			actionMsg = `${done}: ${JSON.stringify(r)}`;
			actionOk = true;
			await refresh();
		} catch (e) {
			actionMsg = `${done} failed: ${errText(e)}`;
			actionOk = false;
		} finally {
			busy = false;
		}
	}

	function resolve(o: ReturnType<typeof orphanRow>, action: 'book' | 'dismiss') {
		const b = orphanResolveBody(action, dismissReason[o.id] ?? '');
		if ('error' in b) return ((actionMsg = b.error), (actionOk = false));
		void act(orphanConfirmText(o, b.body), () => consoleApi.resolveOrphan(o.id, b.body), `orphan ${action}`);
	}

	function bookPayout() {
		const p = payoutBody(book, payoutDay, payoutAmount);
		if ('error' in p) return ((actionMsg = p.error), (actionOk = false));
		void act(payoutConfirmText(p.body), () => consoleApi.bookPayout(p.body), 'payout');
	}

	async function refresh() {
		const f = from || undefined;
		const t = to || undefined;
		const [p, l, o] = await Promise.allSettled([consoleApi.pnlBreakdown(book, f, t), consoleApi.lots(book), consoleApi.orphans('open')]);
		if (p.status === 'fulfilled') ((pnl = p.value), (pnlError = ''));
		else pnlError = errText(p.reason);
		if (l.status === 'fulfilled') ((lots = l.value.lots as Record<string, unknown>[]), (lotsError = ''));
		else lotsError = errText(l.reason);
		if (o.status === 'fulfilled') ((orphans = o.value.orphans), (orphansError = ''));
		else orphansError = errText(o.reason);
	}

	/** The statement carries every fill of the range: loaded on demand, not polled. */
	async function loadStatement() {
		try {
			statement = await consoleApi.statement(book, from || undefined, to || undefined);
			statementError = '';
		} catch (e) {
			statementError = errText(e);
		}
	}

	function apply() {
		void refresh();
		void loadStatement();
	}

	onMount(() => {
		void loadStatement();
		return poll(refresh, () => runtime.pollMs * 15);
	});
</script>

<ViewHeader title="Ledger" spec="42 §4" note="fa-ledger reports derived from fills and cash movements: P&L with FX split, lots, statements, orphans" />

<div class="toolbar">
	<label>book <input bind:value={book} size="8" /></label>
	<label>from <input type="date" bind:value={from} /></label>
	<label>to <input type="date" bind:value={to} /></label>
	<button onclick={apply}>apply</button>
	<a href={consoleApi.statementCsvUrl(book, from || undefined, to || undefined)} download>statement CSV</a>
</div>

<div class="grid cols">
	<section class="panel">
		<div class="panel-head"><h2>P&amp;L breakdown <span class="dim">({book})</span></h2><span class="dim">EUR · FX split into realised / unrealised / cash</span></div>
		{#if pnlError}<p class="badge danger">{pnlError}</p>{/if}
		{#if pnl}
			<table class="parts">
				<tbody>
					{#each PARTS as k}
						{@const v = pnl.book_pnl[k] ?? 0}
						<tr class:total={k === 'net_eur'}>
							<th>{label(k)}</th>
							<td class="mono" class:pos={v > 0 && k !== 'fees_eur'} class:neg={v < 0 || (k === 'fees_eur' && v > 0)}>{eur(k === 'fees_eur' ? -v : v)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<p class="dim small">Fees shown as a cost. FX total {eur(pnl.book_pnl.fx_eur)}.</p>
		{:else if !pnlError}
			<Empty producer="GET /api/trading/pnl/breakdown" hint="" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Per strategy <span class="dim">({strategies.length})</span></h2></div>
		{#if strategies.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>strategy</th><th>realised</th><th>unrealised</th><th>FX</th><th>fees</th><th>net</th></tr></thead>
					<tbody>
						{#each strategies as [id, p] (id)}
							<tr>
								<td class="mono" title={id}>{short(id, 12)}</td>
								<td class="mono">{num(p.realised_eur, 2)}</td>
								<td class="mono">{num(p.unrealised_eur, 2)}</td>
								<td class="mono" title="realised {num(p.fx_realised_eur, 2)} · unrealised {num(p.fx_unrealised_eur, 2)} · cash {num(p.fx_cash_eur, 2)}">{num(p.fx_eur, 2)}</td>
								<td class="mono dim">{num(p.fees_eur, 2)}</td>
								<td class="mono" class:pos={p.net_eur > 0} class:neg={p.net_eur < 0}>{num(p.net_eur, 2)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="dim">No strategy attribution.</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Tax lots <span class="dim">({lots.length})</span></h2><span class="dim">FIFO, open lots now</span></div>
		{#if lotsError}<p class="badge danger">{lotsError}</p>{/if}
		{#if lots.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>instrument</th><th>open / qty</th><th>price</th><th>ccy · FX→EUR</th><th>cost EUR</th><th>opened</th></tr></thead>
					<tbody>
						{#each lots as l, i (String(l.lot_id ?? i))}
							{@const qty = Number(l.open_qty ?? 0)}
							{@const px = Number(l.price ?? 0)}
							{@const fxr = Number(l.fx_to_eur ?? 1)}
							<tr>
								<td class="mono" title="{l.instrument} · lot {l.lot_id}">{short(String(l.instrument ?? ''), 10)}</td>
								<td class="mono">{num(qty, 4)} <span class="dim">/ {num(Number(l.qty ?? 0), 4)}</span></td>
								<td class="mono">{num(px, 4)}</td>
								<td class="mono dim">{l.ccy} · {num(fxr, 4)}</td>
								<td class="mono">{eur(qty * px * fxr)}</td>
								<td class="mono dim">{String(l.opened_at ?? '').slice(0, 19).replace('T', ' ')}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !lotsError}
			<p class="dim">No open lots (the book is flat).</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Statement</h2><span class="dim">{String(statement?.method ?? '')}</span></div>
		{#if statementError}<p class="badge danger">{statementError}</p>{/if}
		{#if statement}
			<table>
				<thead><tr><th></th><th>opening</th><th>closing</th></tr></thead>
				<tbody>
					<tr><th>at</th><td class="mono dim">{String(opening?.at ?? '—')}</td><td class="mono dim">{String(closing?.at ?? 'now')}</td></tr>
					<tr><th>cash</th><td class="mono">{eur(opening?.cash_eur as number)}</td><td class="mono">{eur(closing?.cash_eur as number)}</td></tr>
					<tr><th>equity</th><td class="mono">{eur(opening?.equity_eur as number)}</td><td class="mono">{eur(closing?.equity_eur as number)}</td></tr>
					<tr><th>positions</th><td class="mono">{(opening?.positions as unknown[])?.length ?? 0}</td><td class="mono">{(closing?.positions as unknown[])?.length ?? 0}</td></tr>
					<tr><th>FX marks</th><td></td><td class="mono dim">{Object.entries(((closing?.marks as Record<string, unknown>)?.fx ?? {}) as Record<string, number>).map(([c, r]) => `${c} ${r.toFixed(4)}`).join(' · ') || '—'}</td></tr>
				</tbody>
			</table>
			<p class="dim small">{fills.length} fills ({fxFills} in foreign currency) · fees {eur(feeSum)}. Download the CSV for the full fill list.</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Orphan fills <span class="dim">({orphans.length} open)</span></h2></div>
		{#if orphansError}<p class="badge danger">{orphansError}</p>{/if}
		{#if actionMsg}<p class="badge" class:ok={actionOk} class:danger={!actionOk}>{actionMsg}</p>{/if}
		{#if orphanRows.length}
			<table>
				<thead><tr><th>fill</th><th>order</th><th>instrument</th><th>qty @ price</th><th>book</th><th>resolve</th></tr></thead>
				<tbody>
					{#each orphanRows as o (o.id)}
						<tr>
							<td class="mono">{short(o.venue_fill_id || o.id, 16)}</td>
							<td class="mono dim">{short(o.client_order_id, 14)}</td>
							<td class="mono">{o.instrument ? short(o.instrument, 12) : `${o.venue_symbol ?? '?'} (${o.instrument_status})`}</td>
							<td class="mono">{o.qty} @ {o.price} {o.ccy ?? ''}</td>
							<td class="mono dim">{o.book ?? 'paper'}</td>
							<td class="actions">
								<button disabled={busy || !canBook(o)} title={canBook(o) ? 'book it as a normal fill' : 'instrument not resolved yet'} onclick={() => resolve(o, 'book')}>book</button>
								<input placeholder="dismiss reason" bind:value={dismissReason[o.id]} />
								<button class="danger" disabled={busy} onclick={() => resolve(o, 'dismiss')}>dismiss</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<p class="dim small"><code>POST /api/trading/orphans/&#123;id&#125;/resolve</code> (spec 42 §8); every action asks first and is audited.</p>
		{:else if !orphansError}
			<p class="dim">No orphan fills.</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Book a payout <span class="dim">(book {book})</span></h2></div>
		<div class="toolbar">
			<label>day <input type="date" bind:value={payoutDay} /></label>
			<label>amount EUR <input inputmode="decimal" placeholder="0.00" bind:value={payoutAmount} /></label>
			<button disabled={busy || !payoutAmount} onclick={bookPayout}>book payout…</button>
		</div>
		<p class="dim small"><code>POST /api/trading/payouts</code> (spec 42 §6): a cash movement out of the book, after a confirmation; the risk-engine's daily quota (<code>[payout]</code>) is advisory, booking stays an operator action.</p>
	</section>
</div>

<style>
	.actions {
		display: flex;
		gap: 6px;
		align-items: center;
		flex-wrap: wrap;
	}
	.actions input {
		width: 140px;
	}
	.toolbar {
		display: flex;
		gap: 12px;
		align-items: center;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}
	.toolbar label {
		display: flex;
		gap: 6px;
		align-items: center;
		color: var(--text-dim);
	}
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
	}
	.tall {
		max-height: 420px;
	}
	.parts th {
		position: static;
		text-transform: capitalize;
	}
	.total th,
	.total td {
		font-weight: 700;
		border-top: 2px solid var(--border);
	}
	.small {
		font-size: 11px;
	}
	.pos {
		color: var(--ok);
	}
	.neg {
		color: var(--danger);
	}
</style>
