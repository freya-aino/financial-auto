<script lang="ts">
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import {
		ago,
		eur,
		live,
		pct,
		poll,
		short,
		type OrderView,
		type Position,
		type RiskView,
		type SignalView
	} from '../../lib/api';

	let risk = $state<RiskView | null>(null);
	let positions = $state<Position[]>([]);
	let cash = $state(0);
	let equity = $state(0);
	let orders = $state<OrderView[]>([]);
	let signals = $state<SignalView[]>([]);
	let error = $state('');
	let now = $state(Date.now());

	let killBusy = $state(false);
	let killMsg = $state('');

	// manual paper order ticket
	let oInst = $state('');
	let oSide = $state<'buy' | 'sell'>('buy');
	let oQty = $state(1);
	let oLimit = $state<number | null>(null);
	let oBusy = $state(false);
	let oMsg = $state('');
	let oErr = $state('');

	const figures = $derived(risk?.reported ?? risk?.paper ?? null);
	const figuresSource = $derived(risk?.reported ? 'risk service' : 'paper book');
	const knownInstruments = $derived(
		[
			...new Set([
				...positions.map((p) => p.instrument_id),
				...signals.flatMap((s) => s.exposures.map((e) => e.instrument_id)),
				...orders.map((o) => o.instrument_id)
			])
		].filter(Boolean)
	);

	$effect(() => {
		if (!oInst && knownInstruments.length) oInst = knownInstruments[0];
	});

	async function refresh() {
		now = Date.now();
		try {
			const [r, p, o, s] = await Promise.all([live.risk(), live.positions(), live.orders(100), live.signals()]);
			risk = r;
			positions = p.positions;
			cash = p.cash_eur;
			equity = p.equity_eur;
			orders = o.orders;
			signals = s.signals;
			error = '';
		} catch (e) {
			error = String(e);
		}
	}

	async function kill() {
		const reason = prompt('KILL SWITCH: halt the whole portfolio. Reason?', 'manual kill switch');
		if (reason === null) return;
		killBusy = true;
		killMsg = '';
		try {
			await live.halt(reason);
			killMsg = 'portfolio halted';
			await refresh();
		} catch (e) {
			killMsg = e instanceof Error ? e.message : String(e);
		} finally {
			killBusy = false;
		}
	}

	async function resume() {
		if (!confirm('Re-arm trading for the portfolio?')) return;
		killBusy = true;
		killMsg = '';
		try {
			await live.resume();
			killMsg = 'portfolio resumed';
			await refresh();
		} catch (e) {
			killMsg = e instanceof Error ? e.message : String(e);
		} finally {
			killBusy = false;
		}
	}

	async function submitOrder(ev: SubmitEvent) {
		ev.preventDefault();
		oBusy = true;
		oMsg = '';
		oErr = '';
		try {
			const r = await live.paperOrder({
				instrument_id: oInst.trim(),
				side: oSide,
				quantity: Number(oQty),
				...(oLimit ? { limit_price: Number(oLimit) } : {})
			});
			oMsg = `paper intent ${short(r.intent_id, 10)} sent`;
			await refresh();
		} catch (e) {
			oErr = e instanceof Error ? e.message : String(e);
		} finally {
			oBusy = false;
		}
	}

	const statusClass = (s: string) =>
		s === 'filled' ? 'ok' : s === 'rejected' || s === 'canceled' ? 'danger' : 'warn';
	const signed = (v: number | null | undefined) => (v === null || v === undefined ? '' : v >= 0 ? 'pos' : 'neg');

	onMount(() => poll(refresh, 2000));
</script>

<ViewHeader title="Trading Desk" spec="51 §6.5" note="paper only; no live broker is connected" />
{#if error}<p class="badge danger">{error}</p>{/if}

<div class="killbar panel" class:halted={risk?.portfolio_halted}>
	<div class="modes">
		<span class="badge ok mode">PAPER</span>
		<button class="live" disabled title="no live broker connected">LIVE</button>
		<span class="dim">Kill switch: halts the portfolio and blocks manual orders (spec 40).</span>
	</div>
	<div class="actions">
		{#if killMsg}<span class="badge">{killMsg}</span>{/if}
		{#if risk?.portfolio_halted}
			<span class="badge danger">PORTFOLIO HALTED</span>
			<button onclick={resume} disabled={killBusy}>Resume</button>
		{/if}
		<button class="kill" onclick={kill} disabled={killBusy || risk?.portfolio_halted}>KILL SWITCH</button>
	</div>
</div>

<div class="tiles">
	<div class="panel"><span class="dim">equity · {figuresSource}</span><b class="mono">{eur(figures?.equity_eur)}</b></div>
	<div class="panel"><span class="dim">deployable (net of payout quota)</span><b class="mono">{eur(figures?.deployable_eur)}</b></div>
	<div class="panel"><span class="dim">gross exposure</span><b class="mono">{eur(figures?.gross_exposure_eur)}</b></div>
	<div class="panel"><span class="dim">drawdown</span><b class="mono" class:neg={(figures?.drawdown ?? 0) > 0.05}>{pct(figures?.drawdown)}</b></div>
	<div class="panel">
		<span class="dim">{risk?.reported ? 'daily P&L' : 'P&L since start'}</span>
		<b class="mono {signed(risk?.reported?.daily_pnl_eur ?? risk?.paper.pnl_eur)}">{eur(risk?.reported?.daily_pnl_eur ?? risk?.paper.pnl_eur)}</b>
	</div>
	<div class="panel"><span class="dim">paper book equity / cash</span><b class="mono">{eur(equity)} <span class="dim small">/ {eur(cash)}</span></b></div>
</div>

<div class="grid two">
	<section class="panel">
		<div class="panel-head"><h2>Positions</h2><span class="dim">paper book, EUR, from fills seen by control-api</span></div>
		{#if positions.length}
			<table>
				<thead><tr><th>instrument</th><th>qty</th><th>avg cost</th><th>mark</th><th>value</th><th>unrealised</th><th>realised</th></tr></thead>
				<tbody>
					{#each positions as p (p.instrument_id)}
						<tr>
							<td class="mono" title={p.instrument_id}>{short(p.instrument_id, 12)}</td>
							<td class="mono {signed(p.quantity)}">{p.quantity}</td>
							<td class="mono">{eur(p.avg_cost_eur)}</td>
							<td class="mono">{eur(p.mark_eur)}</td>
							<td class="mono">{eur(p.market_value_eur)}</td>
							<td class="mono {signed(p.unrealised_eur)}">{eur(p.unrealised_eur)}</td>
							<td class="mono {signed(p.realised_eur)}">{eur(p.realised_eur)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<Empty producer="order.event.* fills (broker-gateway paper engine)" hint="place a paper order below" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Manual paper order</h2><span class="dim">order.approved.paper → broker-gateway</span></div>
		<form class="ticket" onsubmit={submitOrder}>
			<label class="wide">instrument
				<input list="instruments" bind:value={oInst} placeholder="instrument UUID" />
				<datalist id="instruments">{#each knownInstruments as i}<option value={i}></option>{/each}</datalist>
			</label>
			<label>side
				<select bind:value={oSide}><option value="buy">buy</option><option value="sell">sell</option></select>
			</label>
			<label>quantity <input type="number" min="0" step="any" bind:value={oQty} /></label>
			<label>limit (optional) <input type="number" min="0" step="any" bind:value={oLimit} placeholder="market" /></label>
			<div class="wide row">
				<button type="submit" disabled={oBusy || !oInst || risk?.portfolio_halted}>
					{oBusy ? 'sending…' : `Paper ${oSide}`}
				</button>
				{#if risk?.portfolio_halted}<span class="badge danger">halted</span>{/if}
				{#if oMsg}<span class="badge ok">{oMsg}</span>{/if}
				{#if oErr}<span class="badge danger">{oErr}</span>{/if}
			</div>
		</form>
		<h3 class="sub">Active halts</h3>
		{#if risk?.halts.length}
			<table>
				<thead><tr><th>scope</th><th>reason</th><th>since</th><th>manual</th></tr></thead>
				<tbody>
					{#each risk.halts as h (h.scope)}
						<tr>
							<td class="mono">{h.scope}</td>
							<td>{h.reason}</td>
							<td class="mono dim">{ago(h.since, now)} ago</td>
							<td>{h.manual ? 'yes' : 'no'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<p class="dim">none</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Orders & fills</h2><span class="dim">order.intent / approved / event</span></div>
		{#if orders.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>intent</th><th>instrument</th><th>side</th><th>qty</th><th>algo</th><th>status</th><th>fill</th><th>strategies</th><th>age</th></tr></thead>
					<tbody>
						{#each orders as o (o.intent_id)}
							<tr title={o.reason}>
								<td class="mono">{short(o.intent_id, 10)}</td>
								<td class="mono" title={o.instrument_id}>{short(o.instrument_id, 10)}</td>
								<td class={o.side === 'buy' ? 'pos' : 'neg'}>{o.side}</td>
								<td class="mono">{o.quantity}</td>
								<td class="mono dim">{o.exec_algo}{o.limit_price ? ` @${o.limit_price}` : ''}</td>
								<td><span class="badge {statusClass(o.status)}">{o.status}</span> <span class="badge">{o.mode}</span></td>
								<td class="mono">{o.filled_quantity ? `${o.filled_quantity} @ ${o.avg_fill_price?.toFixed(4)}` : '—'}</td>
								<td class="mono dim">{o.strategy_ids.join(', ')}</td>
								<td class="mono dim">{ago(o.updated_at, now)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<Empty producer="order.* (allocator / risk / broker-gateway)" hint="place a paper order above" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Strategy signals</h2><span class="dim">target exposure, mean ± σ in [−1, 1]</span></div>
		{#if signals.length}
			<div class="scroll tall">
				{#each signals as s (s.strategy_id)}
					<div class="signal">
						<div class="signal-head">
							<b class="mono">{s.strategy_id}</b>
							<span class="dim mono">{s.genome_id} · horizon {s.horizon} · {ago(s.as_of, now)} ago</span>
						</div>
						{#each s.exposures as e}
							{@const lo = Math.max(-1, e.mean - e.stddev)}
							{@const hi = Math.min(1, e.mean + e.stddev)}
							<div class="exp">
								<span class="mono dim" title={e.instrument_id}>{short(e.instrument_id, 10)}</span>
								<div class="axis">
									<div class="zero"></div>
									<div class="band" style="left: {(lo + 1) * 50}%; width: {(hi - lo) * 50}%"></div>
									<div class="mean {e.mean >= 0 ? 'pos-bg' : 'neg-bg'}" style="left: {(e.mean + 1) * 50}%"></div>
								</div>
								<span class="mono {signed(e.mean)}">{e.mean.toFixed(2)} ± {e.stddev.toFixed(2)}</span>
							</div>
						{/each}
					</div>
				{/each}
			</div>
		{:else}
			<Empty producer="signal.exposure.* (strategy population)" />
		{/if}
	</section>
</div>

<style>
	.killbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		flex-wrap: wrap;
		margin-bottom: 12px;
		border-color: var(--danger);
	}
	.killbar.halted {
		background: color-mix(in srgb, var(--danger) 12%, var(--bg-raised));
	}
	.modes,
	.actions {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.mode {
		font-weight: 700;
		letter-spacing: 0.06em;
	}
	.live {
		font-size: 11px;
		letter-spacing: 0.06em;
	}
	.kill {
		background: var(--danger);
		color: #fff;
		border: none;
		font-weight: 700;
		letter-spacing: 0.08em;
		padding: 8px 18px;
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
		gap: 12px;
		margin-bottom: 12px;
	}
	.tiles > div {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.tiles b {
		font-size: 18px;
	}
	.small {
		font-size: 12px;
	}
	.two {
		grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
	}
	.tall {
		max-height: 420px;
	}
	.pos {
		color: var(--ok);
	}
	.neg {
		color: var(--danger);
	}
	.ticket {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 8px;
	}
	.ticket label {
		display: flex;
		flex-direction: column;
		gap: 2px;
		color: var(--text-dim);
	}
	.wide {
		grid-column: 1 / -1;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.sub {
		font-size: 12px;
		margin: 14px 0 6px;
	}
	.signal {
		padding: 6px 0;
		border-bottom: 1px solid var(--border);
	}
	.signal-head {
		display: flex;
		gap: 8px;
		align-items: baseline;
		margin-bottom: 4px;
	}
	.exp {
		display: grid;
		grid-template-columns: 110px 1fr 110px;
		gap: 8px;
		align-items: center;
	}
	.axis {
		position: relative;
		height: 12px;
		background: var(--bg-inset);
		border-radius: 3px;
	}
	.zero {
		position: absolute;
		left: 50%;
		top: 0;
		bottom: 0;
		width: 1px;
		background: var(--border);
	}
	.band {
		position: absolute;
		top: 3px;
		bottom: 3px;
		background: var(--text-dim);
		opacity: 0.35;
		border-radius: 2px;
	}
	.mean {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 3px;
		margin-left: -1px;
	}
	.pos-bg {
		background: var(--ok);
	}
	.neg-bg {
		background: var(--danger);
	}
</style>
