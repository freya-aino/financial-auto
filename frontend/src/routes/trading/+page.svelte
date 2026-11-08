<script lang="ts">
	import { onMount } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import Gauge from '../../lib/Gauge.svelte';
	import KillSwitch from '../../lib/KillSwitch.svelte';
	import Sparkline from '../../lib/Sparkline.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { drawdownSeries, loadHistory, pushSample, saveHistory } from '../../lib/series';
	import { deriveStrategies, STRATEGY_ACTIONS } from '../../lib/strategies';
	import { onBus } from '../../lib/ws.svelte';
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

	// Client-side history of the polled risk figures for the charts (spec 51 §6.5). The API has
	// no /api/trading/pnl yet (501), so the ring buffer is the only source; the newest 300
	// samples survive reloads via localStorage.
	const history = loadHistory();
	let samples = $state(history.toArray());
	const times = $derived(samples.map((s) => s.t));
	const drawdowns = $derived(drawdownSeries(samples));
	const strategies = $derived(deriveStrategies(signals, orders, now));
	const actionTitle = (action: string) =>
		`POST /api/trading/strategies/{id}/${action} is not implemented yet (501, spec 50 §4.7)`;
	const fmtPct = (v: number) => pct(v, 2);
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
			const f = r.reported ?? r.paper;
			const added = pushSample(history, {
				t: now,
				equity: f.equity_eur,
				pnl: r.reported?.daily_pnl_eur ?? r.paper.pnl_eur ?? 0,
				drawdown: f.drawdown,
				exposure: f.gross_exposure_eur
			});
			if (added) {
				samples = history.toArray();
				saveHistory(history);
			}
		} catch (e) {
			error = String(e);
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

	onMount(() => {
		const stopPoll = poll(refresh, 2000);
		const stopBus = onBus(['order.>', 'risk.>', 'signal.exposure.>'], refresh);
		return () => {
			stopPoll();
			stopBus();
		};
	});
</script>

<ViewHeader title="Trading Desk" spec="51 §6.5" note="paper only; no live broker is connected" />
{#if error}<p class="badge danger">{error}</p>{/if}

<div class="killbar panel" class:halted={risk?.portfolio_halted}>
	<div class="modes">
		<span class="badge ok mode">PAPER</span>
		<button class="live" disabled title="no live broker connected">LIVE</button>
		<span class="dim">Kill switch: halts the portfolio and blocks manual orders (spec 40).</span>
	</div>
	<KillSwitch {risk} onchanged={refresh} />
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

<div class="grid charts">
	<section class="panel">
		<div class="panel-head"><h2>Equity & P&L</h2><span class="dim">{figuresSource}, sampled every 2 s · {samples.length} pts</span></div>
		<Sparkline {times} values={samples.map((s) => s.equity)} format={eur} label="equity" fill />
		<Sparkline {times} values={samples.map((s) => s.pnl)} format={eur} baseline={0} tone="signed" label={risk?.reported ? 'daily P&L' : 'P&L since start'} />
	</section>
	<section class="panel">
		<div class="panel-head"><h2>Drawdown</h2><span class="dim">reported · peak-to-trough from sampled equity</span></div>
		<Sparkline {times} values={samples.map((s) => s.drawdown)} format={fmtPct} tone="danger" label="drawdown (reported)" fill />
		<Sparkline {times} values={drawdowns} format={fmtPct} tone="danger" label="drawdown (client, from equity samples)" />
	</section>
	<section class="panel">
		<div class="panel-head"><h2>Exposure vs caps</h2><span class="dim">GET /api/trading/limits → 501; caps not reported</span></div>
		{#if figures}
			<div class="gauges">
				<Gauge label="gross exposure" value={figures.gross_exposure_eur} reference={figures.equity_eur} format={eur} />
				<Gauge label="deployable used" value={figures.equity_eur - figures.deployable_eur} reference={figures.equity_eur} format={eur} />
				<Gauge label="drawdown" value={figures.drawdown} reference={1} format={fmtPct} />
				{#each positions.slice(0, 6) as p (p.instrument_id)}
					<Gauge label="{short(p.instrument_id, 10)} exposure" value={p.market_value_eur ?? p.quantity * p.avg_cost_eur} reference={figures.equity_eur} format={eur} />
				{/each}
			</div>
		{:else}
			<Empty producer="risk.state.* (risk service) or the paper book" />
		{/if}
	</section>
</div>

<section class="panel strategies">
	<div class="panel-head">
		<h2>Strategies</h2>
		<span class="dim">derived from signal.exposure.* and order.* (strategy_ids); GET /api/trading/strategies → 501</span>
	</div>
	{#if strategies.length}
		<div class="scroll tall">
			<table>
				<thead>
					<tr><th>strategy</th><th>genome</th><th>horizon</th><th>last signal</th><th>instruments</th><th>mean |exposure|</th><th>largest</th><th>orders</th><th>filled</th><th>rejected</th><th>actions</th></tr>
				</thead>
				<tbody>
					{#each strategies as s (s.strategy_id)}
						<tr>
							<td class="mono">{s.strategy_id}</td>
							<td class="mono dim">{s.genome_id || '—'}</td>
							<td class="mono">{s.horizon || '—'}</td>
							<td class="mono dim">{s.last_signal_ms === null ? '—' : `${ago(new Date(now - s.last_signal_ms).toISOString(), now)} ago`}</td>
							<td class="mono">{s.instruments}</td>
							<td>
								<div class="bar"><div style="width: {100 * Math.min(1, s.mean_abs_exposure)}%"></div></div>
								<span class="mono">{s.mean_abs_exposure.toFixed(2)}</span>
							</td>
							<td class="mono {signed(s.max_exposure?.mean)}" title={s.max_exposure?.instrument_id}>
								{s.max_exposure ? `${s.max_exposure.mean.toFixed(2)} ${short(s.max_exposure.instrument_id, 8)}` : '—'}
							</td>
							<td class="mono">{s.orders}</td>
							<td class="mono">{s.filled}</td>
							<td class="mono {s.rejected ? 'neg' : ''}">{s.rejected}</td>
							<td class="actions">
								{#each STRATEGY_ACTIONS as a}
									<button type="button" disabled title={actionTitle(a)}>{a}</button>
								{/each}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="dim">Actions are disabled: <code>POST /api/trading/strategies/&#123;id&#125;/pause|resume|flatten|promote</code> answer 501 until the strategy runtime exists (spec 50 §4.7). Status, sleeve and gate checks need <code>GET /api/trading/strategies</code>.</p>
	{:else}
		<Empty producer="signal.exposure.* (strategy population)" />
	{/if}
</section>

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
	.modes {
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
	.charts {
		grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
		margin-bottom: 12px;
	}
	.charts .panel {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.gauges {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.strategies {
		margin-bottom: 12px;
	}
	.strategies p {
		margin: 8px 0 0;
	}
	.actions {
		white-space: nowrap;
	}
	.actions button {
		font-size: 11px;
		padding: 1px 6px;
		margin-right: 3px;
	}
	.bar {
		display: inline-block;
		width: 70px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
		margin-right: 6px;
	}
	.bar > div {
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
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
