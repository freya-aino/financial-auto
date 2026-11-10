<script lang="ts">
	// Trading Desk (spec 51 §6.5, D-051). Live state arrives on /ws/trading (`positions`, `risk`,
	// `halts`, `order.<intent>`, `fill.<intent>`); REST polling is the fallback while the socket is down.
	import { onMount, untrack } from 'svelte';
	import Empty from '../../lib/Empty.svelte';
	import Gauge from '../../lib/Gauge.svelte';
	import KillSwitch from '../../lib/KillSwitch.svelte';
	import Limits from '../../lib/Limits.svelte';
	import Pnl from '../../lib/Pnl.svelte';
	import Reconciliation from '../../lib/Reconciliation.svelte';
	import Sparkline from '../../lib/Sparkline.svelte';
	import StrategiesTable from '../../lib/StrategiesTable.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import { drawdownSeries, loadHistory, pushSample, saveHistory } from '../../lib/series';
	import { deriveStrategies } from '../../lib/strategies';
	import { runtime } from '../../lib/runtime.svelte';
	import { onBus, tradingWs } from '../../lib/ws.svelte';
	import {
		ago,
		eur,
		live,
		pct,
		poll,
		short,
		type Limits as LimitsT,
		type OrderView,
		type Position,
		type RiskView,
		type SignalView,
		type StrategiesView
	} from '../../lib/api';

	let risk = $state<RiskView | null>(null);
	let positions = $state<Position[]>([]);
	let cash = $state(0);
	let equity = $state(0);
	let orders = $state<OrderView[]>([]);
	let signals = $state<SignalView[]>([]);
	let strategiesView = $state<StrategiesView | null>(null);
	let strategiesError = $state('');
	let limits = $state<LimitsT | null>(null);
	let error = $state('');
	let now = $state(Date.now());
	let fillVersion = $state(0);

	// flatten-all
	let flatMsg = $state('');
	let flatOk = $state(true);
	let flatBusy = $state(false);

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

	// Client-side history of the risk figures for the charts (spec 51 §6.5): sampled from the
	// `risk` frames (5 Hz max) or the polled payload; the newest 300 samples survive reloads.
	const history = loadHistory();
	let samples = $state(history.toArray());
	const times = $derived(samples.map((s) => s.t));
	const drawdowns = $derived(drawdownSeries(samples));
	/** Fallback rows when GET /api/trading/strategies is unavailable (503): derived from signals + orders. */
	const derivedStrategies = $derived(deriveStrategies(signals, orders, now));
	const strategyIds = $derived(strategiesView?.strategies.map((s) => s.strategy_id) ?? derivedStrategies.map((s) => s.strategy_id));
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
	const resting = $derived(orders.filter((o) => o.status === 'ack' || o.status === 'approved' || o.status === 'partial_fill'));
	const byClass = $derived.by(() => {
		// exposure per asset class is not reported; group positions by instrument against the class caps we do know
		return positions.map((p) => ({ id: p.instrument_id, value: p.market_value_eur ?? p.quantity * p.avg_cost_eur }));
	});
	const haltClass = (scope: string) => (scope === 'portfolio' ? 'danger' : scope === 'broker' || scope.startsWith('source:') ? 'warn' : '');

	$effect(() => {
		if (!oInst && knownInstruments.length) oInst = knownInstruments[0];
	});

	// ---- /ws/trading topics (latest value per topic, coalesced per animation frame)
	const riskTopic = tradingWs.topic<RiskView>('risk');
	const positionsTopic = tradingWs.topic<{ cash_eur: number; equity_eur: number; positions: Position[] }>('positions');
	const haltsTopic = tradingWs.topic<{ halts: RiskView['halts'] }>('halts');
	const orderTopics = tradingWs.topics<OrderView>('order.>');
	const fillTopic = tradingWs.topic('fill.>');

	$effect(() => {
		const r = riskTopic.data;
		if (!r) return;
		risk = r;
		sample(r);
	});
	$effect(() => {
		const p = positionsTopic.data;
		if (!p) return;
		positions = p.positions;
		cash = p.cash_eur;
		equity = p.equity_eur;
	});
	$effect(() => {
		const h = haltsTopic.data;
		const r = untrack(() => risk);
		if (h && r) risk = { ...r, halts: h.halts, portfolio_halted: h.halts.some((x) => x.scope === 'portfolio') };
	});
	$effect(() => {
		if (!orderTopics.count) return;
		const merged = new Map(untrack(() => orders).map((o) => [o.intent_id, o]));
		for (const f of orderTopics.entries.values()) merged.set(f.data.intent_id, f.data);
		orders = [...merged.values()].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at)).slice(0, 200);
		now = Date.now();
	});
	$effect(() => {
		if (fillTopic.count) fillVersion = fillTopic.count;
	});

	function sample(r: RiskView) {
		const f = r.reported ?? r.paper;
		const t = Date.now();
		const added = pushSample(history, {
			t,
			equity: f.equity_eur,
			pnl: r.reported?.daily_pnl_eur ?? r.paper.pnl_eur ?? 0,
			drawdown: f.drawdown,
			exposure: f.gross_exposure_eur
		});
		if (added) {
			samples = history.toArray();
			saveHistory(history);
		}
	}

	async function refresh() {
		now = Date.now();
		try {
			const [r, p, o, s] = await Promise.all([live.risk(), live.positions(), live.orders(150), live.signals()]);
			risk = r;
			positions = p.positions;
			cash = p.cash_eur;
			equity = p.equity_eur;
			orders = o.orders;
			signals = s.signals;
			error = '';
			sample(r);
		} catch (e) {
			error = String(e);
		}
	}

	async function refreshStrategies() {
		try {
			strategiesView = await live.strategies();
			strategiesError = '';
		} catch (e) {
			strategiesError = e instanceof Error ? e.message : String(e);
		}
		try {
			limits = (await live.limits()).limits;
		} catch {
			// the Limits panel reports its own error
		}
	}

	async function flattenAll() {
		if (!confirm('FLATTEN ALL: every active strategy is flattened and paused (positions closed; no portfolio halt). Continue?')) return;
		flatBusy = true;
		try {
			const r = await live.flattenAll();
			flatMsg = `flattening ${r.flattening.length}: ${r.flattening.join(', ') || 'none active'}`;
			flatOk = true;
			await Promise.all([refresh(), refreshStrategies()]);
		} catch (e) {
			flatMsg = e instanceof Error ? e.message : String(e);
			flatOk = false;
		} finally {
			flatBusy = false;
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
			oMsg = `paper intent ${short(r.intent_id, 10)} sent (order.intent.paper → risk-engine)`;
			await refresh();
		} catch (e) {
			oErr = e instanceof Error ? e.message : String(e);
		} finally {
			oBusy = false;
		}
	}

	const statusClass = (s: string) =>
		s === 'filled' ? 'ok' : s === 'rejected' || s === 'canceled' ? 'danger' : s === 'ack' ? '' : 'warn';
	const signed = (v: number | null | undefined) => (v === null || v === undefined ? '' : v >= 0 ? 'pos' : 'neg');
	/** Last fill event carrying execution-cost fields (D-051), when control-api forwards them. */
	const costOf = (o: OrderView) => {
		const ev = [...o.events].reverse().find((e) => e.arrival_price != null || e.slippage_bps != null);
		return { arrival: o.arrival_price ?? ev?.arrival_price ?? null, slip: o.slippage_bps ?? ev?.slippage_bps ?? null };
	};

	onMount(() => {
		const stopPoll = poll(refresh, () => (tradingWs.live ? runtime.pollMs * 10 : runtime.pollMs));
		const stopStrat = poll(refreshStrategies, () => runtime.pollMs * 3);
		const stopBus = onBus(['signal.exposure.>', 'risk.halt.>', 'risk.resume.>'], () => void Promise.all([refresh(), refreshStrategies()]), 400);
		return () => {
			stopPoll();
			stopStrat();
			stopBus();
			riskTopic.close();
			positionsTopic.close();
			haltsTopic.close();
			orderTopics.close();
			fillTopic.close();
		};
	});
</script>

<ViewHeader title="Trading Desk" spec="51 §6.5" note="paper only; no live broker is connected · live on /ws/trading, REST polling as fallback" />
{#if error}<p class="badge danger">{error}</p>{/if}

<div class="killbar panel" class:halted={risk?.portfolio_halted}>
	<div class="modes">
		<span class="badge ok mode">PAPER</span>
		<button class="live" disabled title="no live broker connected (D-029): live trading is impossible from this UI">LIVE</button>
		<span class="badge" class:ok={tradingWs.live} class:warn={!tradingWs.live} title="/ws/trading · {tradingWs.frames} frames · {tradingWs.lagged} superseded{tradingWs.flushMs ? ` · flush ${tradingWs.flushMs} ms` : ''}">ws {tradingWs.status}</span>
		<span class="dim">Kill switch halts the portfolio and blocks orders (spec 40). Flatten closes every active strategy and pauses it (no halt).</span>
	</div>
	<div class="modes">
		<button class="flatten" onclick={flattenAll} disabled={flatBusy || !risk} title="POST /api/trading/flatten → ctl.risk.flatten">FLATTEN ALL</button>
		{#if flatMsg}<span class="badge" class:ok={flatOk} class:danger={!flatOk} title={flatMsg}>{flatMsg.slice(0, 80)}</span>{/if}
		<KillSwitch {risk} onchanged={refresh} />
	</div>
</div>

<div class="tiles">
	<div class="panel"><span class="dim">equity · {figuresSource}</span><b class="mono">{eur(figures?.equity_eur)}</b></div>
	<div class="panel"><span class="dim">deployable (net of payout quota)</span><b class="mono">{eur(figures?.deployable_eur)}</b></div>
	<div class="panel"><span class="dim">gross exposure</span><b class="mono">{eur(figures?.gross_exposure_eur)}</b></div>
	<div class="panel"><span class="dim">drawdown{limits ? ` · cap ${pct(limits.max_drawdown, 0)}` : ''}</span><b class="mono" class:neg={(figures?.drawdown ?? 0) > (limits?.max_drawdown ?? 0.1) * 0.5}>{pct(figures?.drawdown)}</b></div>
	<div class="panel">
		<span class="dim">{risk?.reported ? 'daily P&L' : 'P&L since start'}</span>
		<b class="mono {signed(risk?.reported?.daily_pnl_eur ?? risk?.paper.pnl_eur)}">{eur(risk?.reported?.daily_pnl_eur ?? risk?.paper.pnl_eur)}</b>
	</div>
	<div class="panel"><span class="dim">paper book equity / cash</span><b class="mono">{eur(equity)} <span class="dim small">/ {eur(cash)}</span></b></div>
</div>

<div class="grid charts">
	<section class="panel">
		<div class="panel-head"><h2>Equity & P&L</h2><span class="dim">{figuresSource}, {tradingWs.live ? 'from risk frames' : 'sampled by polling'} · {samples.length} pts</span></div>
		<Sparkline {times} values={samples.map((s) => s.equity)} format={eur} label="equity" fill />
		<Sparkline {times} values={samples.map((s) => s.pnl)} format={eur} baseline={0} tone="signed" label={risk?.reported ? 'daily P&L' : 'P&L since start'} />
	</section>
	<section class="panel">
		<div class="panel-head"><h2>Drawdown</h2><span class="dim">reported · peak-to-trough from sampled equity</span></div>
		<Sparkline {times} values={samples.map((s) => s.drawdown)} format={fmtPct} tone="danger" label="drawdown (reported)" fill />
		<Sparkline {times} values={drawdowns} format={fmtPct} tone="danger" label="drawdown (client, from equity samples)" />
	</section>
	<section class="panel">
		<div class="panel-head"><h2>Exposure vs caps</h2><span class="dim">{limits ? 'caps from GET /api/trading/limits' : 'caps not loaded'}</span></div>
		{#if figures}
			<div class="gauges">
				<Gauge label="gross exposure" value={figures.gross_exposure_eur} cap={limits?.portfolio_gross_cap ? limits.portfolio_gross_cap * figures.equity_eur : null} reference={figures.equity_eur} format={eur} />
				<Gauge label="drawdown" value={figures.drawdown} cap={limits?.max_drawdown ?? null} reference={1} format={fmtPct} />
				<Gauge label="daily loss" value={Math.max(0, -(risk?.reported?.daily_pnl_eur ?? risk?.paper.pnl_eur ?? 0)) / Math.max(1, figures.equity_eur)} cap={limits?.max_daily_loss ?? null} reference={1} format={fmtPct} />
				<Gauge label="deployable used" value={figures.equity_eur - figures.deployable_eur} reference={figures.equity_eur} format={eur} />
				{#each byClass.slice(0, 6) as p (p.id)}
					<Gauge label="{short(p.id, 10)} position" value={p.value} cap={limits?.max_position_frac ? limits.max_position_frac * figures.equity_eur : null} reference={figures.equity_eur} format={eur} />
				{/each}
			</div>
		{:else}
			<Empty producer="risk.state.* (risk service) or the paper book" />
		{/if}
	</section>
</div>

<StrategiesTable view={strategiesView} error={strategiesError} {now} onchanged={() => void Promise.all([refresh(), refreshStrategies()])} />
{#if strategiesError && derivedStrategies.length}
	<section class="panel strategies">
		<div class="panel-head"><h2>Strategies (derived fallback)</h2><span class="dim">from signal.exposure.* and order.* while GET /api/trading/strategies is unavailable</span></div>
		<table>
			<thead><tr><th>strategy</th><th>genome</th><th>horizon</th><th>last signal</th><th>instruments</th><th>mean |exposure|</th><th>orders</th><th>filled</th><th>rejected</th></tr></thead>
			<tbody>
				{#each derivedStrategies as s (s.strategy_id)}
					<tr>
						<td class="mono">{s.strategy_id}</td>
						<td class="mono dim">{s.genome_id || '—'}</td>
						<td class="mono">{s.horizon || '—'}</td>
						<td class="mono dim">{s.last_signal_ms === null ? '—' : `${ago(new Date(now - s.last_signal_ms).toISOString(), now)} ago`}</td>
						<td class="mono">{s.instruments}</td>
						<td class="mono">{s.mean_abs_exposure.toFixed(2)}</td>
						<td class="mono">{s.orders}</td>
						<td class="mono">{s.filled}</td>
						<td class="mono {s.rejected ? 'neg' : ''}">{s.rejected}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</section>
{/if}

<div class="grid two">
	<Limits onchanged={refreshStrategies} />
	<Pnl strategies={strategyIds} version={fillVersion} />
	<Reconciliation {now} version={fillVersion} />

	<section class="panel">
		<div class="panel-head"><h2>Positions</h2><span class="dim">paper book, EUR{tradingWs.live ? ' · positions frames' : ' · polled'}</span></div>
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
		<div class="panel-head"><h2>Manual paper order</h2><span class="dim">order.intent.paper → risk-engine → broker-gateway (D-051)</span></div>
		<form class="ticket" onsubmit={submitOrder}>
			<label class="wide">instrument
				<input list="instruments" bind:value={oInst} placeholder="instrument UUID" />
				<datalist id="instruments">{#each knownInstruments as i}<option value={i}></option>{/each}</datalist>
			</label>
			<label>side
				<select bind:value={oSide}><option value="buy">buy</option><option value="sell">sell</option></select>
			</label>
			<label>quantity <input type="number" min="0" step="any" bind:value={oQty} /></label>
			<label>limit (optional; rests as ack until crossed) <input type="number" min="0" step="any" bind:value={oLimit} placeholder="market" /></label>
			<div class="wide row">
				<button type="submit" disabled={oBusy || !oInst || risk?.portfolio_halted}>
					{oBusy ? 'sending…' : `Paper ${oSide}`}
				</button>
				{#if risk?.portfolio_halted}<span class="badge danger">halted</span>{/if}
				{#if oMsg}<span class="badge ok">{oMsg}</span>{/if}
				{#if oErr}<span class="badge danger">{oErr}</span>{/if}
			</div>
		</form>
		<h3 class="sub">Active halts <span class="dim">({risk?.halts.length ?? 0}; scopes portfolio · daily_loss · strategy:&lt;id&gt; · source:&lt;id&gt; · broker)</span></h3>
		{#if risk?.halts.length}
			<table>
				<thead><tr><th>scope</th><th>reason</th><th>since</th><th>manual</th></tr></thead>
				<tbody>
					{#each risk.halts as h (h.scope)}
						<tr>
							<td><span class="badge {haltClass(h.scope)}">{h.scope}</span></td>
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
		<div class="panel-head">
			<h2>Orders & fills</h2>
			<span class="dim">{tradingWs.live ? 'order.<intent> frames' : 'polled'}{resting.length ? ` · ${resting.length} resting/open` : ''}</span>
		</div>
		{#if orders.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>intent</th><th>instrument</th><th>side</th><th>qty</th><th>algo</th><th>status</th><th>fill</th><th>arrival</th><th>slippage</th><th>fees</th><th>strategies</th><th>age</th></tr></thead>
					<tbody>
						{#each orders as o (o.intent_id)}
							{@const c = costOf(o)}
							<tr title={o.reason} class:resting={o.status === 'ack'}>
								<td class="mono">{short(o.intent_id, 10)}</td>
								<td class="mono" title={o.instrument_id}>{short(o.instrument_id, 10)}</td>
								<td class={o.side === 'buy' ? 'pos' : 'neg'}>{o.side}</td>
								<td class="mono">{o.quantity}</td>
								<td class="mono dim">{o.exec_algo}{o.limit_price ? ` @${o.limit_price}` : ''}</td>
								<td>
									<span class="badge {statusClass(o.status)}" title={o.status === 'ack' ? 'resting limit order (fills passively when crossed, D-051)' : ''}>{o.status}{o.status === 'ack' && o.limit_price ? ' · resting' : ''}</span>
									<span class="badge">{o.mode}</span>
								</td>
								<td class="mono">{o.filled_quantity ? `${o.filled_quantity} @ ${o.avg_fill_price?.toFixed(4)}` : '—'}</td>
								<td class="mono dim">{c.arrival === null ? '—' : c.arrival.toFixed(4)}</td>
								<td class="mono {c.slip === null ? 'dim' : c.slip > 0 ? 'neg' : 'pos'}">{c.slip === null ? '—' : `${c.slip.toFixed(1)} bps`}</td>
								<td class="mono dim">{o.fees ? eur(o.fees) : '—'}</td>
								<td class="mono dim">{o.strategy_ids.join(', ')}</td>
								<td class="mono dim">{ago(o.updated_at, now)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="dim small">arrival / slippage come from <code>OrderEvent.arrival_price</code> / <code>slippage_bps</code> (D-051) when control-api's order view forwards them; shown as — otherwise.</p>
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
	.flatten {
		background: var(--warn);
		color: #000;
		border: none;
		font-weight: 700;
		letter-spacing: 0.06em;
		padding: 8px 14px;
	}
	tr.resting td {
		background: color-mix(in srgb, var(--warn) 6%, transparent);
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
