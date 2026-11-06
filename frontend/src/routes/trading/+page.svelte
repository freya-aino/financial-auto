<script lang="ts">
	import Planned from '../../lib/Planned.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
</script>

<ViewHeader title="Trading Desk" spec="51 §6.5" note="paper mode; ledger and risk not wired yet (specs 40, 42)" />

<div class="killbar panel">
	<div>
		<h2>Kill switch</h2>
		<span class="dim">Flattens all books and halts every strategy (spec 40).</span>
	</div>
	<button
		class="kill"
		disabled
		title="Disabled until fa-risk and broker-gateway are implemented (POST /api/trading/kill-switch)"
	>
		KILL SWITCH
	</button>
</div>

<div class="grid cols">
	<Planned
		title="Positions"
		items={['Paper / shadow / live book toggle', 'Per-strategy attribution', 'Exposure vs caps per asset class and venue']}
		endpoints={['GET /api/trading/positions']}
	/>
	<Planned
		title="P&L & risk"
		items={[
			'P&L and drawdown charts (EUR)',
			'Limit headroom gauges: drawdown, daily loss, concentration, leverage',
			'Liquidity cap net of payout quota'
		]}
		endpoints={['GET /api/trading/pnl', 'GET /api/trading/limits', 'WS /ws/trading']}
	/>
	<Planned
		title="Strategies & promotion"
		items={[
			'Status, sleeve, shadow progress (calendar days, horizons)',
			'Gate checklist: min shadow time, deflated Sharpe, PBO',
			'Pause / resume / flatten / promote'
		]}
		endpoints={['GET /api/trading/strategies']}
	/>
	<Planned
		title="Blotter & reconciliation"
		items={['Orders and fills with slippage', 'Risk events feed', 'Reconciliation status and diffs', 'Manual order ticket with risk-check preview']}
		endpoints={['GET /api/trading/orders', 'GET /api/trading/reconciliation', 'GET /api/audit']}
	/>
</div>

<style>
	.killbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 12px;
		border-color: var(--danger);
	}

	.kill {
		background: var(--danger);
		color: #fff;
		border: none;
		font-weight: 700;
		letter-spacing: 0.08em;
		padding: 8px 18px;
	}

	.cols {
		grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
	}
</style>
