<script lang="ts">
	// Strategies table (spec 51 §6.5, D-051): lifecycle status, sleeve, weight, drawdown, shadow
	// progress, gate checklist with missing items, latest signal; pause/resume/flatten/promote/shadow.
	import { ago, eur, live, pct, short, type StrategiesView, type StrategyAction, type StrategyView } from './api';

	interface Props {
		view: StrategiesView | null;
		error?: string;
		now: number;
		onchanged?: () => void;
	}
	let { view, error = '', now, onchanged }: Props = $props();

	let msg = $state<Record<string, { ok: boolean; text: string }>>({});
	let busy = $state<Record<string, boolean>>({});
	let gateOpen = $state<string | null>(null);

	/** Actions sensible per lifecycle state (the risk-engine enforces; this avoids pointless 409s). */
	function actionsFor(s: StrategyView): StrategyAction[] {
		switch (s.status) {
			case 'active':
				return ['pause', 'flatten', 'shadow'];
			case 'paused':
				return ['resume', 'flatten', 'shadow'];
			case 'shadow':
				return ['promote', 'pause'];
			case 'flattening':
				return ['pause'];
			default:
				return ['pause', 'resume', 'flatten', 'promote', 'shadow'];
		}
	}

	const CONFIRM: Record<StrategyAction, string> = {
		pause: 'Pause: the strategy stops receiving allocations (signals are ignored).',
		resume: 'Resume allocations for this strategy.',
		flatten: 'Flatten: close its sub-book positions and pause it.',
		promote: 'Promote shadow → active (the gate must be green; D-029: no live sleeve).',
		shadow: 'Move to shadow: virtual book only, gate progress starts counting.'
	};

	async function act(s: StrategyView, action: StrategyAction) {
		if (!confirm(`${action} ${s.strategy_id}?\n${CONFIRM[action]}\nAudited by the risk-engine (actor ui).`)) return;
		busy = { ...busy, [s.strategy_id]: true };
		try {
			await live.strategyAction(s.strategy_id, action);
			msg = { ...msg, [s.strategy_id]: { ok: true, text: `${action} ok` } };
			onchanged?.();
		} catch (e) {
			msg = { ...msg, [s.strategy_id]: { ok: false, text: e instanceof Error ? e.message : String(e) } };
		} finally {
			busy = { ...busy, [s.strategy_id]: false };
		}
	}

	const statusClass = (s: string) => (s === 'active' ? 'ok' : s === 'shadow' ? '' : s === 'paused' ? 'warn' : 'danger');
	const signed = (v: number | null | undefined) => (v === null || v === undefined ? '' : v >= 0 ? 'pos' : 'neg');
	const meanAbs = (s: StrategyView) => {
		const xs = s.signal?.exposures.map((e) => Math.abs(e.mean)) ?? [];
		return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
	};
	const gateTitle = (s: StrategyView) => (s.gate ? (s.gate.green ? 'gate green' : `gate not green — missing: ${s.gate.missing.join(', ')}`) : 'no gate data');
</script>

<section class="panel strategies">
	<div class="panel-head">
		<h2>Strategies <span class="dim">({view?.count ?? 0})</span></h2>
		<span class="dim">
			GET /api/trading/strategies · source {view?.source ?? '—'}
			{#if view}· {Object.entries(view.status_counts).map(([k, v]) => `${v} ${k}`).join(', ')}{/if}
			{#if view?.signals_without_state.length}· <span class="badge warn">{view.signals_without_state.length} signals without lifecycle state</span>{/if}
		</span>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if view?.strategies.length}
		<div class="scroll tall">
			<table>
				<thead>
					<tr><th>strategy</th><th>status</th><th>sleeve</th><th>weight</th><th>P&amp;L</th><th>drawdown</th><th>shadow</th><th>gate</th><th>last signal</th><th>mean |exposure|</th><th>actions</th></tr>
				</thead>
				<tbody>
					{#each view.strategies as s (s.strategy_id)}
						<tr title={s.last_reason}>
							<td class="mono">{s.strategy_id}<br /><span class="dim">{s.signal?.genome_id ?? ''}{s.signal?.horizon ? ` · ${s.signal.horizon}` : ''}</span></td>
							<td>
								<span class="badge {statusClass(s.status)}">{s.status}</span>
								{#if s.since}<br /><span class="dim mono">{ago(s.since, now)}</span>{/if}
							</td>
							<td class="mono">{eur(s.sleeve_eur)}</td>
							<td class="mono">{pct(s.weight, 1)}</td>
							<td class="mono {signed(s.pnl_eur)}" title="peak {eur(s.peak_pnl_eur)}">{eur(s.pnl_eur)}</td>
							<td>
								<div class="bar" title="max {pct(s.max_drawdown)}"><div class:warnbg={s.drawdown > 0.03} style="width: {Math.min(100, 100 * s.drawdown / 0.05)}%"></div></div>
								<span class="mono">{pct(s.drawdown)}</span>
							</td>
							<td class="mono dim">
								{#if s.status === 'shadow' || s.shadow.since}
									{s.shadow.days === null ? '—' : `${s.shadow.days.toFixed(1)} d`} · {s.shadow.horizons_completed} horizons
								{:else}
									not in shadow
								{/if}
							</td>
							<td>
								{#if s.gate}
									<button class="gate" class:ok={s.gate.green} class:warn={!s.gate.green} title={gateTitle(s)} onclick={() => (gateOpen = gateOpen === s.strategy_id ? null : s.strategy_id)}>
										{s.gate.green ? 'green' : `${s.gate.missing.length} missing`}
									</button>
								{:else}
									<span class="dim">—</span>
								{/if}
							</td>
							<td class="mono dim">{s.signal?.as_of ? `${ago(s.signal.as_of, now)} ago` : '—'}</td>
							<td>
								<div class="bar"><div style="width: {100 * Math.min(1, meanAbs(s))}%"></div></div>
								<span class="mono">{meanAbs(s).toFixed(2)}</span>
							</td>
							<td class="actions">
								{#each actionsFor(s) as a}
									{@const blocked = a === 'promote' && s.gate && !s.gate.green}
									<button disabled={busy[s.strategy_id] || !!blocked} onclick={() => act(s, a)} title={blocked ? `promotion blocked — missing: ${s.gate?.missing.join(', ')}` : `POST /api/trading/strategies/${s.strategy_id}/${a}`}>{a}</button>
								{/each}
								{#if msg[s.strategy_id]}<span class="badge" class:ok={msg[s.strategy_id].ok} class:danger={!msg[s.strategy_id].ok} title={msg[s.strategy_id].text}>{msg[s.strategy_id].text.slice(0, 60)}</span>{/if}
							</td>
						</tr>
						{#if gateOpen === s.strategy_id && s.gate}
							<tr class="detail">
								<td colspan="11">
									<ul class="gate-list">
										{#each s.gate.items as it (it.name)}
											<li>
												<span class="badge" class:ok={it.pass === true} class:danger={it.pass === false && !it.advisory} class:warn={it.pass === false && it.advisory}>{it.pass === null ? 'unknown' : it.pass ? 'pass' : 'fail'}</span>
												<span class="mono">{it.name}</span>{#if it.advisory}<span class="dim"> (advisory)</span>{/if}
												<span class="dim">— {it.detail}</span>
											</li>
										{/each}
									</ul>
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		</div>
	{:else if !error}
		<p class="dim">no strategies (risk-engine <code>ctl.risk.strategy list</code> / <code>ledger.strategy_state</code>)</p>
	{/if}
</section>

<style>
	.strategies {
		margin-bottom: 12px;
	}
	.tall {
		max-height: 440px;
	}
	.bar {
		display: inline-block;
		width: 60px;
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
	.bar > .warnbg {
		background: var(--warn);
	}
	.gate {
		font-size: 11px;
		padding: 1px 6px;
		border-radius: 9px;
	}
	.gate.ok {
		color: var(--ok);
		border-color: var(--ok);
	}
	.gate.warn {
		color: var(--warn);
		border-color: var(--warn);
	}
	.actions {
		white-space: nowrap;
	}
	.actions button {
		font-size: 11px;
		padding: 1px 6px;
		margin-right: 3px;
	}
	.detail td {
		background: var(--bg-inset);
	}
	.gate-list {
		margin: 0;
		padding: 0 0 0 4px;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 2px;
		font-size: 12px;
	}
	.pos {
		color: var(--ok);
	}
	.neg {
		color: var(--danger);
	}
</style>
