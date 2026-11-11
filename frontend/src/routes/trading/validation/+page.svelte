<script lang="ts">
	// Holdouts & shadow (spec 41 §6.4, spec 40 §6.6 item 4): lock a completed holdout segment,
	// spend it on one promotion replay of a candidate, and compare a shadow strategy's realised
	// Sharpe with its backtest. Each mutation: validate → confirm → show the API's answer.
	import { onMount } from 'svelte';
	import Empty from '../../../lib/Empty.svelte';
	import Validation from '../../../lib/Validation.svelte';
	import ViewHeader from '../../../lib/ViewHeader.svelte';
	import { live, num, short, type Holdout, type StrategyView } from '../../../lib/api';
	import { errText } from '../../../lib/console';
	import { TIMESCALES, evaluateConfirmText, holdoutConfirmText, holdoutLockBody, operatorApi, type ShadowView } from '../../../lib/operator';

	let holdouts = $state<Holdout[]>([]);
	let holdoutError = $state('');
	let strategies = $state<StrategyView[]>([]);
	let msg = $state('');
	let msgOk = $state(true);
	let busy = $state(false);

	// lock form
	let lTs = $state('1h');
	let lStart = $state('');
	let lEnd = $state('');
	let lNote = $state('');

	// evaluate form
	let eHoldout = $state('');
	let eCandidate = $state('');
	let eSnapshot = $state('');
	let eBudget = $state('');
	let eSeed = $state('');
	let lastCandidate = $state('');

	// shadow
	let sStrategy = $state('');
	let sMin = $state('');
	let shadow = $state<ShadowView | null>(null);
	let shadowError = $state('');

	const shadowStrategies = $derived(strategies.filter((s) => s.status === 'shadow' || s.status === 'active' || s.status === 'holdout_passed'));

	async function refresh() {
		const [h, s] = await Promise.allSettled([live.holdout(), live.strategies()]);
		if (h.status === 'fulfilled') ((holdouts = h.value.holdouts), (holdoutError = ''));
		else holdoutError = errText(h.reason);
		if (s.status === 'fulfilled') strategies = s.value.strategies;
	}

	async function run(question: string, call: () => Promise<unknown>, done: string, after?: () => void) {
		if (!confirm(question)) return;
		busy = true;
		try {
			const r = await call();
			msg = `${done}: ${JSON.stringify(r)}`;
			msgOk = true;
			after?.();
			await refresh();
		} catch (e) {
			msg = `${done} failed: ${errText(e)}`;
			msgOk = false;
		} finally {
			busy = false;
		}
	}

	function lock(ev: SubmitEvent) {
		ev.preventDefault();
		const b = holdoutLockBody(lTs, lStart, lEnd, lNote);
		if ('error' in b) return ((msg = b.error), (msgOk = false));
		void run(holdoutConfirmText(b.body), () => operatorApi.lockHoldout(b.body), 'holdout locked');
	}

	function evaluate(ev: SubmitEvent) {
		ev.preventDefault();
		const h = holdouts.find((x) => String(x.id) === eHoldout);
		if (!h) return ((msg = 'pick a locked holdout'), (msgOk = false));
		const candidate = eCandidate.trim();
		if (!candidate) return ((msg = 'candidate (genome text or gp-… strategy id) is required'), (msgOk = false));
		const budget = eBudget.trim() ? Number(eBudget) : undefined;
		const seed = eSeed.trim() ? Number(eSeed) : undefined;
		if ((budget !== undefined && !(budget >= 0)) || (seed !== undefined && !Number.isInteger(seed))) return ((msg = 'budget_ms must be ≥ 0 and seed an integer'), (msgOk = false));
		const body = { candidate, ...(eSnapshot.trim() ? { snapshot: eSnapshot.trim() } : {}), ...(budget !== undefined ? { budget_ms: budget } : {}), ...(seed !== undefined ? { seed } : {}) };
		void run(evaluateConfirmText({ id: String(h.id), timescale: String(h.timescale ?? '') }, candidate), () => operatorApi.evaluateHoldout(String(h.id), body), 'holdout evaluation queued', () => (lastCandidate = candidate));
	}

	async function loadShadow(ev?: SubmitEvent) {
		ev?.preventDefault();
		if (!sStrategy.trim()) return;
		try {
			shadow = await operatorApi.shadow(sStrategy.trim(), sMin.trim() ? Number(sMin) : undefined);
			shadowError = '';
		} catch (e) {
			shadow = null;
			shadowError = errText(e);
		}
	}

	onMount(() => {
		void refresh();
	});

	const ratioClass = (r: number | null | undefined) => (r === null || r === undefined ? '' : r >= 0.5 ? 'ok' : r >= 0 ? 'warn' : 'danger');
</script>

<ViewHeader title="Holdouts & shadow" spec="41 §6.4 · 40 §6.6" note="POST /api/validation/holdout (+ /:id/evaluate), GET /api/validation/shadow/:strategy" />

{#if msg}<p class="badge" class:ok={msgOk} class:danger={!msgOk}>{msg}</p>{/if}

<div class="grid cols">
	<section class="panel">
		<div class="panel-head"><h2>Locked holdouts <span class="dim">({holdouts.length})</span></h2><span class="dim">immutable once locked</span></div>
		{#if holdoutError}<p class="badge danger">{holdoutError}</p>{/if}
		{#if holdouts.length}
			<table>
				<thead><tr><th>id</th><th>timescale</th><th>start</th><th>end</th><th>locked</th><th>note</th></tr></thead>
				<tbody>
					{#each holdouts as h (String(h.id))}
						<tr class:sel={String(h.id) === eHoldout} onclick={() => (eHoldout = String(h.id))}>
							<td class="mono">{short(String(h.id), 8)}</td>
							<td class="mono">{h.timescale}</td>
							<td class="mono dim">{h.start?.slice(0, 16)}</td>
							<td class="mono dim">{h.end?.slice(0, 16)}</td>
							<td class="mono dim">{h.locked_at?.slice(0, 16)}</td>
							<td>{h.note ?? ''}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else if !holdoutError}
			<Empty producer="validation.holdout (lock one below)" hint="" />
		{/if}

		<h3 class="sub">Lock a holdout</h3>
		<form class="form" onsubmit={lock}>
			<label>timescale <select bind:value={lTs}>{#each TIMESCALES as t}<option value={t}>{t}</option>{/each}</select></label>
			<label>start <input type="datetime-local" bind:value={lStart} /></label>
			<label>end (not in the future) <input type="datetime-local" bind:value={lEnd} /></label>
			<label class="wide">note <input bind:value={lNote} placeholder="why this segment" /></label>
			<div class="wide"><button type="submit" class="danger" disabled={busy}>Lock holdout…</button> <span class="dim small">409 when it overlaps a locked range of the timescale</span></div>
		</form>
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Evaluate on a holdout</h2><span class="dim">one evo.eval.strategy.holdout job</span></div>
		<form class="form" onsubmit={evaluate}>
			<label class="wide">holdout
				<select bind:value={eHoldout}>
					<option value="">— pick —</option>
					{#each holdouts as h (String(h.id))}<option value={String(h.id)}>{h.timescale} {h.start?.slice(0, 10)} → {h.end?.slice(0, 10)} ({short(String(h.id), 8)})</option>{/each}
				</select>
			</label>
			<label class="wide">candidate (genome inline table or gp-… strategy id)
				<textarea class="mono" rows="3" bind:value={eCandidate} placeholder={'gp-… or { id = "…", … }'}></textarea>
			</label>
			<label>snapshot (optional) <input class="mono" bind:value={eSnapshot} /></label>
			<label>budget ms (optional) <input type="number" min="0" bind:value={eBudget} /></label>
			<label>seed (optional) <input type="number" step="1" bind:value={eSeed} /></label>
			<div class="wide"><button type="submit" class="danger" disabled={busy || !eHoldout}>Evaluate…</button> <span class="dim small">the result lands in the candidate's reports → holdout</span></div>
		</form>
		{#if lastCandidate}<Validation candidate={lastCandidate} />{/if}
	</section>

	<section class="panel wide">
		<div class="panel-head"><h2>Shadow vs backtest</h2><span class="dim">realised shadow Sharpe (fill-marked) ÷ latest backtest Sharpe</span></div>
		<form class="controls" onsubmit={loadShadow}>
			<input class="mono" list="shadow-strategies" bind:value={sStrategy} placeholder="strategy id" size="36" aria-label="strategy" />
			<datalist id="shadow-strategies">{#each shadowStrategies as s}<option value={s.strategy_id}>{s.status}</option>{/each}</datalist>
			<input type="number" min="1" bind:value={sMin} placeholder="min horizons (20)" aria-label="min horizons" />
			<button type="submit" disabled={!sStrategy.trim()}>Check</button>
		</form>
		{#if shadowStrategies.length}
			<div class="chips">
				{#each shadowStrategies as s (s.strategy_id)}
					<button class="chip" class:sel={sStrategy === s.strategy_id} onclick={() => ((sStrategy = s.strategy_id), loadShadow())}>{short(s.strategy_id, 14)} <span class="dim">{s.status}</span></button>
				{/each}
			</div>
		{/if}
		{#if shadowError}<p class="badge danger">{shadowError}</p>{/if}
		{#if shadow}
			{@const c = shadow.consistency}
			<div class="grid two">
				<table>
					<tbody>
						<tr><th>available</th><td>{#if c.available}<span class="badge ok">yes</span>{:else}<span class="badge warn">no</span> <span class="dim">{c.reason ?? ''}</span>{/if}</td></tr>
						<tr><th>horizons</th><td class="mono">{c.n_horizons} / {c.min_horizons}</td></tr>
						<tr><th>backtest Sharpe</th><td class="mono">{num(c.backtest_sharpe ?? null, 3)} <span class="dim">{c.backtest_trial_at ? `trial ${c.backtest_trial_at.slice(0, 16)}` : ''}</span></td></tr>
						<tr><th>shadow Sharpe</th><td class="mono">{num(c.shadow_sharpe ?? null, 3)} <span class="dim">{c.n_shadow_obs ? `${c.n_shadow_obs} bars of ${c.bar}` : ''}{c.n_fills !== undefined ? ` · ${c.n_fills} fills` : ''}</span></td></tr>
						<tr><th>ratio</th><td>{#if c.ratio !== undefined && c.ratio !== null}<span class="badge {ratioClass(c.ratio)}">{c.ratio.toFixed(2)}</span>{:else}—{/if}</td></tr>
						<tr><th>source</th><td class="dim small">{c.source}</td></tr>
					</tbody>
				</table>
				<pre class="mono state">{JSON.stringify(shadow.state, null, 2)}</pre>
			</div>
		{/if}
	</section>
</div>

<style>
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
	}
	.wide {
		grid-column: 1 / -1;
	}
	.two {
		grid-template-columns: 1fr 1fr;
	}
	.sub {
		font-size: 12px;
		margin: 14px 0 6px;
	}
	.form {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 8px;
	}
	.form input,
	.form select,
	.form textarea {
		width: 100%;
		min-width: 0;
	}
	.form label {
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
		color: var(--text-dim);
	}
	.controls {
		display: flex;
		gap: 6px;
		align-items: center;
		flex-wrap: wrap;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin: 6px 0;
	}
	.chip {
		font-size: 11px;
	}
	.sel {
		background: var(--bg-inset);
	}
	tbody tr {
		cursor: pointer;
	}
	.state {
		margin: 0;
		max-height: 260px;
		overflow: auto;
		font-size: 11px;
	}
	.small {
		font-size: 11px;
	}
</style>
