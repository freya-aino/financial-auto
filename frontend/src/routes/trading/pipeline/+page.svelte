<script lang="ts">
	// Strategy pipeline (D-071, spec 35 §6.4, spec 41 §6.4): candidate → holdout_passed → shadow →
	// active → retired, run by risk-engine every 60 s. Per evolved strategy: stage, last_reason,
	// selection PBO / DSR and the single holdout outcome; the round's status (policy, holdouts,
	// actions, rejections, managed live set); locked holdouts; the paper allocator tournament.
	import { onMount } from 'svelte';
	import Empty from '../../../lib/Empty.svelte';
	import Validation from '../../../lib/Validation.svelte';
	import ViewHeader from '../../../lib/ViewHeader.svelte';
	import { live, num, pct, poll, type StrategiesView } from '../../../lib/api';
	import { since } from '../../../lib/alerts';
	import { consoleApi, errText, type TournamentView } from '../../../lib/console';
	import { PIPELINE_STAGES, evidence, rankAllocators, stageCounts, verdict, type PipelinePolicy, type StrategyWithPipeline } from '../../../lib/riskdesk';
	import { runtime } from '../../../lib/runtime.svelte';
	import { onBus } from '../../../lib/ws.svelte';

	let strategies = $state<StrategiesView | null>(null);
	let strategiesError = $state('');
	let status = $state<Record<string, unknown> | null>(null);
	let statusError = $state('');
	let tournament = $state<TournamentView | null>(null);
	let tournamentError = $state('');
	let showAll = $state(false);
	let candidate = $state('');
	let runBusy = $state(false);
	let runMsg = $state('');
	let runOk = $state(true);
	let now = $state(Date.now());

	const rows = $derived((strategies?.strategies ?? []) as StrategyWithPipeline[]);
	const evolved = $derived(rows.filter((s) => s.pipeline || s.evolved));
	const shown = $derived(showAll ? rows : evolved);
	const counts = $derived(stageCounts(evolved));
	const policy = $derived((status?.policy ?? {}) as PipelinePolicy);
	const ranked = $derived(rankAllocators(tournament?.entries ?? [], tournament?.active ?? null));
	const holdouts = $derived(Object.entries((status?.holdouts ?? {}) as Record<string, Record<string, unknown>>));
	const actions = $derived((status?.actions ?? []) as Record<string, unknown>[]);
	const rejected = $derived((status?.rejected ?? []) as Record<string, unknown>[]);
	const waiting = $derived(Object.entries((status?.waiting ?? {}) as Record<string, unknown>));
	const liveSet = $derived((status?.live_set ?? null) as Record<string, unknown> | null);

	const stageClass = (s: string) => (s === 'active' ? 'ok' : s === 'retired' ? 'danger' : s === 'candidate' || s === 'holdout_passed' || s === 'shadow' ? 'warn' : '');
	const passBadge = (v: boolean | null) => (v === null ? '' : v ? 'ok' : 'danger');
	const ms = (v: unknown) => (typeof v === 'number' ? new Date(v).toISOString() : null);

	async function refresh() {
		now = Date.now();
		const [s, p, t] = await Promise.allSettled([live.strategies(), consoleApi.pipeline(), consoleApi.allocators()]);
		if (s.status === 'fulfilled') {
			strategies = s.value;
			strategiesError = '';
		} else strategiesError = errText(s.reason);
		if (p.status === 'fulfilled') {
			status = p.value;
			statusError = '';
		} else statusError = errText(p.reason);
		if (t.status === 'fulfilled') {
			tournament = t.value;
			tournamentError = '';
		} else tournamentError = errText(t.reason);
	}

	async function runRound() {
		if (!confirm('Run one strategy-pipeline round now (ctl.risk.pipeline {action: run})? It may request holdout evaluations, admit or retire strategies and rewrite the managed live set. Audited as pipeline.run.')) return;
		runBusy = true;
		try {
			status = await consoleApi.pipelineRun();
			runMsg = `round done: ${((status.actions as unknown[]) ?? []).length} actions`;
			runOk = true;
		} catch (e) {
			runMsg = errText(e);
			runOk = false;
		} finally {
			runBusy = false;
			void refresh();
		}
	}

	onMount(() => {
		const stop = poll(refresh, () => runtime.pollMs * 10);
		const off = onBus(['risk.halt.>', 'risk.resume.>'], refresh, 1000);
		return () => {
			stop();
			off();
		};
	});
</script>

<ViewHeader title="Strategy pipeline" spec="35 §6.4 · 41 §6.4 · 40 §6.2" note="risk-engine lifecycle round (ctl.risk.pipeline), state in ledger.strategy_state, holdouts locked per timescale" />

<div class="stages panel">
	{#each PIPELINE_STAGES as st, i}
		<div class="stage">
			<span class="badge {stageClass(st)}">{st}</span>
			<b class="mono">{counts[st] ?? 0}</b>
		</div>
		{#if i < PIPELINE_STAGES.length - 1}<span class="dim arrow">→</span>{/if}
	{/each}
	{#if counts.paused}<span class="badge">paused · {counts.paused}</span>{/if}
	<span class="spacer"></span>
	{#if status}
		<span class="dim mono" title="last round">round {since(ms(status.at_ms), now)} ago · {status.took_ms ?? '—'} ms · {status.enabled ? 'enabled' : 'DISABLED'}</span>
	{/if}
	<button onclick={runRound} disabled={runBusy}>{runBusy ? 'running…' : 'run round now'}</button>
	{#if runMsg}<span class="badge" class:ok={runOk} class:danger={!runOk}>{runMsg}</span>{/if}
	{#if statusError}<span class="badge danger" title="GET /api/trading/pipeline">{statusError}</span>{/if}
</div>

<section class="panel">
	<div class="panel-head">
		<h2>Evolved strategies <span class="dim">({evolved.length}{showAll ? ` of ${rows.length}` : ''})</span></h2>
		<label class="dim"><input type="checkbox" bind:checked={showAll} /> include hand-written baselines</label>
	</div>
	{#if strategiesError}<p class="badge danger">{strategiesError}</p>{/if}
	{#if shown.length}
		<div class="scroll tall">
			<table>
				<thead>
					<tr>
						<th>strategy</th><th>stage</th><th>since</th><th>last reason</th>
						<th title="selection PBO ≤ pbo_max">PBO</th><th title="selection deflated Sharpe ≥ dsr_min_candidate">DSR sel.</th>
						<th title="single holdout evaluation (DSR ≥ holdout_min_dsr, Sharpe > 0, n ≥ holdout_min_obs)">holdout</th>
						<th>gate</th><th>P&amp;L</th>
					</tr>
				</thead>
				<tbody>
					{#each shown as s (s.strategy_id)}
						{@const e = evidence(s, policy)}
						{@const pboOk = verdict(e.selection.pbo, policy.pbo_max, 'max')}
						{@const dsrOk = verdict(e.selection.dsr, policy.dsr_min_candidate, 'min')}
						<tr onclick={() => (candidate = s.strategy_id)} class:sel={candidate === s.strategy_id} title="click: load its trial reports in the Validation card">
							<td class="mono" title={(s.pipeline?.genome as string) ?? ''}>{s.strategy_id}<div class="dim small">{s.pipeline ? `${s.pipeline.timescale} · gen ${String(s.pipeline.generation).slice(-8)}` : 'baseline'}</div></td>
							<td><span class="badge {stageClass(s.status)}">{s.status}</span>{#if s.prev_status}<div class="dim small">from {s.prev_status}</div>{/if}</td>
							<td class="mono dim">{since(s.since, now)}</td>
							<td class="reason">{s.last_reason || '—'}{#if s.pipeline?.note}<div class="dim small">{s.pipeline.note}</div>{/if}</td>
							<td><span class="badge {passBadge(pboOk)}" title={pboOk === null ? 'unknown blocks (D-071)' : `max ${policy.pbo_max}`}>{num(e.selection.pbo, 2)}</span></td>
							<td><span class="badge {passBadge(dsrOk)}" title={dsrOk === null ? 'unknown blocks' : `min ${policy.dsr_min_candidate}`}>{num(e.selection.dsr, 2)}</span></td>
							<td>
								{#if s.pipeline?.holdout}
									<span class="badge {passBadge(e.holdout.pass)}">DSR {num(e.holdout.dsr, 2)} · SR {num(e.holdout.sharpe, 3)} · n {e.holdout.n_obs ?? '—'}</span>
								{:else if s.pipeline?.holdout_job_id}
									<span class="badge warn" title={String(s.pipeline.holdout_job_id)}>requested {since(ms(s.pipeline.holdout_requested_ms), now)} ago</span>
								{:else}
									<span class="dim">{s.pipeline ? `not requested (${s.pipeline.holdout_attempts ?? 0} attempts)` : 'n/a'}</span>
								{/if}
							</td>
							<td>
								{#if s.gate}<span class="badge" class:ok={s.gate.green} class:danger={!s.gate.green} title={s.gate.items.map((i) => `${i.name}: ${i.pass === null ? 'unknown' : i.pass ? 'pass' : 'fail'} (${i.detail})`).join('\n')}>{s.gate.green ? 'green' : `${s.gate.missing.length} missing`}</span>{:else}—{/if}
							</td>
							<td class="mono" class:pos={s.pnl_eur > 0} class:neg={s.pnl_eur < 0}>{num(s.pnl_eur, 2)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else if strategies}
		<p class="dim">No evolved strategy has entered the pipeline yet (candidates need known PBO ≤ {policy.pbo_max ?? 'pbo_max'} and selection DSR ≥ {policy.dsr_min_candidate ?? 'dsr_min_candidate'}).</p>
	{:else}
		<Empty producer="GET /api/trading/strategies (risk-engine)" hint="" />
	{/if}
	<p class="dim small">Lifecycle actions (pause / resume / flatten / promote / retire) stay in the Trading Desk strategies table with its gate checklist.</p>
</section>

<div class="grid cols">
	<section class="panel">
		<div class="panel-head"><h2>Last round</h2></div>
		{#if status}
			<h3 class="sub">Holdouts per timescale</h3>
			<table>
				<thead><tr><th>timescale</th><th>state</th><th>window</th></tr></thead>
				<tbody>
					{#each holdouts as [ts, h]}
						<tr>
							<td class="mono">{ts}</td>
							<td><span class="badge" class:ok={h.locked === true} class:warn={h.locked !== true}>{h.locked ? 'locked' : (h.status ?? h.error ?? 'missing')}</span></td>
							<td class="mono dim">{ms(h.start_ms)?.slice(0, 16) ?? '—'} → {ms(h.end_ms)?.slice(0, 16) ?? '—'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<h3 class="sub">Actions ({actions.length}) · discovered {status.discovered ?? 0} · rejected this round {status.rejected_this_round ?? 0}</h3>
			{#if actions.length}
				<ul class="mono small">
					{#each actions as a}<li>{a.strategy_id}: <b>{a.action}</b> {a.holdout_id ? `holdout ${a.holdout_id}` : ''} {a.detail ? JSON.stringify(a.detail).slice(0, 140) : ''}</li>{/each}
				</ul>
			{:else}<p class="dim">No actions in the last round.</p>{/if}
			{#if waiting.length}
				<h3 class="sub">Waiting</h3>
				<ul class="mono small">{#each waiting as [id, w]}<li>{id}: {typeof w === 'string' ? w : JSON.stringify(w)}</li>{/each}</ul>
			{/if}
			{#if liveSet}
				<h3 class="sub">Managed live set</h3>
				<p class="mono small">{liveSet.key} · {liveSet.evolved ?? 0} evolved · {liveSet.written ? 'written' : liveSet.unchanged ? 'unchanged' : (liveSet.reason ?? 'not written')} · strategy-runtime {status.runtime_reachable ? 'reachable' : 'unreachable'}</p>
			{/if}
			<h3 class="sub">Policy</h3>
			<div class="policy">
				{#each Object.entries(policy) as [k, v]}<span class="badge mono">{k} = {String(v)}</span>{/each}
			</div>
		{:else}
			<Empty producer="ctl.risk.pipeline (risk-engine with Postgres)" hint="" />
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head"><h2>Rejected candidates <span class="dim">({rejected.length}, newest first)</span></h2></div>
		{#if rejected.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>candidate</th><th>generation</th><th>reason</th></tr></thead>
					<tbody>
						{#each rejected as r, i (i)}
							<tr>
								<td class="mono">{r.strategy_id ?? r.genome_id ?? r.id ?? '—'}</td>
								<td class="mono dim">{String(r.generation ?? '').slice(-8)}</td>
								<td class="small">{r.reason ?? JSON.stringify(r)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="dim">No rejections recorded.</p>
		{/if}
	</section>

	<section class="panel">
		<div class="panel-head">
			<h2>Allocator tournament</h2>
			{#if tournament}<span class="dim">serving <b class="mono">{tournament.active ?? '—'}</b> · cost {tournament.cost_bps} bps</span>{/if}
		</div>
		{#if tournamentError}<p class="badge danger">{tournamentError}</p>{/if}
		{#if ranked.length}
			<table>
				<thead><tr><th>#</th><th>allocator</th><th>cum. return</th><th>Sharpe/step</th><th>max DD</th><th>turnover</th><th>costs</th><th>steps</th></tr></thead>
				<tbody>
					{#each ranked as e (e.name)}
						<tr class:serving={e.serving}>
							<td class="mono">{e.rank}</td>
							<td class="mono">{e.name}{#if e.serving} <span class="badge ok">serving</span>{/if}</td>
							<td class="mono" class:pos={e.cumulative_return > 0} class:neg={e.cumulative_return < 0}>{pct(e.cumulative_return, 2)}{#if e.lead < 0}<span class="dim"> ({pct(e.lead, 2)})</span>{/if}</td>
							<td class="mono">{num(e.sharpe, 3)}</td>
							<td class="mono">{pct(e.max_drawdown, 2)}</td>
							<td class="mono dim">{num(e.turnover, 1)}</td>
							<td class="mono dim">{pct(e.costs, 2)}</td>
							<td class="mono dim">{e.steps}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<p class="dim small">{tournament?.basis}. Switching the serving allocator is a limits change (ctl.risk.limits allocator), not done from here.</p>
		{:else if !tournamentError}
			<Empty producer="ctl.risk.allocators (risk-engine)" hint="" />
		{/if}
	</section>

	<Validation {candidate} />
</div>

<style>
	.stages {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
		margin-bottom: 12px;
	}
	.stage {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.arrow {
		font-size: 14px;
	}
	.spacer {
		flex: 1;
	}
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
		margin-top: 12px;
	}
	.tall {
		max-height: 520px;
	}
	.sub {
		font-size: 12px;
		margin: 10px 0 4px;
	}
	.small {
		font-size: 11px;
	}
	.reason {
		max-width: 320px;
	}
	.policy {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	ul {
		margin: 0;
		padding-left: 18px;
	}
	.pos {
		color: var(--ok);
	}
	.neg {
		color: var(--danger);
	}
	tr.sel td {
		color: var(--accent);
	}
	tr.serving td {
		background: color-mix(in srgb, var(--accent) 6%, transparent);
	}
</style>
