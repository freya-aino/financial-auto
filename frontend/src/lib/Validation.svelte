<script lang="ts">
	// Validation card (spec 41, D-052): locked holdout periods and the trial registry of one
	// candidate (`GET /api/validation/holdout`, `GET /api/validation/reports/{candidate}`).
	import { onMount } from 'svelte';
	import { ago, live, short, type Holdout, type ValidationReports } from './api';
	import { directionFor, plottableKeys } from './pareto';
	import { runtime } from './runtime.svelte';

	interface Props {
		/** Candidate preselected from the Pareto view / generations table. */
		candidate?: string;
	}
	let { candidate = '' }: Props = $props();

	let holdouts = $state<Holdout[]>([]);
	let holdoutError = $state('');
	let note = $state('');
	let input = $state('');
	let reports = $state<ValidationReports | null>(null);
	let reportsError = $state('');
	let busy = $state(false);
	let now = $state(Date.now());

	$effect(() => {
		if (candidate && candidate !== input) {
			input = candidate;
			void load();
		}
	});

	const keys = $derived(plottableKeys(reports?.trials ?? [], Object.keys(reports?.directions ?? {})));

	async function loadHoldout() {
		try {
			const h = await live.holdout();
			holdouts = h.holdouts;
			note = h.note ?? '';
			holdoutError = '';
		} catch (e) {
			holdoutError = e instanceof Error ? e.message : String(e);
		}
	}

	async function load(ev?: Event) {
		ev?.preventDefault();
		const c = input.trim();
		if (!c) return;
		busy = true;
		now = Date.now();
		try {
			reports = await live.validationReports(c, 30);
			reportsError = '';
		} catch (e) {
			reportsError = e instanceof Error ? e.message : String(e);
			reports = null;
		} finally {
			busy = false;
		}
	}

	onMount(() => {
		void loadHoldout();
		const t = setInterval(loadHoldout, runtime.pollMs * 30);
		return () => clearInterval(t);
	});
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Validation</h2>
		<span class="dim">trial registry (validation.trial) · holdout (immutable)</span>
	</div>
	<h3 class="sub">Locked holdout periods <span class="dim">({holdouts.length})</span></h3>
	{#if holdoutError}<p class="badge danger">{holdoutError}</p>{/if}
	{#if holdouts.length}
		<table>
			<thead><tr><th>id</th><th>from</th><th>to</th><th>locked</th><th>note</th></tr></thead>
			<tbody>
				{#each holdouts as h, i (h.id ?? i)}
					<tr>
						<td class="mono">{h.id ?? i}</td>
						<td class="mono">{String(h.start ?? h.window_start ?? '—').slice(0, 16)}</td>
						<td class="mono">{String(h.end ?? h.window_end ?? '—').slice(0, 16)}</td>
						<td class="mono dim">{String(h.locked_at ?? h.created_at ?? '—').slice(0, 16)}</td>
						<td class="dim">{h.note ?? ''}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{:else if !holdoutError}
		<p class="dim">none locked yet{note ? ` — ${note}` : ''}</p>
	{/if}

	<h3 class="sub">Candidate reports</h3>
	<form class="row" onsubmit={load}>
		<input bind:value={input} class="mono" placeholder="gp-… strategy id, or a genome string" size="40" />
		<button type="submit" disabled={busy || !input.trim()}>{busy ? '…' : 'Load'}</button>
	</form>
	{#if reportsError}<p class="badge danger">{reportsError}</p>{/if}
	{#if reports}
		<p class="dim small">
			<span class="badge">{reports.kind}</span> {reports.n_trials} trials
			{#if reports.latest}· latest {ago(reports.latest.created_at, now)} ago in {short(reports.latest.generation, 10)} · Sharpe {reports.latest.sharpe?.toFixed(4) ?? '—'} · deflated {reports.latest.objectives.deflated_sharpe?.toFixed(4) ?? '—'}{/if}
		</p>
		<div class="scroll trials">
			<table>
				<thead><tr><th>when</th><th>generation</th><th>loop/rung</th>{#each keys as k}<th title={directionFor(k, reports.directions)}>{k} {directionFor(k, reports.directions) === 'min' ? '↓' : '↑'}</th>{/each}<th>window</th><th>report</th></tr></thead>
				<tbody>
					{#each reports.trials as t (t.id)}
						<tr>
							<td class="mono dim">{ago(t.created_at, now)}</td>
							<td class="mono">{short(t.generation, 10)}</td>
							<td class="mono">{t.gen_index}/{t.rung}</td>
							{#each keys as k}<td class="mono">{typeof t.objectives[k] === 'number' ? t.objectives[k].toFixed(4) : '—'}</td>{/each}
							<td class="mono dim">{t.window_start?.slice(5, 16)} → {t.window_end?.slice(5, 16)}</td>
							<td class="mono dim" title={t.report_ref ?? ''}>{t.report_ref ? short(t.report_ref.split('/').pop() ?? '', 12) : '—'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</section>

<style>
	.sub {
		font-size: 12px;
		margin: 8px 0 6px;
	}
	.row {
		display: flex;
		gap: 6px;
	}
	.small {
		font-size: 11px;
		margin: 6px 0;
	}
	.trials {
		max-height: 260px;
	}
</style>
