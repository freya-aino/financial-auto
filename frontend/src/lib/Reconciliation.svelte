<script lang="ts">
	// Reconciliation runs (`GET /api/trading/reconciliation`, D-047/D-051): memory book vs Postgres.
	import { onMount } from 'svelte';
	import { ago, eur, live, short, type ReconciliationRun } from './api';
	import { runtime } from './runtime.svelte';

	interface Props {
		now: number;
		version?: number;
	}
	let { now, version = 0 }: Props = $props();

	let runs = $state<ReconciliationRun[]>([]);
	let latest = $state<string | null>(null);
	let error = $state('');

	async function load() {
		try {
			const r = await live.reconciliation(10);
			runs = r.runs;
			latest = r.latest_status;
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}
	$effect(() => {
		void version;
		void load();
	});
	onMount(() => {
		const t = setInterval(load, runtime.pollMs * 15);
		return () => clearInterval(t);
	});
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Reconciliation</h2>
		{#if latest}<span class="badge" class:ok={latest === 'ok'} class:danger={latest === 'diff'}>latest {latest}</span>{:else}<span class="dim">no runs</span>{/if}
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if runs.length}
		<div class="scroll body">
			<table>
				<thead><tr><th>run</th><th>when</th><th>status</th><th>cash memory / postgres</th><th>position diffs</th></tr></thead>
				<tbody>
					{#each runs as r (r.id)}
						<tr>
							<td class="mono">{short(r.id, 8)}</td>
							<td class="mono dim">{ago(r.started_at, now)} ago</td>
							<td><span class="badge" class:ok={r.status === 'ok'} class:danger={r.status === 'diff'}>{r.status}</span></td>
							<td class="mono">{r.diffs?.cash ? `${eur(r.diffs.cash.memory)} / ${eur(r.diffs.cash.postgres)}` : '—'}</td>
							<td class="mono">
								{#each r.diffs?.positions ?? [] as p}
									<div title={p.instrument_id}>{short(p.instrument_id, 10)}: {p.memory.toFixed(4)} vs {p.postgres.toFixed(4)}</div>
								{:else}
									<span class="dim">none</span>
								{/each}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="dim small">A <code>diff</code> run halts scope <code>broker</code> (D-051); resume from the halts list once the books agree.</p>
	{:else if !error}
		<p class="dim">no reconciliation_run rows yet (risk-engine every reconcile_secs)</p>
	{/if}
</section>

<style>
	.body {
		max-height: 240px;
	}
	.small {
		font-size: 11px;
		margin: 6px 0 0;
	}
</style>
