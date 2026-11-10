<script lang="ts">
	// Realised P&L from persisted fills (`GET /api/trading/pnl?range=&strategy=`, D-047).
	import { onMount } from 'svelte';
	import { eur, live, short, type PnlView } from './api';
	import { loadPref, savePref } from './prefs';
	import { runtime } from './runtime.svelte';

	interface Props {
		strategies: string[];
		/** Bumps to refetch (e.g. on fill frames). */
		version?: number;
	}
	let { strategies, version = 0 }: Props = $props();

	let range = $state(loadPref('trading.pnl.range', '1d'));
	let strategy = $state(loadPref('trading.pnl.strategy', ''));
	let view = $state<PnlView | null>(null);
	let error = $state('');
	$effect(() => savePref('trading.pnl.range', range));
	$effect(() => savePref('trading.pnl.strategy', strategy));

	async function load() {
		try {
			view = await live.pnl(range, strategy || undefined);
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}
	$effect(() => {
		void range;
		void strategy;
		void version;
		void load();
	});

	const signed = (v: number | null | undefined) => (v === null || v === undefined ? '' : v >= 0 ? 'pos' : 'neg');
	const maxAbs = $derived(Math.max(1e-9, ...(view?.by_instrument.map((b) => Math.abs(b.realised_eur)) ?? [])));

	onMount(() => {
		const t = setInterval(load, runtime.pollMs * 15);
		return () => clearInterval(t);
	});
</script>

<section class="panel">
	<div class="panel-head">
		<h2>P&amp;L</h2>
		<div class="pickers">
			<select bind:value={range}>{#each ['1h', '6h', '1d', '7d', '30d', 'all'] as r}<option value={r}>{r}</option>{/each}</select>
			<select bind:value={strategy}>
				<option value="">all strategies</option>
				{#each strategies as s}<option value={s}>{s}</option>{/each}
			</select>
		</div>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if view}
		<div class="tiles">
			<div><span class="dim">realised</span><b class="mono {signed(view.realised_eur)}">{eur(view.realised_eur)}</b></div>
			<div><span class="dim">fees</span><b class="mono neg">{eur(-view.fees_eur)}</b></div>
			<div><span class="dim">net</span><b class="mono {signed(view.net_eur)}">{eur(view.net_eur)}</b></div>
			<div><span class="dim">turnover · fills</span><b class="mono">{eur(view.turnover_eur)} <span class="dim small">· {view.fills}</span></b></div>
		</div>
		{#if view.by_instrument.length}
			<table>
				<thead><tr><th>instrument</th><th>realised</th><th>net qty in range</th><th>avg cost</th></tr></thead>
				<tbody>
					{#each view.by_instrument as b (b.instrument_id)}
						<tr>
							<td class="mono" title={b.instrument_id}>{short(b.instrument_id, 12)}</td>
							<td>
								<div class="bar"><div class={signed(b.realised_eur)} style="width: {(100 * Math.abs(b.realised_eur)) / maxAbs}%"></div></div>
								<span class="mono {signed(b.realised_eur)}">{eur(b.realised_eur)}</span>
							</td>
							<td class="mono">{b.net_quantity_in_range.toFixed(4)}</td>
							<td class="mono dim">{eur(b.avg_cost_eur)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
		<p class="dim small">{view.book} · {view.range.from.slice(0, 16)} → {view.range.to.slice(0, 16)}{view.strategy ? ` · strategy ${view.strategy}` : ''}. {view.note ?? ''}</p>
	{/if}
</section>

<style>
	.pickers {
		display: flex;
		gap: 6px;
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
		gap: 8px;
		margin-bottom: 10px;
	}
	.tiles > div {
		display: flex;
		flex-direction: column;
		background: var(--bg-inset);
		border-radius: 4px;
		padding: 6px 8px;
	}
	.tiles b {
		font-size: 15px;
	}
	.small {
		font-size: 11px;
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
		border-radius: 3px;
		background: var(--text-dim);
	}
	.bar > .pos {
		background: var(--ok);
	}
	.bar > .neg {
		background: var(--danger);
	}
	.pos {
		color: var(--ok);
	}
	.neg {
		color: var(--danger);
	}
	p {
		margin: 8px 0 0;
	}
</style>
