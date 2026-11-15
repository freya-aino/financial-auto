<script lang="ts">
	// Region detail (spec 51 §6.3): `GET /api/latent/regions/{id}/reconstructions` — the region's
	// stored episodes (latest exit first), each with the instrument's 1 m market path while it sat
	// in the region and the decoder outputs (forecast quantiles, retrieved analogs, refs) of that
	// instrument, snapshot and timescale inside the episode window.
	import Empty from './Empty.svelte';
	import Info from './Info.svelte';
	import QuantileStrip from './QuantileStrip.svelte';
	import Sparkline from './Sparkline.svelte';
	import { live, short, type RegionReconstructions } from './api';
	import { errText } from './console';

	interface Props {
		regionId: string;
	}
	let { regionId }: Props = $props();

	const LIMITS = [5, 10, 20, 50];
	let limit = $state(10);
	let view = $state<RegionReconstructions | null>(null);
	let error = $state('');
	let busy = $state(false);

	async function load(id: string, n: number) {
		busy = true;
		try {
			const v = await live.regionReconstructions(id, { limit: n, max_bars: 240, max_outputs: 10 });
			if (id !== regionId) return;
			view = v;
			error = '';
		} catch (e) {
			if (id !== regionId) return;
			view = null;
			error = errText(e);
		} finally {
			busy = false;
		}
	}
	$effect(() => {
		void load(regionId, limit);
	});

	const fmtTime = (ms: number | null) => (ms === null ? '—' : new Date(ms).toISOString().slice(5, 16).replace('T', ' '));
	const fmtDur = (a: number | null, b: number | null) => {
		if (a === null || b === null) return '';
		const m = Math.round((b - a) / 60000);
		return m < 120 ? `${m} min` : `${(m / 60).toFixed(1)} h`;
	};
	const change = (bars: { close: number }[]) => (bars.length > 1 && bars[0].close ? bars[bars.length - 1].close / bars[0].close - 1 : null);
	const withOutputs = $derived((view?.episodes ?? []).filter((e) => e.decoder_outputs.length).length);
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Region detail <code>{regionId}</code><Info k="latent.regionDetail" /></h2>
		<div class="controls">
			<label>episodes
				<select bind:value={limit}>{#each LIMITS as k}<option value={k}>{k}</option>{/each}</select>
			</label>
			<button onclick={() => load(regionId, limit)} disabled={busy}>{busy ? '…' : 'refresh'}</button>
		</div>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if view && view.episodes.length}
		<p class="dim small">
			{view.episodes.length} episodes · decoder outputs from {view.decoder_outputs.source} ({view.decoder_outputs.matched} matched in {withOutputs} episodes, ≤ {view.decoder_outputs.per_episode} each)
			{#if view.decoder_outputs.error}<span class="badge warn">{view.decoder_outputs.error}</span>{/if}
		</p>
		<div class="episodes scroll tall">
			{#each view.episodes as ep, i (i)}
				{@const ch = change(ep.bars)}
				<div class="ep">
					<div class="ep-head">
						<span class="mono" title={ep.instrument_id}>{short(ep.instrument_id || '—', 8)}</span>
						<span class="mono dim">{fmtTime(ep.t_enter_ms)} → {fmtTime(ep.t_exit_ms)} · {fmtDur(ep.t_enter_ms, ep.t_exit_ms)}</span>
						{#if ch !== null}<span class="mono" class:pos={ch > 0} class:neg={ch < 0}>{(100 * ch).toFixed(2)}%</span>{/if}
						<span class="badge">{ep.source}{ep.truncated ? ' · truncated' : ''}</span>
						{#if ep.error}<span class="badge warn" title={ep.error}>{ep.error.slice(0, 40)}</span>{/if}
					</div>
					{#if ep.bars.length > 1}
						<Sparkline label="close (1 m)" times={ep.bars.map((b) => b.ts_ms)} values={ep.bars.map((b) => b.close)} height={40} fill />
					{:else if !ep.error}
						<p class="dim small">no market path in the episode window</p>
					{/if}
					{#each ep.decoder_outputs as o (o.decoder_id + o.family + (o.as_of ?? o.received_at))}
						<div class="out">
							<span class="badge">{o.family}</span>
							<span class="mono dim">{o.decoder_id}{o.genome_id ? ` · ${short(o.genome_id, 12)}` : ''} · {o.as_of?.slice(11, 19) ?? ''}</span>
							{#if o.quantiles.length}<QuantileStrip quantiles={o.quantiles} />{/if}
							{#if o.analogs.length}
								<span class="dim analogs" title={o.analogs.map((a) => `${a.window_id}: similarity ${a.similarity.toFixed(3)}, ${a.time_distance_days.toFixed(0)} d ago`).join('\n')}>
									{o.analogs.length} analogs · best {Math.max(...o.analogs.map((a) => a.similarity)).toFixed(3)} ({o.analogs[0].window_id})
								</span>
							{/if}
						</div>
					{/each}
				</div>
			{/each}
		</div>
	{:else if view}
		<Empty producer="RegionEpisode rows for {regionId} (model-runtime occupancy stage)" hint="episodes are written when an instrument leaves the region" />
	{:else if busy}
		<p class="dim">loading…</p>
	{/if}
</section>

<style>
	.controls {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.controls label {
		display: flex;
		gap: 4px;
		align-items: center;
		color: var(--text-dim);
	}
	.small {
		font-size: 11px;
		margin: 0 0 6px;
	}
	.ep {
		border-top: 1px solid var(--border);
		padding: 6px 0;
	}
	.ep-head,
	.out {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
		font-size: 11px;
	}
	.out {
		margin-top: 3px;
	}
	.analogs {
		font-size: 11px;
	}
	.pos {
		color: var(--accent);
	}
	.neg {
		color: var(--danger);
	}
</style>
