<script lang="ts">
	// Genome inspector (spec 51 §6.4): `GET /api/evolution/genomes/{id}` + `/outputs`
	// (presigned eval reports open in a new tab; decoder outputs via QuantileStrip / Histogram).
	import Histogram from './Histogram.svelte';
	import LineageTree from './LineageTree.svelte';
	import Modal from './Modal.svelte';
	import QuantileStrip from './QuantileStrip.svelte';
	import { ago, live, short, type GenomeDetail, type GenomeOutputs } from './api';
	import { plottableKeys } from './pareto';

	interface Props {
		genomeId: string | null;
		onclose: () => void;
		/** Open another genome (lineage tree click); the drawer shows it in place. */
		onopen?: (id: string) => void;
	}
	let { genomeId, onclose, onopen }: Props = $props();

	let detail = $state<GenomeDetail | null>(null);
	let outputs = $state<GenomeOutputs | null>(null);
	let error = $state('');
	let busy = $state(false);
	let now = $state(Date.now());

	$effect(() => {
		if (!genomeId) return;
		detail = null;
		outputs = null;
		error = '';
		busy = true;
		now = Date.now();
		const id = genomeId;
		Promise.allSettled([live.genome(id), live.genomeOutputs(id, 20)]).then(([d, o]) => {
			if (genomeId !== id) return;
			if (d.status === 'fulfilled') detail = d.value;
			else error = d.reason instanceof Error ? d.reason.message : String(d.reason);
			if (o.status === 'fulfilled') outputs = o.value;
			busy = false;
		});
	});

	const keys = $derived(plottableKeys(detail?.fitness ?? []));
	const fitnessByGen = $derived([...(detail?.fitness ?? [])].sort((a, b) => a.generation - b.generation));
	const scores = $derived((k: string) => fitnessByGen.map((f) => f.objectives[k]).filter((v) => typeof v === 'number'));
	const genomeText = $derived(detail?.params ?? detail?.genome?.params ?? '');
	/** Inline-TOML strategies get a light pretty-print (one key per line); kv genomes split on `;`. */
	const pretty = $derived(
		genomeText.startsWith('{')
			? genomeText.replace(/^\{\s*/, '{\n  ').replace(/\s*\}$/, '\n}').replace(/,\s+(?=[a-z_]+\s*=)/g, ',\n  ')
			: genomeText.replace(/;/g, ';\n')
	);
</script>

<Modal open={genomeId !== null} title="Genome {genomeId ? short(genomeId, 24) : ''}" {onclose} wide>
	{#if busy}<p class="dim">loading GET /api/evolution/genomes/{genomeId} …</p>{/if}
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if detail}
		<div class="grid">
			<section>
				<h3>Definition <span class="dim">{detail.genome ? `${detail.genome.kind} / ${detail.genome.family} · created ${ago(detail.genome.created_at, now)} ago` : 'not in Helix'}</span></h3>
				{#if detail.helix?.error}<p class="badge warn">helix: {detail.helix.error}</p>{/if}
				<pre class="mono def">{pretty || '—'}</pre>
				{#if detail.parents.length}
					<p class="dim small">parents: <span class="mono">{detail.parents.map((p) => (typeof p === 'object' && p && 'genome_id' in p ? String((p as { genome_id: string }).genome_id) : String(p))).join(', ')}</span></p>
				{/if}
			</section>
			<section>
				<h3>Fitness <span class="dim">({detail.fitness.length} records{detail.latest_objectives ? ' · latest job objectives below' : ''})</span></h3>
				{#if fitnessByGen.length}
					<div class="scroll fit">
						<table>
							<thead><tr><th>gen</th><th>rank</th>{#each keys as k}<th>{k}</th>{/each}</tr></thead>
							<tbody>
								{#each fitnessByGen as f, i (f.$id ?? i)}
									<tr>
										<td class="mono">{f.generation}</td>
										<td class="mono" class:ok={f.pareto_rank === 0}>{f.pareto_rank}</td>
										{#each keys as k}<td class="mono">{typeof f.objectives[k] === 'number' ? f.objectives[k].toFixed(4) : '—'}</td>{/each}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
					{#if keys.length && scores(keys[0]).length > 2}
						<div class="hists">
							{#each keys.slice(0, 2) as k}<div><Histogram values={scores(k)} label={k} bins={8} /></div>{/each}
						</div>
					{/if}
				{:else}
					<p class="dim">no FitnessRecord rows (evo-worker upserts them to Helix after evaluation)</p>
				{/if}
				{#if detail.latest_objectives}
					<p class="mono small">{Object.entries(detail.latest_objectives).map(([k, v]) => `${k}=${v.toFixed(4)}`).join(' · ')}</p>
				{/if}
			</section>
		</div>
		<section>
			<h3>Eval reports <span class="dim">(fa-artifacts/evals/&lt;generation&gt;/&lt;job&gt;.json, presigned {outputs?.presign.ttl_secs ?? 3600}s · {outputs?.presign.configured ? outputs.presign.endpoint : 'presign not configured'})</span></h3>
			{#if outputs?.presign.error}<p class="badge warn">{outputs.presign.error}</p>{/if}
			{#if outputs?.eval_reports.length}
				<table>
					<thead><tr><th>job</th><th>generation</th><th>status</th><th>report</th></tr></thead>
					<tbody>
						{#each outputs.eval_reports as r (r.job_id)}
							<tr>
								<td class="mono">{short(r.job_id, 12)}</td>
								<td class="mono dim">{short(r.generation_id, 12)}</td>
								<td><span class="badge" class:ok={r.status === 'ok'}>{r.status}</span></td>
								<td>{#if r.url}<a href={r.url} target="_blank" rel="noopener noreferrer">open report ↗</a>{:else}<span class="dim mono">{r.key}</span>{/if}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{:else}
				<p class="dim">no eval reports known to control-api for this genome (in-memory jobs only, D-032; validation reports are in the registry)</p>
			{/if}
		</section>
		<section>
			<h3>Decoder outputs <span class="dim">({outputs?.outputs.length ?? 0})</span></h3>
			{#if outputs?.outputs.length}
				<div class="outs">
					{#each outputs.outputs as o (o.received_at + o.decoder_id + o.family)}
						<div class="out">
							<div class="row"><span class="badge">{o.family}</span><span class="mono dim">{o.decoder_id} · {o.timescale} · {ago(o.received_at, now)} ago</span></div>
							{#if o.quantiles.length}<div class="row"><span class="dim">forecast</span><QuantileStrip quantiles={o.quantiles} /></div>{/if}
							{#if o.analogs.length}<div class="row"><span class="dim">analogs</span><Histogram values={o.analogs.map((a) => a.similarity)} label="analog similarity" bins={8} /></div>{/if}
							{#if o.urls?.length}
								<div class="row refs">
									{#each o.urls as u}
										{#if typeof u === 'string'}<span class="mono dim">{u}</span>{:else if u.url}<a href={u.url} target="_blank" rel="noopener noreferrer">{u.ref.split('/').pop()} ↗</a>{:else}<span class="mono dim">{u.ref}</span>{/if}
									{/each}
								</div>
							{:else if o.output_refs.length}
								<div class="row mono dim refs">{o.output_refs.join(' · ')}</div>
							{/if}
						</div>
					{/each}
				</div>
			{:else}
				<p class="dim">no decoder.output.* frames from this genome in control-api's window</p>
			{/if}
		</section>
	{/if}
	{#if genomeId && !busy}<LineageTree {genomeId} {onopen} />{/if}
</Modal>

<style>
	.grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}
	@media (max-width: 900px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
	section {
		margin-bottom: 14px;
	}
	h3 {
		font-size: 12px;
		margin-bottom: 6px;
	}
	.def {
		background: var(--bg-inset);
		padding: 8px;
		border-radius: 4px;
		font-size: 11px;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 260px;
		overflow: auto;
	}
	.fit {
		max-height: 200px;
	}
	.hists {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
		margin-top: 8px;
	}
	.ok {
		color: var(--ok);
	}
	.small {
		font-size: 11px;
	}
	.outs {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.out {
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 8px;
		background: var(--bg-inset);
	}
	.row {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 4px;
	}
	.row :global(.hist) {
		height: 40px;
		flex: 1;
	}
	.refs {
		flex-wrap: wrap;
		word-break: break-all;
	}
</style>
