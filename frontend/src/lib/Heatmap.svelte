<script lang="ts">
	// Square matrix heatmap (expert correlation, spec 51 §6.3). `diverging`: the sign matters
	// (red negative, green positive, opacity ∝ |value|); `sequential`: only the magnitude
	// matters (one-hue ramp of |value|, projection.ts `seqColor`; the diagonal is left blank).
	import { heatColor, seqColor, SEQ_BREAKS } from './projection';

	interface Props {
		matrix: number[][];
		labels: string[];
		title?: string;
		digits?: number;
		scale?: 'diverging' | 'sequential';
	}
	let { matrix, labels, title = 'matrix', digits = 2, scale = 'diverging' }: Props = $props();
	const n = $derived(labels.length);
</script>

{#if n}
	<div class="heat" role="img" aria-label={title} style="grid-template-columns: auto repeat({n}, 1fr)">
		<div></div>
		{#each labels as l}<div class="lab mono dim">{l}</div>{/each}
		{#each matrix as row, i}
			<div class="lab mono dim">{labels[i]}</div>
			{#each row as v, j}
				{#if scale === 'sequential'}
					{@const c = seqColor(v)}
					<div class="cell" class:strong={c.strong && i !== j} title="{labels[i]} × {labels[j] ?? ''}: |{v.toFixed(4)}|" data-step={i === j ? null : c.step}>
						{#if i !== j}<div class="fill" style="background: {c.color}"></div>{/if}
						<span class="mono">{i === j ? '' : v.toFixed(digits)}</span>
					</div>
				{:else}
					{@const h = heatColor(v)}
					<div class="cell" title="{labels[i]} × {labels[j] ?? ''}: {v.toFixed(4)}">
						<div class="fill" style="background: {h.color}; opacity: {Math.max(0.06, h.alpha)}"></div>
						<span class="mono">{v.toFixed(digits)}</span>
					</div>
				{/if}
			{/each}
		{/each}
	</div>
	{#if scale === 'sequential'}
		<div class="legend small dim" aria-label="|value| scale">
			|value|
			{#each [0, ...SEQ_BREAKS] as b, k}
				<span class="key"><i style="background: var(--seq-{k})"></i>{k < SEQ_BREAKS.length ? `${b}–${SEQ_BREAKS[k]}` : `≥ ${b}`}</span>
			{/each}
		</div>
	{/if}
{/if}

<style>
	.heat {
		display: grid;
		gap: 2px;
	}
	.lab {
		font-size: 11px;
		align-self: center;
		justify-self: center;
		padding: 2px 4px;
	}
	.cell {
		position: relative;
		aspect-ratio: 1.6;
		min-height: 28px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 3px;
		background: var(--bg-inset);
		overflow: hidden;
	}
	.fill {
		position: absolute;
		inset: 0;
	}
	.cell span {
		position: relative;
		font-size: 11px;
		color: var(--seq-ink, inherit);
	}
	.cell.strong span {
		color: var(--seq-ink-strong);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
		margin-top: 6px;
		font-size: 11px;
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.key i {
		display: inline-block;
		width: 12px;
		height: 10px;
		border-radius: 2px;
	}
</style>
