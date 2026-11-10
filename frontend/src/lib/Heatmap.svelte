<script lang="ts">
	// Square matrix heatmap (expert correlation, spec 51 §6.3) with the diverging colour of
	// projection.ts: red negative, green positive, opacity ∝ |value|.
	import { heatColor } from './projection';

	interface Props {
		matrix: number[][];
		labels: string[];
		title?: string;
		digits?: number;
	}
	let { matrix, labels, title = 'matrix', digits = 2 }: Props = $props();
	const n = $derived(labels.length);
</script>

{#if n}
	<div class="heat" role="img" aria-label={title} style="grid-template-columns: auto repeat({n}, 1fr)">
		<div></div>
		{#each labels as l}<div class="lab mono dim">{l}</div>{/each}
		{#each matrix as row, i}
			<div class="lab mono dim">{labels[i]}</div>
			{#each row as v, j}
				{@const h = heatColor(v)}
				<div class="cell" title="{labels[i]} × {labels[j] ?? ''}: {v.toFixed(4)}">
					<div class="fill" style="background: {h.color}; opacity: {Math.max(0.06, h.alpha)}"></div>
					<span class="mono">{v.toFixed(digits)}</span>
				</div>
			{/each}
		{/each}
	</div>
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
	}
</style>
