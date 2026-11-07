<script lang="ts">
	interface Props {
		values: number[];
		bins?: number;
		label: string;
	}
	let { values, bins = 12, label }: Props = $props();

	const hist = $derived.by(() => {
		if (!values.length) return { counts: [] as number[], lo: 0, hi: 0, max: 1 };
		const lo = Math.min(...values);
		const hi = Math.max(...values);
		const w = (hi - lo || 1) / bins;
		const counts = Array(bins).fill(0) as number[];
		for (const v of values) counts[Math.min(bins - 1, Math.floor((v - lo) / w))]++;
		return { counts, lo, hi, max: Math.max(...counts) };
	});
</script>

<div class="hist" role="img" aria-label="histogram of {label}">
	{#each hist.counts as c}
		<div class="bar" style="height: {(100 * c) / hist.max}%" title={String(c)}></div>
	{/each}
</div>
<div class="axis mono dim"><span>{hist.lo.toFixed(2)}</span><span>{label}</span><span>{hist.hi.toFixed(2)}</span></div>

<style>
	.hist {
		display: flex;
		align-items: flex-end;
		gap: 2px;
		height: 120px;
	}
	.bar {
		flex: 1;
		background: var(--accent);
		opacity: 0.75;
		min-height: 1px;
	}
	.axis {
		display: flex;
		justify-content: space-between;
	}
</style>
