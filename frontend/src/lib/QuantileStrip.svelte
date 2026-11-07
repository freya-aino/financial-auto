<script lang="ts">
	// Forecast quantiles as a box-like strip: outer quantiles = whiskers, inner = box, median tick.
	interface Props {
		quantiles: number[];
	}
	let { quantiles }: Props = $props();

	const W = 160;
	const H = 18;
	const q = $derived([...quantiles].sort((a, b) => a - b));
	const lo = $derived(q[0]);
	const hi = $derived(q[q.length - 1]);
	const x = (v: number) => 4 + ((v - lo) / (hi - lo || 1)) * (W - 8);
	const inner = $derived(q.length >= 4 ? [q[1], q[q.length - 2]] : [lo, hi]);
	const mid = $derived(q[Math.floor(q.length / 2)]);
</script>

{#if q.length}
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="forecast quantiles">
		<line x1={x(lo)} x2={x(hi)} y1={H / 2} y2={H / 2} class="whisker" />
		<rect x={x(inner[0])} y={3} width={Math.max(1, x(inner[1]) - x(inner[0]))} height={H - 6} class="box" />
		<line x1={x(mid)} x2={x(mid)} y1={2} y2={H - 2} class="mid" />
		<title>{q.map((v) => v.toFixed(2)).join(' · ')}</title>
	</svg>
{/if}

<style>
	.whisker {
		stroke: var(--text-dim);
	}
	.box {
		fill: var(--accent);
		fill-opacity: 0.35;
		stroke: var(--accent);
	}
	.mid {
		stroke: var(--text);
		stroke-width: 2;
	}
</style>
