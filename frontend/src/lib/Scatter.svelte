<script lang="ts">
	// Minimal SVG scatter for Pareto views (spec 51 §6.4).
	interface Point {
		x: number;
		y: number;
		label: string;
		highlight?: boolean;
		/** CSS colour overriding the default (projection points coloured by region). */
		color?: string;
	}
	interface Props {
		points: Point[];
		xLabel: string;
		yLabel: string;
		/** Point radius for coloured (non-highlight) points. */
		radius?: number;
		onselect?: (index: number) => void;
	}
	let { points, xLabel, yLabel, radius = 3.5, onselect }: Props = $props();

	const W = 460;
	const H = 300;
	const P = 36;
	const ext = (v: number[]) => {
		const lo = Math.min(...v);
		const hi = Math.max(...v);
		const pad = (hi - lo || 1) * 0.08;
		return [lo - pad, hi + pad] as const;
	};
	const xs = $derived(ext(points.map((p) => p.x)));
	const ys = $derived(ext(points.map((p) => p.y)));
	const sx = (v: number) => P + ((v - xs[0]) / (xs[1] - xs[0])) * (W - 2 * P);
	const sy = (v: number) => H - P - ((v - ys[0]) / (ys[1] - ys[0])) * (H - 2 * P);
</script>

<svg viewBox="0 0 {W} {H}" role="img" aria-label="{yLabel} vs {xLabel}">
	<line x1={P} y1={H - P} x2={W - P} y2={H - P} class="axis" />
	<line x1={P} y1={P} x2={P} y2={H - P} class="axis" />
	<text x={W / 2} y={H - 8} text-anchor="middle">{xLabel}</text>
	<text x={12} y={H / 2} text-anchor="middle" transform="rotate(-90 12 {H / 2})">{yLabel}</text>
	<text x={P} y={H - P + 14} text-anchor="start">{xs[0].toFixed(2)}</text>
	<text x={W - P} y={H - P + 14} text-anchor="end">{xs[1].toFixed(2)}</text>
	<text x={P - 4} y={H - P} text-anchor="end">{ys[0].toFixed(2)}</text>
	<text x={P - 4} y={P + 4} text-anchor="end">{ys[1].toFixed(2)}</text>
	{#each points as p, i}
		<circle
			cx={sx(p.x)}
			cy={sy(p.y)}
			r={p.highlight ? 5 : radius}
			class:front={p.highlight}
			class:colored={!!p.color}
			class:clickable={!!onselect}
			style={p.color ? `fill: ${p.color}` : undefined}
			onclick={() => onselect?.(i)}
			role={onselect ? 'button' : undefined}
		>
			<title>{p.label}: {xLabel}={p.x.toFixed(3)}, {yLabel}={p.y.toFixed(3)}</title>
		</circle>
	{/each}
</svg>

<style>
	svg {
		width: 100%;
		display: block;
	}
	.axis {
		stroke: var(--border);
	}
	text {
		font: 10px var(--mono);
		fill: var(--text-dim);
	}
	circle {
		fill: var(--text-dim);
		fill-opacity: 0.6;
	}
	circle.front {
		fill: var(--accent);
		fill-opacity: 1;
	}
	circle.colored {
		fill-opacity: 0.8;
	}
	circle.clickable {
		cursor: pointer;
	}
</style>
