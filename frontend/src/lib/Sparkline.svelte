<script lang="ts">
	// Single-series SVG line chart with a hover crosshair (spec 51 §6.5 P&L / drawdown charts).
	import { linePath, yFor } from './series';

	interface Props {
		times: number[];
		values: number[];
		format?: (v: number) => string;
		/** Draws a dashed reference line at this value (e.g. 0 for P&L) and forces it into range. */
		baseline?: number | null;
		/** CSS colour of the line; `signed` colours by the last value's sign relative to the baseline. */
		tone?: 'accent' | 'danger' | 'signed';
		fill?: boolean;
		height?: number;
		label: string;
	}
	let { times, values, format = (v) => v.toFixed(2), baseline = null, tone = 'accent', fill = false, height = 72, label }: Props =
		$props();

	const W = 400;
	const H = $derived(height);
	const scaled = $derived(linePath(values, W, H, baseline === null ? [] : [baseline]));
	const area = $derived(scaled.path ? `${scaled.path} L${W} ${H} L0 ${H} Z` : '');
	const last = $derived(values[values.length - 1]);
	const cls = $derived(
		tone === 'signed' ? ((last ?? 0) < (baseline ?? 0) ? 'danger' : 'accent') : tone
	);

	let hover = $state<number | null>(null);
	function move(ev: PointerEvent) {
		const el = ev.currentTarget as SVGSVGElement;
		const r = el.getBoundingClientRect();
		const x = ((ev.clientX - r.left) / r.width) * W;
		if (!values.length) return;
		hover = Math.max(0, Math.min(values.length - 1, Math.round((x / W) * (values.length - 1))));
	}
	const tip = $derived(hover === null || !scaled.pts[hover] ? null : { ...scaled.pts[hover], v: values[hover], t: times[hover] });
	const span = $derived(times.length > 1 ? Math.round((times[times.length - 1] - times[0]) / 60000) : 0);
</script>

<div class="spark">
	<div class="head">
		<span class="dim">{label}</span>
		<span class="mono" class:dim={tip === null}>
			{tip ? `${format(tip.v)} · ${new Date(tip.t).toLocaleTimeString()}` : last === undefined ? '—' : format(last)}
		</span>
	</div>
	{#if values.length > 1}
		<svg
			viewBox="0 0 {W} {H}"
			preserveAspectRatio="none"
			style="height: {H}px"
			role="img"
			aria-label="{label} over the last {span} min"
			onpointermove={move}
			onpointerleave={() => (hover = null)}
		>
			{#if fill}<path d={area} class="area {cls}" />{/if}
			{#if baseline !== null}
				<line x1="0" x2={W} y1={yFor(baseline, scaled, H)} y2={yFor(baseline, scaled, H)} class="base" />
			{/if}
			<path d={scaled.path} class="line {cls}" vector-effect="non-scaling-stroke" />
			{#if tip}
				<line x1={tip.x} x2={tip.x} y1="0" y2={H} class="cross" vector-effect="non-scaling-stroke" />
				<circle cx={tip.x} cy={tip.y} r="3.5" class="dot {cls}" vector-effect="non-scaling-stroke" />
			{/if}
		</svg>
		<div class="axis mono dim"><span>{format(scaled.lo)}</span><span>{span} min</span><span>{format(scaled.hi)}</span></div>
	{:else}
		<div class="waiting dim">collecting samples…</div>
	{/if}
</div>

<style>
	.spark {
		display: flex;
		flex-direction: column;
		gap: 4px;
		min-width: 0;
	}
	.head,
	.axis {
		display: flex;
		justify-content: space-between;
		gap: 8px;
	}
	svg {
		width: 100%;
		display: block;
		overflow: visible;
		touch-action: none;
	}
	.line {
		fill: none;
		stroke-width: 2px;
		stroke-linejoin: round;
	}
	.line.accent,
	.dot.accent {
		stroke: var(--accent);
	}
	.line.danger,
	.dot.danger {
		stroke: var(--danger);
	}
	.area {
		opacity: 0.12;
	}
	.area.accent {
		fill: var(--accent);
	}
	.area.danger {
		fill: var(--danger);
	}
	.base {
		stroke: var(--border);
		stroke-dasharray: 4 4;
	}
	.cross {
		stroke: var(--text-dim);
		stroke-width: 1px;
	}
	.dot {
		fill: var(--bg-raised);
		stroke-width: 2px;
	}
	.waiting {
		height: 72px;
		display: flex;
		align-items: center;
		justify-content: center;
		border: 1px dashed var(--border);
		border-radius: var(--radius);
	}
</style>
