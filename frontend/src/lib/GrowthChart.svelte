<script lang="ts">
	// Dependency-free multi-series SVG line chart for the /growth page (spec 51 §6, round 9):
	// shared y scale, real time on x (so sparse young series sit where their samples are), gaps
	// broken instead of bridged, dashed cap reference lines, hover crosshair with per-series
	// readout, and an honest empty state naming the PromQL the chart reads.
	import { assignRegionColors } from './projection';
	import { partitionCaps, splitSegments, valueRange, fmtTime, type Cap, type RangeKey, type Series } from './growth';

	interface Props {
		label: string;
		series: Series[];
		/** [startMs, endMs] of the queried range (the x domain, not the data extent). */
		domain: [number, number];
		/** Query step in ms; gaps wider than 2.5 steps break the line. */
		stepMs: number;
		rangeKey: RangeKey;
		format?: (v: number) => string;
		/** Dashed reference lines (e.g. stream caps); far-away caps are listed as text instead. */
		caps?: Cap[];
		/** The PromQL this chart reads, shown in the empty state. */
		query: string;
		error?: string;
		height?: number;
		/** Keep 0 in the y range (counts and bytes read from a zero base). */
		zero?: boolean;
	}
	let { label, series, domain, stepMs, rangeKey, format = (v) => v.toFixed(1), caps = [], query, error = '', height = 110, zero = true }: Props = $props();

	const W = 400;
	const H = $derived(height);
	const drawn = $derived(series.filter((s) => s.pts.length > 0));
	const colors = $derived(assignRegionColors(series.map((s) => s.name)));
	const dataHi = $derived(drawn.reduce((m, s) => s.pts.reduce((m2, p) => Math.max(m2, p[1]), m), 0));
	const capSplit = $derived(partitionCaps(caps, dataHi));
	const range = $derived(valueRange(drawn, [...(zero ? [0] : []), ...capSplit.drawn.map((c) => c.value)]));

	const x = $derived((t: number) => ((t - domain[0]) / Math.max(1, domain[1] - domain[0])) * W);
	const y = $derived((v: number) => (range ? H - ((v - range.lo) / (range.hi - range.lo)) * H : H / 2));
	const paths = $derived(
		drawn.map((s) => ({
			name: s.name,
			color: colors.get(s.name) ?? 'var(--accent)',
			d: splitSegments(s.pts, 2.5 * stepMs)
				.map((run) =>
					run.length === 1
						? `M${(x(run[0][0]) - 1.5).toFixed(1)} ${y(run[0][1]).toFixed(1)} h3`
						: run.map(([t, v], i) => `${i ? 'L' : 'M'}${x(t).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')
				)
				.join(' ')
		}))
	);

	let hover = $state<number | null>(null);
	function move(ev: PointerEvent) {
		const el = ev.currentTarget as SVGSVGElement;
		const r = el.getBoundingClientRect();
		hover = domain[0] + ((ev.clientX - r.left) / r.width) * (domain[1] - domain[0]);
	}
	/** Per-series value nearest the hovered time (within 2.5 steps), newest-first legend order. */
	const tip = $derived.by(() => {
		if (hover === null) return null;
		const t = hover;
		const rows = drawn
			.map((s) => {
				let best: [number, number] | null = null;
				for (const p of s.pts) if (best === null || Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
				return best && Math.abs(best[0] - t) <= 2.5 * stepMs ? { name: s.name, color: colors.get(s.name) ?? '', t: best[0], v: best[1] } : null;
			})
			.filter((r) => r !== null);
		if (!rows.length) return null;
		return { t: rows[0].t, x: x(rows[0].t), rows: rows.sort((a, b) => b.v - a.v).slice(0, 8) };
	});
</script>

<div class="chart">
	<div class="head">
		<span class="dim">{label}</span>
		{#if drawn.length === 1 && drawn[0].pts.length}
			<span class="mono">{format(drawn[0].pts[drawn[0].pts.length - 1][1])}</span>
		{/if}
	</div>
	{#if error}
		<p class="badge danger">{error}</p>
	{:else if drawn.length && range}
		<svg
			viewBox="0 0 {W} {H}"
			preserveAspectRatio="none"
			style="height: {H}px"
			role="img"
			aria-label="{label} from {fmtTime(domain[0], rangeKey)} to {fmtTime(domain[1], rangeKey)}"
			onpointermove={move}
			onpointerleave={() => (hover = null)}
		>
			{#each capSplit.drawn as c (c.name + c.value)}
				<line x1="0" x2={W} y1={y(c.value)} y2={y(c.value)} class="cap" vector-effect="non-scaling-stroke" />
			{/each}
			{#each paths as p (p.name)}
				<path d={p.d} class="line" style="stroke: {p.color}" vector-effect="non-scaling-stroke" />
			{/each}
			{#if tip}
				<line x1={tip.x} x2={tip.x} y1="0" y2={H} class="cross" vector-effect="non-scaling-stroke" />
				{#each tip.rows as r (r.name)}
					<circle cx={x(r.t)} cy={y(r.v)} r="3" class="dot" style="stroke: {r.color}" vector-effect="non-scaling-stroke" />
				{/each}
			{/if}
		</svg>
		<div class="axis mono dim">
			<span>{format(range.lo)} – {format(range.hi)}</span>
			<span>{fmtTime(domain[0], rangeKey)} → {fmtTime(domain[1], rangeKey)}</span>
		</div>
		{#if tip}
			<div class="tipbox mono">
				<span class="dim">{new Date(tip.t).toLocaleString()}</span>
				{#each tip.rows as r (r.name)}
					<span><i style="background: {r.color}"></i>{r.name}: {format(r.v)}</span>
				{/each}
			</div>
		{:else if drawn.length > 1}
			<div class="legend mono">
				{#each drawn as s (s.name)}
					<span><i style="background: {colors.get(s.name)}"></i>{s.name}</span>
				{/each}
			</div>
		{/if}
		{#each capSplit.drawn as c (c.name + c.value)}
			<p class="dim small">cap {format(c.value)} ({c.name})</p>
		{/each}
		{#each capSplit.noted as c (c.name + c.value)}
			<p class="dim small">cap {format(c.value)} ({c.name}) — off scale, {dataHi > 0 ? `${((100 * dataHi) / c.value).toFixed(1)}% used` : 'unused'}</p>
		{/each}
	{:else}
		<div class="empty dim">
			<span>no series yet in this range</span>
			<code class="mono">{query}</code>
		</div>
	{/if}
</div>

<style>
	.chart {
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
		border-bottom: 1px solid var(--border);
	}
	.line {
		fill: none;
		stroke-width: 2px;
		stroke-linejoin: round;
	}
	.cap {
		stroke: var(--warn);
		stroke-dasharray: 4 4;
		stroke-width: 1px;
	}
	.cross {
		stroke: var(--text-dim);
		stroke-width: 1px;
	}
	.dot {
		fill: var(--bg-raised);
		stroke-width: 2px;
	}
	.legend,
	.tipbox {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
	}
	.legend i,
	.tipbox i {
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 2px;
		margin-right: 4px;
	}
	.empty {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 14px;
		border: 1px dashed var(--border);
		border-radius: var(--radius);
	}
	.empty code {
		word-break: break-all;
	}
	.small {
		font-size: 11px;
		margin: 0;
	}
</style>
