<script lang="ts">
	// Horizontal headroom gauge: value vs cap (spec 51 §6.5 exposure vs caps). When the API
	// reports no cap, the bar is drawn against `reference` and labelled accordingly.
	interface Props {
		label: string;
		value: number;
		cap?: number | null;
		/** Used as the full scale when there is no cap (e.g. equity for gross exposure). */
		reference?: number;
		format: (v: number) => string;
	}
	let { label, value, cap = null, reference = 1, format }: Props = $props();

	const scale = $derived(cap ?? reference);
	const ratio = $derived(scale > 0 ? Math.abs(value) / scale : 0);
	const width = $derived(Math.min(1, ratio) * 100);
	const level = $derived(cap === null ? 'none' : ratio >= 1 ? 'danger' : ratio >= 0.8 ? 'warn' : 'ok');
</script>

<div class="gauge">
	<div class="row">
		<span class="dim">{label}</span>
		<span class="mono">{format(value)}<span class="dim"> / {cap === null ? `${format(scale)} (no cap reported)` : format(cap)}</span></span>
	</div>
	<div class="track" role="meter" aria-label={label} aria-valuemin="0" aria-valuemax={scale} aria-valuenow={Math.abs(value)}>
		<div class="bar {level}" style="width: {width}%"></div>
		{#if cap !== null}<div class="cap"></div>{/if}
	</div>
	<div class="row mono dim">
		<span>{(ratio * 100).toFixed(1)}% of {cap === null ? 'reference' : 'cap'}</span>
		{#if cap !== null}<span>headroom {format(Math.max(0, cap - Math.abs(value)))}</span>{/if}
	</div>
</div>

<style>
	.gauge {
		display: flex;
		flex-direction: column;
		gap: 3px;
	}
	.row {
		display: flex;
		justify-content: space-between;
		gap: 8px;
	}
	.track {
		position: relative;
		height: 10px;
		background: var(--bg-inset);
		border-radius: 5px;
		overflow: hidden;
	}
	.bar {
		height: 100%;
		border-radius: 5px;
		background: var(--text-dim);
		transition: width 0.3s;
	}
	.bar.ok {
		background: var(--ok);
	}
	.bar.warn {
		background: var(--warn);
	}
	.bar.danger {
		background: var(--danger);
	}
	.cap {
		position: absolute;
		right: 0;
		top: 0;
		bottom: 0;
		width: 2px;
		background: var(--danger);
	}
</style>
