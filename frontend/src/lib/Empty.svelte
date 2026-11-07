<script lang="ts">
	// Empty state naming the producer that would fill this panel.
	interface Props {
		producer: string;
		hint?: string;
	}
	let { producer, hint = 'cargo run -p control-api --example demo_publisher' }: Props = $props();
</script>

<div class="empty">
	<span>No data yet: waiting for <code>{producer}</code>.</span>
	{#if hint}<span class="dim">For synthetic data: <code>{hint}</code></span>{/if}
</div>

<style>
	.empty {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 16px;
		border: 1px dashed var(--border);
		border-radius: var(--radius);
		color: var(--text-dim);
	}
</style>
