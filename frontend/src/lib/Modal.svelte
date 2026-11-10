<script lang="ts">
	// Minimal dialog wrapper (native <dialog>) for the Backfill / Resolve / Genome drawers.
	import type { Snippet } from 'svelte';

	interface Props {
		open: boolean;
		title: string;
		onclose: () => void;
		wide?: boolean;
		children: Snippet;
	}
	let { open, title, onclose, wide = false, children }: Props = $props();

	let el = $state<HTMLDialogElement | null>(null);
	$effect(() => {
		if (!el) return;
		if (open && !el.open) el.showModal();
		else if (!open && el.open) el.close();
	});
</script>

<dialog bind:this={el} class:wide onclose={onclose} oncancel={onclose}>
	{#if open}
		<div class="head">
			<h3>{title}</h3>
			<button type="button" onclick={onclose} aria-label="close">×</button>
		</div>
		<div class="body">{@render children()}</div>
	{/if}
</dialog>

<style>
	dialog {
		background: var(--bg-raised);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0;
		min-width: min(520px, 92vw);
		max-width: 92vw;
		max-height: 90vh;
	}
	dialog.wide {
		min-width: min(900px, 92vw);
	}
	dialog::backdrop {
		background: rgb(0 0 0 / 0.55);
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 10px 14px;
		border-bottom: 1px solid var(--border);
	}
	h3 {
		font-size: 13px;
	}
	.body {
		padding: 14px;
		overflow: auto;
		max-height: calc(90vh - 48px);
	}
</style>
