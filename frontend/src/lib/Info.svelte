<script lang="ts">
	// A small ⓘ next to a heading or value: click opens a short explanation, outside click or
	// Esc closes it. Text comes from the help registry (lib/help) by key, so copy stays
	// reviewable and help.test.ts can check every key resolves. Deliberately quiet: dim glyph,
	// one popover at a time, no hover flicker.
	import { help } from './help';

	let { k, text = '' }: { k?: string; text?: string } = $props();
	let open = $state(false);
	let root: HTMLSpanElement | undefined = $state();
	const body = $derived(text || (k ? help(k) : ''));

	function onWindowClick(e: MouseEvent) {
		if (open && root && !root.contains(e.target as Node)) open = false;
	}
	function onKey(e: KeyboardEvent) {
		if (e.key === 'Escape') open = false;
	}
</script>

<svelte:window onclick={onWindowClick} onkeydown={onKey} />

{#if body}
	<span class="info" bind:this={root}>
		<button
			type="button"
			class="glyph"
			class:open
			aria-expanded={open}
			aria-label="explain"
			onclick={(e) => ((e.stopPropagation(), (open = !open)))}>i</button
		>
		{#if open}
			<span class="pop" role="note">{body}</span>
		{/if}
	</span>
{/if}

<style>
	.info {
		position: relative;
		display: inline-flex;
		vertical-align: middle;
		margin-left: 4px;
	}
	.glyph {
		all: unset;
		cursor: pointer;
		width: 13px;
		height: 13px;
		border-radius: 50%;
		border: 1px solid var(--border);
		color: var(--text-dim);
		font: italic 600 9px/1 serif;
		display: inline-flex;
		align-items: center;
		justify-content: center;
	}
	.glyph:hover,
	.glyph.open {
		color: var(--text);
		border-color: var(--text-dim);
	}
	.pop {
		position: absolute;
		top: calc(100% + 6px);
		left: -8px;
		z-index: 40;
		width: max-content;
		max-width: 280px;
		padding: 7px 9px;
		font: 400 11.5px/1.45 var(--sans, system-ui);
		font-style: normal;
		color: var(--text);
		background: var(--bg-inset);
		border: 1px solid var(--border);
		border-radius: 6px;
		box-shadow: 0 4px 16px rgb(0 0 0 / 0.25);
		white-space: normal;
	}
</style>
