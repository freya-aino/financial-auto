<script lang="ts">
	interface Props {
		title: string;
		spec: string;
		note?: string;
	}
	let { title, spec, note = '' }: Props = $props();
</script>

<div class="view-head">
	<h1>{title}</h1>
	<span class="dim">spec {spec}</span>
	{#if note}<span class="dim">· {note}</span>{/if}
</div>

<style>
	.view-head {
		display: flex;
		align-items: baseline;
		gap: 10px;
		margin-bottom: 12px;
	}
</style>
