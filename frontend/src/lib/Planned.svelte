<script lang="ts">
	// Placeholder panel for a view section that's specified but not built yet (spec 51).
	interface Props {
		title: string;
		items: string[];
		endpoints?: string[];
	}
	let { title, items, endpoints = [] }: Props = $props();
</script>

<section class="panel planned">
	<div class="panel-head">
		<h2>{title}</h2>
		<span class="badge">planned</span>
	</div>
	<ul>
		{#each items as item}
			<li>{item}</li>
		{/each}
	</ul>
	{#if endpoints.length}
		<div class="endpoints">
			{#each endpoints as e}
				<code>{e}</code>
			{/each}
		</div>
	{/if}
</section>

<style>
	.planned {
		border-style: dashed;
	}

	ul {
		margin: 0;
		padding-left: 18px;
		color: var(--text-dim);
	}

	li + li {
		margin-top: 2px;
	}

	.endpoints {
		margin-top: 10px;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.endpoints code {
		background: var(--bg-inset);
		padding: 1px 6px;
		border-radius: 3px;
		color: var(--text-dim);
	}
</style>
