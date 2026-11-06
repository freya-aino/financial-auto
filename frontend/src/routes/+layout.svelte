<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { api, type Health } from '../lib/api';

	let { children } = $props();

	const nav = [
		{ href: '/', label: 'Ops' },
		{ href: '/latent/', label: 'Latent & Graph' },
		{ href: '/evolution/', label: 'Evolution Lab' },
		{ href: '/trading/', label: 'Trading Desk' }
	];

	let health = $state<Health | null>(null);
	let healthError = $state(false);

	onMount(() => {
		const poll = async () => {
			try {
				health = await api.health();
				healthError = false;
			} catch {
				healthError = true;
			}
		};
		poll();
		const t = setInterval(poll, 5000);
		return () => clearInterval(t);
	});
</script>

<div class="shell">
	<header>
		<div class="brand">
			<img src="/favicon.svg" alt="" width="18" height="18" />
			<h1>financial-auto</h1>
			<span class="badge">paper</span>
		</div>
		<nav>
			{#each nav as item}
				<a href={item.href} class:active={page.url.pathname === item.href}>{item.label}</a>
			{/each}
		</nav>
		<div class="status">
			{#if healthError}
				<span class="badge danger">control-api unreachable</span>
			{:else if health}
				<span class="badge" class:ok={health.nats.connected} class:danger={!health.nats.connected}>
					NATS {health.nats.connected ? 'up' : 'down'}
				</span>
				{#each Object.entries(health.dependencies) as [name, up]}
					<span class="badge" class:ok={up} class:warn={!up}>{name}</span>
				{/each}
			{/if}
		</div>
	</header>
	<main>
		{@render children()}
	</main>
</div>

<style>
	.shell {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}

	header {
		display: flex;
		align-items: center;
		gap: 24px;
		padding: 8px 16px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-raised);
		flex-wrap: wrap;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	nav {
		display: flex;
		gap: 4px;
	}

	nav a {
		color: var(--text-dim);
		text-decoration: none;
		padding: 4px 10px;
		border-radius: 4px;
	}

	nav a:hover {
		color: var(--text);
		background: var(--bg-inset);
	}

	nav a.active {
		color: var(--text);
		background: var(--bg-inset);
		box-shadow: inset 0 -2px 0 var(--accent);
	}

	.status {
		margin-left: auto;
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}

	main {
		padding: 16px;
		flex: 1;
	}
</style>
