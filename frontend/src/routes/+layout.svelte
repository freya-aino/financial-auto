<script lang="ts">
	// Global chrome (spec 51 §6.1): mode, kill switch, equity & P&L, dependency health, halts.
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import KillSwitch from '../lib/KillSwitch.svelte';
	import { api, eur, live, poll, type Health, type RiskView } from '../lib/api';

	let { children } = $props();

	const nav = [
		{ href: '/', label: 'Ops' },
		{ href: '/latent/', label: 'Latent & Graph' },
		{ href: '/evolution/', label: 'Evolution Lab' },
		{ href: '/trading/', label: 'Trading Desk' }
	];

	let health = $state<Health | null>(null);
	let healthError = $state(false);
	let risk = $state<RiskView | null>(null);
	let riskError = $state(false);

	const figures = $derived(risk?.reported ?? risk?.paper ?? null);
	const pnl = $derived(risk?.reported?.daily_pnl_eur ?? risk?.paper.pnl_eur ?? null);

	async function pollHealth() {
		try {
			health = await api.health();
			healthError = false;
		} catch {
			healthError = true;
		}
	}

	async function pollRisk() {
		try {
			risk = await live.risk();
			riskError = false;
		} catch {
			riskError = true;
		}
	}

	onMount(() => {
		const stopHealth = poll(pollHealth, 5000);
		const stopRisk = poll(pollRisk, 2000);
		return () => {
			stopHealth();
			stopRisk();
		};
	});
</script>

<div class="shell">
	<header class:halted={risk?.portfolio_halted}>
		<div class="brand">
			<img src="/favicon.svg" alt="" width="18" height="18" />
			<h1>financial-auto</h1>
			<span class="badge ok mode" title="Live trading is impossible from this UI (D-029)">PAPER</span>
		</div>
		<nav>
			{#each nav as item}
				<a href={item.href} class:active={page.url.pathname === item.href}>{item.label}</a>
			{/each}
		</nav>
		<div class="figures" class:stale={riskError}>
			{#if figures}
				<span title="equity ({risk?.reported ? 'risk service' : 'paper book'})"><span class="dim">equity</span> <b class="mono">{eur(figures.equity_eur)}</b></span>
				<span title={risk?.reported ? 'daily P&L' : 'P&L since control-api start'}>
					<span class="dim">P&amp;L</span>
					<b class="mono" class:pos={(pnl ?? 0) > 0} class:neg={(pnl ?? 0) < 0}>{eur(pnl)}</b>
				</span>
				{#if risk?.halts.length}<a href="/trading/" class="badge danger">{risk.halts.length} halt{risk.halts.length === 1 ? '' : 's'}</a>{/if}
			{/if}
		</div>
		<div class="status">
			{#if healthError}
				<span class="badge danger">control-api unreachable</span>
			{:else if health}
				<span class="badge" class:ok={health.nats.connected} class:danger={!health.nats.connected}>
					NATS {health.nats.connected ? 'up' : 'down'}
				</span>
				{#each Object.entries(health.dependencies) as [name, up]}
					<span class="dot" class:ok={up} class:warn={!up} title="{name}: {up ? 'reachable' : 'unreachable'}">{name}</span>
				{/each}
			{/if}
			<KillSwitch {risk} compact onchanged={pollRisk} />
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
		gap: 20px;
		padding: 8px 16px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-raised);
		flex-wrap: wrap;
		position: sticky;
		top: 0;
		z-index: 10;
	}

	header.halted {
		border-bottom: 2px solid var(--danger);
		background: color-mix(in srgb, var(--danger) 10%, var(--bg-raised));
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.mode {
		font-weight: 700;
		letter-spacing: 0.06em;
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

	.figures {
		display: flex;
		gap: 14px;
		align-items: center;
	}

	.figures.stale {
		opacity: 0.5;
	}

	.figures a {
		text-decoration: none;
	}

	.pos {
		color: var(--ok);
	}

	.neg {
		color: var(--danger);
	}

	.status {
		margin-left: auto;
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
		align-items: center;
	}

	.dot {
		font-size: 11px;
		color: var(--text-dim);
		padding-left: 12px;
		position: relative;
	}

	.dot::before {
		content: '';
		position: absolute;
		left: 0;
		top: 50%;
		width: 7px;
		height: 7px;
		margin-top: -3.5px;
		border-radius: 50%;
		background: var(--warn);
	}

	.dot.ok::before {
		background: var(--ok);
	}

	main {
		padding: 16px;
		flex: 1;
	}
</style>
