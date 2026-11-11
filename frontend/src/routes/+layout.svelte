<script lang="ts">
	// Global chrome (spec 51 §6.1): mode, kill switch, equity & P&L, dependency health, halts.
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import AlertBanner from '../lib/AlertBanner.svelte';
	import KillSwitch from '../lib/KillSwitch.svelte';
	import { api, eur, live, poll, type Health, type RiskView } from '../lib/api';
	import { runtime } from '../lib/runtime.svelte';
	import { bus, onBus, sockets, tradingWs } from '../lib/ws.svelte';

	let { children } = $props();

	// Sections with their sub-pages (spec 51 §4; panels that outgrew a view got their own route).
	const nav: { href: string; label: string; sub?: { href: string; label: string }[] }[] = [
		{ href: '/', label: 'Ops', sub: [{ href: '/', label: 'Overview' }, { href: '/alerts/', label: 'Alerts' }, { href: '/refdata/', label: 'Reference data' }] },
		{ href: '/latent/', label: 'Latent & Graph', sub: [{ href: '/latent/', label: 'Explorer' }, { href: '/latent/mcmc/', label: 'MCMC' }, { href: '/latent/snapshots/', label: 'Encoder snapshots' }] },
		{ href: '/evolution/', label: 'Evolution Lab', sub: [{ href: '/evolution/', label: 'Lab' }, { href: '/evolution/decoders/', label: 'Neural decoders' }] },
		{ href: '/trading/', label: 'Trading Desk', sub: [{ href: '/trading/', label: 'Desk' }, { href: '/trading/pipeline/', label: 'Strategy pipeline' }, { href: '/trading/risk/', label: 'Risk & halts' }, { href: '/trading/ledger/', label: 'Ledger' }] }
	];
	const path = $derived(page.url.pathname);
	const section = $derived(nav.find((n) => n.sub?.some((s) => s.href === path)) ?? nav.find((n) => n.href !== '/' && path.startsWith(n.href)) ?? nav[0]);

	let health = $state<Health | null>(null);
	let healthError = $state(false);
	let risk = $state<RiskView | null>(null);
	let riskError = $state(false);

	let now = $state(Date.now());
	const figures = $derived(risk?.reported ?? risk?.paper ?? null);
	const pnl = $derived(risk?.reported?.daily_pnl_eur ?? risk?.paper.pnl_eur ?? null);
	/** Sockets that are wanted but not open (spec 51 §8 banner), with their reconnect countdown. */
	const staleSockets = $derived(
		Object.entries(sockets)
			.filter(([, s]) => s.stale)
			.map(([name, s]) => ({
				name,
				status: s.status,
				attempt: s.attempt,
				error: s.error,
				retryIn: s.nextRetryAt === null ? null : Math.max(0, Math.ceil((s.nextRetryAt - now) / 1000)),
				reconnect: () => s.reconnectNow()
			}))
	);
	const anyStale = $derived(staleSockets.length > 0);
	const lagged = $derived(bus.dropped + tradingWs.lagged + sockets.latent.lagged + sockets.evolution.lagged);

	// The top-bar risk figures come from /ws/trading (`risk` topic) while it is open; REST is the fallback.
	const riskTopic = tradingWs.topic<RiskView>('risk');
	$effect(() => {
		if (riskTopic.data) {
			risk = riskTopic.data;
			riskError = false;
		}
	});

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
		void runtime.load();
		const stopHealth = poll(pollHealth, 5000);
		const stopRisk = poll(pollRisk, () => (tradingWs.live ? runtime.pollMs * 10 : runtime.pollMs));
		const tick = setInterval(() => (now = Date.now()), 500);
		const stopBus = onBus('risk.>', pollRisk);
		return () => {
			stopHealth();
			stopRisk();
			stopBus();
			clearInterval(tick);
			riskTopic.close();
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
				<a href={item.href} class:active={section === item}>{item.label}</a>
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
			<span class="dot sock" class:ok={tradingWs.live} class:warn={!tradingWs.live} title="/ws/trading {tradingWs.status}{tradingWs.flushMs ? ` · flush ${tradingWs.flushMs} ms` : ''}{lagged ? ` · ${lagged} frames superseded/dropped across sockets` : ''}">ws</span>
			<KillSwitch {risk} compact onchanged={pollRisk} />
		</div>
	</header>
	{#if section.sub}
		<nav class="subnav" aria-label="{section.label} pages">
			{#each section.sub as item}
				<a href={item.href} class:active={path === item.href}>{item.label}</a>
			{/each}
		</nav>
	{/if}
	<AlertBanner />
	{#if anyStale || healthError || riskError}
		<div class="stale-banner" role="status">
			{#if healthError || riskError}
				<span class="badge danger">control-api unreachable</span>
				<span>Figures are stale; polling continues every {(runtime.pollMs / 1000).toFixed(1)} s.</span>
			{/if}
			{#each staleSockets as s (s.name)}
				<span class="badge warn">/ws/{s.name} {s.status}</span>
				<span>
					{#if s.status === 'connecting'}Connecting…{:else if s.retryIn !== null}Reconnecting in {s.retryIn}s (attempt {s.attempt}).{/if}
					{#if s.error}<span class="dim">{s.error}</span>{/if}
				</span>
				<button type="button" onclick={s.reconnect}>Retry now</button>
			{/each}
			{#if anyStale}<span class="dim">Live updates paused on those streams; REST polling every {(runtime.pollMs / 1000).toFixed(1)} s is the fallback.</span>{/if}
		</div>
	{/if}
	<main class:stale={anyStale}>
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

	.subnav {
		display: flex;
		gap: 2px;
		padding: 2px 16px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-inset);
		font-size: 12px;
		flex-wrap: wrap;
	}

	.subnav a {
		padding: 3px 10px;
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

	main.stale :global(.panel) {
		border-style: dashed;
	}

	.stale-banner {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
		padding: 6px 16px;
		background: color-mix(in srgb, var(--warn) 12%, var(--bg-raised));
		border-bottom: 1px solid var(--warn);
		font-size: 12px;
	}
</style>
