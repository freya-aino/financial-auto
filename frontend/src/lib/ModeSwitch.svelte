<script lang="ts">
	// The real-money switch (D-121). Paper and live are one execution path; this is the button.
	//
	// It is deliberately unglamorous and deliberately loud. The switch is only offered when the
	// server would accept it (`canGoLive` / `canGoPaper` mirror the risk engine's and the
	// gateway's own gates), every blocker is listed when it is not, and the confirmation names the
	// sleeve and the per-order cap — the amounts actually at risk — before anything is sent.
	//
	// The result always comes from the API, never optimistically: a refusal is shown as the server
	// worded it, because the reason ("the book is not flat", "no credentials for kraken") is what
	// the operator has to act on.
	import Info from './Info.svelte';
	import { consoleApi, errText, type TradingModeView } from './console';
	import { canGoLive, canGoPaper, liveBlockers, modeLabel, switchConfirmText, venueLines, venueReconLine } from './mode';

	let { onchanged }: { onchanged?: () => void } = $props();

	let mode = $state<TradingModeView | null>(null);
	let error = $state('');
	let msg = $state('');
	let ok = $state(true);
	let busy = $state(false);

	const blockers = $derived(liveBlockers(mode));
	const venues = $derived(venueLines(mode));
	const reconLine = $derived(venueReconLine(mode));
	const live = $derived(mode?.mode === 'live');

	export async function refresh() {
		try {
			mode = await consoleApi.tradingMode();
			error = '';
		} catch (e) {
			error = errText(e);
		}
	}

	$effect(() => {
		void refresh();
	});

	async function to(want: 'live' | 'paper') {
		if (!mode || busy) return;
		if (!confirm(switchConfirmText(mode, want))) return;
		// Going live asks twice. The second prompt is typed, not clicked: this is the one action
		// in the platform that spends money, and a stray double-click must not do it.
		if (want === 'live') {
			const typed = prompt('Type REAL MONEY to confirm:', '');
			if (typed !== 'REAL MONEY') {
				msg = 'not switched';
				ok = true;
				return;
			}
		}
		busy = true;
		try {
			const r = await consoleApi.setTradingMode(want);
			const now = (r.mode as { mode?: string })?.mode ?? want;
			msg = `mode is ${now}${r.audited ? ' (audited)' : ' (audit not written: no Postgres)'}`;
			ok = true;
		} catch (e) {
			msg = errText(e);
			ok = false;
		} finally {
			busy = false;
			await refresh();
			onchanged?.();
		}
	}
</script>

<div class="mode" class:live>
	<div class="row">
		<span class="label">Trading mode</span>
		<span class="badge" class:danger={live} class:ok={!live}>{modeLabel(mode)}</span>
		<Info text="Paper and live are one execution path: the same allocator, the same risk chain, the same adapters. Switching stamps every new intent `live` and sends it to a real venue, capped by the sleeve and the per-order cap in risk.toml. The book must be flat in either direction." />
	</div>

	{#if error}
		<p class="danger">The trading mode could not be read: {error}</p>
	{:else if mode}
		{#if live}
			<p class="warn">
				This platform is trading <strong>real money</strong>. At most {mode.sleeve_eur.toLocaleString()} EUR is
				deployable and no order may exceed {mode.max_order_eur.toLocaleString()} EUR.
			</p>
			<button class="danger" onclick={() => to('paper')} disabled={busy || !canGoPaper(mode)}>
				{busy ? 'switching…' : 'Back to paper money'}
			</button>
			{#if !canGoPaper(mode)}
				<ul class="blockers">
					{#each mode.switch_blockers as b, i (i)}<li>{b}</li>{/each}
				</ul>
			{/if}
		{:else if canGoLive(mode)}
			<button class="danger" onclick={() => to('live')} disabled={busy}>
				{busy ? 'switching…' : 'Switch to real money'}
			</button>
			<p class="dim">
				Sleeve {mode.sleeve_eur.toLocaleString()} EUR · {mode.max_order_eur.toLocaleString()} EUR per order.
			</p>
		{:else}
			<p class="dim">Real money is not available yet:</p>
			<ul class="blockers">
				{#each blockers as b, i (i)}<li>{b}</li>{/each}
			</ul>
		{/if}
		{#if reconLine}<p class="dim">{reconLine}</p>{/if}
		{#if venues.length}
			<ul class="venues">
				{#each venues as v, i (i)}<li>{v}</li>{/each}
			</ul>
		{/if}
	{/if}
	{#if msg}<span class="badge" class:ok class:danger={!ok}>{msg}</span>{/if}
</div>

<style>
	.mode {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		min-width: 18rem;
	}
	.mode.live {
		border-left: 3px solid var(--bad, #e06c6c);
		padding-left: 0.6rem;
	}
	.row {
		display: flex;
		align-items: baseline;
		gap: 0.4rem;
	}
	.label {
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--muted);
	}
	p {
		margin: 0;
		font-size: 0.85rem;
	}
	.warn {
		color: var(--warn, #e0b000);
	}
	.danger {
		color: var(--bad, #e06c6c);
	}
	button.danger {
		color: inherit;
		border-color: var(--bad, #e06c6c);
		font-weight: 600;
		align-self: flex-start;
	}
	.blockers,
	.venues {
		margin: 0;
		padding-left: 1.1rem;
		font-size: 0.8rem;
		color: var(--muted);
	}
	.dim {
		color: var(--muted);
	}
</style>
