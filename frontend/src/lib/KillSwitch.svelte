<script lang="ts">
	// Kill switch + re-arm (spec 51 §6.1, §8). Confirm dialog first; the shown result comes
	// from the API response, never optimistically. Compact mode fits the top bar.
	import { live, type RiskView } from './api';

	interface Props {
		risk: RiskView | null;
		compact?: boolean;
		onchanged?: () => void | Promise<void>;
	}
	let { risk, compact = false, onchanged }: Props = $props();

	let busy = $state(false);
	let msg = $state('');
	let ok = $state(true);
	let timer: ReturnType<typeof setTimeout> | undefined;

	async function run(success: string, action: () => Promise<unknown>) {
		busy = true;
		clearTimeout(timer);
		try {
			await action();
			msg = success;
			ok = true;
			await onchanged?.();
		} catch (e) {
			msg = e instanceof Error ? e.message : String(e);
			ok = false;
		} finally {
			busy = false;
			timer = setTimeout(() => (msg = ''), 6000);
		}
	}

	function kill() {
		const reason = prompt('KILL SWITCH: halt the whole portfolio and block all orders. Reason?', 'manual kill switch');
		if (reason === null) return;
		run('portfolio halted', () => live.halt(reason));
	}

	function resume() {
		if (!confirm('Re-arm trading for the portfolio?')) return;
		run('portfolio resumed', () => live.resume());
	}
</script>

<div class="ks" class:compact>
	{#if msg}<span class="badge" class:ok class:danger={!ok}>{msg}</span>{/if}
	{#if risk?.portfolio_halted}
		<span class="badge danger halted">PORTFOLIO HALTED</span>
		<button onclick={resume} disabled={busy}>Resume</button>
	{/if}
	<button class="kill" onclick={kill} disabled={busy || !risk || risk.portfolio_halted} title={risk ? 'Halt the portfolio (risk.halt.portfolio)' : 'control-api unreachable'}>
		KILL SWITCH
	</button>
</div>

<style>
	.ks {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.halted {
		font-weight: 700;
	}
	.kill {
		background: var(--danger);
		color: #fff;
		border: none;
		font-weight: 700;
		letter-spacing: 0.08em;
		padding: 8px 18px;
	}
	.compact .kill {
		padding: 3px 10px;
		font-size: 11px;
	}
</style>
