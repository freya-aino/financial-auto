<script lang="ts">
	// Global alert banner (D-066, spec 51 §6.1): `GET /api/alerts` every 15 s, `ops.alert`
	// transitions from the bus tap in between. A dead alert pipeline (no Watchdog) is shown as an
	// alert of its own. Hidden while everything is quiet.
	import { onMount } from 'svelte';
	import { poll } from './api';
	import { alertBanner, applyAlertEvent } from './alerts';
	import { consoleApi, errText, type AlertEvent, type AlertsView } from './console';
	import { bus } from './ws.svelte';

	let view = $state<AlertsView | null>(null);
	let error = $state('');
	let now = $state(Date.now());
	const banner = $derived(alertBanner(view, error, now));

	async function refresh() {
		now = Date.now();
		try {
			view = await consoleApi.alerts(20);
			error = '';
		} catch (e) {
			error = errText(e);
		}
	}

	onMount(() => {
		const stop = poll(refresh, 15_000);
		const off = bus.subscribe('ops.alert', (f) => {
			const ev = f.data.decoded as AlertEvent | null;
			if (view && ev && typeof ev === 'object') view = { ...view, active: applyAlertEvent(view.active, ev) };
		});
		return () => {
			stop();
			off();
		};
	});
</script>

{#if banner.level !== 'ok'}
	<div class="alert-banner {banner.level}" role="alert">
		<a href="/alerts/" class="badge {banner.level}">
			{#if banner.critical || banner.warning}{banner.critical} critical · {banner.warning} warning{:else}alerts{/if}
		</a>
		{#each banner.items as item (item.key)}
			<span class="item {item.level}" title={item.title ?? ''}>{item.text}</span>
		{/each}
		<a href="/alerts/" class="more">details →</a>
	</div>
{/if}

<style>
	.alert-banner {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
		padding: 5px 16px;
		font-size: 12px;
		border-bottom: 1px solid var(--warn);
		background: color-mix(in srgb, var(--warn) 10%, var(--bg-raised));
	}
	.alert-banner.danger {
		border-bottom-color: var(--danger);
		background: color-mix(in srgb, var(--danger) 12%, var(--bg-raised));
	}
	.item.danger {
		color: var(--danger);
	}
	.item.warn {
		color: var(--warn);
	}
	.badge {
		text-decoration: none;
		font-weight: 600;
	}
	.more {
		margin-left: auto;
	}
</style>
