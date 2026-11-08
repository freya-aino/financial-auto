<script lang="ts">
	// Live NATS tap over the shared /ws/bus connection (spec 50 §4.3, 51 §6.2). Rows are
	// appended once per animation frame; the socket itself is owned by `bus` (ws.svelte.ts).
	import { onDestroy } from 'svelte';
	import type { BusFrame } from './api';
	import { loadPref, savePref } from './prefs';
	import { bus, frameBatcher } from './ws.svelte';

	const MAX_ROWS = 200;

	let subjects = $state(loadPref('ops.tap', 'md.>,svc.heartbeat.>'));
	let rows = $state<BusFrame[]>([]);
	let paused = $state(false);
	let rate = $state(0);

	let count = 0;
	const rateTimer = setInterval(() => {
		rate = count;
		count = 0;
	}, 1000);

	const append = frameBatcher<BusFrame>((frames) => {
		if (paused) return;
		rows = [...frames.reverse(), ...rows].slice(0, MAX_ROWS);
	});

	let off: (() => void) | null = null;
	function tap() {
		off?.();
		rows = [];
		savePref('ops.tap', subjects);
		off = bus.subscribe(subjects, (f) => {
			count++;
			append(f);
		});
	}

	function summary(decoded: unknown): string {
		if (!decoded || typeof decoded !== 'object') return '';
		const d = decoded as Record<string, unknown>;
		if (d.type === 'heartbeat') return `${d.service} ${JSON.stringify(d.counters)}`;
		if (d.type === 'envelope') return `${d.source_id} ${JSON.stringify(d.payload)}`;
		return JSON.stringify(d);
	}

	tap();
	onDestroy(() => {
		clearInterval(rateTimer);
		off?.();
	});
</script>

<section class="panel tap">
	<div class="panel-head">
		<h2>Bus tap</h2>
		<form
			onsubmit={(e) => {
				e.preventDefault();
				tap();
			}}
		>
			<input class="mono" bind:value={subjects} size="32" aria-label="subjects" />
			<button type="submit">Tap</button>
			<button type="button" onclick={() => (paused = !paused)}>{paused ? 'Resume' : 'Pause'}</button>
		</form>
		<div class="meters">
			<span class="badge" class:ok={bus.status === 'open'} class:warn={bus.status === 'connecting'} class:danger={bus.status === 'closed'}>
				{bus.status}
			</span>
			<span class="mono">{rate} msg/s</span>
			{#if bus.dropped}<span class="badge warn" title="server rate limit (200 msg/s per socket)">{bus.dropped} dropped</span>{/if}
			<span class="dim" title="all patterns on the shared socket: {bus.patterns.join(', ')}">{bus.patterns.length} patterns</span>
		</div>
	</div>
	{#if bus.error}<p class="badge danger">{bus.error}</p>{/if}
	<div class="scroll body">
		<table class="mono">
			<thead><tr><th>time</th><th>subject</th><th>bytes</th><th>decoded</th></tr></thead>
			<tbody>
				{#each rows as r (r.seq)}
					<tr>
						<td class="dim">{r.ts.slice(11, 23)}</td>
						<td>{r.data.subject}</td>
						<td class="dim">{r.data.size}</td>
						<td class="decoded">{summary(r.data.decoded)}</td>
					</tr>
				{:else}
					<tr><td colspan="4" class="dim">no messages yet</td></tr>
				{/each}
			</tbody>
		</table>
	</div>
</section>

<style>
	form {
		display: flex;
		gap: 6px;
	}

	.meters {
		display: flex;
		gap: 8px;
		align-items: center;
	}

	.panel-head {
		flex-wrap: wrap;
	}

	.body {
		max-height: 360px;
	}

	.decoded {
		max-width: 520px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
