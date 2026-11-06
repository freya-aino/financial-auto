<script lang="ts">
	// Live NATS tap over /ws/bus (spec 50 §4.3, 51 §6.2).
	import { onDestroy } from 'svelte';
	import { busTapUrl, type BusFrame } from './api';

	const MAX_ROWS = 200;

	let subjects = $state('md.>,svc.heartbeat.>');
	let rows = $state<BusFrame[]>([]);
	let connected = $state(false);
	let paused = $state(false);
	let rate = $state(0);
	let dropped = $state(0);
	let error = $state('');

	let ws: WebSocket | null = null;
	let count = 0;
	const rateTimer = setInterval(() => {
		rate = count;
		count = 0;
	}, 1000);

	function connect() {
		ws?.close();
		error = '';
		rows = [];
		dropped = 0;
		const sock = new WebSocket(busTapUrl(subjects));
		ws = sock;
		sock.onopen = () => (connected = true);
		sock.onclose = () => {
			if (ws === sock) connected = false;
		};
		sock.onerror = () => (error = 'websocket error');
		sock.onmessage = (ev) => {
			const msg = JSON.parse(ev.data);
			if (msg.op === 'lagged') {
				dropped += msg.dropped;
				return;
			}
			if (msg.op === 'error') {
				error = msg.error;
				return;
			}
			count++;
			if (!paused) rows = [msg as BusFrame, ...rows].slice(0, MAX_ROWS);
		};
	}

	function summary(decoded: unknown): string {
		if (!decoded || typeof decoded !== 'object') return '';
		const d = decoded as Record<string, unknown>;
		if (d.type === 'heartbeat') return `${d.service} ${JSON.stringify(d.counters)}`;
		if (d.type === 'envelope') return `${d.source_id} ${JSON.stringify(d.payload)}`;
		return JSON.stringify(d);
	}

	connect();
	onDestroy(() => {
		clearInterval(rateTimer);
		ws?.close();
	});
</script>

<section class="panel tap">
	<div class="panel-head">
		<h2>Bus tap</h2>
		<form
			onsubmit={(e) => {
				e.preventDefault();
				connect();
			}}
		>
			<input class="mono" bind:value={subjects} size="32" aria-label="subjects" />
			<button type="submit">Tap</button>
			<button type="button" onclick={() => (paused = !paused)}>{paused ? 'Resume' : 'Pause'}</button>
		</form>
		<div class="meters">
			<span class="badge" class:ok={connected} class:danger={!connected}>{connected ? 'live' : 'closed'}</span>
			<span class="mono">{rate} msg/s</span>
			{#if dropped}<span class="badge warn">{dropped} dropped</span>{/if}
		</div>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
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
