<script lang="ts">
	// Live NATS tap over the shared /ws/bus connection (spec 50 §4.3, 51 §6.2). Rows are
	// appended once per animation frame; the socket itself is owned by `bus` (ws.svelte.ts).
	import { onDestroy } from 'svelte';
	import type { BusFrame } from './api';
	import { SubjectRates, sparkPoints, type SubjectRate } from './busrates';
	import { loadPref, savePref } from './prefs';
	import { bus, frameBatcher } from './ws.svelte';

	const MAX_ROWS = 200;

	let subjects = $state(loadPref('ops.tap', 'md.>,svc.heartbeat.>'));
	let rows = $state<BusFrame[]>([]);
	let paused = $state(false);
	let rate = $state(0);

	// per-subject rates (spec 51 §6.2): 60 s window, subjects folded to `depth` tokens
	let depth = $state(loadPref('ops.tap.depth', 3));
	const rates = new SubjectRates(60); // depth set by the effect below
	let perSubject = $state<SubjectRate[]>([]);
	$effect(() => {
		savePref('ops.tap.depth', depth);
		rates.reset(depth);
		perSubject = [];
	});

	let count = 0;
	const rateTimer = setInterval(() => {
		rate = count;
		count = 0;
		rates.tick();
		perSubject = rates.top(12);
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
		rates.reset(depth);
		off = bus.subscribe(subjects, (f) => {
			count++;
			rates.hit(f.data.subject);
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
	<div class="rates">
		<div class="rates-head">
			<span class="dim">msg/s per subject, last 60 s</span>
			<label class="dim">fold to
				<select bind:value={depth} aria-label="subject depth">
					<option value={0}>full subject</option>
					{#each [1, 2, 3, 4] as d}<option value={d}>{d} token{d > 1 ? 's' : ''}</option>{/each}
				</select>
			</label>
		</div>
		{#if perSubject.length}
			<table class="mono">
				<tbody>
					{#each perSubject as r (r.key)}
						<tr>
							<td class="subj" title={r.key}>{r.key}</td>
							<td><svg viewBox="0 0 120 18" class="mini" preserveAspectRatio="none" role="img" aria-label="{r.key} rate, peak {Math.max(...r.series)} msg/s"><polyline points={sparkPoints(r.series, 120, 18)} /></svg></td>
							<td class="num">{r.last}/s</td>
							<td class="num dim">avg {r.mean.toFixed(1)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<p class="dim">waiting for messages on the tapped subjects…</p>
		{/if}
	</div>
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

	.rates {
		margin-bottom: 8px;
	}
	.rates-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 11px;
	}
	.subj {
		max-width: 260px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.mini {
		width: 120px;
		height: 18px;
		display: block;
	}
	.mini polyline {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.5;
		vector-effect: non-scaling-stroke;
	}
	.num {
		text-align: right;
	}

	.decoded {
		max-width: 520px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
