<script lang="ts">
	// JetStream streams and consumer lag (spec 51 §6.2 "JetStream consumer lag", D-046).
	import { onMount } from 'svelte';
	import Empty from './Empty.svelte';
	import { api, bytes, poll, type BusStream } from './api';
	import { runtime } from './runtime.svelte';

	let streams = $state<BusStream[]>([]);
	let error = $state('');
	let open = $state<string | null>(null);

	const totalMsgs = $derived(streams.reduce((a, s) => a + s.messages, 0));
	const totalBytes = $derived(streams.reduce((a, s) => a + s.bytes, 0));
	const maxBytes = $derived(Math.max(1, ...streams.map((s) => s.bytes)));
	const lagging = $derived(streams.flatMap((s) => s.consumers).filter((c) => c.num_pending > 0 || c.ack_pending > 0).length);

	async function refresh() {
		try {
			streams = (await api.streams()).streams.sort((a, b) => a.name.localeCompare(b.name));
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	const age = (secs: number) => (secs === 0 ? 'work queue' : secs >= 86400 ? `${Math.round(secs / 86400)} d` : `${Math.round(secs / 3600)} h`);

	onMount(() => poll(refresh, () => runtime.pollMs * 5));
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Streams</h2>
		<span class="dim">GET /api/bus/streams · {streams.length} streams · {totalMsgs.toLocaleString()} msgs · {bytes(totalBytes)}{lagging ? ` · ${lagging} consumer${lagging === 1 ? '' : 's'} with pending` : ''}</span>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if streams.length}
		<table>
			<thead><tr><th>stream</th><th>subjects</th><th>retention</th><th>messages</th><th>bytes</th><th>seq</th><th>consumers</th></tr></thead>
			<tbody>
				{#each streams as s (s.name)}
					<tr class:open={open === s.name} onclick={() => (open = open === s.name ? null : s.name)}>
						<td class="mono">{s.name}</td>
						<td class="mono dim">{s.subjects.join(', ')}</td>
						<td class="dim">{s.retention} · {s.storage} · {age(s.max_age_secs)}</td>
						<td class="mono">{s.messages.toLocaleString()}</td>
						<td>
							<div class="bar"><div style="width: {(100 * s.bytes) / maxBytes}%"></div></div>
							<span class="mono">{bytes(s.bytes)}</span>
						</td>
						<td class="mono dim">{s.first_seq}–{s.last_seq}</td>
						<td class="mono">
							{#if s.consumers.length}
								{#each s.consumers as c (c.name)}
									<span class="badge" class:warn={c.num_pending > 0 || c.ack_pending > 0} class:ok={!c.num_pending && !c.ack_pending} title="ack_pending {c.ack_pending} · num_pending {c.num_pending} · redelivered {c.redelivered} · waiting {c.waiting}">
										{c.name}{c.durable ? '' : ' (ephemeral)'} · pending {c.num_pending} · ack {c.ack_pending}
									</span>
								{/each}
							{:else}
								<span class="dim">none</span>
							{/if}
						</td>
					</tr>
					{#if open === s.name && s.consumers.length}
						<tr class="detail">
							<td colspan="7">
								<table>
									<thead><tr><th>consumer</th><th>durable</th><th>filter</th><th>ack pending</th><th>num pending</th><th>redelivered</th><th>waiting pulls</th><th>delivered seq</th><th>ack floor</th></tr></thead>
									<tbody>
										{#each s.consumers as c (c.name)}
											<tr>
												<td class="mono">{c.name}</td>
												<td>{c.durable ? 'yes' : 'no'}</td>
												<td class="mono dim">{c.filter_subjects.join(', ') || '—'}</td>
												<td class="mono" class:warn={c.ack_pending > 0}>{c.ack_pending}</td>
												<td class="mono" class:warn={c.num_pending > 0}>{c.num_pending}</td>
												<td class="mono">{c.redelivered}</td>
												<td class="mono">{c.waiting}</td>
												<td class="mono dim">{c.delivered_seq}</td>
												<td class="mono dim">{c.ack_floor_seq}</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</td>
						</tr>
					{/if}
				{/each}
			</tbody>
		</table>
	{:else if !error}
		<Empty producer="JetStream (GET /api/bus/streams needs NATS)" hint="docker compose up nats" />
	{/if}
</section>

<style>
	tr:not(.detail) {
		cursor: pointer;
	}
	.detail td {
		background: var(--bg-inset);
	}
	.detail table th {
		background: var(--bg-inset);
	}
	.bar {
		display: inline-block;
		width: 60px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
		margin-right: 6px;
	}
	.bar > div {
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
	}
	.warn {
		color: var(--warn);
	}
	.badge {
		margin: 1px 2px 1px 0;
	}
</style>
