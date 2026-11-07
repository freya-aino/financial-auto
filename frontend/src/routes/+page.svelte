<script lang="ts">
	import { onMount } from 'svelte';
	import BusTap from '../lib/BusTap.svelte';
	import ViewHeader from '../lib/ViewHeader.svelte';
	import JobsGpu from '../lib/JobsGpu.svelte';
	import { api, type ServiceStatus, type Source } from '../lib/api';
	import { loadPref, savePref } from '../lib/prefs';

	let services = $state<ServiceStatus[]>([]);
	let servicesError = $state('');
	let sources = $state<Source[]>([]);
	let sourcesError = $state('');
	let now = $state(Date.now());

	let category = $state(loadPref('ops.category', ''));
	let status = $state(loadPref('ops.status', ''));
	let capability = $state(loadPref('ops.capability', ''));
	let query = $state('');
	let ingestMsg = $state<Record<string, { ok: boolean; text: string }>>({});

	$effect(() => savePref('ops.category', category));
	$effect(() => savePref('ops.status', status));
	$effect(() => savePref('ops.capability', capability));

	async function ingest(id: string, action: 'start' | 'stop') {
		try {
			await api.ingest(id, action);
			ingestMsg = { ...ingestMsg, [id]: { ok: true, text: `${action} requested` } };
		} catch (e) {
			ingestMsg = { ...ingestMsg, [id]: { ok: false, text: e instanceof Error ? e.message : String(e) } };
		}
	}

	const categories = $derived([...new Set(sources.map((s) => s.category))].sort());
	const statuses = $derived([...new Set(sources.map((s) => s.status))].sort());
	const filtered = $derived(
		sources.filter(
			(s) =>
				(!category || s.category === category) &&
				(!status || s.status === status) &&
				(!capability || s.capabilities.includes(capability)) &&
				(!query || `${s.id} ${s.name} ${s.notes}`.toLowerCase().includes(query.toLowerCase()))
		)
	);

	function age(iso: string): string {
		const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
		return s < 60 ? `${s}s` : s < 3600 ? `${Math.round(s / 60)}m` : `${Math.round(s / 3600)}h`;
	}

	onMount(() => {
		const pollServices = async () => {
			try {
				services = (await api.services()).services;
				servicesError = '';
			} catch (e) {
				servicesError = String(e);
			}
		};
		api
			.sources()
			.then((c) => (sources = c.sources))
			.catch((e) => (sourcesError = String(e)));
		pollServices();
		const t1 = setInterval(pollServices, 2000);
		const t2 = setInterval(() => (now = Date.now()), 1000);
		return () => {
			clearInterval(t1);
			clearInterval(t2);
		};
	});
</script>

<ViewHeader title="Ops" spec="51 §6.2" />

<div class="grid top">
	<section class="panel">
		<div class="panel-head">
			<h2>Services</h2>
			<span class="dim">from svc.heartbeat.&gt;</span>
		</div>
		{#if servicesError}<p class="badge danger">{servicesError}</p>{/if}
		<table>
			<thead><tr><th>service</th><th>instance</th><th>version</th><th>last seen</th><th>counters</th></tr></thead>
			<tbody>
				{#each services as s (s.service)}
					{@const stale = now - Date.parse(s.last_seen) > 10_000}
					<tr>
						<td><span class="badge" class:ok={!stale} class:warn={stale}>{s.service}</span></td>
						<td class="mono dim">{s.instance_id.slice(0, 8)}</td>
						<td class="mono">{s.version}</td>
						<td class="mono">{age(s.last_seen)} ago</td>
						<td class="mono">
							{#each Object.entries(s.counters) as [k, v]}<span class="counter">{k}={v}</span>{/each}
						</td>
					</tr>
				{:else}
					<tr><td colspan="5" class="dim">no heartbeats received yet</td></tr>
				{/each}
			</tbody>
		</table>
	</section>
	<JobsGpu />
</div>

<BusTap />

<section class="panel sources">
	<div class="panel-head">
		<h2>Source catalog <span class="dim">({filtered.length}/{sources.length})</span></h2>
		<div class="filters">
			<input placeholder="search" bind:value={query} />
			<select bind:value={category} aria-label="category">
				<option value="">all categories</option>
				{#each categories as c}<option value={c}>{c}</option>{/each}
			</select>
			<select bind:value={status} aria-label="status">
				<option value="">all statuses</option>
				{#each statuses as s}<option value={s}>{s}</option>{/each}
			</select>
			<select bind:value={capability} aria-label="capability">
				<option value="">data + execution</option>
				<option value="data">data</option>
				<option value="execution">execution</option>
			</select>
		</div>
	</div>
	{#if sourcesError}<p class="badge danger">{sourcesError}</p>{/if}
	<div class="scroll catalog">
		<table>
			<thead>
				<tr><th>id</th><th>name</th><th>category</th><th>status</th><th>capabilities</th><th>assets</th><th>transport</th><th>auth</th><th>ingest</th><th>notes</th></tr>
			</thead>
			<tbody>
				{#each filtered as s (s.id)}
					<tr>
						<td class="mono">{s.id}</td>
						<td>{s.name}</td>
						<td>{s.category}</td>
						<td>
							<span class="badge" class:ok={s.status === 'live'} class:warn={s.status === 'stub'}>{s.status}</span>
						</td>
						<td>{s.capabilities.join(', ')}</td>
						<td class="dim">{s.asset_classes.join(', ')}</td>
						<td class="dim">{s.transport.join(', ')}</td>
						<td class="dim">{s.auth}</td>
						<td class="ingest">
							{#if s.capabilities.includes('data') && s.status !== 'planned'}
								<button onclick={() => ingest(s.id, 'start')} disabled={s.status === 'disabled'}>start</button>
								<button onclick={() => ingest(s.id, 'stop')}>stop</button>
								{#if ingestMsg[s.id]}
									<span class="badge" class:ok={ingestMsg[s.id].ok} class:warn={!ingestMsg[s.id].ok} title={ingestMsg[s.id].text}>{ingestMsg[s.id].ok ? 'sent' : 'not yet: ' + ingestMsg[s.id].text.slice(0, 60)}</span>
								{/if}
							{:else}
								<span class="dim">—</span>
							{/if}
						</td>
						<td class="dim notes">{s.notes}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</section>

<style>
	.top {
		grid-template-columns: 2fr 1fr;
		margin-bottom: 12px;
	}

	@media (max-width: 900px) {
		.top {
			grid-template-columns: 1fr;
		}
	}

	.counter + .counter {
		margin-left: 8px;
	}

	.sources {
		margin-top: 12px;
	}

	.filters {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}

	.catalog {
		max-height: 520px;
	}

	.notes {
		max-width: 360px;
	}

	.ingest {
		white-space: nowrap;
	}

	.ingest button {
		font-size: 11px;
		padding: 1px 6px;
	}
</style>
