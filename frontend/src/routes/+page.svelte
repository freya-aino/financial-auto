<script lang="ts">
	import { onMount } from 'svelte';
	import BusTap from '../lib/BusTap.svelte';
	import ViewHeader from '../lib/ViewHeader.svelte';
	import Planned from '../lib/Planned.svelte';
	import { api, type ServiceStatus, type Source } from '../lib/api';

	let services = $state<ServiceStatus[]>([]);
	let servicesError = $state('');
	let sources = $state<Source[]>([]);
	let sourcesError = $state('');
	let now = $state(Date.now());

	let category = $state('');
	let status = $state('');
	let capability = $state('');
	let query = $state('');

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
	<Planned
		title="Jobs & GPU"
		items={[
			'Work-queue jobs by priority class (live > adapters > evolution > training)',
			'Running job, preemptions, cancel',
			'GPU arbiter state (D-006)'
		]}
		endpoints={['GET /api/jobs', 'GET /api/gpu']}
	/>
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
				<tr><th>id</th><th>name</th><th>category</th><th>status</th><th>capabilities</th><th>assets</th><th>transport</th><th>auth</th><th>notes</th></tr>
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
</style>
