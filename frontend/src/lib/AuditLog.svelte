<script lang="ts">
	// Audit log (`GET /api/audit?since=`): every operator and system mutation (D-047/D-051/D-054/D-055).
	import { onMount } from 'svelte';
	import { ago, api, poll, type AuditEntry } from './api';
	import { loadPref, savePref } from './prefs';
	import { runtime } from './runtime.svelte';

	let since = $state(loadPref('ops.audit.since', '24h'));
	let filter = $state('');
	let entries = $state<AuditEntry[]>([]);
	let error = $state('');
	let now = $state(Date.now());
	let open = $state<number | null>(null);
	$effect(() => savePref('ops.audit.since', since));

	const shown = $derived(
		[...entries]
			.reverse()
			.filter((e) => !filter || `${e.actor} ${e.action} ${e.subject ?? ''}`.toLowerCase().includes(filter.toLowerCase()))
	);
	const actions = $derived([...new Set(entries.map((e) => e.action))].sort());

	async function refresh() {
		now = Date.now();
		try {
			entries = (await api.audit(since, 300)).entries;
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	const actorClass = (a: string) => (a === 'ui' || a.startsWith('ui') ? 'ok' : a.startsWith('system') ? '' : 'warn');

	onMount(() => poll(refresh, () => runtime.pollMs * 5));
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Audit log <span class="dim">({shown.length}/{entries.length})</span></h2>
		<div class="filters">
			<select bind:value={since} onchange={refresh}>
				{#each ['1h', '6h', '24h', '7d', '30d'] as s}<option value={s}>last {s}</option>{/each}
			</select>
			<input bind:value={filter} placeholder="filter actor / action / subject" list="audit-actions" />
			<datalist id="audit-actions">{#each actions as a}<option value={a}></option>{/each}</datalist>
		</div>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if shown.length}
		<div class="scroll body">
			<table>
				<thead><tr><th>when</th><th>actor</th><th>action</th><th>subject</th><th>payload</th></tr></thead>
				<tbody>
					{#each shown as e (e.id)}
						<tr onclick={() => (open = open === e.id ? null : e.id)}>
							<td class="mono dim" title={e.at}>{ago(e.at, now)} ago</td>
							<td><span class="badge {actorClass(e.actor)}">{e.actor}</span></td>
							<td class="mono">{e.action}</td>
							<td class="mono dim">{e.subject ?? '—'}</td>
							<td class="mono payload" class:open={open === e.id}>{open === e.id ? JSON.stringify(e.payload, null, 1) : JSON.stringify(e.payload)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else if !error}
		<p class="dim">no audit rows in the last {since} (audit.log in Postgres)</p>
	{/if}
</section>

<style>
	.filters {
		display: flex;
		gap: 6px;
	}
	.body {
		max-height: 320px;
	}
	tbody tr {
		cursor: pointer;
	}
	.payload {
		max-width: 420px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 11px;
	}
	.payload.open {
		white-space: pre-wrap;
		word-break: break-all;
	}
</style>
