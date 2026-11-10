<script lang="ts">
	// Services table (spec 51 §6.2): heartbeats with the gauge groups of D-053/D-055
	// (strategy-runtime `primitives.*`, orchestrator `validation.*` / `pareto.*`, model-runtime `hub.*`).
	import { onMount } from 'svelte';
	import { api, poll, type ServiceStatus } from './api';
	import { runtime } from './runtime.svelte';
	import { fmtCounter, groupCounters, primitiveShares } from './services';
	import { onBus } from './ws.svelte';

	let services = $state<ServiceStatus[]>([]);
	let error = $state('');
	let now = $state(Date.now());
	let expanded = $state<string | null>(null);

	async function refresh() {
		try {
			services = (await api.services()).services;
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	function age(iso: string): string {
		const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
		return s < 60 ? `${s}s` : s < 3600 ? `${Math.round(s / 60)}m` : `${Math.round(s / 3600)}h`;
	}

	onMount(() => {
		const stopPoll = poll(refresh, () => runtime.pollMs);
		const stopBus = onBus('svc.heartbeat.>', refresh, 300);
		const tick = setInterval(() => (now = Date.now()), 1000);
		return () => {
			stopPoll();
			stopBus();
			clearInterval(tick);
		};
	});
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Services</h2>
		<span class="dim">from svc.heartbeat.&gt; · click a row for every counter</span>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	<table>
		<thead><tr><th>service</th><th>instance</th><th>version</th><th>last seen</th><th>gauges</th></tr></thead>
		<tbody>
			{#each services as s (s.service)}
				{@const stale = now - Date.parse(s.last_seen) > 10_000}
				{@const g = groupCounters(s.counters)}
				{@const prim = primitiveShares(s.counters)}
				<tr onclick={() => (expanded = expanded === s.service ? null : s.service)}>
					<td><span class="badge" class:ok={!stale} class:warn={stale}>{s.service}</span></td>
					<td class="mono dim">{s.instance_id.slice(0, 8)}</td>
					<td class="mono">{s.version}</td>
					<td class="mono">{age(s.last_seen)} ago</td>
					<td class="gauges">
						{#if prim}
							<div class="prim" title="primitive calls answered from the latent graph / decoder cache / price fallback (D-053)">
								<span class="dim">primitives</span>
								<div class="stack">
									<div class="graph" style="flex: {prim.graph}" title="graph {(100 * prim.graph).toFixed(0)}%"></div>
									<div class="decoder" style="flex: {prim.decoder}" title="decoder {(100 * prim.decoder).toFixed(0)}%"></div>
									<div class="fallback" style="flex: {prim.fallback}" title="fallback {(100 * prim.fallback).toFixed(0)}%"></div>
								</div>
								<span class="mono">graph {(100 * prim.graph).toFixed(0)}% · decoder {(100 * prim.decoder).toFixed(0)}% · fallback {(100 * prim.fallback).toFixed(0)}%</span>
							</div>
						{/if}
						{#each g.gauges.filter((x) => x.prefix !== 'primitives') as grp (grp.prefix)}
							<div class="grp">
								<span class="dim">{grp.prefix}</span>
								{#each grp.items as it (it.key)}<span class="mono counter">{it.key}=<b>{fmtCounter(it.value)}</b></span>{/each}
							</div>
						{/each}
						{#if expanded === s.service}
							<div class="grp">
								{#each g.other as it (it.key)}<span class="mono counter">{it.key}={fmtCounter(it.value)}</span>{/each}
							</div>
							{#each Object.entries(s.counters).filter(([k]) => g.folded.some((f) => k.startsWith(f.prefix + '.'))) as [k, v] (k)}
								<span class="mono counter dim">{k}={fmtCounter(v)}</span>
							{/each}
						{:else}
							<div class="grp">
								{#each g.other.slice(0, 8) as it (it.key)}<span class="mono counter">{it.key}={fmtCounter(it.value)}</span>{/each}
								{#if g.other.length > 8}<span class="dim">+{g.other.length - 8} more</span>{/if}
								{#each g.folded as f (f.prefix)}<span class="badge" title="{f.count} counters, sum {fmtCounter(f.sum)}">{f.prefix}.* ×{f.count}</span>{/each}
							</div>
						{/if}
					</td>
				</tr>
			{:else}
				<tr><td colspan="5" class="dim">no heartbeats received yet (svc.heartbeat.&gt;)</td></tr>
			{/each}
		</tbody>
	</table>
</section>

<style>
	tbody tr {
		cursor: pointer;
	}
	.gauges {
		display: flex;
		flex-direction: column;
		gap: 3px;
		font-size: 11px;
	}
	.grp {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 8px;
		align-items: center;
	}
	.counter b {
		font-weight: 600;
	}
	.prim {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.stack {
		display: flex;
		width: 120px;
		height: 8px;
		border-radius: 4px;
		overflow: hidden;
		background: var(--bg-inset);
	}
	.stack .graph {
		background: var(--accent);
	}
	.stack .decoder {
		background: #7aa2f7;
	}
	.stack .fallback {
		background: var(--warn);
	}
</style>
