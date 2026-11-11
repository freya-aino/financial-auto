<script lang="ts">
	// Services table (spec 51 §6.2): heartbeats with the gauge groups of D-053/D-055
	// (strategy-runtime `primitives.*`, orchestrator `validation.*` / `pareto.*`, model-runtime `hub.*`).
	import { onMount } from 'svelte';
	import { ApiError, api, poll, type ServiceStatus } from './api';
	import { runtime } from './runtime.svelte';
	import { SERVICE_ACTIONS, actionBody, actionGate, confirmText, describeResult, fmtCounter, groupCounters, primitiveShares, type ServiceAction, type Supervisor } from './services';
	import { onBus } from './ws.svelte';

	let services = $state<ServiceStatus[]>([]);
	let error = $state('');
	let now = $state(Date.now());
	let expanded = $state<string | null>(null);
	let supervisor = $state<Supervisor | null>(null);
	let busy = $state<string | null>(null);
	let msg = $state('');
	let msgOk = $state(true);

	/** Compose services without a heartbeat row (e.g. stopped ones) still get a control row. */
	const silent = $derived((supervisor?.services ?? []).filter((n) => !services.some((s) => s.service === n)));
	let silentContainers = $state<Record<string, ServiceStatus['container']>>({});
	let silentInfo = $state<Record<string, string>>({});
	let inspectedAt = 0;

	async function inspectSilent(names: string[]) {
		const out: Record<string, ServiceStatus['container']> = {};
		const info: Record<string, string> = {};
		await Promise.all(
			names.map(async (n) => {
				try {
					const c = await api.serviceContainer(n);
					out[n] = { name: c.name, id: c.id, image: c.image, state: c.state, status: c.state };
					info[n] = [c.exit_code !== null && !c.running ? `exit ${c.exit_code}` : '', c.restart_count ? `${c.restart_count} restarts` : '', c.health ?? ''].filter(Boolean).join(' · ');
				} catch {
					out[n] = null;
				}
			})
		);
		silentContainers = out;
		silentInfo = info;
	}

	async function act(service: string, action: ServiceAction) {
		if (!confirm(confirmText(service, action))) return;
		busy = `${service}:${action}`;
		msg = '';
		try {
			const r = await api.serviceAction(service, action, actionBody(service, action));
			msg = describeResult(service, action, r);
			msgOk = true;
		} catch (e) {
			msg = e instanceof ApiError ? describeResult(service, action, null, { status: e.status, body: e.body, message: e.message }) : String(e);
			msgOk = false;
		} finally {
			busy = null;
			inspectedAt = 0;
			void refresh();
		}
	}

	async function refresh() {
		try {
			const r = await api.services();
			services = r.services;
			supervisor = r.supervisor ?? null;
			error = '';
			const names = (supervisor?.available ? (supervisor.services ?? []) : []).filter((n) => !r.services.some((x) => x.service === n));
			// docker inspect at most every 10 s (refresh also runs on every heartbeat burst)
			if (names.length && Date.now() - inspectedAt > 10_000) {
				inspectedAt = Date.now();
				void inspectSilent(names);
			}
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
		<span class="badge" class:ok={supervisor?.available} class:warn={!supervisor?.available} title={supervisor?.available ? `compose project ${supervisor.project} via ${supervisor.socket}` : (supervisor?.detail ?? 'GET /api/services did not report a supervisor')}>
			docker {supervisor?.available ? 'control on' : 'control off'}
		</span>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if msg}<p class="badge" class:ok={msgOk} class:danger={!msgOk}>{msg}</p>{/if}
	{#snippet controls(name: string, container: ServiceStatus['container'])}
		<td class="ctl" onclick={(e) => e.stopPropagation()}>
			<span class="mono dim" title={container?.status ?? ''}>{container?.state ?? '—'}</span>
			{#each SERVICE_ACTIONS as a (a)}
				{@const g = actionGate(name, a, container, supervisor)}
				<button class:danger={a !== 'start'} disabled={!g.enabled || busy !== null} title={g.why || `${a} ${name} (POST /api/services/${name}/${a})`} onclick={() => act(name, a)}>
					{busy === `${name}:${a}` ? '…' : a}
				</button>
			{/each}
		</td>
	{/snippet}
	<table>
		<thead><tr><th>service</th><th>instance</th><th>version</th><th>last seen</th><th>container</th><th>gauges</th></tr></thead>
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
					{@render controls(s.service, s.container)}
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
				<tr><td colspan="6" class="dim">no heartbeats received yet (svc.heartbeat.&gt;)</td></tr>
			{/each}
			{#each silent as name (name)}
				<tr>
					<td><span class="badge warn">{name}</span></td>
					<td class="dim" colspan="3">no heartbeat (compose service){silentInfo[name] ? ` · ${silentInfo[name]}` : ''}</td>
					{@render controls(name, silentContainers[name])}
					<td></td>
				</tr>
			{/each}
		</tbody>
	</table>
</section>

<style>
	.ctl {
		white-space: nowrap;
	}
	.ctl button {
		font-size: 11px;
		padding: 1px 6px;
	}
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
