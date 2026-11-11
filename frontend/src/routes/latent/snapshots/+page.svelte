<script lang="ts">
	// Encoder snapshots (D-070, spec 31, 32, 36 §6): every snapshot per timescale with its family
	// (statistical / neural fsq_hybrid · jepa_vae), expert health on the 15 % time-ordered split
	// (K_eff, residual correlation, per-expert skill and usage), the full promotion checklist with
	// the neural items, shadow gauges, and the shadow / promote / rollback controls. Every action
	// keeps the confirm + audit flow (latent.snapshot.* in audit.log).
	import { onMount } from 'svelte';
	import Empty from '../../../lib/Empty.svelte';
	import Heatmap from '../../../lib/Heatmap.svelte';
	import ViewHeader from '../../../lib/ViewHeader.svelte';
	import { live, num, pct, poll, type Snapshot } from '../../../lib/api';
	import { since } from '../../../lib/alerts';
	import { consoleApi, errText, type SnapshotDetail } from '../../../lib/console';
	import { checklistRows, expertHealth, isNeuralFamily, missingNeuralChecks } from '../../../lib/models';
	import { loadPref, savePref } from '../../../lib/prefs';
	import { runtime } from '../../../lib/runtime.svelte';

	let snapshots = $state<Snapshot[]>([]);
	let listError = $state('');
	let selected = $state(loadPref('snapshots.selected', ''));
	let tsFilter = $state(loadPref('snapshots.ts', ''));
	let detail = $state<SnapshotDetail | null>(null);
	let detailError = $state('');
	let busy = $state(false);
	let msg = $state('');
	let ok = $state(true);
	let now = $state(Date.now());
	$effect(() => savePref('snapshots.selected', selected));
	$effect(() => savePref('snapshots.ts', tsFilter));

	const timescales = $derived([...new Set(snapshots.map((s) => s.timescale))].sort());
	const shown = $derived(snapshots.filter((s) => !tsFilter || s.timescale === tsFilter));
	const health = $derived(expertHealth(detail?.manifest ?? null));
	const checks = $derived(checklistRows(detail));
	const missing = $derived(missingNeuralChecks(detail));
	const snap = $derived(detail?.snapshot ?? null);
	const shadowGauges = $derived(Object.entries(detail?.shadow_gauges ?? {}));
	const labels = $derived((health?.resid_corr ?? []).map((_, i) => `E${i}`));
	const metrics = $derived(((detail?.manifest?.metrics ?? {}) as Record<string, number>));
	const params = $derived(((detail?.manifest?.params ?? {}) as Record<string, unknown>));
	const statusClass = (s: string) => (s === 'active' ? 'ok' : s === 'candidate' || s === 'shadow' ? 'warn' : s === 'retired' ? '' : 'danger');

	async function refreshList() {
		now = Date.now();
		try {
			snapshots = (await live.snapshots()).snapshots.sort((a, b) => a.timescale.localeCompare(b.timescale) || b.created_at.localeCompare(a.created_at));
			listError = '';
			if (!selected && snapshots.length) selected = (snapshots.find((s) => s.status === 'candidate') ?? snapshots[0]).snapshot_id;
		} catch (e) {
			listError = errText(e);
		}
	}

	async function refreshDetail() {
		if (!selected) return;
		try {
			detail = await consoleApi.snapshot(selected);
			detailError = '';
		} catch (e) {
			detailError = errText(e);
			detail = null;
		}
	}

	$effect(() => {
		void selected;
		void refreshDetail();
	});

	async function run(label: string, question: string, action: () => Promise<unknown>) {
		if (!confirm(question)) return;
		busy = true;
		try {
			await action();
			msg = `${label} ok`;
			ok = true;
		} catch (e) {
			msg = `${label}: ${errText(e)}`;
			ok = false;
		} finally {
			busy = false;
			await Promise.all([refreshList(), refreshDetail()]);
		}
	}

	function promote() {
		if (!snap) return;
		const gate = checks.passed ? 'The checklist is green.' : `The checklist is NOT green (${checks.rows.filter((r) => r.state !== 'pass').length} items open); the API refuses unless it passes.`;
		void run('promote', `Promote ${snap.snapshot_id} to the active ${snap.timescale} snapshot? Other active snapshots of ${snap.timescale} are retired. ${gate} This is audited (latent.snapshot.promote).`, () =>
			consoleApi.snapshotAction(snap.snapshot_id, 'promote')
		);
	}

	function rollback() {
		if (!snap) return;
		void run('rollback', `Roll back ${snap.snapshot_id}? It is retired and the newest other retired ${snap.timescale} snapshot becomes active again. This is audited (latent.snapshot.rollback).`, () =>
			consoleApi.snapshotAction(snap.snapshot_id, 'rollback')
		);
	}

	function shadow(start: boolean) {
		if (!snap) return;
		void run(
			start ? 'shadow start' : 'shadow stop',
			start
				? `Serve ${snap.snapshot_id} in shadow next to the active ${snap.timescale} encoder (model-runtime, ctl.model)? Shadow minutes and region agreement start counting. Audited (latent.snapshot.shadow.start).`
				: `Stop shadow serving of ${snap.snapshot_id}? Shadow gauges stop counting. Audited (latent.snapshot.shadow.stop).`,
			() => (start ? consoleApi.shadowStart(snap.snapshot_id) : consoleApi.shadowStop(snap.snapshot_id))
		);
	}

	onMount(() => {
		const a = poll(refreshList, () => runtime.pollMs * 15);
		const b = poll(refreshDetail, () => runtime.pollMs * 10);
		return () => {
			a();
			b();
		};
	});
</script>

<ViewHeader title="Encoder snapshots" spec="31 · 32 §6.4 · 36 §6" note="HelixDB snapshots + manifest; checklist from control-api; shadow / promote / rollback via ctl.model (confirm + audit)" />

<div class="layout">
	<section class="panel list">
		<div class="panel-head">
			<h2>Snapshots <span class="dim">({shown.length})</span></h2>
			<select bind:value={tsFilter} aria-label="timescale">
				<option value="">all timescales</option>
				{#each timescales as t}<option value={t}>{t}</option>{/each}
			</select>
		</div>
		{#if listError}<p class="badge danger">{listError}</p>{/if}
		{#if shown.length}
			<div class="scroll tall">
				<table>
					<thead><tr><th>snapshot</th><th>family</th><th>status</th><th>age</th></tr></thead>
					<tbody>
						{#each shown as s (s.snapshot_id)}
							<tr class:sel={s.snapshot_id === selected} onclick={() => (selected = s.snapshot_id)}>
								<td class="mono">{s.snapshot_id}<div class="dim small">{s.timescale} · gen {s.generation}</div></td>
								<td><span class="badge" class:ok={isNeuralFamily(s.encoder_family)}>{s.encoder_family}</span></td>
								<td><span class="badge {statusClass(s.status)}">{s.status}</span></td>
								<td class="mono dim">{since(s.created_at, now)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if !listError}
			<Empty producer="GET /api/latent/snapshots (HelixDB Snapshot nodes)" hint="" />
		{/if}
	</section>

	<div class="detail">
		{#if detailError}<p class="badge danger">{detailError}</p>{/if}
		{#if snap}
			<section class="panel head">
				<div>
					<h2 class="mono">{snap.snapshot_id}</h2>
					<span class="badge {statusClass(snap.status)}">{snap.status}</span>
					<span class="badge" class:ok={isNeuralFamily(snap.encoder_family)}>{snap.encoder_family}{isNeuralFamily(snap.encoder_family) ? ' (neural)' : ''}</span>
					<span class="dim">{snap.timescale} · generation {snap.generation} · parent {String(detail?.manifest?.parent_snapshot_id ?? '—')} · manifest from {detail?.manifest_source}</span>
				</div>
				<div class="actions">
					<button onclick={() => shadow(true)} disabled={busy || snap.status === 'active'} title="POST /api/latent/snapshots/{snap.snapshot_id}/shadow">shadow…</button>
					<button onclick={() => shadow(false)} disabled={busy || snap.status === 'active'} title="POST /api/latent/snapshots/{snap.snapshot_id}/shadow/stop">stop shadow…</button>
					<button onclick={promote} disabled={busy || snap.status === 'active'} title="POST /api/latent/snapshots/{snap.snapshot_id}/promote (refused unless the checklist passes)">promote…</button>
					<button onclick={rollback} disabled={busy || snap.status === 'retired'} title="POST /api/latent/snapshots/{snap.snapshot_id}/rollback">rollback…</button>
					{#if msg}<span class="badge" class:ok class:danger={!ok}>{msg}</span>{/if}
				</div>
			</section>

			<div class="grid two">
				<section class="panel">
					<div class="panel-head">
						<h2>Promotion checklist</h2>
						<span class="badge" class:ok={checks.passed} class:danger={!checks.passed}>{checks.passed ? 'passes' : 'blocked'}</span>
					</div>
					<table>
						<thead><tr><th>item</th><th>value</th><th>threshold</th><th>detail</th></tr></thead>
						<tbody>
							{#each checks.rows as c (c.name)}
								<tr>
									<td><span class="badge" class:ok={c.state === 'pass'} class:danger={c.state === 'fail'} class:warn={c.state === 'unknown'}>{c.state === 'pass' ? '✓' : c.state === 'fail' ? '✗' : '?'}</span> <span class="mono">{c.name}</span></td>
									<td class="mono">{num(c.value, 3)}</td>
									<td class="mono dim">{c.threshold ?? '—'}</td>
									<td class="small dim">{c.detail}</td>
								</tr>
							{/each}
							{#each missing as m}
								<tr><td><span class="badge warn">?</span> <span class="mono">{m}</span></td><td colspan="3" class="small dim">expected for neural snapshots (D-070) but not reported by the checklist</td></tr>
							{/each}
						</tbody>
					</table>
					{#if checks.notEvaluated.length}
						<h3 class="sub">Not evaluated</h3>
						<ul class="small dim">{#each checks.notEvaluated as n}<li>{n}</li>{/each}</ul>
					{/if}
					{#if shadowGauges.length}
						<h3 class="sub">Shadow gauges (model-runtime heartbeat)</h3>
						<div class="gauges">{#each shadowGauges as [k, v]}<span class="badge mono">{k} = {num(v, 3)}</span>{/each}</div>
					{/if}
				</section>

				<section class="panel">
					<div class="panel-head">
						<h2>Expert health</h2>
						<span class="dim">validation split{health?.n_windows ? `, ${health.n_windows} windows` : ''}</span>
					</div>
					{#if health}
						<div class="kpis">
							<div><span class="dim">K_eff</span><b class="mono" class:neg={(health.k_eff ?? 0) < 2}>{num(health.k_eff, 2)}</b><span class="dim small">≥ 2{health.pre_prune_k_eff !== null ? ` · pre-prune ${num(health.pre_prune_k_eff, 2)}` : ''}</span></div>
							<div><span class="dim">max |resid corr|</span><b class="mono" class:neg={(health.resid_corr_max ?? 0) > 0.1}>{num(health.resid_corr_max, 3)}</b><span class="dim small">≤ 0.1</span></div>
							<div><span class="dim">min skill</span><b class="mono" class:neg={Math.min(...health.skill) < 0.02}>{health.skill.length ? num(Math.min(...health.skill), 3) : '—'}</b><span class="dim small">≥ 0.02</span></div>
							<div><span class="dim">mixture skill</span><b class="mono">{num(health.mix_skill, 3)}</b><span class="dim small">{health.pruned ? `${health.pruned} pruned` : ''}</span></div>
						</div>
						<table>
							<thead><tr><th>expert</th><th>usage</th><th>skill vs naive</th></tr></thead>
							<tbody>
								{#each health.skill as sk, i}
									<tr>
										<td class="mono">E{i}</td>
										<td><div class="bar"><div style="width: {100 * (health.usage[i] ?? 0)}%"></div></div> <span class="mono">{pct(health.usage[i] ?? null, 1)}</span></td>
										<td class="mono" class:neg={sk < 0.02} class:pos={sk >= 0.02}>{num(sk, 3)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
						{#if health.resid_corr?.length}
							<h3 class="sub">Residual correlation between experts</h3>
							<Heatmap matrix={health.resid_corr} {labels} title="residual correlation" />
						{/if}
					{:else}
						<p class="dim">{isNeuralFamily(snap.encoder_family) ? 'The manifest carries no health block.' : 'Statistical snapshot: expert-space health is not part of its checklist (see Latent & Graph → experts for live routing statistics).'}</p>
					{/if}
				</section>

				<section class="panel">
					<div class="panel-head"><h2>Training</h2></div>
					<table>
						<tbody>
							{#each [['backbone', params.backbone], ['context steps', params.context_steps], ['d_model', params.d_model], ['expert_dim', params.expert_dim], ['experts', metrics.n_experts], ['params', metrics.n_params], ['device', params.device]] as [k, v]}
								<tr><th>{k}</th><td class="mono">{v ?? '—'}</td></tr>
							{/each}
							{#each [['train loss', metrics.train_loss], ['val loss', metrics.val_loss], ['val SSL loss', metrics.val_ssl_loss], ['val pred loss', metrics.val_pred_loss], ['carried mass', metrics.carried_mass], ['parent probe coverage', metrics.parent_probe_coverage], ['regions', metrics.regions], ['train seconds', metrics.train_seconds]] as [k, v]}
								<tr><th>{k}</th><td class="mono">{typeof v === 'number' ? num(v, 4) : '—'}</td></tr>
							{/each}
						</tbody>
					</table>
				</section>
			</div>
		{:else if !detailError}
			<Empty producer="GET /api/latent/snapshots/{selected || '…'}" hint="" />
		{/if}
	</div>
</div>

<style>
	.layout {
		display: grid;
		grid-template-columns: minmax(360px, 1fr) 2.4fr;
		gap: 12px;
		align-items: start;
	}
	@media (max-width: 1100px) {
		.layout {
			grid-template-columns: 1fr;
		}
	}
	.detail {
		display: flex;
		flex-direction: column;
		gap: 12px;
		min-width: 0;
	}
	.head {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		flex-wrap: wrap;
		align-items: center;
	}
	.head h2 {
		text-transform: none;
		color: var(--text);
		display: inline;
		margin-right: 8px;
	}
	.actions {
		display: flex;
		gap: 6px;
		align-items: center;
		flex-wrap: wrap;
	}
	.two {
		grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
	}
	.tall {
		max-height: 75vh;
	}
	tr.sel td {
		color: var(--accent);
	}
	tr {
		cursor: default;
	}
	.small {
		font-size: 11px;
	}
	.sub {
		font-size: 12px;
		margin: 10px 0 4px;
	}
	.kpis {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 8px;
		margin-bottom: 10px;
	}
	.kpis > div {
		display: flex;
		flex-direction: column;
	}
	.kpis b {
		font-size: 18px;
	}
	.gauges {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
	}
	.bar {
		display: inline-block;
		width: 90px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
	}
	.bar > div {
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
	}
	.neg {
		color: var(--danger);
	}
	.pos {
		color: var(--ok);
	}
	ul {
		margin: 0;
		padding-left: 18px;
	}
</style>
