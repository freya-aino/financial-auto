<script lang="ts">
	// Generations table (spec 51 §6.4, D-049/D-052): REST list + `/ws/evolution` generation frames
	// + registry statistics, with pause / resume / stop actions.
	import { ago, live, short, type GenerationFrame, type GenerationRow, type Population } from './api';
	import { mergeGenerations, runActions, type GenerationTableRow } from './generations';

	interface Props {
		rows: GenerationRow[];
		frames: Map<string, GenerationFrame>;
		populations: Population[];
		now: number;
		onchanged?: () => void;
		onpick?: (row: GenerationTableRow) => void;
	}
	let { rows, frames, populations, now, onchanged, onpick }: Props = $props();

	let msg = $state<Record<string, { ok: boolean; text: string }>>({});
	let busy = $state<Record<string, boolean>>({});
	let open = $state<string | null>(null);

	const table = $derived(mergeGenerations(rows, frames, populations));

	async function act(id: string, action: 'pause' | 'resume' | 'stop') {
		if (action === 'stop' && !confirm(`Stop generation ${short(id, 12)}? Pending jobs are cancelled (STATUS_CANCELLED).`)) return;
		busy = { ...busy, [id]: true };
		try {
			await live.runAction(id, action);
			msg = { ...msg, [id]: { ok: true, text: `${action} ok` } };
			onchanged?.();
		} catch (e) {
			msg = { ...msg, [id]: { ok: false, text: e instanceof Error ? e.message : String(e) } };
		} finally {
			busy = { ...busy, [id]: false };
		}
	}

	const statusClass = (s: string) => (s === 'done' ? 'ok' : s === 'running' ? '' : s === 'paused' ? 'warn' : 'danger');
	const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${(100 * v).toFixed(0)}%`);
</script>

<div class="scroll tall">
	<table>
		<thead>
			<tr><th>generation</th><th>kind</th><th>status</th><th>loop</th><th>rung</th><th>candidates</th><th>front sizes</th><th>survivors / rung</th><th>deflated Sharpe</th><th>PBO</th><th>artifact</th><th>updated</th><th>actions</th></tr>
		</thead>
		<tbody>
			{#each table as g (g.id)}
				<tr class:open={open === g.id} onclick={() => ((open = open === g.id ? null : g.id), onpick?.(g))} title={g.source === 'ws' ? 'live frame from /ws/evolution' : 'REST row'}>
					<td class="mono">{short(g.id, 12)} {#if g.source === 'ws'}<span class="dot" title="live"></span>{/if}</td>
					<td class="mono dim">{g.kind || '—'}</td>
					<td><span class="badge {statusClass(g.status)}">{g.status}</span></td>
					<td class="mono">{g.generation + 1}/{Math.max(1, g.max_generations)}</td>
					<td class="mono">{g.rung + 1}/{Math.max(1, g.rungs)}{g.final ? ' ✓' : ''}</td>
					<td class="mono">{g.candidates}</td>
					<td class="mono dim">{g.front_sizes.join(' / ') || '—'}</td>
					<td class="mono">
						{#if g.survivors_by_rung.length}
							{#each g.survivors_by_rung as r (r.generation * 1000 + r.rung)}
								<span class="badge" title="loop {r.generation} rung {r.rung}: {r.ids.join(', ')}">{r.ids.length}</span>
							{/each}
						{:else}
							{g.survivors}
						{/if}
					</td>
					<td class="mono" title={g.validation ? `n_trials ${g.validation.n_trials} · best ${g.validation.best_candidate ?? ''}` : 'no validation.generation_stats yet'}>{g.validation?.deflated_sharpe?.toFixed(4) ?? '—'}</td>
					<td class="mono" class:warn={(g.validation?.pbo ?? 0) > 0.5}>{pct(g.validation?.pbo)}</td>
					<td class="mono dim" title={g.artifact_key ?? ''}>{g.artifact_key ? short(g.artifact_key.split('/').pop() ?? g.artifact_key, 14) : '—'}</td>
					<td class="mono dim">{ago(g.updated_at, now)}</td>
					<td class="actions" onclick={(e) => e.stopPropagation()}>
						{#each runActions(g.status) as a}
							<button disabled={busy[g.id]} onclick={() => act(g.id, a)} title="POST /api/evolution/runs/{g.id}/{a}">{a}</button>
						{/each}
						{#if msg[g.id]}<span class="badge" class:ok={msg[g.id].ok} class:danger={!msg[g.id].ok} title={msg[g.id].text}>{msg[g.id].text.slice(0, 50)}</span>{/if}
					</td>
				</tr>
				{#if open === g.id && frames.get(g.id)}
					{@const f = frames.get(g.id)!}
					<tr class="detail">
						<td colspan="13">
							<div class="dgrid">
								<div>
									<b>Pareto front</b> <span class="dim">({f.pareto_front.length} · objectives {f.objective_keys.join(', ')})</span>
									<ul class="mono front">
										{#each f.pareto_front.slice(0, 12) as m}
											<li title={m.candidate}>{m.candidate.length > 70 ? `${m.candidate.slice(0, 69)}…` : m.candidate} <span class="dim">{Object.entries(m.objectives).filter(([k]) => f.objective_keys.includes(k)).map(([k, v]) => `${k}=${v.toFixed(3)}`).join(' ')}</span></li>
										{/each}
									</ul>
								</div>
								<div>
									<b>Survivors</b> <span class="dim">({f.survivors.length}; next population {f.next_population || '—'}; vary job {f.vary_job_id ? short(f.vary_job_id, 10) : '—'}; elapsed {(f.elapsed_ms / 1000).toFixed(1)} s)</span>
									<ul class="mono front">{#each f.survivors.slice(0, 12) as s}<li title={s}>{s.length > 70 ? `${s.slice(0, 69)}…` : s}</li>{/each}</ul>
								</div>
							</div>
						</td>
					</tr>
				{/if}
			{:else}
				<tr><td colspan="13" class="dim">no generations yet (jobs.result.evo.generation; start one below)</td></tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.tall {
		max-height: 420px;
	}
	tbody tr:not(.detail) {
		cursor: pointer;
	}
	tr.open td {
		background: var(--bg-inset);
	}
	.detail td {
		background: var(--bg-inset);
	}
	.dot {
		display: inline-block;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--ok);
		vertical-align: middle;
	}
	.warn {
		color: var(--warn);
	}
	.actions {
		white-space: nowrap;
	}
	.actions button {
		font-size: 11px;
		padding: 1px 6px;
		margin-right: 3px;
	}
	.badge + .badge {
		margin-left: 2px;
	}
	.dgrid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
		font-size: 12px;
	}
	.front {
		margin: 4px 0 0;
		padding-left: 16px;
		font-size: 11px;
	}
</style>
