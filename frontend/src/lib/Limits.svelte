<script lang="ts">
	// Risk limits (spec 40 §6.5, D-051): GET/PUT /api/trading/limits, tighten-only. Loosening
	// fields are flagged before sending; a 422 `loosen:` reply is shown inline.
	import { onMount } from 'svelte';
	import { live, type Limits } from './api';
	import { diffLimits, limitsBody, loosening, parseLoosenError } from './limits';
	import { runtime } from './runtime.svelte';

	interface Props {
		onchanged?: () => void;
	}
	let { onchanged }: Props = $props();

	let current = $state<Limits | null>(null);
	let draft = $state<Limits | null>(null);
	let note = $state('');
	let error = $state('');
	let loosenMsg = $state('');
	let ok = $state('');
	let busy = $state(false);

	const body = $derived(current && draft ? limitsBody(current, draft) : {});
	const changes = $derived(current ? diffLimits(current, body) : []);
	const willLoosen = $derived(loosening(changes));
	const scalarKeys = $derived(Object.keys(current ?? {}).filter((k) => typeof current?.[k] === 'number').sort());

	async function load() {
		try {
			const r = await live.limits();
			current = r.limits;
			draft = structuredClone(r.limits);
			note = r.note ?? '';
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	async function save(ev: SubmitEvent) {
		ev.preventDefault();
		if (!Object.keys(body).length) return;
		if (!confirm(`Apply ${changes.length} limit change(s)? ${changes.map((c) => `${c.field}: ${c.from} → ${c.to}`).join(', ')}`)) return;
		busy = true;
		loosenMsg = '';
		ok = '';
		error = '';
		try {
			const r = await live.putLimits(body);
			current = r.limits;
			draft = structuredClone(r.limits);
			ok = 'limits applied (audited by the risk-engine)';
			onchanged?.();
		} catch (e) {
			const m = e instanceof Error ? e.message : String(e);
			const l = parseLoosenError(m);
			if (l) loosenMsg = l;
			else error = m;
		} finally {
			busy = false;
		}
	}

	const step = (k: string) => (k.endsWith('_eur') ? 100 : k === 'liquidity_k' ? 0.001 : 0.01);

	onMount(() => {
		void load();
		const t = setInterval(load, runtime.pollMs * 30);
		return () => clearInterval(t);
	});
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Risk limits</h2>
		<span class="dim">{note || 'PUT /api/trading/limits may only tighten'}</span>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if current && draft}
		<form onsubmit={save} class="lim">
			<div class="cols">
				<table>
					<thead><tr><th>limit</th><th>value</th><th>current</th></tr></thead>
					<tbody>
						{#each scalarKeys as k (k)}
							{@const ch = changes.find((c) => c.field === k)}
							<tr class:loosen={ch?.kind === 'loosen'} class:tighten={ch?.kind === 'tighten'}>
								<td class="mono">{k}</td>
								<td><input type="number" step={step(k)} min="0" bind:value={draft[k] as number} /></td>
								<td class="mono dim">{current[k]}</td>
							</tr>
						{/each}
					</tbody>
				</table>
				<table>
					<thead><tr><th>leverage cap</th><th>value</th><th>current</th></tr></thead>
					<tbody>
						{#each Object.keys(current.leverage_cap).sort() as cls (cls)}
							{@const ch = changes.find((c) => c.field === `leverage_cap.${cls}`)}
							<tr class:loosen={ch?.kind === 'loosen'} class:tighten={ch?.kind === 'tighten'}>
								<td class="mono">{cls}</td>
								<td><input type="number" step="0.1" min="0" bind:value={draft.leverage_cap[cls]} /></td>
								<td class="mono dim">{current.leverage_cap[cls]}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<div class="row">
				<button type="submit" disabled={busy || !changes.length}>{busy ? 'applying…' : `Apply ${changes.length} change${changes.length === 1 ? '' : 's'}`}</button>
				<button type="button" onclick={() => (draft = structuredClone(current))} disabled={!changes.length}>reset</button>
				{#if willLoosen.length}<span class="badge warn" title="the risk-engine answers 422 loosen: for these">would loosen: {willLoosen.join(', ')}</span>{/if}
				{#if ok}<span class="badge ok">{ok}</span>{/if}
			</div>
			{#if loosenMsg}
				<p class="loosen-msg"><span class="badge danger">422 loosen</span> {loosenMsg}</p>
			{/if}
		</form>
	{/if}
</section>

<style>
	.cols {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
	}
	@media (max-width: 800px) {
		.cols {
			grid-template-columns: 1fr;
		}
	}
	input {
		width: 100px;
	}
	tr.loosen td {
		color: var(--danger);
	}
	tr.tighten td {
		color: var(--accent);
	}
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
		margin-top: 8px;
	}
	.loosen-msg {
		margin: 8px 0 0;
		color: var(--danger);
	}
</style>
