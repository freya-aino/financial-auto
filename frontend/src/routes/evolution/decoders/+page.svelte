<script lang="ts">
	// Neural decoders (D-068, spec 34 §4.5–4.6): model-runtime serves the best card per
	// (family, timescale) on decoder.output.<family> and shadows the next on decoder.shadow.<family>,
	// scores both online and promotes a shadow that wins by 2 % after 200 outputs and 30 min.
	// Operator control goes through control-api → ctl.decoders.* (confirmed, audited).
	import { onMount } from 'svelte';
	import Empty from '../../../lib/Empty.svelte';
	import ViewHeader from '../../../lib/ViewHeader.svelte';
	import { num, poll } from '../../../lib/api';
	import { since } from '../../../lib/alerts';
	import { consoleApi, errText, type DecoderCardRow, type DecoderServing, type DecoderSlot, type DecoderVerb } from '../../../lib/console';
	import { cardSummary, genomeParams, shadowAge, slotVerdict } from '../../../lib/models';
	import { runtime } from '../../../lib/runtime.svelte';
	import { bus } from '../../../lib/ws.svelte';

	let serving = $state<DecoderServing | null>(null);
	let servingError = $state('');
	let cards = $state<DecoderCardRow[]>([]);
	let cardsTotal = $state(0);
	let cardsError = $state('');
	let familyFilter = $state('');
	let busy = $state(false);
	let msg = $state('');
	let ok = $state(true);
	let now = $state(Date.now());
	let rates = $state<Record<string, { serving: number; shadow: number }>>({});

	const summaries = $derived(cards.map(cardSummary).filter((c) => !familyFilter || c.output_family === familyFilter || c.family === familyFilter));
	const servingKeys = $derived(new Set((serving?.slots ?? []).flatMap((s) => [s.serving?.key, s.shadow?.key]).filter(Boolean) as string[]));
	const loadErrors = $derived(Array.isArray(serving?.errors) ? serving.errors.map(String) : Array.from({ length: Number(serving?.errors ?? 0) }, () => 'error'));
	const families = $derived([...new Set(cards.map((c) => cardSummary(c).output_family))].sort());
	const roleOf = (key: string) => {
		for (const s of serving?.slots ?? []) {
			if (s.serving?.key === key) return 'serving';
			if (s.shadow?.key === key) return 'shadow';
		}
		return '';
	};

	async function refresh() {
		now = Date.now();
		try {
			serving = await consoleApi.decoderServing();
			servingError = '';
		} catch (e) {
			servingError = errText(e);
		}
	}

	async function refreshCards() {
		try {
			const r = await consoleApi.decoderCards(120);
			cards = r.cards;
			cardsTotal = r.total;
			cardsError = '';
		} catch (e) {
			cardsError = errText(e);
		}
	}

	async function act(verb: DecoderVerb, args: Record<string, string>, what: string) {
		if (!confirm(`${what}\n\nSends ctl.decoders.${verb} ${JSON.stringify(args)} to model-runtime; audited as decoders.${verb} (actor ui).`)) return;
		busy = true;
		try {
			const r = await consoleApi.decoderAction(verb, args);
			msg = `${verb}: ${typeof r.message === 'string' ? r.message : JSON.stringify(r.message)}`;
			ok = true;
		} catch (e) {
			msg = `${verb}: ${errText(e)}`;
			ok = false;
		} finally {
			busy = false;
			await refresh();
		}
	}

	const slotArgs = (s: DecoderSlot) => ({ family: s.family, timescale: s.timescale });

	onMount(() => {
		const stop = poll(refresh, () => runtime.pollMs * 5);
		const stopCards = poll(refreshCards, () => runtime.pollMs * 60);
		const tick = setInterval(() => (now = Date.now()), 1000);
		// outputs per role from the bus tap (decoder.output.* vs decoder.shadow.*), counted per family
		const off = bus.subscribe(['decoder.output.>', 'decoder.shadow.>'], (f) => {
			const parts = f.data.subject.split('.');
			const fam = parts.slice(2).join('.');
			const role = parts[1] === 'shadow' ? 'shadow' : 'serving';
			const cur = rates[fam] ?? { serving: 0, shadow: 0 };
			rates = { ...rates, [fam]: { ...cur, [role]: cur[role] + 1 } };
		});
		return () => {
			stop();
			stopCards();
			off();
			clearInterval(tick);
		};
	});
</script>

<ViewHeader title="Neural decoders" spec="34 §4.5–4.6 · 51 §6.4" note="model-runtime neural serving: serving vs shadow per (family, timescale), online CRPS / loss, ctl.decoders.*" />

<div class="toolbar panel">
	{#if serving}
		<span><b class="mono">{serving.cards}</b> <span class="dim">cards in Garage</span></span>
		<span><b class="mono">{serving.loaded}</b> <span class="dim">loaded candidates</span></span>
		<span class="badge" class:danger={loadErrors.length > 0} title={loadErrors.join('\n')}>load errors {loadErrors.length}</span>
		<span class="badge" title="FA_NEURAL_REPLACE_BASELINE">{serving.replace_baseline ? 'replaces baselines' : 'alongside baselines'}</span>
		<span class="dim">{serving.news} news items in the text timeline</span>
	{/if}
	<button onclick={() => act('reload', {}, 'Reload decoder cards from Garage now (normally every FA_NEURAL_RELOAD_SECS)?')} disabled={busy}>reload cards…</button>
	{#if msg}<span class="badge" class:ok class:danger={!ok}>{msg}</span>{/if}
	{#if servingError}<span class="badge danger" title="GET /api/decoders/serving">{servingError}</span>{/if}
</div>

<section class="panel">
	<div class="panel-head">
		<h2>Slots <span class="dim">({serving?.slots.length ?? 0})</span></h2>
		<span class="dim">online loss: CRPS of forecasts at horizon close, 1 − cos for news, raster CE for charts · lower is better · bus counts since page load</span>
	</div>
	{#if serving?.slots.length}
		<div class="scroll">
			<table>
				<thead><tr><th>family</th><th>ts</th><th>serving</th><th>shadow</th><th>promotion</th><th>bus</th><th></th></tr></thead>
				<tbody>
					{#each serving.slots as s (s.family + s.timescale)}
						{@const v = slotVerdict(s)}
						<tr class:off={s.disabled}>
							<td class="mono">{s.family}{#if s.disabled} <span class="badge danger">off</span>{/if}</td>
							<td class="mono">{s.timescale}</td>
							<td>
								{#if s.serving}
									<div class="mono key" title={s.serving.key}>{genomeParams(s.serving.genome ?? '')[0] || s.serving.key.split('/').pop()}</div>
									<div class="dim small">score {num(s.serving.score, 3)} · n {s.serving.online_n} · loss {num(s.serving.online_loss, 5)}{#if s.pinned === s.serving.key} · <span class="badge warn">pinned</span>{/if}</div>
								{:else}<span class="dim">baseline only</span>{/if}
							</td>
							<td>
								{#if s.shadow}
									<div class="mono key" title={s.shadow.key}>{genomeParams(s.shadow.genome ?? '')[0] || s.shadow.key.split('/').pop()}</div>
									<div class="dim small">score {num(s.shadow.score, 3)} · n {s.shadow.online_n} · loss {num(s.shadow.online_loss, 5)}</div>
									{@const age = shadowAge(s)}
									{#if age}<div class="small" title="shadow age vs the minimum before an automatic promotion"><span class="dim">shadow for</span> <span class="mono" class:pos={age.ripe}>{age.text}</span></div>{/if}
								{:else}<span class="dim">—</span>{/if}
							</td>
							<td>
								<div class="bar" title="scored outputs vs the 200 needed"><div style="width: {100 * v.progress}%" class:go={v.wouldPromote}></div></div>
								<span class="small" class:pos={v.wouldPromote}>{v.text}</span>
							</td>
							<td class="mono dim small">{rates[s.family]?.serving ?? 0} / {rates[s.family]?.shadow ?? 0}</td>
							<td class="actions">
								<button disabled={busy || !s.shadow || s.disabled} onclick={() => act('promote', slotArgs(s), `Promote the shadow of ${s.family}@${s.timescale} now (${s.shadow?.key}), ahead of the online rule? The promotion becomes a pin.`)}>promote…</button>
								{#if s.pinned}<button disabled={busy} onclick={() => act('unpin', slotArgs(s), `Unpin ${s.family}@${s.timescale} and return to the ranking?`)}>unpin…</button>{/if}
								{#if s.disabled}
									<button disabled={busy} onclick={() => act('on', slotArgs(s), `Re-enable neural serving for ${s.family}@${s.timescale}?`)}>on…</button>
								{:else}
									<button disabled={busy} onclick={() => act('off', slotArgs(s), `Disable neural serving for ${s.family}@${s.timescale}? Baselines keep publishing.`)}>off…</button>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else if serving}
		<p class="dim">No neural slot: no loaded card beats its naive baseline yet.</p>
	{:else if !servingError}
		<Empty producer="ctl.decoders.status (model-runtime neural serving)" hint="" />
	{/if}
</section>

<section class="panel cards">
	<div class="panel-head">
		<h2>Decoder cards <span class="dim">({summaries.length} of {cardsTotal}, newest first)</span></h2>
		<select bind:value={familyFilter} aria-label="output family">
			<option value="">all output families</option>
			{#each families as f}<option value={f}>{f}</option>{/each}
		</select>
	</div>
	{#if cardsError}<p class="badge danger">{cardsError}</p>{/if}
	{#if summaries.length}
		<div class="scroll tall">
			<table>
				<thead><tr><th>family → output</th><th>ts</th><th>score</th><th>skill</th><th>coverage</th><th>hit rate</th><th>trade PSR</th><th>generation</th><th>created</th><th></th></tr></thead>
				<tbody>
					{#each summaries as c (c.key)}
						{@const role = roleOf(c.key)}
						<tr>
							<td class="mono" title={c.genome}>{c.family} → {c.output_family}{#if role}<span class="badge" class:ok={role === 'serving'} class:warn={role === 'shadow'}> {role}</span>{/if}</td>
							<td class="mono">{c.timescale}</td>
							<td class="mono">{num(c.score, 3)}</td>
							<td><span class="badge" class:ok={c.beats_baseline === true} class:danger={c.beats_baseline === false}>{num(c.skill, 3)}</span></td>
							<td class="mono dim">{num(c.objectives.coverage, 2)}</td>
							<td class="mono dim">{num(c.objectives.hit_rate, 3)}</td>
							<td class="mono dim">{num(c.objectives.trade_psr, 3)}</td>
							<td class="mono dim">{c.generation.slice(-8)}</td>
							<td class="mono dim">{since(c.created, now)}</td>
							<td>
								<button disabled={busy || servingKeys.has(c.key) && role === 'serving'} onclick={() => act('pin', { key: c.key }, `Pin ${c.key} as the serving decoder of its slot (${c.output_family}@${c.timescale})? Only loaded candidates can be pinned.`)}>pin…</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else if !cardsError}
		<Empty producer="fa-models/decoders/**/*.card.json (evo-worker decoder generations)" hint="" />
	{/if}
</section>

<style>
	.toolbar {
		display: flex;
		gap: 14px;
		align-items: center;
		flex-wrap: wrap;
		margin-bottom: 12px;
	}
	.cards {
		margin-top: 12px;
	}
	.tall {
		max-height: 520px;
	}
	.small {
		font-size: 11px;
	}
	.key {
		max-width: 260px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.bar {
		display: inline-block;
		width: 70px;
		height: 6px;
		background: var(--bg-inset);
		border-radius: 3px;
		vertical-align: middle;
		margin-right: 6px;
	}
	.bar > div {
		height: 100%;
		background: var(--text-dim);
		border-radius: 3px;
	}
	.bar > div.go {
		background: var(--accent);
	}
	.pos {
		color: var(--ok);
	}
	tr.off td {
		opacity: 0.6;
	}
	.actions {
		white-space: nowrap;
	}
	.actions button,
	td button {
		font-size: 11px;
		padding: 1px 6px;
	}
</style>
