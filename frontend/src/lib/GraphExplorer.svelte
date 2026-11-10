<script lang="ts">
	// Graph Explorer (spec 50 §4.4, D-048): HelixDB neighbourhood around any business id and
	// the whitelisted named-query catalog with a parameter form.
	import { onMount } from 'svelte';
	import { ApiError, live, type GraphQueryDef, type Neighborhood } from './api';
	import { loadPref, savePref } from './prefs';

	interface Props {
		/** Default node (the selected snapshot or region). */
		node?: string;
	}
	let { node = '' }: Props = $props();

	let nodeInput = $state('');
	let depth = $state(loadPref('latent.graph.depth', 1));
	let hood = $state<Neighborhood | null>(null);
	let hoodError = $state('');
	let hoodBusy = $state(false);
	$effect(() => savePref('latent.graph.depth', depth));
	$effect(() => {
		if (!nodeInput && node) nodeInput = node;
	});

	let catalog = $state<GraphQueryDef[]>([]);
	let catalogError = $state('');
	let qName = $state(loadPref('latent.graph.query', ''));
	let params = $state<Record<string, string>>({});
	let qResult = $state<unknown>(null);
	let qError = $state('');
	let qBusy = $state(false);
	$effect(() => savePref('latent.graph.query', qName));

	const def = $derived(catalog.find((q) => q.name === qName) ?? null);
	/** `"name: type?"` → {name, type, optional}. */
	const paramDefs = $derived(
		(def?.params ?? []).map((p) => {
			const [name, rest = ''] = p.split(':').map((x) => x.trim());
			return { name, type: rest.replace(/\?$/, ''), optional: rest.endsWith('?') };
		})
	);
	const labelCounts = $derived.by(() => {
		const m = new Map<string, number>();
		for (const n of hood?.nodes ?? []) m.set(n.$label ?? '?', (m.get(n.$label ?? '?') ?? 0) + 1);
		return [...m];
	});

	function nodeName(n: Record<string, unknown>): string {
		for (const k of ['snapshot_id', 'expert_space_id', 'region_id', 'genome_id', 'instrument_id', 'id', 'name']) {
			if (typeof n[k] === 'string' && n[k]) return n[k] as string;
		}
		return `#${n.$id}`;
	}

	async function explore(ev?: Event) {
		ev?.preventDefault();
		if (!nodeInput.trim()) return;
		hoodBusy = true;
		try {
			hood = await live.neighborhood(nodeInput.trim(), depth);
			hoodError = '';
		} catch (e) {
			hoodError = e instanceof Error ? e.message : String(e);
			hood = null;
		} finally {
			hoodBusy = false;
		}
	}

	function coerce(type: string, raw: string): unknown {
		const v = raw.trim();
		if (type === 'integer' || type === 'number') return v === '' ? undefined : Number(v);
		if (type.startsWith('[')) return v === '' ? undefined : v.split(/[\s,]+/).filter(Boolean).map(Number);
		return v === '' ? undefined : v;
	}

	async function runQuery(ev: SubmitEvent) {
		ev.preventDefault();
		if (!def) return;
		qBusy = true;
		qError = '';
		try {
			const body: Record<string, unknown> = {};
			for (const p of paramDefs) {
				const v = coerce(p.type, params[p.name] ?? '');
				if (v !== undefined) body[p.name] = v;
				else if (!p.optional) throw new Error(`${p.name} is required`);
			}
			qResult = (await live.graphQuery(def.name, body)).result;
		} catch (e) {
			qError = e instanceof ApiError && e.body && typeof e.body === 'object' ? `${e.message} ${JSON.stringify(e.body)}` : e instanceof Error ? e.message : String(e);
			qResult = null;
		} finally {
			qBusy = false;
		}
	}

	onMount(async () => {
		try {
			catalog = (await live.graphQueries()).queries;
			if (!qName || !catalog.some((q) => q.name === qName)) qName = catalog[0]?.name ?? '';
			catalogError = '';
		} catch (e) {
			catalogError = e instanceof Error ? e.message : String(e);
		}
	});
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Graph explorer</h2>
		<span class="dim">HelixDB · GET /api/graph/neighborhood · POST /api/graph/query (named catalog only)</span>
	</div>
	<form class="row" onsubmit={explore}>
		<input bind:value={nodeInput} class="mono" placeholder="snapshot / region / expert space / genome / instrument id or numeric $id" size="44" />
		<label>depth <input type="number" min="1" max="4" bind:value={depth} /></label>
		<button type="submit" disabled={hoodBusy}>{hoodBusy ? '…' : 'Explore'}</button>
	</form>
	{#if hoodError}<p class="badge danger">{hoodError}</p>{/if}
	{#if hood}
		<p class="dim small">
			center <span class="badge">{hood.center.$label}</span> <span class="mono">{nodeName(hood.center)}</span> · depth {hood.depth} · {hood.nodes.length} nodes · {hood.edges.length} edges ·
			{#each labelCounts as [l, c]}<span class="badge">{l} {c}</span>{/each}
		</p>
		<div class="scroll hood">
			<table>
				<thead><tr><th>edge</th><th>from</th><th>to</th></tr></thead>
				<tbody>
					{#each hood.edges as e (e.$id)}
						{@const from = hood.nodes.find((n) => n.$id === e.$from)}
						{@const to = hood.nodes.find((n) => n.$id === e.$to)}
						<tr>
							<td class="mono dim">{e.$label}</td>
							<td class="mono"><button class="link" type="button" onclick={() => ((nodeInput = from ? nodeName(from) : String(e.$from)), explore())}>{from ? `${from.$label} ${nodeName(from)}` : e.$from}</button></td>
							<td class="mono"><button class="link" type="button" onclick={() => ((nodeInput = to ? nodeName(to) : String(e.$to)), explore())}>{to ? `${to.$label} ${nodeName(to)}` : e.$to}</button></td>
						</tr>
					{:else}
						<tr><td colspan="3" class="dim">no edges within depth {hood.depth}</td></tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}

	<h3 class="sub">Named queries <span class="dim">({catalog.length} in the whitelist)</span></h3>
	{#if catalogError}<p class="badge danger">{catalogError}</p>{/if}
	<form class="query" onsubmit={runQuery}>
		<label>query
			<select bind:value={qName} onchange={() => ((params = {}), (qResult = null), (qError = ''))}>
				{#each catalog as q (q.name)}<option value={q.name}>{q.name}</option>{/each}
			</select>
		</label>
		{#if def}<span class="dim doc">{def.doc}</span>{/if}
		{#each paramDefs as p (p.name)}
			<label>{p.name} <span class="dim">{p.type}{p.optional ? '?' : ''}</span>
				<input bind:value={params[p.name]} class="mono" placeholder={p.type.startsWith('[') ? 'comma-separated numbers' : p.type} required={!p.optional} />
			</label>
		{/each}
		<button type="submit" disabled={qBusy || !def}>{qBusy ? '…' : 'Run'}</button>
	</form>
	{#if qError}<p class="badge danger">{qError}</p>{/if}
	{#if qResult !== null}
		<pre class="mono result">{JSON.stringify(qResult, null, 1)}</pre>
	{/if}
</section>

<style>
	.row,
	.query {
		display: flex;
		gap: 8px;
		align-items: flex-end;
		flex-wrap: wrap;
	}
	.row label,
	.query label {
		display: flex;
		flex-direction: column;
		gap: 2px;
		color: var(--text-dim);
	}
	.row input[type='number'] {
		width: 60px;
	}
	.small {
		font-size: 11px;
		margin: 8px 0 4px;
	}
	.hood {
		max-height: 220px;
	}
	.link {
		background: none;
		border: none;
		padding: 0;
		color: var(--accent);
		font: inherit;
	}
	.sub {
		font-size: 12px;
		margin: 12px 0 6px;
	}
	.doc {
		align-self: center;
		font-size: 11px;
	}
	.result {
		max-height: 260px;
		overflow: auto;
		font-size: 11px;
		background: var(--bg-inset);
		padding: 8px;
		border-radius: 4px;
		margin: 8px 0 0;
	}
</style>
