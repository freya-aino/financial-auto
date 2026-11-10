<script lang="ts">
	// Runtime-tunable config editor (`GET/PUT /api/config/runtime`, D-055): bounded inputs
	// from the server's `bounds`, shows `changed`, `persisted` and `audited` from the reply.
	import { onMount } from 'svelte';
	import { api, type ConfigPutResult, type ConfigSection } from './api';
	import { runtime } from './runtime.svelte';

	let section = $state<ConfigSection | null>(null);
	let draft = $state<Record<string, number>>({});
	let error = $state('');
	let busy = $state(false);
	let result = $state<ConfigPutResult | null>(null);

	/** Flattens `{ui: {poll_ms}}` to `ui.poll_ms`. */
	function flatten(v: unknown, prefix = ''): Record<string, number> {
		const out: Record<string, number> = {};
		if (v && typeof v === 'object') {
			for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
				if (typeof x === 'number') out[prefix + k] = x;
				else Object.assign(out, flatten(x, `${prefix}${k}.`));
			}
		}
		return out;
	}

	function nest(flat: Record<string, number>): Record<string, unknown> {
		const out: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(flat)) {
			const parts = k.split('.');
			let o = out;
			for (const p of parts.slice(0, -1)) o = (o[p] ??= {}) as Record<string, unknown>;
			o[parts[parts.length - 1]] = v;
		}
		return out;
	}

	const current = $derived(section ? flatten(section.values) : {});
	const changed = $derived(Object.fromEntries(Object.entries(draft).filter(([k, v]) => v !== current[k] && Number.isFinite(v))));
	const bound = (key: string) => section?.bounds.find((b) => b.key === key);
	const outOfBounds = $derived(
		Object.entries(changed).filter(([k, v]) => {
			const b = bound(k);
			return b && (v < b.min || v > b.max);
		}).map(([k]) => k)
	);

	async function load() {
		try {
			section = await api.runtimeConfig();
			draft = flatten(section.values);
			runtime.set(section.values);
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	async function save(ev: SubmitEvent) {
		ev.preventDefault();
		if (!Object.keys(changed).length || outOfBounds.length) return;
		busy = true;
		error = '';
		try {
			result = await api.putRuntimeConfig(nest(changed));
			runtime.set(result.values);
			await load();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	onMount(load);
</script>

<section class="panel">
	<div class="panel-head">
		<h2>Runtime config</h2>
		<span class="dim">GET/PUT /api/config/runtime · applied in-process, persisted to config/runtime.toml, audited</span>
	</div>
	{#if error}<p class="badge danger">{error}</p>{/if}
	{#if section}
		<form onsubmit={save} class="cfg">
			<table>
				<thead><tr><th>key</th><th>value</th><th>bounds</th><th>current</th></tr></thead>
				<tbody>
					{#each Object.keys(current) as key (key)}
						{@const b = bound(key)}
						<tr class:changed={key in changed}>
							<td class="mono">{key}</td>
							<td><input type="number" bind:value={draft[key]} min={b?.min} max={b?.max} step={key.endsWith('_hz') ? 1 : key.endsWith('ms') ? 250 : 1} class:bad={outOfBounds.includes(key)} /></td>
							<td class="mono dim">{b ? `[${b.min}, ${b.max}]` : 'read-only'}</td>
							<td class="mono dim">{current[key]}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<div class="row">
				<button type="submit" disabled={busy || !Object.keys(changed).length || outOfBounds.length > 0}>{busy ? 'saving…' : `Apply ${Object.keys(changed).length} change${Object.keys(changed).length === 1 ? '' : 's'}`}</button>
				<button type="button" onclick={() => (draft = { ...current })} disabled={!Object.keys(changed).length}>reset</button>
				{#if outOfBounds.length}<span class="badge danger">out of bounds: {outOfBounds.join(', ')}</span>{/if}
				{#if result}
					<span class="badge" class:ok={result.changed} class:warn={!result.changed}>{result.changed ? 'applied' : 'no change'}</span>
					<span class="badge" class:ok={result.persisted.ok} class:warn={!result.persisted.ok} title={result.persisted.error ?? result.persisted.path}>persisted: {result.persisted.ok ? 'yes' : 'no'}</span>
					<span class="badge" class:ok={result.audited} class:warn={!result.audited}>audited: {result.audited ? 'yes' : 'no'}</span>
				{/if}
			</div>
		</form>
		<p class="dim small">The UI's own poll fallback reads <code>ui.poll_ms</code> (now {runtime.pollMs} ms); the sockets' flush rates are <code>ws.*_hz</code>.</p>
	{/if}
</section>

<style>
	.cfg input {
		width: 110px;
	}
	.cfg input.bad {
		border-color: var(--danger);
	}
	tr.changed td {
		color: var(--accent);
	}
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
		margin-top: 8px;
	}
	.small {
		font-size: 11px;
		margin: 8px 0 0;
	}
</style>
