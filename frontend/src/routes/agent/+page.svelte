<script lang="ts">
	// Agent (D-117): the LLM suite's operator console. A conversation with the platform's own
	// agent, the tool catalogue it may use, the endpoint form, and the news-embedding mix that
	// feeds the encoder's text expert.
	//
	// Nothing mutating happens without a click here: the suite ends a turn when the model asks for
	// a mutating tool, this page shows the exact call, and `confirm()` plus a re-sent turn is what
	// runs it. While trading mode is `live` the approval buttons are not rendered at all — the
	// server refuses those calls too, so this is the second of two locks, not the only one.
	import { onMount } from 'svelte';
	import Info from '../../lib/Info.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
	import {
		agentApi,
		approvalKey,
		canApprove,
		confirmText,
		embeddingNote,
		endpointLabel,
		groupTools,
		type ConversationSummary,
		type EmbeddingHealth,
		type LlmStatus,
		type PendingCall,
		type ToolRecord,
		type TurnResponse
	} from '../../lib/agent';
	import { errText } from '../../lib/console';

	type Turn = { role: 'you' | 'agent'; text: string; calls: ToolRecord[]; pending: PendingCall[] };

	let status = $state<LlmStatus | null>(null);
	let embeddings = $state<EmbeddingHealth | null>(null);
	let conversations = $state<ConversationSummary[]>([]);
	let turns = $state<Turn[]>([]);
	let conversationId = $state<string | null>(null);
	let draft = $state('');
	let busy = $state(false);
	let error = $state('');
	let statusError = $state('');

	// The endpoint form. `key` is never read back from the server, so an empty field means "leave
	// the stored key alone" rather than "clear it" — the placeholder says which.
	let form = $state({ base_url: '', model: '', embedding_model: '', embedding_dim: 0, api_key: '' });
	let formNote = $state('');

	const tools = $derived(groupTools(status?.tools ?? []));
	const live = $derived(status?.trading_mode === 'live');

	async function loadStatus() {
		try {
			status = await agentApi.status();
			statusError = '';
			form.base_url ||= status.endpoint ?? '';
			form.model ||= status.model ?? '';
			form.embedding_model ||= status.embedding_model ?? '';
			form.embedding_dim ||= status.embedding_dim ?? 0;
		} catch (e) {
			statusError = errText(e);
		}
		try {
			embeddings = await agentApi.embeddings();
		} catch {
			embeddings = null;
		}
		try {
			conversations = (await agentApi.conversations()).conversations;
		} catch {
			conversations = [];
		}
	}

	onMount(loadStatus);

	async function send(approvals: PendingCall[] = [], message = draft.trim()) {
		if (!message || busy) return;
		busy = true;
		error = '';
		if (!approvals.length) turns = [...turns, { role: 'you', text: message, calls: [], pending: [] }];
		try {
			const out: TurnResponse = await agentApi.chat(message, conversationId, approvals);
			conversationId = out.conversation_id;
			turns = [...turns, { role: 'agent', text: out.reply, calls: out.tool_calls, pending: out.pending }];
			draft = '';
			await loadStatus();
		} catch (e) {
			error = errText(e);
		} finally {
			busy = false;
		}
	}

	/** Approves one pending call and re-sends the turn that asked for it. */
	async function approve(call: PendingCall) {
		if (!confirm(confirmText(call))) return;
		// The message is the last thing the operator said; the approval is what changes.
		const lastYou = [...turns].reverse().find((t) => t.role === 'you');
		await send([call], lastYou?.text ?? '');
	}

	async function openConversation(id: string) {
		busy = true;
		try {
			const d = await agentApi.conversation(id);
			conversationId = id;
			turns = d.messages
				.filter((m) => m.role === 'user' || m.role === 'assistant')
				.map((m) => ({ role: m.role === 'user' ? 'you' : 'agent', text: m.content, calls: [], pending: [] }));
		} catch (e) {
			error = errText(e);
		} finally {
			busy = false;
		}
	}

	function newConversation() {
		conversationId = null;
		turns = [];
		error = '';
	}

	async function saveEndpoint() {
		formNote = '';
		const body: Record<string, unknown> = {};
		if (form.base_url !== (status?.endpoint ?? '')) body.base_url = form.base_url;
		if (form.model !== (status?.model ?? '')) body.model = form.model;
		if (form.embedding_model !== (status?.embedding_model ?? '')) body.embedding_model = form.embedding_model;
		if (form.embedding_dim && form.embedding_dim !== status?.embedding_dim) body.embedding_dim = form.embedding_dim;
		if (form.api_key) body.api_key = form.api_key;
		if (!Object.keys(body).length) {
			formNote = 'nothing changed';
			return;
		}
		try {
			const out = await agentApi.setEndpoint(body);
			form.api_key = '';
			formNote = `saved ${out.fields.join(', ')} — ${out.note}`;
			await loadStatus();
		} catch (e) {
			formNote = errText(e);
		}
	}
</script>

<ViewHeader title="Agent" spec="D-117" note={endpointLabel(status)} />

{#if statusError}
	<p class="error">The LLM suite is not answering: {statusError}</p>
{/if}

<div class="grid">
	<section class="chat">
		<header>
			<h2>Conversation</h2>
			<div class="actions">
				{#if conversationId}<code>{conversationId}</code>{/if}
				<button onclick={newConversation} disabled={busy}>New</button>
			</div>
		</header>

		{#if !status?.configured}
			<p class="hint">
				No endpoint is configured, so the agent cannot answer. News is still embedded with the
				deterministic hashed fallback, so the text expert keeps working — enter an endpoint below to
				use a model's embeddings instead.
			</p>
		{/if}

		<ol class="turns">
			{#each turns as t, i (i)}
				<li class={t.role}>
					<span class="who">{t.role}</span>
					<p>{t.text}</p>
					{#if t.calls.length}
						<ul class="calls">
							{#each t.calls as c, j (j)}
								<li>
									<code>{c.tool}</code>
									<span class="outcome {c.outcome}">{c.outcome}</span>
									<code class="args">{JSON.stringify(c.arguments)}</code>
								</li>
							{/each}
						</ul>
					{/if}
					{#each t.pending as p, j (j)}
						<div class="pending">
							<p><strong>Waiting for you:</strong> <code>{p.tool}</code> <code class="args">{JSON.stringify(p.arguments)}</code></p>
							{#if canApprove(status, p)}
								<button class="approve" onclick={() => approve(p)} disabled={busy}>Approve and run</button>
							{:else if live}
								<p class="hint">Trading mode is live: the agent takes no actions until it is back on paper.</p>
							{:else}
								<p class="hint">This call is not an approvable tool.</p>
							{/if}
						</div>
					{/each}
				</li>
			{/each}
		</ol>

		{#if error}<p class="error">{error}</p>{/if}

		<form
			onsubmit={(e) => {
				e.preventDefault();
				send();
			}}
		>
			<textarea
				bind:value={draft}
				rows="3"
				placeholder="Ask about the platform — halts, strategies, generations, P&L…"
				disabled={busy || !status?.configured}
			></textarea>
			<button type="submit" disabled={busy || !draft.trim() || !status?.configured}>
				{busy ? 'Thinking…' : 'Send'}
			</button>
		</form>
	</section>

	<aside>
		<section>
			<h2>Endpoint <Info text="An OpenAI-compatible API. The key is written to the git-ignored config/secrets.toml and is never returned by any route; the suite reads it at start, so a change needs a restart." /></h2>
			<label>Base URL<input bind:value={form.base_url} placeholder="https://api.openai.com/v1" /></label>
			<label>Chat model<input bind:value={form.model} placeholder="gpt-4o-mini" /></label>
			<label>Embedding model<input bind:value={form.embedding_model} placeholder="text-embedding-3-small" /></label>
			<label>Embedding width<input type="number" bind:value={form.embedding_dim} min="1" /></label>
			<label>
				API key
				<input
					type="password"
					bind:value={form.api_key}
					placeholder={status?.key_set ? 'stored — leave blank to keep' : 'not set'}
					autocomplete="off"
				/>
			</label>
			<button onclick={saveEndpoint} disabled={busy}>Save</button>
			{#if formNote}<p class="hint">{formNote}</p>{/if}
		</section>

		<section>
			<h2>News embeddings <Info text="Every news item is embedded once when it arrives and stored with its knowledge time, so the encoder reads stored vectors and replays stay deterministic." /></h2>
			<p>{embeddingNote(embeddings)}</p>
			{#if embeddings?.by_provider.length}
				<table>
					<thead><tr><th>Path</th><th>Items</th><th>Newest</th></tr></thead>
					<tbody>
						{#each embeddings.by_provider as p (p.provider)}
							<tr>
								<td>{p.provider}</td>
								<td class="num">{p.count.toLocaleString()}</td>
								<td>{p.newest ? new Date(p.newest).toISOString().slice(0, 16).replace('T', ' ') : '—'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</section>

		<section>
			<h2>Tools</h2>
			<p class="hint">
				{tools.read.length} read, {tools.mutating.length} mutating{live ? ' (refused in live mode)' : ''}. Promotion,
				trading mode, halts and money are not tools.
			</p>
			<details>
				<summary>Read ({tools.read.length})</summary>
				<ul class="tools">
					{#each tools.read as t (t.name)}<li><code>{t.name}</code> — {t.description}</li>{/each}
				</ul>
			</details>
			<details>
				<summary>Mutating ({tools.mutating.length})</summary>
				<ul class="tools">
					{#each tools.mutating as t (t.name)}<li><code>{t.name}</code> — {t.description}</li>{/each}
				</ul>
			</details>
		</section>

		<section>
			<h2>History</h2>
			{#if conversations.length}
				<ul class="history">
					{#each conversations as c (c.conversation_id)}
						<li>
							<button class="link" onclick={() => openConversation(c.conversation_id)}>
								{c.title || c.conversation_id}
							</button>
							<span class="hint">{c.messages} messages · {new Date(c.updated_at).toISOString().slice(0, 16).replace('T', ' ')}</span>
						</li>
					{/each}
				</ul>
			{:else}
				<p class="hint">No conversations yet.</p>
			{/if}
		</section>
	</aside>
</div>

<style>
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 22rem);
		gap: var(--gap, 1rem);
		align-items: start;
	}
	@media (max-width: 60rem) {
		.grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	section {
		background: var(--panel);
		border: 1px solid var(--line);
		border-radius: 6px;
		padding: 0.75rem 1rem;
		margin-bottom: var(--gap, 1rem);
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.5rem;
	}
	h2 {
		font-size: 0.95rem;
		margin: 0 0 0.5rem;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
		align-items: center;
	}
	.turns {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.turns li {
		border-left: 3px solid var(--line);
		padding-left: 0.75rem;
	}
	.turns li.you {
		border-left-color: var(--accent, #6aa9ff);
	}
	.who {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--muted);
	}
	.turns p {
		margin: 0.25rem 0;
		white-space: pre-wrap;
	}
	.calls,
	.tools,
	.history {
		list-style: none;
		margin: 0.25rem 0 0;
		padding: 0;
		font-size: 0.8rem;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.args {
		color: var(--muted);
	}
	.outcome {
		font-size: 0.7rem;
		padding: 0 0.3rem;
		border-radius: 3px;
		border: 1px solid var(--line);
	}
	.outcome.ok {
		color: var(--ok, #6ac46a);
	}
	.outcome.needs_approval,
	.outcome.refused_live_mode {
		color: var(--warn, #e0b000);
	}
	.outcome.bad_arguments,
	.outcome.unknown_tool {
		color: var(--bad, #e06c6c);
	}
	.pending {
		border: 1px dashed var(--warn, #e0b000);
		border-radius: 4px;
		padding: 0.5rem;
		margin-top: 0.5rem;
	}
	.approve {
		font-weight: 600;
	}
	form {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.75rem;
	}
	textarea {
		flex: 1;
		font: inherit;
		resize: vertical;
	}
	label {
		display: block;
		font-size: 0.8rem;
		margin-bottom: 0.4rem;
	}
	label input {
		display: block;
		width: 100%;
		font: inherit;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.8rem;
	}
	th,
	td {
		text-align: left;
		padding: 0.15rem 0.3rem;
		border-bottom: 1px solid var(--line);
	}
	.num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.hint {
		color: var(--muted);
		font-size: 0.8rem;
	}
	.error {
		color: var(--bad, #e06c6c);
		font-size: 0.85rem;
	}
	.link {
		background: none;
		border: none;
		padding: 0;
		color: var(--accent, #6aa9ff);
		cursor: pointer;
		font: inherit;
		text-align: left;
	}
</style>
