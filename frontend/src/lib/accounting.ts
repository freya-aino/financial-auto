// Organisational books of D-104 (spec 51 §6 round 9): client for the accounting routes plus the
// pure logic of the Accounting panel — segment funds in a fixed order, account balances grouped
// by kind, the trial-balance invariant check and the close-day form (validation + confirmation
// text). The page asks `confirm()` and shows the API's report; nothing here is optimistic.
import { postJson, q } from './api';
import { getApi } from './console';

/** One account of `GET /api/accounting/balances` (`normal_eur` is the balance on the account's normal side). */
export interface AccountBalance {
	normal_eur: number;
	kind: string;
	segment: string;
	external_ref: string | null;
}

export interface BalancesView {
	accounts: Record<string, AccountBalance>;
	segment_funds_eur: Record<string, number>;
	trial_balance: number;
	entries: number;
}

/** A posting of a journal entry: signed EUR, debit positive, credit negative. */
export interface Posting {
	account: string;
	amount: number;
}

export interface JournalEntry {
	id: string;
	ts: string;
	entry_ref: string;
	memo: string;
	postings: Posting[];
}

/** `POST /api/accounting/close` report (idempotent server-side; only finished UTC days close). */
export interface CloseReport {
	period: string;
	revenue_eur: number;
	opex: unknown[];
	allocation: { owner: number; infra: number; reserves: number; retained: number };
	entries_posted: number;
	entries_already: number;
}

export const accountingApi = {
	balances: () => getApi<BalancesView>('/api/accounting/balances'),
	entries: (limit = 20) => getApi<{ count: number; entries: JournalEntry[] }>(`/api/accounting/entries${q({ limit })}`),
	/** Only called after the operator confirmed `closeConfirmText`. */
	close: (day: string) => postJson<{ ok: boolean; report: CloseReport }>('/api/accounting/close', { day })
};

// ---------------------------------------------------------------- pure logic

/** The five organisational segments of D-104, in presentation order. */
export const SEGMENTS = ['trading', 'infrastructure', 'operations', 'owner', 'treasury'];

/** Account kinds in balance-sheet-then-P&L order. */
export const KINDS = ['asset', 'liability', 'equity', 'income', 'expense'];

/** Segment funds as the headline row: the five segments (0 when absent), unknown segments appended. */
export function segmentFunds(funds: Record<string, number>): { segment: string; eur: number }[] {
	const extra = Object.keys(funds).filter((s) => !SEGMENTS.includes(s)).sort();
	return [...SEGMENTS, ...extra].map((segment) => ({ segment, eur: funds[segment] ?? 0 }));
}

export interface AccountRow extends AccountBalance {
	name: string;
}

/** Balances grouped by kind (asset → expense, unknown kinds appended), accounts by name; empty kinds dropped. */
export function groupByKind(accounts: Record<string, AccountBalance>): { kind: string; accounts: AccountRow[] }[] {
	const rows = Object.entries(accounts).map(([name, a]) => ({ name, ...a }));
	const extra = [...new Set(rows.map((r) => r.kind))].filter((k) => !KINDS.includes(k)).sort();
	return [...KINDS, ...extra]
		.map((kind) => ({ kind, accounts: rows.filter((r) => r.kind === kind).sort((a, b) => a.name.localeCompare(b.name)) }))
		.filter((g) => g.accounts.length > 0);
}

/** A non-zero trial balance (beyond float noise, well under a cent) is a broken invariant. */
export function trialBalanceOk(trial: number): boolean {
	return Number.isFinite(trial) && Math.abs(trial) < 0.005;
}

/** Yesterday as a UTC day (the latest day a close can book). */
export function yesterdayUtc(now: Date = new Date()): string {
	return new Date(now.getTime() - 86_400_000).toISOString().slice(0, 10);
}

/** Validates the close-day form: an ISO day strictly before today (UTC) — only finished UTC days close. */
export function closeDayBody(day: string, opts: { today?: string } = {}): { body: { day: string } } | { error: string } {
	const d = day.trim();
	if (!/^\d{4}-\d{2}-\d{2}$/.test(d) || Number.isNaN(Date.parse(`${d}T00:00:00Z`))) return { error: 'day must be YYYY-MM-DD' };
	const today = opts.today ?? new Date().toISOString().slice(0, 10);
	if (d >= today) return { error: `day ${d} is not over yet (UTC): only finished days close` };
	return { body: { day: d } };
}

export function closeConfirmText(day: string): string {
	return `Close the books for ${day} (UTC)? This books the day's trading revenue, accrues opex against the infra fund and allocates the split (owner / infra / reserves / retained) as journal entries. The close is idempotent: a day already closed posts nothing new.`;
}
