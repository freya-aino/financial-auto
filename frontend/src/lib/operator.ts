// Client + form logic for the routes that were API-only until round 8 (spec 51 §6 "Not in the
// UI"): ops job / config history, alert history, holdout lock / evaluate, shadow-vs-backtest
// consistency, identifier and contract-spec overrides, and the FX restatement (dry run first,
// apply only after an explicit confirm). Every mutation keeps the confirm-then-API-result pattern.
import { postJson, q } from './api';
import { getApi, type AlertEvent } from './console';

const enc = encodeURIComponent;

// ---------------------------------------------------------------- ops history (spec 20 §5.4)

export interface JobRun {
	job_id: string;
	kind: string;
	status: string;
	worker_id: string | null;
	error: string | null;
	used_gpu_ms: number;
	output_refs: string[];
	checkpoint_ref: string | null;
	result: unknown;
	stream_seq: number | null;
	finished_at: string;
}

export interface ConfigSnapshotRow {
	id: number;
	service: string;
	section: string;
	digest: string;
	actor: string;
	taken_at: string;
	config?: unknown;
}

// ---------------------------------------------------------------- validation (spec 41)

export interface ShadowView {
	strategy: string;
	state: Record<string, unknown>;
	consistency: {
		available: boolean;
		n_horizons: number;
		min_horizons: number;
		reason?: string;
		backtest_sharpe?: number;
		shadow_sharpe?: number;
		ratio?: number | null;
		n_fills?: number;
		n_shadow_obs?: number;
		bar?: string;
		backtest_trial_at?: string;
		source: string;
	};
}

// ---------------------------------------------------------------- FX restatement (spec 42 §6)

export interface RestatementRequest {
	book: string;
	instrument: string;
	ccy?: string;
	from_ccy?: string;
	check_hours?: number;
}

export interface RestatementReply {
	dry_run: boolean;
	fills_total: number;
	plan: {
		book: string;
		instrument: string;
		from_ccy: string;
		ccy: string;
		planned_at: string;
		fills: Record<string, unknown>[];
		missing_rate: string[];
		by_old_source: Record<string, number>;
		cash_delta_eur: number | string;
		before: Record<string, unknown>;
		after: Record<string, unknown>;
		check_from: string;
		notes: string[];
	};
	applied: { batch_id: string; fills: number; cash_delta_eur: number | string } | null;
}

export const operatorApi = {
	jobRuns: (kind?: string, limit = 100) => getApi<{ runs: JobRun[] }>(`/api/ops/jobs${q({ kind, limit })}`),
	jobSummary: (hours = 24) => getApi<{ since: string; counts: { kind: string; status: string; count: number }[] }>(`/api/ops/jobs/summary${q({ hours })}`),
	configSnapshots: (service?: string, limit = 50, config = false) =>
		getApi<{ snapshots: ConfigSnapshotRow[] }>(`/api/ops/config-snapshots${q({ service, limit, config: config || undefined })}`),
	alertHistory: (since?: string, alertname?: string, limit = 200) => getApi<{ transitions: AlertEvent[] }>(`/api/alerts/history${q({ since, alertname, limit })}`),

	lockHoldout: (body: { timescale: string; start: string; end: string; note?: string }) => postJson<Record<string, unknown>>('/api/validation/holdout', body),
	evaluateHoldout: (id: string, body: { candidate: string; snapshot?: string; budget_ms?: number; seed?: number }) =>
		postJson<{ job_id: string; kind: string; holdout: Record<string, unknown>; audited: boolean; result: string }>(`/api/validation/holdout/${enc(id)}/evaluate`, body),
	shadow: (strategy: string, min_horizons?: number) => getApi<ShadowView>(`/api/validation/shadow/${enc(strategy)}${q({ min_horizons })}`),

	addIdentifier: (body: IdentifierBody) => postJson<{ ok: boolean }>('/api/refdata/identifiers', body),
	addContractSpec: (body: ContractSpecBody) => postJson<{ ok: boolean }>('/api/refdata/contract-specs', body),

	/** Dry run only: never sends `apply`. */
	restateFxDryRun: (req: RestatementRequest, fills = 20) => postJson<RestatementReply>('/api/trading/restatements/fx', { ...req, fills, apply: false }),
	/** Writes the restatement. Only called after the dry run was shown and the operator confirmed. */
	restateFxApply: (req: RestatementRequest, reason: string) =>
		postJson<RestatementReply>('/api/trading/restatements/fx', { ...req, apply: true, confirm: true, reason, fills: 0 })
};

// ---------------------------------------------------------------- pure helpers

/** Job summary rows → kind × status matrix (statuses in a fixed order, unknown ones appended). */
export const JOB_STATUSES = ['ok', 'failed', 'preempted', 'budget_exceeded', 'cancelled'];
export function summaryMatrix(rows: { kind: string; status: string; count: number }[]): { statuses: string[]; kinds: { kind: string; counts: Record<string, number>; total: number }[] } {
	const statuses = [...JOB_STATUSES, ...[...new Set(rows.map((r) => r.status))].filter((s) => !JOB_STATUSES.includes(s)).sort()];
	const byKind = new Map<string, Record<string, number>>();
	for (const r of rows) {
		const m = byKind.get(r.kind) ?? {};
		m[r.status] = (m[r.status] ?? 0) + r.count;
		byKind.set(r.kind, m);
	}
	const kinds = [...byKind].map(([kind, counts]) => ({ kind, counts, total: Object.values(counts).reduce((a, b) => a + b, 0) })).sort((a, b) => b.total - a.total || a.kind.localeCompare(b.kind));
	return { statuses, kinds };
}

/** Consecutive snapshots of one (service, section) whose digest changed: the change log. */
export function digestChanges(rows: ConfigSnapshotRow[]): (ConfigSnapshotRow & { previous: string | null })[] {
	const sorted = [...rows].sort((a, b) => Date.parse(a.taken_at) - Date.parse(b.taken_at) || a.id - b.id);
	const last = new Map<string, string>();
	const out: (ConfigSnapshotRow & { previous: string | null })[] = [];
	for (const r of sorted) {
		const k = `${r.service}/${r.section}`;
		out.push({ ...r, previous: last.get(k) ?? null });
		last.set(k, r.digest);
	}
	return out.reverse();
}

/** `fa_core::Timescale::LADDER`. */
export const TIMESCALES = ['1s', '1m', '1h', '1d', '1w'];

/** `<input type="datetime-local">` value (local time) → RFC 3339 UTC, or null. */
export function localToIso(v: string): string | null {
	if (!v) return null;
	const t = Date.parse(v);
	return Number.isFinite(t) ? new Date(t).toISOString() : null;
}

/** Validates a holdout lock (the API repeats the checks: overlap → 409, future end → 400). */
export function holdoutLockBody(timescale: string, start: string, end: string, note: string, now = Date.now()): { body: { timescale: string; start: string; end: string; note?: string } } | { error: string } {
	const s = localToIso(start);
	const e = localToIso(end);
	if (!timescale.trim()) return { error: 'timescale is required' };
	if (!TIMESCALES.includes(timescale.trim())) return { error: `unknown timescale ${timescale} (one of ${TIMESCALES.join(', ')})` };
	if (!s || !e) return { error: 'start and end are required' };
	if (Date.parse(e) <= Date.parse(s)) return { error: 'end must be after start' };
	if (Date.parse(e) > now) return { error: 'a holdout is a completed segment: end must not be in the future' };
	return { body: { timescale: timescale.trim(), start: s, end: e, ...(note.trim() ? { note: note.trim() } : {}) } };
}

export const holdoutConfirmText = (b: { timescale: string; start: string; end: string }) =>
	`Lock holdout ${b.timescale} ${b.start} → ${b.end}?\n\nLocked holdouts are immutable: selection replays that touch this range are rejected from now on (spec 41 §6.4). Audited as validation.holdout.lock.`;

export const evaluateConfirmText = (holdout: { id: string; timescale: string }, candidate: string) =>
	`Evaluate ${candidate.length > 60 ? candidate.slice(0, 60) + '…' : candidate} on holdout ${holdout.id} (${holdout.timescale})?\n\nThis spends the holdout on one promotion replay (evo.eval.strategy.holdout job). Audited.`;

export interface IdentifierBody {
	instrument_id: string;
	scheme: string;
	value: string;
	valid_from?: string;
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const IDENTIFIER_SCHEMES = ['isin', 'figi', 'cusip', 'sedol', 'ric', 'bbg_ticker', 'coingecko', 'lei'];

export function identifierBody(instrument: string, scheme: string, value: string, validFrom: string): { body: IdentifierBody } | { error: string } {
	if (!UUID.test(instrument.trim())) return { error: 'instrument: a UUID is required' };
	if (!scheme.trim() || !value.trim()) return { error: 'scheme and value are required' };
	const from = validFrom ? localToIso(validFrom) : null;
	if (validFrom && !from) return { error: 'valid_from is not a date' };
	return { body: { instrument_id: instrument.trim(), scheme: scheme.trim().toLowerCase(), value: value.trim(), ...(from ? { valid_from: from } : {}) } };
}

export interface ContractSpecBody {
	instrument_id: string;
	venue: string;
	tick_size?: number;
	lot_size?: number;
	min_notional?: number;
	multiplier?: number;
	max_leverage?: number;
	settlement_ccy?: string;
	shortable?: boolean;
	fee_schedule_ref?: string;
	expiry?: string;
	valid_from?: string;
}

/** Contract-spec override form (empty fields are omitted, i.e. NULL in the new version). */
export function contractSpecBody(f: Record<string, string>): { body: ContractSpecBody } | { error: string } {
	if (!UUID.test((f.instrument_id ?? '').trim())) return { error: 'instrument: a UUID is required' };
	if (!(f.venue ?? '').trim()) return { error: 'venue is required' };
	const body: ContractSpecBody = { instrument_id: f.instrument_id.trim(), venue: f.venue.trim() };
	for (const k of ['tick_size', 'lot_size', 'min_notional', 'multiplier', 'max_leverage'] as const) {
		const v = (f[k] ?? '').trim();
		if (!v) continue;
		const n = Number(v);
		if (!Number.isFinite(n) || n <= 0) return { error: `${k} must be a positive number` };
		body[k] = n;
	}
	if ((f.settlement_ccy ?? '').trim()) {
		const c = f.settlement_ccy.trim().toUpperCase();
		if (!/^[A-Z]{3}$/.test(c)) return { error: 'settlement_ccy must be a 3-letter code' };
		body.settlement_ccy = c;
	}
	if (f.shortable === 'true' || f.shortable === 'false') body.shortable = f.shortable === 'true';
	if ((f.fee_schedule_ref ?? '').trim()) body.fee_schedule_ref = f.fee_schedule_ref.trim();
	for (const k of ['expiry', 'valid_from'] as const) {
		if (!(f[k] ?? '').trim()) continue;
		const iso = localToIso(f[k]);
		if (!iso) return { error: `${k} is not a date` };
		body[k] = iso;
	}
	return { body };
}

export function restatementRequest(f: { book: string; instrument: string; ccy: string; from_ccy: string; check_hours: string }): { req: RestatementRequest } | { error: string } {
	if (!f.book.trim()) return { error: 'book is required' };
	if (!UUID.test(f.instrument.trim())) return { error: 'instrument: a UUID is required' };
	const req: RestatementRequest = { book: f.book.trim(), instrument: f.instrument.trim() };
	for (const k of ['ccy', 'from_ccy'] as const) {
		const c = f[k].trim().toUpperCase();
		if (!c) continue;
		if (!/^[A-Z]{3}$/.test(c)) return { error: `${k} must be a 3-letter code` };
		req[k] = c;
	}
	if (f.check_hours.trim()) {
		const h = Number(f.check_hours);
		if (!Number.isInteger(h) || h <= 0) return { error: 'check_hours must be a positive integer' };
		req.check_hours = h;
	}
	return { req };
}

/** Same request as the shown dry run? (apply is only offered for the plan on screen) */
export const sameRequest = (a: RestatementRequest | null, b: RestatementRequest | null) => !!a && !!b && JSON.stringify(a) === JSON.stringify(b);

export function restatementConfirmText(r: RestatementReply, reason: string): string {
	const p = r.plan;
	return (
		`APPLY the FX restatement of ${r.fills_total} fill(s) on book ${p.book}, instrument ${p.instrument}: ${p.from_ccy} → ${p.ccy}, cash delta ${Number(p.cash_delta_eur).toFixed(2)} EUR?\n\n` +
		`This rewrites booked fills (audited ledger.restatement, reason: "${reason}"). The risk-engine must be restarted afterwards.`
	);
}
