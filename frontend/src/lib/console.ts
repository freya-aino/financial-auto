// Typed client for the D-066..D-071 control-api surface (spec 50 §4.5–§4.7): alerts, MCMC
// queries, region lineage, snapshot detail, neural decoders, strategy pipeline, allocators,
// persisted halts + re-arm, risk events, reference data and ledger reports.
import { ApiError, postJson, q } from './api';

/** GET that keeps the API's `{error, detail}` message (unlike `getJson`). */
export async function getApi<T>(path: string): Promise<T> {
	const res = await fetch(path);
	const data = await res.json().catch(() => ({}));
	if (!res.ok) {
		const msg = data.reason ?? data.error ?? `${res.status} ${res.statusText}`;
		throw new ApiError(res.status, data.detail ? `${msg}: ${data.detail}` : msg, data);
	}
	return data as T;
}

export const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

// ---------------------------------------------------------------- alerts (D-066)

export interface AlertEvent {
	status: 'firing' | 'resolved' | string;
	alertname: string;
	severity: string;
	summary: string;
	fingerprint: string;
	labels: Record<string, string>;
	annotations: Record<string, string>;
	starts_at: string | null;
	ends_at: string | null;
	generator_url: string | null;
	at: string;
	expired?: boolean;
}

export interface ActiveAlert extends AlertEvent {
	last_seen: string;
	refreshes: number;
}

export interface AlertPipeline {
	ok: boolean;
	watchdog_seen_at: string | null;
	posts: number;
	last_post_at: string | null;
	subject: string;
}

export interface AlertsView {
	active: ActiveAlert[];
	recent: AlertEvent[];
	pipeline: AlertPipeline;
}

// ---------------------------------------------------------------- MCMC (D-069)

/** Posterior summary every MCMC answer carries. */
export interface Dist {
	mean: number;
	sd: number;
	lower: number;
	upper: number;
	median: number;
	level: number;
	n: number;
}

export interface McmcContext {
	draws: number;
	expert: number;
	level: number;
	snapshot: string;
	states: number;
}

export interface SamplerScoreRow {
	sampler_id: string;
	log_score: number;
	brier: number;
	ece: number;
	pit_ks: number;
	cost_us: number;
	evals: number;
	pareto: boolean;
	serving: boolean;
}

export interface KernelRow {
	kernel_id: string;
	expert_index: number;
	expert_space_id: string;
	kernel_type: string;
	live_regions: number;
	n_rows: number;
	observation_mass: number;
	relaxation_steps: number;
	drift_kl: number;
	serving_sampler: string;
	snapshot_id: string;
	scores: SamplerScoreRow[];
	stationary: { region_id: string; mean: number; sd: number; lower: number; upper: number }[];
	[k: string]: unknown;
}

export interface RegionProb {
	region_id: string;
	prob: Dist;
}

export interface HierEdgeRow {
	fine_region_id?: string;
	coarse_region_id?: string;
	p_coarse_given_fine?: number;
	p_fine_given_coarse?: number;
	support?: number;
	pmi_bits?: number;
	[k: string]: unknown;
}

export interface CrossPrior {
	source_instrument_id: string;
	target_instrument_id: string;
	correlation?: number;
	lead_corr?: number;
	predictive_gain?: number;
	uncertainty?: number;
	score?: number;
	[k: string]: unknown;
}

export interface HelixRegion {
	region_id: string;
	lineage_id: string;
	expert_space_id: string;
	status: string;
	mass?: number;
	occupancy?: number;
	age?: number;
	method?: string;
	related_region_id?: string;
	created_at?: string;
	updated_at?: string;
	[k: string]: unknown;
}

export interface LineageView {
	region: HelixRegion;
	lineage: HelixRegion[];
	ancestors: (HelixRegion & { kind?: string; overlap?: number; depth?: number })[];
	descendants: (HelixRegion & { kind?: string; overlap?: number; depth?: number })[];
}

export type McmcQuery = 'transition' | 'k-step' | 'neighbors' | 'paths' | 'path-prob' | 'hitting' | 'stationary';

// ---------------------------------------------------------------- snapshots (D-070)

export interface CheckItem {
	name: string;
	ok: boolean;
	value: number | null;
	threshold: number | null;
	detail: string;
}

export interface SnapshotDetail {
	snapshot: {
		snapshot_id: string;
		timescale: string;
		status: string;
		encoder_family: string;
		generation: number;
		created_at: string;
		carried_mass: number | null;
		previous_active: string | null;
		garage_key: string;
	};
	checklist: { passed: boolean; checks: CheckItem[]; thresholds: Record<string, number>; snapshot_id: string; timescale: string } | null;
	not_evaluated: string[];
	shadow_gauges: Record<string, number>;
	manifest: Record<string, unknown> | null;
	manifest_source: string;
	latest_adapter: unknown;
}

// ---------------------------------------------------------------- decoders (D-068)

export interface SlotRole {
	key: string;
	genome: string | null;
	score: number | null;
	online_n: number;
	online_loss: number | null;
}

export interface DecoderSlot {
	family: string;
	timescale: string;
	serving: SlotRole | null;
	shadow: SlotRole | null;
	disabled: boolean;
	pinned: string | null;
}

export interface DecoderServing {
	cards: number;
	loaded: number;
	/** Load errors (model-runtime reports the messages). */
	errors: string[] | number;
	replace_baseline: boolean;
	news: number;
	slots: DecoderSlot[];
}

export interface DecoderCardRow {
	key: string;
	size: number;
	last_modified: string | null;
	card: Record<string, unknown> | null;
}

export type DecoderVerb = 'reload' | 'promote' | 'pin' | 'unpin' | 'off' | 'on';

// ---------------------------------------------------------------- trading: pipeline, allocators, halts, events (D-071)

export interface AllocatorEntry {
	name: string;
	cumulative_return: number;
	sharpe: number | null;
	max_drawdown: number;
	steps: number;
	turnover: number;
	costs: number;
	weights: Record<string, number>;
}

export interface TournamentView {
	active: string | null;
	cost_bps: number;
	basis: string;
	entries: AllocatorEntry[];
}

export interface HaltRow {
	scope: string;
	family: string;
	reason: string;
	since: string | null;
	manual: boolean;
	persisted: boolean;
	live: boolean;
	rearmable: boolean;
}

export interface HaltsView {
	halts: HaltRow[];
	persisted: { book: string; updated_at: string; peak_equity: number; day_start_equity: number; risk_halts: unknown } | null;
	postgres: boolean;
	families: string[];
}

export interface RiskEventRow {
	id: number;
	at: string;
	actor: string;
	action: string;
	kind: string;
	scope: string | null;
	detail: Record<string, unknown> | null;
}

export interface RiskEventsView {
	since: string | null;
	count: number;
	counts: Record<string, number>;
	events: RiskEventRow[];
}

// ---------------------------------------------------------------- refdata + ledger (D-067)

export interface Venue {
	code: string;
	mic: string | null;
	name: string;
	timezone: string;
	kind: string;
	always_open: boolean;
	rule: string | null;
	materialized: { first: string | null; last: string | null; days: number };
}

export interface SessionRow {
	date: string;
	open?: string | null;
	close?: string | null;
	segments?: { kind: string; start: string; end: string }[];
	is_holiday?: boolean;
	holiday?: string | null;
	early_close?: boolean;
	basis: 'table' | 'rule' | 'missing_calendar' | string;
}

export interface CorporateAction {
	id: string;
	instrument_id: string;
	instrument_name?: string | null;
	kind: string;
	ex_date: string;
	record_date: string | null;
	pay_date: string | null;
	ratio: number | null;
	amount: number | null;
	ccy: string | null;
	source_id: string | null;
	recorded_at: string;
	status: string;
	cancelled_at: string | null;
	old_symbol: string | null;
	new_symbol: string | null;
}

export interface PnlParts {
	realised_eur: number;
	unrealised_eur: number;
	fees_eur: number;
	fx_eur: number;
	fx_realised_eur: number;
	fx_unrealised_eur: number;
	fx_cash_eur: number;
	funding_eur: number;
	borrow_eur: number;
	other_eur: number;
	net_eur: number;
}

export interface PnlBreakdown {
	book: string;
	from: string | null;
	to?: string | null;
	book_pnl: PnlParts;
	strategies: Record<string, PnlParts>;
}

export interface Lot {
	[k: string]: unknown;
}

// ---------------------------------------------------------------- client

const enc = encodeURIComponent;

export const consoleApi = {
	alerts: (limit = 50) => getApi<AlertsView>(`/api/alerts${q({ limit })}`),

	kernels: (snapshot: string) => getApi<{ kernels: KernelRow[] }>(`/api/mcmc/${enc(snapshot)}/kernels`),
	mcmc: <T = Record<string, unknown>>(snapshot: string, query: McmcQuery, params: Record<string, string | number | undefined>) =>
		getApi<T & { context?: McmcContext }>(`/api/mcmc/${enc(snapshot)}/${query}${q(params)}`),
	hierarchy: (snapshot: string) => getApi<{ snapshot: string; edges: HierEdgeRow[] }>(`/api/mcmc/${enc(snapshot)}/hierarchy`),
	crossPriors: (timescale: string, limit = 50) =>
		getApi<{ timescale: string; proposals: CrossPrior[] }>(`/api/mcmc/cross-priors${q({ timescale, limit })}`),
	lineage: (region: string, depth = 4) => getApi<LineageView>(`/api/latent/regions/${enc(region)}/lineage${q({ depth })}`),

	snapshot: (id: string) => getApi<SnapshotDetail>(`/api/latent/snapshots/${enc(id)}`),
	shadowStart: (id: string) => postJson<{ ok: boolean }>(`/api/latent/snapshots/${enc(id)}/shadow`),
	shadowStop: (id: string) => postJson<{ ok: boolean }>(`/api/latent/snapshots/${enc(id)}/shadow/stop`),
	snapshotAction: (id: string, action: 'promote' | 'rollback') => postJson<{ ok: boolean }>(`/api/latent/snapshots/${enc(id)}/${action}`),

	decoderServing: () => getApi<DecoderServing>('/api/decoders/serving'),
	decoderCards: (limit = 60) => getApi<{ total: number; count: number; cards: DecoderCardRow[] }>(`/api/decoders/cards${q({ limit })}`),
	decoderAction: (verb: DecoderVerb, args: Record<string, string> = {}) =>
		postJson<{ ok: boolean; verb: string; message: unknown; audited: boolean }>(`/api/decoders/${verb}`, args),

	pipeline: () => getApi<Record<string, unknown>>('/api/trading/pipeline'),
	pipelineRun: () => postJson<Record<string, unknown>>('/api/trading/pipeline/run'),
	allocators: () => getApi<TournamentView>('/api/trading/allocators'),
	halts: () => getApi<HaltsView>('/api/trading/halts'),
	rearm: (scope: string, reason: string) =>
		postJson<{ ok: boolean; resumed: string; audited: boolean }>(`/api/trading/halts/${enc(scope)}/rearm`, { confirm: true, reason }),
	riskEvents: (since = '24h', limit = 200, kind?: string) => getApi<RiskEventsView>(`/api/trading/risk-events${q({ since, limit, kind })}`),

	venues: () => getApi<{ venues: Venue[] }>('/api/refdata/venues'),
	calendar: (venue: string, from?: string, to?: string) =>
		getApi<{ venue: string; timezone: string; rule: string | null; sessions: SessionRow[] }>(`/api/refdata/calendar/${enc(venue)}${q({ from, to })}`),
	session: (venue: string, at?: string) => getApi<{ tradable: boolean; session: Record<string, unknown> }>(`/api/refdata/session${q({ venue, at })}`),
	resolve: (p: { venue?: string; source?: string; symbol: string; at?: string; known_at?: string }) =>
		getApi<{ resolved: { result: 'none' | 'one' | 'ambiguous'; [k: string]: unknown }; [k: string]: unknown }>(`/api/refdata/resolve${q(p)}`),
	instrumentAsOf: (id: string, at?: string, known_at?: string) => getApi<Record<string, unknown>>(`/api/refdata/instruments/${enc(id)}${q({ at, known_at })}`),
	fx: (ccy: string, at?: string, known_at?: string) =>
		getApi<{ ccy: string; at: string; rate_to_eur: number; rate_time: string; source: string }>(`/api/refdata/fx${q({ ccy, at, known_at })}`),
	corporateActions: (instrument_id?: string, known_at?: string) =>
		getApi<{ actions: CorporateAction[]; point_in_time: boolean }>(`/api/refdata/corporate-actions${q({ instrument_id, known_at })}`),
	addCorporateAction: (body: Record<string, unknown>) => postJson<Record<string, unknown>>('/api/refdata/corporate-actions', body),
	cancelCorporateAction: (id: string) => postJson<Record<string, unknown>>(`/api/refdata/corporate-actions/${enc(id)}/cancel`),
	symbolChange: (body: { venue: string; old_symbol: string; new_symbol: string; effective?: string }) =>
		postJson<Record<string, unknown>>('/api/refdata/symbol-change', body),
	materialize: (venue: string, from: string, to: string) =>
		postJson<{ ok: boolean; rows: number }>(`/api/refdata/calendar/${enc(venue)}/materialize`, { from, to }),

	lots: (book = 'paper', at?: string) => getApi<{ book: string; count: number; lots: Lot[] }>(`/api/trading/lots${q({ book, at })}`),
	pnlBreakdown: (book = 'paper', from?: string, to?: string) => getApi<PnlBreakdown>(`/api/trading/pnl/breakdown${q({ book, from, to })}`),
	statement: (book = 'paper', from?: string, to?: string) => getApi<Record<string, unknown>>(`/api/trading/statement${q({ book, from, to, format: 'json' })}`),
	statementCsvUrl: (book = 'paper', from?: string, to?: string) => `/api/trading/statement${q({ book, from, to, format: 'csv' })}`,
	orphans: (status = 'open') => getApi<{ count: number; orphans: Record<string, unknown>[] }>(`/api/trading/orphans${q({ status })}`)
};
