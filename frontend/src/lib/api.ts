// Typed client for control-api (spec 50). Paths are relative; vite proxies them in dev.

export interface ServiceStatus {
	service: string;
	instance_id: string;
	version: string;
	last_seen: string;
	counters: Record<string, number>;
}

/** `ingest.health.<source>` frame as overlaid by `GET /api/sources` (D-054). */
export interface SourceHealth {
	at: string;
	status: 'ok' | 'degraded' | 'down';
	running: boolean;
	mode: string;
	expected_period_s: number;
	last_activity_age_s: number | null;
	last_event_age_s: number | null;
	last_event_at: string | null;
	events_total: number;
	events_window: number;
	errors_total: number;
	errors_window: number;
	error_rate: number;
	limiter_requests: number;
	limiter_saturation: number;
	last_error?: string | null;
	reason?: string | null;
	halted: boolean;
}

export interface Source {
	id: string;
	name: string;
	category: string;
	capabilities: string[];
	asset_classes: string[];
	transport: string[];
	auth: string;
	/** Effective status (Postgres wins over the YAML seed, D-054). */
	status: 'planned' | 'stub' | 'live' | 'disabled';
	yaml_status?: string;
	status_updated_at?: string | null;
	tos_notes: string;
	notes: string;
	running?: boolean;
	health?: SourceHealth | null;
	rate_limit?: { per_second: number; burst: number } | null;
	config?: { symbols?: string[]; queries?: string[]; timeframe?: string; extra?: Record<string, string> } | null;
}

export interface BackfillCursor {
	instrument_id: string;
	timescale: string;
	range_start: string;
	range_end: string;
	cursor: string;
	status: string;
	live: boolean;
	updated_at: string;
}

export interface ConnectorRun {
	id: string;
	started_at: string;
	ended_at: string | null;
	status: string;
	events: number;
	errors: number;
	last_error: string | null;
}

export interface SourceDetail extends Source {
	cursors?: BackfillCursor[];
	runs?: ConnectorRun[];
	recent_errors?: unknown[];
	gaps?: unknown[];
	unresolved?: unknown[];
	quarantine_open?: number;
	status_from_postgres?: boolean;
}

export interface StreamConsumer {
	name: string;
	durable: boolean;
	filter_subjects: string[];
	ack_pending: number;
	num_pending: number;
	redelivered: number;
	waiting: number;
	delivered_seq: number;
	ack_floor_seq: number;
}

export interface BusStream {
	name: string;
	subjects: string[];
	retention: string;
	storage: string;
	max_age_secs: number;
	messages: number;
	bytes: number;
	first_seq: number;
	last_seq: number;
	consumer_count: number;
	consumers: StreamConsumer[];
}

export interface Instrument {
	id: string;
	name: string;
	status: string;
	asset_class: string | null;
	kind: string | null;
	base_ccy: string | null;
	quote_ccy: string | null;
	identifiers: { scheme: string; value: string }[];
	listings: { source_id: string; venue: string; venue_symbol: string }[];
	recorded_at?: string;
}

export interface QuarantineRow {
	instrument_id: string;
	instrument_name: string;
	instrument_status: string;
	source_id: string;
	venue_symbol: string;
	status: 'open' | 'resolved' | 'ignored' | string;
	count: number;
	first_seen: string;
	last_seen: string;
	resolution: string | null;
	resolved_at: string | null;
	sample_raw_ref: string | null;
}

export interface AuditEntry {
	id: number;
	at: string;
	actor: string;
	action: string;
	subject: string | null;
	payload: unknown;
}

export interface RuntimeBound {
	key: string;
	min: number;
	max: number;
}

export interface RuntimeConfig {
	ui: { poll_ms: number };
	bus_tap: { max_msgs_per_sec: number };
	live: { max_jobs: number; max_orders: number; max_decoder_outputs: number; max_generations: number };
	ws: { latent_hz: number; trading_hz: number };
}

export interface ConfigSection<T = RuntimeConfig> {
	section: string;
	tunable: boolean;
	values: T;
	bounds: RuntimeBound[];
}

export interface ConfigPutResult {
	ok: boolean;
	changed: boolean;
	values: RuntimeConfig;
	persisted: { ok: boolean; path: string; error?: string };
	audited: boolean;
}

export interface Health {
	service: string;
	version: string;
	nats: { connected: boolean; url: string };
	dependencies: Record<string, boolean>;
}

export interface BusFrame {
	stream: string;
	seq: number;
	ts: string;
	data: { subject: string; size: number; decoded: unknown };
}

export async function getJson<T>(path: string): Promise<T> {
	const res = await fetch(path);
	if (!res.ok) throw new Error(`${path}: ${res.status} ${res.statusText}`);
	return (await res.json()) as T;
}

export const api = {
	health: () => getJson<Health>('/api/health'),
	services: () => getJson<{ services: ServiceStatus[]; nats_connected: boolean }>('/api/services'),
	sources: () => getJson<{ version: number; sources: Source[] }>('/api/sources'),
	/** `ctl.ingest.*` start/stop. The ingestor has no handler yet, so control-api answers 501 (shown to the operator). */
	ingest: (source: string, action: 'start' | 'stop') => postJson<{ ok: boolean }>(`/api/ingest/${source}/${action}`),
	cancelJob: (id: string) => postJson<{ ok: boolean }>(`/api/jobs/${id}/cancel`),
	sourceDetail: (id: string, limit = 20) => getJson<SourceDetail>(`/api/sources/${id}${q({ limit })}`),
	/** `ctl.ingest.enable|disable`: persists `catalog.source.status` and starts/stops the connector (D-054). */
	sourceEnable: (id: string, enable: boolean) => postJson<{ ok: boolean }>(`/api/sources/${id}/${enable ? 'enable' : 'disable'}`),
	/** Enqueues a `jobs.ingest.backfill` job; follow it with `jobs(kind='ingest.backfill')`. */
	backfill: (id: string, body: { instruments?: string[]; timescale?: string; range?: string; start?: string; end?: string }) =>
		postJson<{ ok: boolean; job_id: string; kind: string }>(`/api/sources/${id}/backfill`, body),
	streams: () => getJson<{ streams: BusStream[] }>('/api/bus/streams'),
	instruments: (query: string, status?: string, limit = 25) =>
		getJson<{ count: number; instruments: Instrument[] }>(`/api/instruments${q({ q: query, status, limit })}`),
	quarantine: (status = 'open', limit = 100) =>
		getJson<{ count: number; quarantine: QuarantineRow[] }>(`/api/instruments/quarantine${q({ status, limit })}`),
	resolveQuarantine: (provisionalId: string, body: Record<string, unknown>) =>
		postJson<{ ok: boolean }>(`/api/instruments/quarantine/${provisionalId}/resolve`, body),
	audit: (since = '24h', limit = 200) => getJson<{ count: number; entries: AuditEntry[]; since: string }>(`/api/audit${q({ since, limit })}`),
	runtimeConfig: () => getJson<ConfigSection>('/api/config/runtime'),
	putRuntimeConfig: (values: unknown) => putJson<ConfigPutResult>('/api/config/runtime', values)
};

/** WebSocket URL for the bus tap, relative to the current origin. */
export function busTapUrl(subjects: string): string {
	const proto = location.protocol === 'https:' ? 'wss' : 'ws';
	return `${proto}://${location.host}/ws/bus?subjects=${encodeURIComponent(subjects)}`;
}

// ---------------------------------------------------------------- live views (spec 50 §4.5–4.7)

export interface ExpertState {
	expert_index: number;
	region_id: string;
	region_lineage_id: string;
	dims: number;
	norm: number;
	uncertainty: number;
}

export interface LatentStateView {
	instrument_id: string;
	snapshot_id: string;
	timescale: string;
	as_of: string | null;
	router_weights: number[];
	experts: ExpertState[];
}

export interface Region {
	region_id: string;
	lineage_id: string;
	snapshot_id: string;
	expert_index: number;
	status: string;
	related_region_ids: string[];
	updated_at: string;
	occupancy: number;
}

export interface GraphNode {
	id: string;
	lineage_id: string | null;
	expert_index: number | null;
	status: string;
	occupancy: number;
}

export interface GraphEdge {
	from: string;
	to: string;
	mean: number;
	lower: number;
	upper: number;
	sampler_id: string;
}

export interface Analog {
	window_id: string;
	window_end: string | null;
	similarity: number;
	time_distance_days: number;
}

export interface DecoderOutputView {
	received_at: string;
	decoder_id: string;
	genome_id: string;
	family: string;
	snapshot_id: string;
	timescale: string;
	as_of: string | null;
	output_refs: string[];
	quantiles: number[];
	analogs: Analog[];
}

export interface JobView {
	job_id: string;
	kind: string;
	status: 'queued' | 'running' | 'ok' | 'failed' | 'preempted' | 'budget_exceeded';
	priority: number;
	snapshot_id: string;
	genome_id: string;
	budget_gpu_ms: number;
	used_gpu_ms: number;
	progress: number;
	message: string;
	worker_id: string;
	error: string;
	score: number | null;
	objectives: Record<string, number>;
	submitted_at: string;
	updated_at: string;
}

export interface GpuStatus {
	available: boolean;
	reason?: string;
	last_seen?: string;
	used_mb?: number | null;
	active?: number | null;
	queued?: number | null;
	capacity_mb?: number | null;
	generations_started?: number | null;
	generations_done?: number | null;
	classes: string[];
}

export interface Position {
	instrument_id: string;
	quantity: number;
	avg_cost_eur: number;
	mark_eur: number | null;
	market_value_eur: number | null;
	unrealised_eur: number | null;
	realised_eur: number;
}

export interface OrderEventView {
	kind: string;
	fill_price: number;
	fill_quantity: number;
	fee: number;
	fee_currency: string;
	event_time: string | null;
	reason: string;
	/** Execution-cost fields of D-051 (`OrderEvent.arrival_price` / `slippage_bps`), when control-api forwards them. */
	arrival_price?: number | null;
	slippage_bps?: number | null;
}

export interface OrderView {
	arrival_price?: number | null;
	slippage_bps?: number | null;
	intent_id: string;
	account_id: string;
	instrument_id: string;
	side: string;
	quantity: number;
	exec_algo: string;
	limit_price: number | null;
	mode: string;
	strategy_ids: string[];
	status: string;
	filled_quantity: number;
	avg_fill_price: number | null;
	fees: number;
	reason: string;
	created_at: string | null;
	updated_at: string;
	events: OrderEventView[];
}

export interface Halt {
	scope: string;
	reason: string;
	since: string | null;
	manual: boolean;
}

export interface RiskFigures {
	equity_eur: number;
	deployable_eur: number;
	gross_exposure_eur: number;
	drawdown: number;
	daily_pnl_eur?: number;
	pnl_eur?: number;
	as_of?: string | null;
	received_at?: string;
}

export interface RiskView {
	mode: 'paper';
	live_available: boolean;
	reported: RiskFigures | null;
	paper: RiskFigures;
	halts: Halt[];
	portfolio_halted: boolean;
}

export interface SignalView {
	strategy_id: string;
	genome_id: string;
	horizon: string;
	as_of: string | null;
	exposures: { instrument_id: string; mean: number; stddev: number }[];
}

// ---------------------------------------------------------------- evolution (D-049, D-052)

export interface StartGenerationBody {
	kind: string;
	snapshot_id: string;
	genomes: string[];
	eta?: number;
	min_budget_ms?: number;
	max_budget_ms?: number;
	max_generations?: number;
	total_budget_ms?: number;
	data_window_hours?: number;
}

/** Row of `GET /api/evolution/generations` (in-memory `generation.summary` aggregate). */
export interface GenerationRow {
	id: string;
	kind: string;
	generation: number;
	max_generations: number;
	rung: number;
	rungs: number;
	final: boolean;
	candidates: number;
	front_sizes: number[];
	/** Survivor count (the list route) — the WS frame carries the ids instead. */
	survivors: number | string[];
	status: 'running' | 'done' | 'paused' | 'stopped' | string;
	at: string | null;
	updated_at: string;
}

export interface ValidationStats {
	n_trials: number;
	sharpe_mean?: number | null;
	sharpe_var?: number | null;
	best_candidate?: string;
	best_sharpe?: number | null;
	expected_max_sharpe?: number | null;
	deflated_sharpe?: number | null;
	pbo?: number | null;
	pbo_trials?: number;
	pbo_blocks?: number;
	loop?: number;
	computed_at?: string;
}

export interface FrontMember {
	candidate: string;
	objectives: Record<string, number>;
	rank: number;
	crowding: number | string;
}

/** `generation.<id>` frame on `/ws/evolution`: the full `GenerationView`. */
export interface GenerationFrame {
	generation_id: string;
	kind: string;
	generation: number;
	max_generations: number;
	rung: number;
	rungs: number;
	final: boolean;
	candidates: number;
	unranked: number;
	objective_keys: string[];
	front_sizes: number[];
	survivors: string[];
	pareto_front: FrontMember[];
	next_population: number;
	vary_job_id: string;
	elapsed_ms: number;
	at: string | null;
	status: string;
	/** Key = generation * 1000 + rung. */
	survivors_by_rung: Record<string, string[]>;
	updated_at: string;
	/** Present when the summary carried them (not stripped by the aggregate). */
	validation?: ValidationStats | null;
	artifact_key?: string;
}

export interface Population {
	id: string;
	kind: string;
	loops: number;
	trials: number;
	candidates: number;
	first_at: string;
	last_at: string;
	stats: ValidationStats | null;
}

export type Direction = 'min' | 'max';

export interface ParetoView {
	id: string;
	kind: string;
	gen: number;
	candidates: number;
	objective_keys: string[];
	directions: Record<string, Direction>;
	front_sizes: number[];
	pareto_front: FrontMember[];
	fronts: FrontMember[][];
	trials?: unknown[];
	stats?: ValidationStats | null;
}

export interface FitnessRecord {
	$id?: number;
	generation: number;
	objectives: Record<string, number>;
	pareto_rank: number;
}

export interface GenomeDetail {
	genome_id: string;
	genome: { $id?: number; created_at: string; family: string; genome_id: string; kind: string; params: string } | null;
	params: string | null;
	fitness: FitnessRecord[];
	parents: unknown[];
	jobs: JobView[];
	latest_objectives: Record<string, number> | null;
	generations: unknown[];
	helix: { error: string } | null;
}

export interface EvalReport {
	job_id: string;
	generation_id: string;
	key: string;
	status: string;
	url: string | null;
}

export interface GenomeOutputs {
	genome_id: string;
	outputs: (DecoderOutputView & { urls?: ({ ref: string; url: string | null } | string)[] })[];
	eval_reports: EvalReport[];
	presign: { configured: boolean; endpoint: string; error: string | null; ttl_secs: number };
}

export interface Holdout {
	id?: string | number;
	start?: string;
	end?: string;
	locked_at?: string;
	note?: string;
	[k: string]: unknown;
}

export interface ValidationTrial {
	id: string;
	job_id: string;
	generation: string;
	gen_index: number;
	rung: number;
	kind: string;
	objectives: Record<string, number>;
	sharpe: number | null;
	n_obs: number;
	window_start: string | null;
	window_end: string | null;
	report_ref: string | null;
	created_at: string;
}

export interface ValidationReports {
	candidate: string;
	kind: string;
	directions: Record<string, Direction>;
	latest: { created_at: string; generation: string; objectives: Record<string, number>; sharpe: number | null; n_obs: number } | null;
	n_trials: number;
	trials: ValidationTrial[];
}

// ---------------------------------------------------------------- persisted latent (D-048, D-055)

export interface Snapshot {
	$id?: number;
	snapshot_id: string;
	timescale: string;
	status: 'active' | 'candidate' | 'retired' | string;
	encoder_family: string;
	generation: number;
	created_at: string;
	garage_key: string;
}

export interface ExpertRouting {
	expert_index: number;
	dims: number;
	messages: number;
	gate_mean: number;
	top_share: number;
	regions_seen: number;
	norm_mean: number;
	uncertainty_mean: number;
}

export interface ExpertRow {
	expert_index: number;
	expert_space_id: string;
	dim: number;
	regions: number;
	regions_live: number;
	helix_id?: number;
	routing: ExpertRouting | null;
}

export interface ExpertsView {
	snapshot: Snapshot | null;
	source: string;
	experts: ExpertRow[];
	covariance: {
		experts: number[];
		matrix: number[][];
		correlation: number[][];
		mean: number[];
		n: number;
		feature: string;
	} | null;
	live: { messages: number; last_as_of: string | null };
}

export interface ProjectionPoint {
	id: number;
	coords: number[];
	region_id: string;
	kind: string;
	instrument_id: string;
	ts: string;
	uncertainty: number;
}

export interface ProjectionView {
	snapshot_id: string;
	expert_space_id: string;
	expert_index: number;
	dim: number;
	dims: number;
	requested: number;
	fetched: number;
	total_in_space: number;
	cached: boolean;
	cache_ttl_secs: number;
	range: { spec: string; from: string | null };
	points: ProjectionPoint[];
	pca: { mean: number[]; components: number[][]; explained_variance_ratio: number[]; total_variance: number; method: string };
}

export interface McmcRegion {
	$id?: number;
	region_id: string;
	lineage_id: string;
	expert_space_id: string;
	status: string;
	density: number;
	radius: number;
	method: string;
	related_region_id: string;
	created_at: string;
	updated_at: string;
}

export interface McmcEdge {
	id?: number;
	from: number;
	to: number;
	from_region_id: string;
	to_region_id: string;
	count: number;
	mean: number;
	lower: number;
	upper: number;
	sampler_id: string;
	updated_at: string;
}

export interface McmcGraph {
	snapshot: Snapshot;
	timescale: string;
	spaces: { $id?: number; expert_space_id: string; index: number; dim: number; snapshot_id: string }[];
	nodes: McmcRegion[];
	edges: McmcEdge[];
}

export type HelixNode = { $id: number; $label?: string; [k: string]: unknown };
export type HelixEdge = { $id: number; $label: string; $from: number; $to: number; [k: string]: unknown };

export interface Neighborhood {
	center: HelixNode;
	depth: number;
	nodes: HelixNode[];
	edges: HelixEdge[];
}

export interface GraphQueryDef {
	name: string;
	doc: string;
	/** `"name: type"` or `"name: type?"` for optional parameters. */
	params: string[];
}

// ---------------------------------------------------------------- strategy lifecycle (D-051)

export type StrategyAction = 'pause' | 'resume' | 'flatten' | 'promote' | 'shadow';

export interface GateItem {
	name: string;
	pass: boolean | null;
	advisory: boolean;
	detail: string;
}

export interface StrategyView {
	strategy_id: string;
	status: 'active' | 'paused' | 'flattening' | 'shadow' | string;
	prev_status: string | null;
	since: string | null;
	first_seen: string | null;
	sleeve_eur: number;
	weight: number;
	pnl_eur: number;
	peak_pnl_eur: number;
	drawdown: number;
	max_drawdown: number;
	last_reason: string;
	shadow: { days: number | null; horizons_completed: number; since: string | null };
	gate: { green: boolean; items: GateItem[]; missing: string[] } | null;
	signal: SignalView | null;
}

export interface StrategiesView {
	source: 'risk-engine' | 'postgres' | string;
	count: number;
	status_counts: Record<string, number>;
	strategies: StrategyView[];
	signals_without_state: SignalView[];
}

export interface Limits {
	max_drawdown: number;
	max_daily_loss: number;
	max_position_frac: number;
	max_class_frac?: number;
	max_venue_frac?: number;
	portfolio_gross_cap?: number;
	strategy_max_drawdown?: number;
	liquidity_k?: number;
	daily_payout_quota_eur?: number;
	leverage_cap: Record<string, number>;
	[k: string]: number | Record<string, number> | undefined;
}

export interface PnlView {
	book: string;
	range: { from: string; to: string };
	strategy: string | null;
	fills: number;
	realised_eur: number;
	fees_eur: number;
	net_eur: number;
	turnover_eur: number;
	by_instrument: { instrument_id: string; realised_eur: number; net_quantity_in_range: number; avg_cost_eur: number }[];
	snapshot: { instrument: string; qty: number; avg_price: number; as_of: string; book: string }[];
	note?: string;
}

export interface ReconciliationRun {
	id: string;
	started_at: string;
	status: 'ok' | 'diff' | string;
	venue_id: string;
	diffs: {
		book?: string;
		cash?: { memory: number; postgres: number };
		positions?: { instrument_id: string; memory: number; postgres: number }[];
	} | null;
}

/** Error carrying the control-api's `{error}` message. */
export class ApiError extends Error {
	constructor(
		public status: number,
		message: string,
		/** The decoded error body (e.g. `{error, known}` from the graph query catalog). */
		public body: unknown = null
	) {
		super(message);
	}
}

async function sendJson<T>(method: 'POST' | 'PUT', path: string, body: unknown): Promise<T> {
	const res = await fetch(path, {
		method,
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(body)
	});
	const data = await res.json().catch(() => ({}));
	if (!res.ok) {
		const msg = data.reason ?? data.error ?? `${res.status} ${res.statusText}`;
		throw new ApiError(res.status, data.detail ? `${msg}: ${data.detail}` : msg, data);
	}
	return data as T;
}

export async function postJson<T>(path: string, body: unknown = {}): Promise<T> {
	return sendJson<T>('POST', path, body);
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
	return sendJson<T>('PUT', path, body);
}

export const q = (params: Record<string, string | number | boolean | undefined>) => {
	const s = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') s.set(k, String(v));
	const str = s.toString();
	return str ? `?${str}` : '';
};

export const live = {
	timescales: () => getJson<{ configured: string[]; seen: string[] }>('/api/latent/timescales'),
	states: (ts: string) => getJson<{ states: LatentStateView[] }>(`/api/latent/timescales/${ts}/states`),
	regions: (ts: string) => getJson<{ regions: Region[] }>(`/api/latent/timescales/${ts}/regions`),
	graph: (ts: string) => getJson<{ nodes: GraphNode[]; edges: GraphEdge[] }>(`/api/latent/timescales/${ts}/transitions`),
	decoderOutputs: (family?: string, limit = 30) =>
		getJson<{ outputs: DecoderOutputView[]; families: string[] }>(`/api/decoders/outputs${q({ family, limit })}`),

	jobs: (kind?: string, limit = 100) =>
		getJson<{ counts: Record<string, Record<string, number>>; jobs: JobView[]; objectives: string[] }>(
			`/api/evolution/jobs${q({ kind, limit })}`
		),
	gpu: () => getJson<GpuStatus>('/api/evolution/gpu'),
	startGeneration: (body: StartGenerationBody) => postJson<{ ok: boolean; generation_id: string }>('/api/evolution/generations', body),
	generations: () => getJson<{ generations: GenerationRow[]; count: number }>('/api/evolution/generations'),
	runAction: (id: string, action: 'pause' | 'resume' | 'stop') => postJson<{ ok: boolean }>(`/api/evolution/runs/${id}/${action}`),
	populations: () => getJson<{ populations: Population[] }>('/api/evolution/populations'),
	pareto: (id: string, gen?: number) => getJson<ParetoView>(`/api/evolution/populations/${id}/pareto${q({ gen })}`),
	genome: (id: string) => getJson<GenomeDetail>(`/api/evolution/genomes/${encodeURIComponent(id)}`),
	genomeOutputs: (id: string, limit = 20) => getJson<GenomeOutputs>(`/api/evolution/genomes/${encodeURIComponent(id)}/outputs${q({ limit })}`),
	holdout: () => getJson<{ holdouts: Holdout[]; immutable: boolean; note?: string }>('/api/validation/holdout'),
	validationReports: (candidate: string, limit = 20) =>
		getJson<ValidationReports>(`/api/validation/reports/${encodeURIComponent(candidate)}${q({ limit })}`),

	snapshots: () => getJson<{ snapshots: Snapshot[]; by_timescale: Record<string, Snapshot[]> }>('/api/latent/snapshots'),
	experts: (snapshot: string) => getJson<ExpertsView>(`/api/latent/${snapshot}/experts`),
	projection: (snapshot: string, p: { expert: number | string; range?: string; n?: number; dims?: 2 | 3; kind?: string }) =>
		getJson<ProjectionView>(`/api/latent/${snapshot}/projection${q({ expert: p.expert, range: p.range, n: p.n, dims: p.dims, kind: p.kind })}`),
	mcmcGraph: (snapshot: string, timescale: string) => getJson<McmcGraph>(`/api/mcmc/${snapshot}/graph${q({ timescale })}`),
	snapshotAction: (id: string, action: 'promote' | 'rollback') => postJson<{ ok: boolean }>(`/api/latent/snapshots/${id}/${action}`),
	neighborhood: (node: string, depth = 1) => getJson<Neighborhood>(`/api/graph/neighborhood${q({ node, depth })}`),
	graphQueries: () => getJson<{ queries: GraphQueryDef[] }>('/api/graph/query'),
	graphQuery: (name: string, params: Record<string, unknown>) => postJson<{ name: string; result: unknown }>('/api/graph/query', { name, params }),

	strategies: () => getJson<StrategiesView>('/api/trading/strategies'),
	strategyAction: (id: string, action: StrategyAction) =>
		postJson<{ ok: boolean; strategy_id: string; action: string; strategy: unknown }>(`/api/trading/strategies/${encodeURIComponent(id)}/${action}`),
	limits: () => getJson<{ limits: Limits; note?: string }>('/api/trading/limits'),
	/** Tighten-only (spec 40 §6.5): the API answers 422 `loosen: <fields>` otherwise. */
	putLimits: (partial: Record<string, unknown>) => putJson<{ ok: boolean; limits: Limits }>('/api/trading/limits', partial),
	flattenAll: () => postJson<{ ok: boolean; flattening: string[] }>('/api/trading/flatten'),
	pnl: (range = '1d', strategy?: string) => getJson<PnlView>(`/api/trading/pnl${q({ range, strategy })}`),
	reconciliation: (limit = 10) => getJson<{ latest_status: string | null; runs: ReconciliationRun[] }>(`/api/trading/reconciliation${q({ limit })}`),

	positions: () => getJson<{ cash_eur: number; equity_eur: number; fees_eur: number; positions: Position[] }>('/api/trading/positions'),
	orders: (limit = 100) => getJson<{ orders: OrderView[] }>(`/api/trading/orders${q({ limit })}`),
	risk: () => getJson<RiskView>('/api/trading/risk'),
	signals: () => getJson<{ signals: SignalView[] }>('/api/trading/signals'),
	halt: (reason: string) => postJson<{ ok: boolean }>('/api/trading/halt', { reason }),
	resume: () => postJson<{ ok: boolean }>('/api/trading/resume'),
	paperOrder: (body: { instrument_id: string; side: 'buy' | 'sell'; quantity: number; limit_price?: number }) =>
		postJson<{ ok: boolean; intent_id: string }>('/api/trading/paper-order', { ...body, mode: 'paper' })
};

/**
 * Runs `fn` now and then every `ms`; returns a cleanup for onMount. `ms` may be a function,
 * re-read after every run, so a view can poll fast while its socket is down and slowly (or not
 * at all: return `null`, re-checked every second) while live frames arrive.
 */
export function poll(fn: () => Promise<void> | void, ms: number | (() => number | null) = 2000): () => void {
	let stopped = false;
	let t: ReturnType<typeof setTimeout> | undefined;
	const interval = () => (typeof ms === 'function' ? ms() : ms);
	const schedule = () => {
		if (stopped) return;
		const next = interval();
		t = setTimeout(next === null ? schedule : run, next ?? 1000);
	};
	const run = async () => {
		if (stopped) return;
		try {
			await fn();
		} finally {
			schedule();
		}
	};
	run();
	return () => {
		stopped = true;
		clearTimeout(t);
	};
}

/** Parses `"<n>m|h|d"` into milliseconds (the API's range grammar); null when invalid. */
export function rangeMs(spec: string): number | null {
	const m = /^(\d+)\s*([mhd])$/.exec(spec.trim());
	if (!m) return null;
	const n = Number(m[1]);
	return n * { m: 60_000, h: 3_600_000, d: 86_400_000 }[m[2] as 'm' | 'h' | 'd'];
}

export const num = (v: number | null | undefined, digits = 3) =>
	v === null || v === undefined || Number.isNaN(v) ? '—' : Math.abs(v) >= 1e6 ? v.toExponential(2) : v.toFixed(digits);

export const bytes = (b: number) => (b < 1024 ? `${b} B` : b < 1 << 20 ? `${(b / 1024).toFixed(1)} KB` : b < 1 << 30 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${(b / (1 << 30)).toFixed(2)} GB`);

/** Short id for tables. */
export const short = (id: string, n = 8) => (id.length > n ? id.replace(/-/g, '').slice(0, n) : id);

export const eur = (v: number | null | undefined) =>
	v === null || v === undefined || Number.isNaN(v)
		? '—'
		: v.toLocaleString('en-GB', { style: 'currency', currency: 'EUR', maximumFractionDigits: 2 });

export const pct = (v: number | null | undefined, digits = 2) =>
	v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`;

export function ago(iso: string | null | undefined, now = Date.now()): string {
	if (!iso) return '—';
	const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
	return s < 60 ? `${s}s` : s < 3600 ? `${Math.round(s / 60)}m` : `${Math.round(s / 3600)}h`;
}
