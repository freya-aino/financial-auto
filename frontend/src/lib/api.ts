// Typed client for control-api (spec 50). Paths are relative; vite proxies them in dev.

export interface ServiceStatus {
	service: string;
	instance_id: string;
	version: string;
	last_seen: string;
	counters: Record<string, number>;
}

export interface Source {
	id: string;
	name: string;
	category: string;
	capabilities: string[];
	asset_classes: string[];
	transport: string[];
	auth: string;
	status: 'planned' | 'stub' | 'live' | 'disabled';
	tos_notes: string;
	notes: string;
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

async function getJson<T>(path: string): Promise<T> {
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
	/** Spec 50 §4.1; 501 until the orchestrator handles cancellation. */
	cancelJob: (id: string) => postJson<{ ok: boolean }>(`/api/jobs/${id}/cancel`)
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
}

export interface OrderView {
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

/** Error carrying the control-api's `{error}` message. */
export class ApiError extends Error {
	constructor(
		public status: number,
		message: string
	) {
		super(message);
	}
}

async function postJson<T>(path: string, body: unknown = {}): Promise<T> {
	const res = await fetch(path, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(body)
	});
	const data = await res.json().catch(() => ({}));
	if (!res.ok) throw new ApiError(res.status, data.reason ?? data.error ?? `${res.status} ${res.statusText}`);
	return data as T;
}

const q = (params: Record<string, string | number | boolean | undefined>) => {
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
	startGeneration: (body: {
		kind: string;
		snapshot_id: string;
		genomes: string[];
		eta?: number;
		min_budget_ms?: number;
		max_budget_ms?: number;
	}) => postJson<{ ok: boolean; generation_id: string }>('/api/evolution/generations', body),

	positions: () => getJson<{ cash_eur: number; equity_eur: number; fees_eur: number; positions: Position[] }>('/api/trading/positions'),
	orders: (limit = 100) => getJson<{ orders: OrderView[] }>(`/api/trading/orders${q({ limit })}`),
	risk: () => getJson<RiskView>('/api/trading/risk'),
	signals: () => getJson<{ signals: SignalView[] }>('/api/trading/signals'),
	halt: (reason: string) => postJson<{ ok: boolean }>('/api/trading/halt', { reason }),
	resume: () => postJson<{ ok: boolean }>('/api/trading/resume'),
	paperOrder: (body: { instrument_id: string; side: 'buy' | 'sell'; quantity: number; limit_price?: number }) =>
		postJson<{ ok: boolean; intent_id: string }>('/api/trading/paper-order', { ...body, mode: 'paper' })
};

/** Runs `fn` now and every `ms`; returns a cleanup for onMount. */
export function poll(fn: () => Promise<void> | void, ms = 2000): () => void {
	let stopped = false;
	const run = async () => {
		if (!stopped) await fn();
	};
	run();
	const t = setInterval(run, ms);
	return () => {
		stopped = true;
		clearInterval(t);
	};
}

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
