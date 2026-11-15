// Pure logic for the /growth page (spec 51 §6, round 9): PromQL builders for the growth series,
// range-picker → `query_range` parameters (within control-api's proxy limits, metrics_routes.rs:
// range ≤ 31 d, ≤ 11 000 points per series), VictoriaMetrics matrices reshaped for the SVG line
// chart, cap/reference-line partitioning and the money panel built from /api/accounting/balances.
import type { VmMatrix } from './pointcloud';

// ---------------------------------------------------------------- time ranges

export type RangeKey = '1h' | '6h' | '24h' | '7d' | '30d';

export const RANGES: { key: RangeKey; seconds: number }[] = [
	{ key: '1h', seconds: 3_600 },
	{ key: '6h', seconds: 6 * 3_600 },
	{ key: '24h', seconds: 86_400 },
	{ key: '7d', seconds: 7 * 86_400 },
	{ key: '30d', seconds: 30 * 86_400 }
];

/** ~240 points per series; never finer than the 15 s scrape interval. */
export function rangeParams(key: RangeKey, nowMs = Date.now()): { start: number; end: number; step: number } {
	const spec = RANGES.find((r) => r.key === key) ?? RANGES[2];
	const end = Math.floor(nowMs / 1000);
	const step = Math.max(15, Math.ceil(spec.seconds / 240));
	return { start: end - spec.seconds, end, step };
}

/** Lookbehind window for `rate()`: a few steps, at least 5 min (heartbeats mirror every ~15 s). */
export function rateWindow(stepS: number): string {
	return `${Math.max(300, 4 * stepS)}s`;
}

// ---------------------------------------------------------------- PromQL builders
// Series verified against the live stack 2026-11-15; a chart whose query matches nothing
// renders the empty state naming the query, never fabricated data.

/** Ingest events per minute per source (`fa_heartbeat_counter{service="ingestor", key="events.<source>"}`). */
export function ingestEventsQuery(stepS: number): string {
	return `sum by (key) (rate(fa_heartbeat_counter{service="ingestor",key=~"events\\\\..*"}[${rateWindow(stepS)}])) * 60`;
}

/** Bars written to the TSDB per minute (ingestor `tsdb.written`). */
export function tsdbWrittenQuery(stepS: number): string {
	return `sum(rate(fa_heartbeat_counter{service="ingestor",key="tsdb.written"}[${rateWindow(stepS)}])) * 60`;
}

/** VictoriaMetrics stored rows (data samples, not index rows). */
export function vmRowsQuery(): string {
	return `sum(vm_rows{type=~"storage/.*"})`;
}

/** VictoriaMetrics on-disk data bytes. */
export function vmBytesQuery(): string {
	return `sum(vm_data_size_bytes)`;
}

/** Active series over the last hour (VM self-metric). */
export function vmSeriesQuery(): string {
	return `sum(vm_cache_entries{type="storage/hour_metric_ids"})`;
}

/** JetStream bytes per stream (nats-exporter); caps come from GET /api/bus/streams. */
export function streamBytesQuery(): string {
	return `sum by (stream_name) (jetstream_stream_total_bytes)`;
}

/** Live latent regions per timescale (model-runtime `regions_live.<tf>`). */
export function regionsLiveQuery(): string {
	return `sum by (key) (fa_heartbeat_counter{service="model-runtime",key=~"regions_live\\\\..*"})`;
}

/** Regions born since start (model-runtime lifecycle counter, cumulative). */
export function regionsBornQuery(): string {
	return `sum(fa_heartbeat_counter{service="model-runtime",key="lifecycle.born"})`;
}

/** Encoder snapshots seen in lineage by strategy-runtime (cumulative). */
export function snapshotsQuery(): string {
	return `sum(fa_heartbeat_counter{service="strategy-runtime",key="lineage.snapshots"})`;
}

/** Strategies loaded (total and neural) in strategy-runtime. */
export function strategiesQuery(): string {
	return `sum by (key) (fa_heartbeat_counter{service="strategy-runtime",key=~"strategies(\\\\.neural)?"})`;
}

/** Genome / generation counters from the evolution services — no such series exists on the
 *  freshly bootstrapped stack yet, so this slot renders its empty state naming the query. */
export function generationsQuery(): string {
	return `sum by (service,key) (fa_heartbeat_counter{service=~"evo-orchestrator|evo-worker",key=~"(generations|genomes|promotions).*"})`;
}

// ---------------------------------------------------------------- matrix → chart series

export interface Series {
	name: string;
	/** [ms since epoch, value], finite values only, ascending time. */
	pts: [number, number][];
}

/** `events.kraken` → `kraken`; leaves keys without the prefix alone. */
export function stripPrefix(key: string, prefix: string): string {
	return key.startsWith(prefix) ? key.slice(prefix.length) : key;
}

/** A `query_range` matrix → named series sorted by name (stable colour assignment). */
export function seriesFromMatrix(resp: VmMatrix, nameOf: (metric: Record<string, string>) => string = defaultName): Series[] {
	const out: Series[] = [];
	for (const r of resp.data?.result ?? []) {
		const pts: [number, number][] = [];
		for (const [t, v] of r.values ?? []) {
			const x = Number(v);
			if (Number.isFinite(x)) pts.push([t * 1000, x]);
		}
		out.push({ name: nameOf(r.metric ?? {}), pts });
	}
	return out.sort((a, b) => a.name.localeCompare(b.name));
}

function defaultName(metric: Record<string, string>): string {
	const keys = Object.keys(metric).filter((k) => k !== '__name__');
	if (keys.length === 0) return metric.__name__ ?? 'value';
	return keys
		.sort()
		.map((k) => metric[k])
		.join(' ');
}

export function hasData(series: Series[]): boolean {
	return series.some((s) => s.pts.length > 0);
}

/** Splits a series into contiguous runs so the line breaks across gaps (sparse young series
 *  must not draw a misleading bridge over hours with no samples). */
export function splitSegments(pts: [number, number][], maxGapMs: number): [number, number][][] {
	const runs: [number, number][][] = [];
	let run: [number, number][] = [];
	for (const p of pts) {
		if (run.length && p[0] - run[run.length - 1][0] > maxGapMs) {
			runs.push(run);
			run = [];
		}
		run.push(p);
	}
	if (run.length) runs.push(run);
	return runs;
}

/** Value range over all series (plus forced extra values); null when no points. */
export function valueRange(series: Series[], include: number[] = []): { lo: number; hi: number } | null {
	let lo = Infinity;
	let hi = -Infinity;
	for (const s of series)
		for (const [, v] of s.pts) {
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
	if (!Number.isFinite(lo)) return null;
	for (const v of include) {
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	if (lo === hi) {
		lo -= 1;
		hi += 1;
	}
	return { lo, hi };
}

// ---------------------------------------------------------------- caps (reference lines)

export interface Cap {
	name: string;
	value: number;
}

/** Caps close enough to the data to draw as reference lines without flattening the chart
 *  (≤ `factor` × data max); the rest are listed as text so the headroom stays honest. */
export function partitionCaps(caps: Cap[], dataHi: number, factor = 3): { drawn: Cap[]; noted: Cap[] } {
	const drawn: Cap[] = [];
	const noted: Cap[] = [];
	for (const c of caps) (c.value > 0 && c.value <= dataHi * factor ? drawn : noted).push(c);
	return { drawn, noted };
}

/** Stream caps for the NATS chart, deduplicated by value (most streams share the 1 GiB cap). */
export function streamCaps(streams: { name: string; max_bytes: number; planned_max_bytes?: number | null }[]): Cap[] {
	const byValue = new Map<number, string[]>();
	for (const s of streams) {
		const cap = s.max_bytes > 0 ? s.max_bytes : (s.planned_max_bytes ?? 0);
		if (cap > 0) byValue.set(cap, [...(byValue.get(cap) ?? []), s.name]);
	}
	return [...byValue]
		.sort((a, b) => a[0] - b[0])
		.map(([value, names]) => ({ value, name: names.length > 2 ? `${names.length} streams` : names.join(', ') }));
}

// ---------------------------------------------------------------- money panel (D-104)

export interface Account {
	kind: string;
	segment: string;
	normal_eur: number;
	external_ref: string | null;
}

export interface BalancesView {
	accounts: Record<string, Account>;
	segment_funds_eur: Record<string, number>;
	trial_balance: number;
	entries?: number;
}

/** The five organisational segments of D-104, in presentation order. */
export const SEGMENTS = ['trading', 'infrastructure', 'operations', 'owner', 'treasury'] as const;

export interface MoneyView {
	segments: { segment: string; funds: number }[];
	/** Accounts of kind `expense` are cumulative opex. */
	expenses: { account: string; eur: number; external_ref: string | null }[];
	opexTotal: number;
	trialBalance: number;
	/** No allocation happened yet: every segment fund except trading is absent or zero. */
	allRetained: boolean;
}

export function moneyView(b: BalancesView): MoneyView {
	const funds = b.segment_funds_eur ?? {};
	const segments = SEGMENTS.map((segment) => ({ segment, funds: funds[segment] ?? 0 }));
	const expenses = Object.entries(b.accounts ?? {})
		.filter(([, a]) => a.kind === 'expense')
		.map(([account, a]) => ({ account, eur: a.normal_eur, external_ref: a.external_ref }))
		.sort((a, b2) => b2.eur - a.eur || a.account.localeCompare(b2.account));
	const opexTotal = expenses.reduce((sum, e) => sum + e.eur, 0);
	const allRetained =
		(funds.trading ?? 0) !== 0 && segments.every((s) => s.segment === 'trading' || Math.abs(s.funds) < 0.005);
	return { segments, expenses, opexTotal, trialBalance: b.trial_balance ?? 0, allRetained };
}

// ---------------------------------------------------------------- formatting

/** Compact count for chart axes: 1500 → `1.5k`, 2_300_000 → `2.3M`. */
export function fmtCount(v: number): string {
	const a = Math.abs(v);
	if (a >= 1e9) return `${(v / 1e9).toFixed(1)}G`;
	if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
	if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
	return a >= 10 || Number.isInteger(v) ? `${Math.round(v)}` : v.toFixed(1);
}

/** Axis time label: time of day inside a day, date beyond. */
export function fmtTime(ms: number, rangeKey: RangeKey): string {
	const d = new Date(ms);
	return rangeKey === '7d' || rangeKey === '30d'
		? d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
		: d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}
