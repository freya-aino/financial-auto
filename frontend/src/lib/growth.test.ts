import { describe, expect, it } from 'vitest';
import {
	SEGMENTS,
	fmtCount,
	generationsQuery,
	hasData,
	ingestEventsQuery,
	moneyView,
	partitionCaps,
	rangeParams,
	rateWindow,
	regionsLiveQuery,
	seriesFromMatrix,
	splitSegments,
	streamBytesQuery,
	streamCaps,
	stripPrefix,
	valueRange,
	vmBytesQuery,
	type BalancesView
} from './growth';

const NOW = 1_794_743_000_000; // 2026-11-15

describe('range → query_range parameters', () => {
	it('targets ~240 points and stays inside the proxy limits (31 d, 11 000 points)', () => {
		for (const key of ['1h', '6h', '24h', '7d', '30d'] as const) {
			const { start, end, step } = rangeParams(key, NOW);
			expect(end).toBe(Math.floor(NOW / 1000));
			expect(end - start).toBeLessThanOrEqual(31 * 86_400);
			expect((end - start) / step).toBeLessThanOrEqual(11_000);
			expect((end - start) / step).toBeLessThanOrEqual(241);
			expect(Number.isInteger(step)).toBe(true);
			expect(step).toBeGreaterThanOrEqual(15);
		}
		expect(rangeParams('1h', NOW).step).toBe(15);
		expect(rangeParams('24h', NOW).step).toBe(360);
	});

	it('rate window is a few steps, at least 5 min', () => {
		expect(rateWindow(15)).toBe('300s');
		expect(rateWindow(360)).toBe('1440s');
	});
});

describe('promql builders', () => {
	it('ingest events: per-source rate over the step-scaled window', () => {
		const q = ingestEventsQuery(360);
		expect(q).toContain('service="ingestor"');
		expect(q).toContain('key=~"events\\\\..*"');
		expect(q).toContain('[1440s]');
		expect(q).toContain('sum by (key)');
	});
	it('store and bus queries name the exporter series', () => {
		expect(vmBytesQuery()).toContain('vm_data_size_bytes');
		expect(streamBytesQuery()).toContain('jetstream_stream_total_bytes');
		expect(streamBytesQuery()).toContain('sum by (stream_name)');
		expect(regionsLiveQuery()).toContain('regions_live');
		// no genome/generation series exists on a fresh stack; the query is still honest about
		// what it would read, so the empty state can name it
		expect(generationsQuery()).toContain('evo-orchestrator|evo-worker');
	});
});

describe('matrix → series', () => {
	const matrix = {
		data: {
			result: [
				{ metric: { key: 'events.yahoo' }, values: [[100, '2'], [115, '3']] as [number, string][] },
				{ metric: { key: 'events.kraken' }, values: [[100, '5'], [115, 'NaN']] as [number, string][] }
			]
		}
	};
	it('names, sorts and converts to ms, dropping non-finite samples', () => {
		const s = seriesFromMatrix(matrix, (m) => stripPrefix(m.key ?? '', 'events.'));
		expect(s.map((x) => x.name)).toEqual(['kraken', 'yahoo']);
		expect(s[1].pts).toEqual([
			[100_000, 2],
			[115_000, 3]
		]);
		expect(s[0].pts).toEqual([[100_000, 5]]);
		expect(hasData(s)).toBe(true);
		expect(hasData(seriesFromMatrix({ data: { result: [] } }))).toBe(false);
	});
	it('default naming joins label values and falls back to the metric name', () => {
		expect(seriesFromMatrix({ data: { result: [{ metric: {}, values: [[1, '1']] }] } })[0].name).toBe('value');
		expect(seriesFromMatrix({ data: { result: [{ metric: { service: 'a', key: 'b' }, values: [] }] } })[0].name).toBe('b a');
	});

	it('splits a sparse series at gaps instead of bridging them', () => {
		const pts: [number, number][] = [
			[0, 1],
			[15, 1],
			[500, 2],
			[515, 2]
		];
		expect(splitSegments(pts, 40)).toEqual([
			[
				[0, 1],
				[15, 1]
			],
			[
				[500, 2],
				[515, 2]
			]
		]);
		expect(splitSegments([], 40)).toEqual([]);
	});

	it('value range spans all series plus forced values; flat series get padding', () => {
		const s = [{ name: 'a', pts: [[0, 3]] as [number, number][] }];
		expect(valueRange(s, [0])).toEqual({ lo: 0, hi: 3 });
		expect(valueRange(s)).toEqual({ lo: 2, hi: 4 });
		expect(valueRange([])).toBeNull();
	});
});

describe('caps', () => {
	it('draws near caps, notes far ones (a 1 GiB cap must not flatten a 5 MB series)', () => {
		const { drawn, noted } = partitionCaps(
			[
				{ name: 'MD', value: 10 },
				{ name: 'LATENT', value: 1 << 30 }
			],
			5
		);
		expect(drawn).toEqual([{ name: 'MD', value: 10 }]);
		expect(noted[0].name).toBe('LATENT');
	});
	it('stream caps dedupe by value and ignore uncapped streams', () => {
		const caps = streamCaps([
			{ name: 'MD', max_bytes: 1024 },
			{ name: 'TEXT', max_bytes: 1024 },
			{ name: 'LATENT', max_bytes: 1024 },
			{ name: 'JOBS', max_bytes: -1, planned_max_bytes: null },
			{ name: 'DLQ', max_bytes: 2048 }
		]);
		expect(caps).toEqual([
			{ name: '3 streams', value: 1024 },
			{ name: 'DLQ', value: 2048 }
		]);
	});
});

describe('money panel (D-104)', () => {
	const balances: BalancesView = {
		accounts: {
			'assets:trading-book': { kind: 'asset', segment: 'trading', normal_eur: 100_000, external_ref: 'ledger:paper' },
			'equity:contributed': { kind: 'equity', segment: 'treasury', normal_eur: 100_000, external_ref: null },
			'expense:infra:hetzner': { kind: 'expense', segment: 'infrastructure', normal_eur: 12.5, external_ref: 'invoice-1' },
			'expense:data:polygon': { kind: 'expense', segment: 'operations', normal_eur: 30, external_ref: null }
		},
		segment_funds_eur: { trading: 100_000 },
		trial_balance: 0,
		entries: 3
	};

	it('shows all five segments (absent = 0) and cumulative opex from expense accounts', () => {
		const m = moneyView(balances);
		expect(m.segments.map((s) => s.segment)).toEqual([...SEGMENTS]);
		expect(m.segments[0]).toEqual({ segment: 'trading', funds: 100_000 });
		expect(m.segments[1]).toEqual({ segment: 'infrastructure', funds: 0 });
		expect(m.expenses.map((e) => e.account)).toEqual(['expense:data:polygon', 'expense:infra:hetzner']);
		expect(m.opexTotal).toBeCloseTo(42.5);
		expect(m.trialBalance).toBe(0);
	});

	it('flags "all retained" only while every non-trading segment fund is absent or zero', () => {
		expect(moneyView(balances).allRetained).toBe(true);
		expect(moneyView({ ...balances, segment_funds_eur: { trading: 99_000, owner: 1_000 } }).allRetained).toBe(false);
		expect(moneyView({ ...balances, segment_funds_eur: {} }).allRetained).toBe(false);
	});
});

describe('formatting', () => {
	it('compact counts', () => {
		expect(fmtCount(0)).toBe('0');
		expect(fmtCount(7.25)).toBe('7.3');
		expect(fmtCount(1_500)).toBe('1.5k');
		expect(fmtCount(2_300_000)).toBe('2.3M');
		expect(fmtCount(3_400_000_000)).toBe('3.4G');
	});
});
