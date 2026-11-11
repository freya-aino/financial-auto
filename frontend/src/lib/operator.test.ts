import { describe, expect, it } from 'vitest';
import { contractSpecBody, digestChanges, holdoutLockBody, identifierBody, restatementConfirmText, restatementRequest, sameRequest, summaryMatrix } from './operator';

const id = '0b6f8c1e-2d3a-4b5c-8d9e-0f1a2b3c4d5e';

describe('ops history helpers', () => {
	it('builds a kind × status matrix', () => {
		const m = summaryMatrix([
			{ kind: 'evo.eval.strategy', status: 'ok', count: 5 },
			{ kind: 'evo.eval.strategy', status: 'failed', count: 1 },
			{ kind: 'ingest.backfill', status: 'weird', count: 2 }
		]);
		expect(m.statuses.slice(-1)).toEqual(['weird']);
		expect(m.kinds[0]).toMatchObject({ kind: 'evo.eval.strategy', total: 6 });
		expect(m.kinds[1].counts.weird).toBe(2);
	});
	it('links each config snapshot to the previous digest of its section', () => {
		const rows = [
			{ id: 2, service: 'risk', section: 'runtime', digest: 'b', actor: 'a', taken_at: '2026-11-11T10:05:00Z' },
			{ id: 1, service: 'risk', section: 'runtime', digest: 'a', actor: 'a', taken_at: '2026-11-11T10:00:00Z' }
		];
		const out = digestChanges(rows);
		expect(out[0]).toMatchObject({ id: 2, previous: 'a' });
		expect(out[1].previous).toBeNull();
	});
});

describe('operator forms', () => {
	it('validates a holdout lock', () => {
		const now = Date.parse('2026-11-11T12:00:00Z');
		expect('error' in holdoutLockBody('5m', '2026-10-01T00:00', '2026-10-02T00:00', '', now)).toBe(true);
		expect('error' in holdoutLockBody('1h', '2026-10-02T00:00', '2026-10-01T00:00', '', now)).toBe(true);
		expect('error' in holdoutLockBody('1h', '2026-10-01T00:00', '2026-12-01T00:00', '', now)).toBe(true);
		const ok = holdoutLockBody('1h', '2026-10-01T00:00', '2026-10-08T00:00', ' oct ', now);
		expect('body' in ok && ok.body.note).toBe('oct');
	});
	it('validates identifier and contract-spec overrides', () => {
		expect('error' in identifierBody('x', 'isin', 'US0378331005', '')).toBe(true);
		const i = identifierBody(id, 'ISIN', ' US0378331005 ', '');
		expect('body' in i && i.body).toEqual({ instrument_id: id, scheme: 'isin', value: 'US0378331005' });
		expect('error' in contractSpecBody({ instrument_id: id, venue: '' })).toBe(true);
		expect('error' in contractSpecBody({ instrument_id: id, venue: 'XNAS', tick_size: '-1' })).toBe(true);
		const c = contractSpecBody({ instrument_id: id, venue: 'XNAS', tick_size: '0.01', lot_size: '', settlement_ccy: 'usd', shortable: 'false' });
		expect('body' in c && c.body).toEqual({ instrument_id: id, venue: 'XNAS', tick_size: 0.01, settlement_ccy: 'USD', shortable: false });
	});
	it('restatement: request validation, apply only for the shown plan, explicit confirm text', () => {
		expect('error' in restatementRequest({ book: 'paper', instrument: 'x', ccy: '', from_ccy: '', check_hours: '' })).toBe(true);
		const r = restatementRequest({ book: 'paper', instrument: id, ccy: 'usd', from_ccy: '', check_hours: '24' });
		expect('req' in r && r.req).toEqual({ book: 'paper', instrument: id, ccy: 'USD', check_hours: 24 });
		const req = 'req' in r ? r.req : null;
		expect(sameRequest(req, req ? { ...req } : null)).toBe(true);
		expect(sameRequest(req, req ? { ...req, ccy: 'GBP' } : null)).toBe(false);
		const text = restatementConfirmText(
			{ dry_run: true, fills_total: 3, applied: null, plan: { book: 'paper', instrument: id, from_ccy: 'EUR', ccy: 'USD', planned_at: '', fills: [], missing_rate: [], by_old_source: {}, cash_delta_eur: '-1.5', before: {}, after: {}, check_from: '', notes: [] } },
			'quote ccy fix'
		);
		expect(text).toContain('APPLY');
		expect(text).toContain('-1.50 EUR');
		expect(text).toContain('restarted');
	});
});
