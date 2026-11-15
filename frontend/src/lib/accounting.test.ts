import { describe, expect, it } from 'vitest';
import { closeConfirmText, closeDayBody, groupByKind, segmentFunds, trialBalanceOk, yesterdayUtc, type AccountBalance } from './accounting';

const accounts: Record<string, AccountBalance> = {
	'assets:trading-book': { normal_eur: 100000, kind: 'asset', segment: 'trading', external_ref: 'ledger:paper' },
	'equity:contributed': { normal_eur: 100000, kind: 'equity', segment: 'treasury', external_ref: null },
	'expense:infra': { normal_eur: 12.5, kind: 'expense', segment: 'infrastructure', external_ref: null },
	'assets:bank': { normal_eur: 50, kind: 'asset', segment: 'treasury', external_ref: 'bank:main' },
	'other:odd': { normal_eur: 1, kind: 'suspense', segment: 'operations', external_ref: null }
};

describe('organisational books', () => {
	it('lists the five segments in order, zero-filled, unknown ones appended', () => {
		const rows = segmentFunds({ trading: 100000, growth: 7 });
		expect(rows.map((r) => r.segment)).toEqual(['trading', 'infrastructure', 'operations', 'owner', 'treasury', 'growth']);
		expect(rows[0].eur).toBe(100000);
		expect(rows[1].eur).toBe(0);
		expect(rows[5].eur).toBe(7);
	});

	it('groups accounts by kind in balance-sheet order, names sorted, unknown kinds last', () => {
		const groups = groupByKind(accounts);
		expect(groups.map((g) => g.kind)).toEqual(['asset', 'equity', 'expense', 'suspense']);
		expect(groups[0].accounts.map((a) => a.name)).toEqual(['assets:bank', 'assets:trading-book']);
		expect(groups[0].accounts[1].external_ref).toBe('ledger:paper');
		expect(groupByKind({})).toEqual([]);
	});

	it('flags a non-zero trial balance as a broken invariant', () => {
		expect(trialBalanceOk(0)).toBe(true);
		expect(trialBalanceOk(1e-9)).toBe(true);
		expect(trialBalanceOk(0.01)).toBe(false);
		expect(trialBalanceOk(-3)).toBe(false);
		expect(trialBalanceOk(NaN)).toBe(false);
	});
});

describe('close-day form', () => {
	const today = '2026-11-15';

	it('defaults to yesterday UTC', () => {
		expect(yesterdayUtc(new Date('2026-11-15T00:30:00Z'))).toBe('2026-11-14');
		expect(yesterdayUtc(new Date('2026-01-01T12:00:00Z'))).toBe('2025-12-31');
	});

	it('accepts only a finished UTC day', () => {
		expect(closeDayBody(' 2026-11-14 ', { today })).toEqual({ body: { day: '2026-11-14' } });
		expect('error' in closeDayBody('14/11/2026', { today })).toBe(true);
		expect('error' in closeDayBody('2026-13-40', { today })).toBe(true);
		expect('error' in closeDayBody(today, { today })).toBe(true);
		expect('error' in closeDayBody('2026-11-16', { today })).toBe(true);
	});

	it('names the day and what the close books', () => {
		const t = closeConfirmText('2026-11-14');
		expect(t).toContain('2026-11-14');
		expect(t).toContain('revenue');
		expect(t).toContain('opex');
		expect(t).toContain('owner / infra / reserves / retained');
		expect(t).toContain('idempotent');
	});
});
