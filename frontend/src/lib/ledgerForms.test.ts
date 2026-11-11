import { describe, expect, it } from 'vitest';
import { canBook, orphanConfirmText, orphanResolveBody, orphanRow, payoutBody, payoutConfirmText } from './ledgerForms';

const raw = {
	id: 'o1',
	book: 'paper',
	venue_fill_id: 'vf-9',
	client_order_id: 'co-1',
	instrument: '0190-x',
	instrument_status: 'resolved',
	venue_symbol: 'XBTEUR',
	qty: '-0.5',
	price: '61234.5',
	ccy: 'EUR',
	status: 'open'
};

describe('orphan resolution form', () => {
	it('builds book and dismiss bodies; dismiss needs a reason', () => {
		expect(orphanResolveBody('book', '')).toEqual({ body: { action: 'book' } });
		expect('error' in orphanResolveBody('dismiss', ' x ')).toBe(true);
		expect(orphanResolveBody('dismiss', ' duplicate of vf-8 ')).toEqual({ body: { action: 'dismiss', reason: 'duplicate of vf-8' } });
	});

	it('only offers booking when the instrument is known', () => {
		const o = orphanRow(raw);
		expect(canBook(o)).toBe(true);
		expect(canBook(orphanRow({ ...raw, instrument: null, instrument_status: 'unresolved' }))).toBe(false);
		expect(canBook(orphanRow({ ...raw, status: 'booked' }))).toBe(false);
	});

	it('names the fill and the consequence in the confirmation', () => {
		const o = orphanRow(raw);
		expect(orphanConfirmText(o, { action: 'book' })).toContain('vf-9');
		expect(orphanConfirmText(o, { action: 'book' })).toContain('writes a fill');
		expect(orphanConfirmText(o, { action: 'dismiss', reason: 'dup' })).toContain('"dup"');
	});
});

describe('payout form', () => {
	const today = '2026-11-11';
	it('validates day and amount', () => {
		expect(payoutBody('', '2026-11-10', '12,345', { today })).toEqual({ body: { book: 'paper', day: '2026-11-10', amount_eur: 12.35 } });
		expect('error' in payoutBody('paper', '11/10/2026', 5, { today })).toBe(true);
		expect('error' in payoutBody('paper', '2026-11-12', 5, { today })).toBe(true);
		expect('error' in payoutBody('paper', today, 0, { today })).toBe(true);
		expect('error' in payoutBody('paper', today, 'abc', { today })).toBe(true);
		expect('error' in payoutBody('paper', today, 101, { today, max: 100 })).toBe(true);
		expect('body' in payoutBody('paper', today, 100, { today, max: 100 })).toBe(true);
	});

	it('states the amount, book and day before booking', () => {
		const t = payoutConfirmText({ book: 'paper', day: today, amount_eur: 50 });
		expect(t).toContain('50.00 EUR');
		expect(t).toContain('paper');
		expect(t).toContain(today);
	});
});
