import { describe, expect, it } from 'vitest';
import { canSubmit, localChecks, pretradeBody, readResult, ticketKey } from './ticket';

const id = '0b6f8c1e-2d3a-4b5c-8d9e-0f1a2b3c4d5e';

describe('order ticket pre-trade check', () => {
	it('validates locally', () => {
		expect(localChecks({ instrument_id: id, side: 'buy', quantity: 1 })).toEqual([]);
		expect(localChecks({ instrument_id: 'x', side: 'buy', quantity: 0 })).toHaveLength(2);
		expect(localChecks({ instrument_id: id, side: 'sell', quantity: 1, limit_price: -1 })[0]).toContain('limit');
	});
	it('a result only counts for the ticket it was made for', () => {
		const d = { instrument_id: id, side: 'buy' as const, quantity: 2 };
		const k = ticketKey(d);
		expect(canSubmit({ kind: 'passed', key: k, reasons: [], at: 0 }, k)).toBe(true);
		expect(canSubmit({ kind: 'passed', key: k, reasons: [], at: 0 }, ticketKey({ ...d, quantity: 3 }))).toBe(false);
		expect(canSubmit({ kind: 'rejected', key: k, reasons: ['x'], at: 0 }, k)).toBe(false);
		expect(canSubmit({ kind: 'unavailable', key: k, message: '404' }, k)).toBe(true);
		expect(canSubmit({ kind: 'none' }, k)).toBe(false);
	});
	it('reads {ok, reasons} and tolerant variants', () => {
		expect(readResult({ ok: true, reasons: [] })).toEqual({ ok: true, reasons: [] });
		expect(readResult({ ok: false, reasons: ['pre-trade: order notional 9000 EUR > max_order_frac'] }).ok).toBe(false);
		expect(readResult({ ok: true, checks: [{ name: 'band', ok: false, detail: '12% away' }] })).toEqual({ ok: false, reasons: ['band: 12% away'] });
		expect(readResult({ ok: false, reason: 'no mark' }).reasons).toEqual(['no mark']);
	});
	it('builds the body', () => {
		expect(pretradeBody({ instrument_id: ` ${id} `, side: 'sell', quantity: 2, limit_price: null })).toEqual({ instrument_id: id, side: 'sell', quantity: 2, mode: 'paper' });
	});
});
