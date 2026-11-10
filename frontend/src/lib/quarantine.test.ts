import { describe, expect, it } from 'vitest';
import { emptyResolveForm, isinChecksumOk, validateResolve } from './quarantine';

describe('quarantine resolve form', () => {
	it('validates ISIN check digits', () => {
		expect(isinChecksumOk('US0378331005')).toBe(true); // Apple
		expect(isinChecksumOk('DE0007164600')).toBe(true); // SAP
		expect(isinChecksumOk('US0378331006')).toBe(false);
		expect(isinChecksumOk('US037833100')).toBe(false);
	});

	it('requires asset class and kind to activate and normalises identifiers', () => {
		const f = { ...emptyResolveForm(), isin: ' us0378331005 ', asset_class: 'equity', kind: 'equity', base_ccy: 'usd' };
		const v = validateResolve(f);
		expect(v.errors).toEqual([]);
		expect(v.body).toEqual({ asset_class: 'equity', kind: 'equity', isin: 'US0378331005', base_ccy: 'USD' });
		const missing = validateResolve({ ...emptyResolveForm(), isin: 'XX' });
		expect(missing.body).toBeNull();
		expect(missing.errors.length).toBe(3);
	});

	it('rejects bad FIGI and currencies', () => {
		const v = validateResolve({ ...emptyResolveForm(), figi: 'nope', asset_class: 'fx', kind: 'fx_pair', quote_ccy: 'EURO' });
		expect(v.errors).toHaveLength(2);
	});

	it('merge needs a UUID target, ignore needs nothing', () => {
		expect(validateResolve({ ...emptyResolveForm(), mode: 'merge', merge_into: 'abc' }).body).toBeNull();
		expect(validateResolve({ ...emptyResolveForm(), mode: 'merge', merge_into: '92AB6643-9787-5E8C-9D02-A6AF84B38C02' }).body).toEqual({
			instrument_id: '92ab6643-9787-5e8c-9d02-a6af84b38c02'
		});
		expect(validateResolve({ ...emptyResolveForm(), mode: 'ignore' }).body).toEqual({ ignore: true });
	});
});
