import { describe, expect, it } from 'vitest';
import { SubjectRates, sparkPoints, subjectKey } from './busrates';

describe('per-subject bus rates', () => {
	it('folds subjects to a token depth', () => {
		expect(subjectKey('md.trade.binance.btc', 2)).toBe('md.trade.>');
		expect(subjectKey('md.trade', 2)).toBe('md.trade');
		expect(subjectKey('a.b.c', 0)).toBe('a.b.c');
	});
	it('counts per second, ranks by window mean and drops idle keys', () => {
		const r = new SubjectRates(3, 2);
		for (let i = 0; i < 5; i++) r.hit('md.trade.x');
		r.hit('svc.heartbeat.a');
		r.tick();
		r.hit('svc.heartbeat.b');
		r.tick();
		const top = r.top();
		expect(top[0].key).toBe('md.trade.>');
		expect(top[0].series).toEqual([0, 5, 0]);
		expect(top[0].last).toBe(0);
		expect(top[1]).toMatchObject({ key: 'svc.heartbeat.>', total: 2, last: 1 });
		r.tick();
		r.tick();
		r.tick();
		expect(r.size).toBe(0);
	});
	it('caps tracked keys', () => {
		const r = new SubjectRates(5, 0, 2);
		['a', 'b', 'c'].forEach((s) => r.hit(s));
		r.tick();
		expect(r.size).toBe(2);
	});
	it('scales spark points to the box', () => {
		expect(sparkPoints([0, 2], 10, 10)).toBe('0.0,9.0 10.0,1.0');
		expect(sparkPoints([], 10, 10)).toBe('');
	});
});
