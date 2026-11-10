import { describe, expect, it } from 'vitest';
import type { Limits } from './api';
import { diffLimits, limitsBody, loosening, parseLoosenError } from './limits';

const current: Limits = {
	max_drawdown: 0.1,
	max_daily_loss: 0.02,
	max_position_frac: 0.05,
	liquidity_k: 0.01,
	daily_payout_quota_eur: 0,
	leverage_cap: { equity: 1.5, fx: 3 }
};

describe('limits', () => {
	it('classifies smaller caps as tightening and larger as loosening', () => {
		const d = diffLimits(current, { max_drawdown: 0.05, max_daily_loss: 0.03, leverage_cap: { fx: 2, equity: 2 } });
		expect(d.map((c) => [c.field, c.kind])).toEqual([
			['max_drawdown', 'tighten'],
			['max_daily_loss', 'loosen'],
			['leverage_cap.fx', 'tighten'],
			['leverage_cap.equity', 'loosen']
		]);
		expect(loosening(d)).toEqual(['max_daily_loss', 'leverage_cap.equity']);
	});

	it('treats liquidity_k the other way round and payout quota as free', () => {
		expect(diffLimits(current, { liquidity_k: 0.02 })[0].kind).toBe('tighten');
		expect(diffLimits(current, { liquidity_k: 0.005 })[0].kind).toBe('loosen');
		expect(diffLimits(current, { daily_payout_quota_eur: 500 })[0].kind).toBe('tighten');
		expect(diffLimits(current, { max_drawdown: 0.1 })[0].kind).toBe('same');
		expect(diffLimits(current, { brand_new: 1 })).toEqual([]);
	});

	it('builds a body with only the changed fields', () => {
		const draft: Limits = { ...current, max_drawdown: 0.08, leverage_cap: { equity: 1.5, fx: 2 } };
		expect(limitsBody(current, draft)).toEqual({ max_drawdown: 0.08, leverage_cap: { fx: 2 } });
		expect(limitsBody(current, { ...current })).toEqual({});
	});

	it('extracts the loosen message from a 422 reply', () => {
		expect(parseLoosenError('loosen: max_drawdown: 0.1 → 0.5 would loosen')).toBe('max_drawdown: 0.1 → 0.5 would loosen');
		expect(parseLoosenError('invalid: unknown field')).toBeNull();
	});
});
