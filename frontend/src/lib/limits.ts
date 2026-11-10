// Risk limits form logic (spec 40 §6.5, D-051): PUT /api/trading/limits may only tighten.
// The server is authoritative (422 `loosen: <fields>`); this mirrors `RiskConfig::check_tightens`
// so the UI can warn before sending and explain the reply.
import type { Limits } from './api';

/** Fields where a smaller value is tighter (fractions / caps / multipliers). */
export const TIGHTER_WHEN_SMALLER = new Set([
	'max_drawdown',
	'max_daily_loss',
	'max_position_frac',
	'max_class_frac',
	'max_venue_frac',
	'portfolio_gross_cap',
	'strategy_max_drawdown',
	'leverage_cap'
]);

/** Fields where a larger value is tighter (`liquidity_k` penalises more impact). */
export const TIGHTER_WHEN_LARGER = new Set(['liquidity_k']);

/** Fields that are neither (operator preferences the server accepts in either direction). */
export const FREE_FIELDS = new Set(['daily_payout_quota_eur']);

export interface LimitChange {
	field: string;
	from: number;
	to: number;
	/** `tighten | loosen | same | unknown` (unknown field → server decides). */
	kind: 'tighten' | 'loosen' | 'same' | 'unknown';
}

/** Compares a proposed partial limits object against the current one, per scalar field. */
export function diffLimits(current: Limits, proposed: Record<string, unknown>): LimitChange[] {
	const out: LimitChange[] = [];
	for (const [field, value] of Object.entries(proposed)) {
		if (field === 'leverage_cap' && value && typeof value === 'object') {
			for (const [cls, to] of Object.entries(value as Record<string, unknown>)) {
				const from = current.leverage_cap?.[cls];
				if (typeof to !== 'number' || typeof from !== 'number') continue;
				out.push({ field: `leverage_cap.${cls}`, from, to, kind: classify('leverage_cap', from, to) });
			}
			continue;
		}
		const from = current[field];
		if (typeof value !== 'number' || typeof from !== 'number') continue;
		out.push({ field, from, to: value, kind: classify(field, from, value) });
	}
	return out;
}

function classify(field: string, from: number, to: number): LimitChange['kind'] {
	if (from === to) return 'same';
	if (FREE_FIELDS.has(field)) return 'tighten';
	if (TIGHTER_WHEN_SMALLER.has(field)) return to < from ? 'tighten' : 'loosen';
	if (TIGHTER_WHEN_LARGER.has(field)) return to > from ? 'tighten' : 'loosen';
	return 'unknown';
}

/** Only the changed fields, as the body for PUT (unchanged values are left out). */
export function limitsBody(current: Limits, draft: Limits): Record<string, unknown> {
	const body: Record<string, unknown> = {};
	for (const [k, v] of Object.entries(draft)) {
		if (k === 'leverage_cap') {
			const lc: Record<string, number> = {};
			for (const [cls, cap] of Object.entries((v as Record<string, number>) ?? {})) {
				if (typeof cap === 'number' && cap !== current.leverage_cap?.[cls]) lc[cls] = cap;
			}
			if (Object.keys(lc).length) body.leverage_cap = lc;
			continue;
		}
		if (typeof v === 'number' && Number.isFinite(v) && v !== current[k]) body[k] = v;
	}
	return body;
}

/** Fields the server would refuse (`loosen:`) — shown before sending. */
export function loosening(changes: LimitChange[]): string[] {
	return changes.filter((c) => c.kind === 'loosen').map((c) => c.field);
}

/** Extracts the `loosen: …` message of a 422 reply, or null for any other error. */
export function parseLoosenError(message: string): string | null {
	const m = /loosen:\s*(.*)$/s.exec(message);
	return m ? m[1].trim() : null;
}
