// Manual order ticket with a pre-submit risk check (spec 51 §6.5, spec 40 §6.1 step 5). The
// check is a dry run of `fa_risk::pretrade::check_order` against the risk-engine's book via
// `POST /api/trading/pretrade` (control-api, pending: see `PENDING_ROUTES` in openapi.test.ts);
// nothing is published. Submit is enabled only once the *current* ticket has a result: passed,
// or the check route is unavailable (the risk-engine still runs the same check on the intent).

export interface TicketDraft {
	instrument_id: string;
	side: 'buy' | 'sell';
	quantity: number;
	limit_price?: number | null;
}

/** `{ok, reasons}` from the pre-trade route (`reasons` empty when ok). */
export interface PretradeResult {
	ok: boolean;
	reasons: string[];
	[k: string]: unknown;
}

export type CheckState =
	| { kind: 'none' }
	| { kind: 'checking'; key: string }
	| { kind: 'passed' | 'rejected'; key: string; reasons: string[]; at: number }
	| { kind: 'unavailable'; key: string; message: string };

/** Identity of a ticket: any edit invalidates an earlier check. */
export function ticketKey(d: TicketDraft): string {
	return [d.instrument_id.trim(), d.side, Number(d.quantity), d.limit_price ? Number(d.limit_price) : ''].join('|');
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Client-side sanity before asking the API (same wording style as the risk-engine). */
export function localChecks(d: TicketDraft): string[] {
	const out: string[] = [];
	if (!UUID.test(d.instrument_id.trim())) out.push('instrument: an instrument UUID is required');
	const q = Number(d.quantity);
	if (!Number.isFinite(q) || q <= 0) out.push(`quantity ${d.quantity} is not a finite positive number`);
	if (d.limit_price !== null && d.limit_price !== undefined && String(d.limit_price) !== '') {
		const lp = Number(d.limit_price);
		if (!Number.isFinite(lp) || lp <= 0) out.push(`limit price ${d.limit_price} is not positive`);
	}
	return out;
}

/** Request body of the check (and of the paper order, minus `mode`). */
export function pretradeBody(d: TicketDraft): Record<string, unknown> {
	return {
		instrument_id: d.instrument_id.trim(),
		side: d.side,
		quantity: Number(d.quantity),
		...(d.limit_price ? { limit_price: Number(d.limit_price) } : {}),
		mode: 'paper'
	};
}

/** Status codes meaning "the check route is not there (yet)", not "the order is bad". */
export const UNAVAILABLE = [404, 405, 501, 502, 503];

/** Normalises the API reply (tolerates `reason` / `reasons` / `checks[].ok=false`). */
export function readResult(r: Partial<PretradeResult> & { reason?: string; checks?: { name?: string; ok?: boolean; detail?: string }[] }): { ok: boolean; reasons: string[] } {
	const reasons = [...(r.reasons ?? []), ...(r.reason ? [r.reason] : []), ...(r.checks ?? []).filter((c) => c.ok === false).map((c) => `${c.name ?? 'check'}: ${c.detail ?? 'failed'}`)];
	return { ok: r.ok === true && !reasons.length, reasons };
}

/** Whether Submit is allowed for the ticket `key`. */
export function canSubmit(s: CheckState, key: string): boolean {
	return (s.kind === 'passed' || s.kind === 'unavailable') && s.key === key;
}
