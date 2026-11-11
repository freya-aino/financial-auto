// Operator forms of the ledger page (spec 42 §8 orphan fills, §6 payouts; D-072): validation,
// request bodies and the confirmation text. The page asks `confirm()` with the text and only
// then posts; nothing here talks to the API, so the logic is tested without touching a book.

/** `POST /api/trading/orphans/{id}/resolve` body (`fa_ledger::OrphanAction`, tag `action`). */
export type OrphanResolve = { action: 'book' } | { action: 'dismiss'; reason: string };

export interface OrphanRow {
	id: string;
	book: string | null;
	venue_fill_id: string;
	client_order_id: string;
	instrument: string | null;
	instrument_status: string;
	venue_symbol: string | null;
	qty: string;
	price: string;
	ccy: string | null;
	status: string;
}

const str = (v: unknown) => (v === null || v === undefined ? '' : String(v));

/** Normalises a `GET /api/trading/orphans` row (money fields arrive as decimal strings). */
export function orphanRow(o: Record<string, unknown>): OrphanRow {
	return {
		id: str(o.id),
		book: o.book === null || o.book === undefined ? null : str(o.book),
		venue_fill_id: str(o.venue_fill_id ?? o.fill_id),
		client_order_id: str(o.client_order_id),
		instrument: o.instrument ? str(o.instrument) : o.instrument_id ? str(o.instrument_id) : null,
		instrument_status: str(o.instrument_status || 'resolved'),
		venue_symbol: o.venue_symbol ? str(o.venue_symbol) : null,
		qty: str(o.qty ?? o.quantity),
		price: str(o.price),
		ccy: o.ccy ? str(o.ccy) : null,
		status: str(o.status || 'open')
	};
}

/** Body of a resolution, or the reason it cannot be sent. */
export function orphanResolveBody(action: 'book' | 'dismiss', reason: string): { body: OrphanResolve } | { error: string } {
	if (action === 'book') return { body: { action: 'book' } };
	const r = reason.trim();
	if (r.length < 3) return { error: 'a dismissal needs a reason (≥ 3 characters, kept in the audit log)' };
	return { body: { action: 'dismiss', reason: r } };
}

/** Whether booking can work: the ledger books an orphan only once its instrument is known. */
export function canBook(o: OrphanRow): boolean {
	return o.status === 'open' && !!o.instrument && o.instrument_status === 'resolved';
}

export function orphanConfirmText(o: OrphanRow, body: OrphanResolve): string {
	const what = `orphan fill ${o.venue_fill_id || o.id} (${o.qty} @ ${o.price}${o.ccy ? ` ${o.ccy}` : ''}, order ${o.client_order_id || '?'}) in book ${o.book ?? 'paper'}`;
	return body.action === 'book'
		? `Book ${what} into the ledger? This writes a fill (cash, position, fees) and is audited.`
		: `Dismiss ${what}? It stays out of the ledger with the reason "${body.reason}" (audited).`;
}

/** `POST /api/trading/payouts` body (`{book?, day, amount_eur}`). */
export interface PayoutBody {
	book: string;
	day: string;
	amount_eur: number;
}

/** Validates the payout form: ISO day not in the future, positive finite EUR amount ≤ `max` (when known). */
export function payoutBody(book: string, day: string, amount: string | number, opts: { today?: string; max?: number | null } = {}): { body: PayoutBody } | { error: string } {
	const b = book.trim() || 'paper';
	if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || Number.isNaN(Date.parse(`${day}T00:00:00Z`))) return { error: 'day must be YYYY-MM-DD' };
	const today = opts.today ?? new Date().toISOString().slice(0, 10);
	if (day > today) return { error: `day ${day} is in the future` };
	const a = typeof amount === 'number' ? amount : Number(String(amount).replace(',', '.').trim());
	if (!Number.isFinite(a) || a <= 0) return { error: 'amount must be a positive number of EUR' };
	if (opts.max !== null && opts.max !== undefined && a > opts.max + 1e-9) return { error: `amount ${a.toFixed(2)} EUR exceeds the open quota ${opts.max.toFixed(2)} EUR` };
	return { body: { book: b, day, amount_eur: Math.round(a * 100) / 100 } };
}

export function payoutConfirmText(p: PayoutBody): string {
	return `Book a payout of ${p.amount_eur.toFixed(2)} EUR from book ${p.book} on ${p.day}? This is a cash movement out of the book (audited as ledger.payout) and cannot be undone from the console.`;
}
