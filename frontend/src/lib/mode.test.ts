import { describe, expect, it } from 'vitest';
import { canGoLive, canGoPaper, liveBlockers, modeLabel, switchConfirmText, venueLines, venueReconLine } from './mode';
import type { TradingModeView } from './console';

const view = (over: Partial<TradingModeView> = {}): TradingModeView => ({
	mode: 'paper',
	live_permitted: true,
	live_blocked_by: null,
	sleeve_eur: 500,
	max_order_eur: 50,
	deployable_eur: 500,
	switch_blockers: [],
	venue_reconciliation: { status: 'paper: no venue holds this book' },
	venues: [{ venue: 'kraken', configured: true, open_orders: 0 }],
	venues_ready: true,
	...over
});

describe('the real-money switch', () => {
	it('is offered only when nothing blocks it', () => {
		expect(canGoLive(view())).toBe(true);
		expect(canGoLive(null)).toBe(false);
		expect(canGoLive(view({ mode: 'live' }))).toBe(false);
		expect(canGoLive(view({ switch_blockers: ['the book is not flat: X +1'] }))).toBe(false);
		expect(canGoLive(view({ venues_ready: false, venues: [{ venue: 'kraken', configured: false, open_orders: 0, detail: 'kraken: missing credentials: api_key' }] }))).toBe(
			false
		);
		expect(canGoLive(view({ live_permitted: false, live_blocked_by: '[limits.live] sleeve_eur must be positive' }))).toBe(false);
	});

	it('names every blocker, including the ones the server would also refuse on', () => {
		const m = view({
			live_permitted: false,
			live_blocked_by: '[limits.live] sleeve_eur must be a positive number of EUR',
			switch_blockers: ['halts are active: portfolio'],
			venues_ready: false,
			venues: [
				{ venue: 'kraken', configured: false, open_orders: 0, detail: 'kraken: missing credentials: api_key' },
				{ venue: 'lemon', configured: false, open_orders: 0 }
			]
		});
		const b = liveBlockers(m);
		expect(b[0]).toContain('sleeve_eur');
		expect(b.join(' ')).toContain('halts are active');
		expect(b.join(' ')).toContain('kraken, lemon');
		// A blocker the server already reported is not repeated.
		expect(liveBlockers(view({ live_permitted: false, live_blocked_by: 'x', switch_blockers: ['x'] }))).toEqual(['x']);
		// Not knowing is itself a blocker: silence must not read as "ready".
		expect(liveBlockers(null)[0]).toContain('unknown');
		expect(liveBlockers(view({ venues_ready: null, venues_error: 'gateway down' }))).toContain('gateway down');
	});

	it('lets you stop without asking a venue, but not with an open book', () => {
		expect(canGoPaper(view({ mode: 'live' }))).toBe(true);
		expect(canGoPaper(view({ mode: 'live', venues_ready: false }))).toBe(true);
		expect(canGoPaper(view({ mode: 'live', switch_blockers: ['the book is not flat: X +1'] }))).toBe(false);
		expect(canGoPaper(view())).toBe(false);
	});

	it('says what is at risk before it is', () => {
		const t = switchConfirmText(view(), 'live');
		expect(t).toContain('REAL MONEY');
		expect(t).toContain('500 EUR');
		expect(t).toContain('50 EUR');
		expect(t).toContain('audit log');
		expect(switchConfirmText(view({ mode: 'live' }), 'paper')).toContain('flat');
	});

	it('labels the state the way the header shows it', () => {
		expect(modeLabel(view({ mode: 'live' }))).toContain('REAL MONEY');
		expect(modeLabel(view({ mode: 'live' }))).toContain('sleeve 500 EUR');
		expect(modeLabel(view())).toContain('live permitted');
		expect(modeLabel(view({ live_permitted: false }))).toBe('paper');
		expect(modeLabel(null)).toBe('unknown');
		expect(venueLines(view())).toEqual(['kraken: ready']);
		expect(venueLines(view({ venues: [{ venue: 'oanda', configured: false, open_orders: 0, detail: 'oanda: missing credentials: account_id' }] }))[0]).toContain(
			'account_id'
		);
		expect(venueLines(view({ venues: [{ venue: 'kraken', configured: true, open_orders: 2 }] }))[0]).toContain('2 open order(s)');
	});
});

describe('the venue reconciliation line', () => {
	it('never lets an incomplete pass read as a clean one', () => {
		expect(venueReconLine(view({ venue_reconciliation: { status: 'ok', compared: 2 } }))).toContain('2 position(s) agree');
		// a pass that could not ask a venue proves nothing, and must not be silent
		const incomplete = venueReconLine(
			view({ venue_reconciliation: { status: 'incomplete', unavailable: [{ venue: 'lemon', why: 'did not answer' }] } })
		);
		expect(incomplete).toContain('not reconciled');
		expect(incomplete).toContain('lemon did not answer');
		expect(venueReconLine(view({ venue_reconciliation: { status: 'unreachable', detail: 'no answer in time' } }))).toContain(
			'no answer in time'
		);
		expect(venueReconLine(view({ venue_reconciliation: { status: 'diff', diffs: [{}, {}] } }))).toContain('halted');
		// holdings outside the symbol mapping are reported, not treated as the platform's
		expect(
			venueReconLine(view({ venue_reconciliation: { status: 'ok', compared: 1, foreign: [{ symbol: 'DOGEEUR' }] } }))
		).toContain('outside the symbol mapping');
		expect(venueReconLine(view({ venue_reconciliation: { status: 'pending' } }))).toContain('not reconciled yet');
	});
});
