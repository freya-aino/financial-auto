// The real-money switch (D-121): the rules the UI applies before it will offer it, and the words
// it uses when it does.
//
// This logic lives here rather than in the page because it is the last thing between a click and
// real money, and it is worth testing. None of it is a substitute for the server's own gates — the
// risk engine checks the owner's numbers, a flat book, unfilled intents and halts; the broker
// gateway checks credentials — this is the second of two locks, and it exists so that the button
// is not offered when it cannot work and never offered without saying what will happen.
import type { TradingModeView } from './console';

/** Everything that currently prevents a switch to live, newest reason first. */
export function liveBlockers(m: TradingModeView | null): string[] {
	if (!m) return ['the trading mode is unknown: control-api did not answer'];
	const out = [...(m.switch_blockers ?? [])];
	if (!m.live_permitted && m.live_blocked_by && !out.includes(m.live_blocked_by)) out.unshift(m.live_blocked_by);
	if (m.venues_ready === false) {
		const missing = (m.venues ?? []).filter((v) => !v.configured).map((v) => v.venue);
		out.push(
			missing.length > 0
				? `no credentials for ${missing.join(', ')}: enter them under [venues.<id>] in config/secrets.toml`
				: 'no execution venue has credentials'
		);
	}
	if (m.venues_ready === null) out.push(m.venues_error ?? 'the broker-gateway did not report its venues');
	return out;
}

/** Whether the switch to live may be offered at all. */
export function canGoLive(m: TradingModeView | null): boolean {
	return !!m && m.mode === 'paper' && liveBlockers(m).length === 0;
}

/**
 * Whether the switch back to paper may be offered.
 *
 * Stopping is never blocked by a venue — only by the risk engine's own rules, and the flat-book
 * rule among them: a real position must not become a paper one by a button press.
 */
export function canGoPaper(m: TradingModeView | null): boolean {
	return !!m && m.mode === 'live' && (m.switch_blockers ?? []).length === 0;
}

/** What the operator confirms. It names the sleeve, because that is the amount at risk. */
export function switchConfirmText(m: TradingModeView, to: 'live' | 'paper'): string {
	if (to === 'paper') {
		return 'Switch back to paper money? New intents will be paper from that moment; the book must already be flat.';
	}
	return [
		`Switch this platform to REAL MONEY?`,
		`At most ${eurText(m.sleeve_eur)} may be deployed, and no single order may exceed ${eurText(m.max_order_eur)}.`,
		`Orders will be sent to a live venue and filled with real money. This is written to the audit log.`
	].join('\n\n');
}

/** A short line for the status badge. */
export function modeLabel(m: TradingModeView | null): string {
	if (!m) return 'unknown';
	if (m.mode === 'live') return `REAL MONEY · sleeve ${eurText(m.sleeve_eur)} · ${eurText(m.max_order_eur)} per order`;
	return m.live_permitted ? 'paper (live permitted by risk.toml)' : 'paper';
}

/** Venues as one line each: whether they could take an order, and what is missing when not. */
export function venueLines(m: TradingModeView | null): string[] {
	if (!m?.venues) return [];
	return m.venues.map((v) =>
		v.configured
			? `${v.venue}: ready${v.open_orders ? ` · ${v.open_orders} open order(s)` : ''}`
			: `${v.venue}: ${v.detail ?? 'no credentials'}`
	);
}

function eurText(v: number): string {
	return `${(v ?? 0).toLocaleString('en-GB', { maximumFractionDigits: 2 })} EUR`;
}
