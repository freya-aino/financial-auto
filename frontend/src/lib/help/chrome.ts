// Global chrome (+layout.svelte): the top bar everyone sees.
export const chrome: Record<string, string> = {
	'chrome.mode':
		'The mode the risk engine reports, never a constant. PAPER: fills on the simulated venue at real prices, no real money. REAL MONEY: intents go to a venue, capped by the sleeve in risk.toml. UNKNOWN: not yet read. The switch is on Trading → Risk & halts.',
	'chrome.equity':
		'Paper-book equity and day P&L from the live in-memory view, updated over WebSocket. The authoritative, exact numbers live in Trading → Ledger.',
	'chrome.kill':
		'Halts all order flow at the risk engine. Positions are not flattened automatically. Asks for confirmation; re-arm from Trading → Risk & halts.',
	'chrome.deps':
		'Reachability of each backing store (a fast TCP check only). Real usage and health live in Ops → Infra.',
	'chrome.halts': 'Active trading halts. Any halt blocks new orders in its scope until it is re-armed.',
};
