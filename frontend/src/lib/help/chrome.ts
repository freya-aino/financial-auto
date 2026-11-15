// Global chrome (+layout.svelte): the top bar everyone sees.
export const chrome: Record<string, string> = {
	'chrome.mode':
		'paper: orders fill on the simulated venue at real market prices; no real money moves. live trading is disabled by design until the owner enables it.',
	'chrome.equity':
		'Paper-book equity and day P&L from the live in-memory view, updated over WebSocket. The authoritative, exact numbers live in Trading → Ledger.',
	'chrome.kill':
		'Halts all order flow at the risk engine. Positions are not flattened automatically. Asks for confirmation; re-arm from Trading → Risk & halts.',
	'chrome.deps':
		'Reachability of each backing store (a fast TCP check only). Real usage and health live in Ops → Infra.',
	'chrome.halts': 'Active trading halts. Any halt blocks new orders in its scope until it is re-armed.',
};
